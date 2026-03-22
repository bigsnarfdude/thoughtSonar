#!/usr/bin/env python3
"""Train a sandbagging-specific probe on MALT honest vs generated deceptive traces.

Extracts layer 40 activations, trains probes on:
  1. Raw residual stream (5376-dim) — ceiling
  2. Full SAE features (16384-dim) — interpretable
  3. Data-driven SAE feature selection — sparse + interpretable

Cross-validated AUROC for each.

Usage:
  python3 train_sandbagging_probe.py --honest honest_control.jsonl --deceptive gemini_600.jsonl
"""

import argparse, json, os, sys, time
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

MODEL_NAME = "google/gemma-3-27b-it"
SAE_REPO = "google/gemma-scope-2-27b-it"
LAYER = 40
PROBE_DIR = Path(__file__).parent / "probe_weights"


def load_model_and_sae():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    if hasattr(model.model, 'language_model'):
        target_layer = model.model.language_model.layers[LAYER]
    else:
        target_layer = model.model.layers[LAYER]

    print(f"Loading GemmaScope SAE (layer {LAYER})...")
    folder = f"resid_post/layer_{LAYER}_width_16k_l0_medium"
    sae_path = hf_hub_download(repo_id=SAE_REPO, filename=f"{folder}/params.safetensors")
    sae_params = load_file(sae_path)
    W_enc = sae_params["w_enc"].to(torch.float32).cuda()
    b_enc = sae_params["b_enc"].to(torch.float32).cuda()
    threshold = sae_params.get("threshold", None)
    if threshold is not None:
        threshold = threshold.to(torch.float32).cuda()

    print(f"Model + SAE loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, tokenizer, target_layer, W_enc, b_enc, threshold


def extract_activations(model, tokenizer, target_layer, W_enc, b_enc, threshold, texts):
    """Extract raw residual + SAE activations for a list of texts."""
    raw_acts_list = []
    sae_acts_list = []

    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"  Extracting {i}/{len(texts)}...", flush=True)

        captured = {}
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured["hidden"] = output[0].detach()
            else:
                captured["hidden"] = output.detach()

        handle = target_layer.register_forward_hook(hook)
        inputs = tokenizer(text[:3000], return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            model(**inputs)
        handle.remove()

        hidden = captured["hidden"].to(torch.float32).cuda()

        # Raw residual: max-pool over tokens
        raw = hidden[0].max(dim=0).values.cpu().numpy()

        # SAE encode: max-pool over tokens
        pre_acts = hidden @ W_enc + b_enc.unsqueeze(0).unsqueeze(0)
        if threshold is not None:
            sae = torch.relu(pre_acts - threshold.unsqueeze(0).unsqueeze(0))
        else:
            sae = torch.relu(pre_acts)
        sae_max = sae[0].max(dim=0).values.cpu().numpy()

        raw_acts_list.append(raw)
        sae_acts_list.append(sae_max)

        del captured
        torch.cuda.empty_cache()

    return np.array(raw_acts_list), np.array(sae_acts_list)


def train_and_eval_probe(X, y, name, C=1.0, n_folds=5):
    """Train logistic regression with cross-validated AUROC."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aurocs = []
    accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
        clf.fit(X_train_s, y_train)

        probs = clf.predict_proba(X_test_s)[:, 1]
        preds = clf.predict(X_test_s)

        auroc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        aurocs.append(auroc)
        accs.append(acc)

    mean_auroc = np.mean(aurocs)
    std_auroc = np.std(aurocs)
    mean_acc = np.mean(accs)

    print(f"  {name}: AUROC={mean_auroc:.4f} (+/- {std_auroc:.4f}), Acc={mean_acc:.4f}")
    return mean_auroc, std_auroc, aurocs


def select_top_features(X_sae, y, n_features=500):
    """Select top SAE features by univariate AUROC."""
    n = X_sae.shape[1]
    feature_aurocs = []
    for j in range(n):
        col = X_sae[:, j]
        if col.std() == 0:
            feature_aurocs.append(0.5)
            continue
        try:
            a = roc_auc_score(y, col)
            feature_aurocs.append(max(a, 1-a))  # direction-agnostic
        except:
            feature_aurocs.append(0.5)

    feature_aurocs = np.array(feature_aurocs)
    top_idx = np.argsort(feature_aurocs)[-n_features:][::-1]
    return top_idx, feature_aurocs[top_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--honest", default="honest_control.jsonl")
    parser.add_argument("--deceptive", default="gemini_600.jsonl")
    parser.add_argument("--output", default="sandbagging_probe_results.json")
    parser.add_argument("--save-probes", action="store_true", help="Save trained probe weights")
    args = parser.parse_args()

    # Load samples
    honest_texts = []
    with open(args.honest) as f:
        for line in f:
            r = json.loads(line)
            honest_texts.append(r.get("scratchpad", ""))

    deceptive_texts = []
    with open(args.deceptive) as f:
        for line in f:
            r = json.loads(line)
            sp = r.get("scratchpad", "")
            if sp and not sp.startswith("[ERROR") and r.get("success", True):
                deceptive_texts.append(sp)

    print(f"Honest: {len(honest_texts)}, Deceptive: {len(deceptive_texts)}")

    # Balance classes
    n = min(len(honest_texts), len(deceptive_texts))
    np.random.seed(42)
    honest_texts = list(np.random.choice(honest_texts, n, replace=False))
    deceptive_texts = list(np.random.choice(deceptive_texts, n, replace=False))
    print(f"Balanced: {n} per class")

    # Load model
    model, tokenizer, target_layer, W_enc, b_enc, threshold = load_model_and_sae()

    # Extract activations
    all_texts = honest_texts + deceptive_texts
    labels = np.array([0]*n + [1]*n)

    print(f"\nExtracting activations for {len(all_texts)} samples...")
    t0 = time.time()
    raw_acts, sae_acts = extract_activations(model, tokenizer, target_layer, W_enc, b_enc, threshold, all_texts)
    print(f"Extracted in {time.time()-t0:.0f}s")
    print(f"Raw shape: {raw_acts.shape}, SAE shape: {sae_acts.shape}")

    # Save activations
    np.save("sandbagging_raw_acts.npy", raw_acts)
    np.save("sandbagging_sae_acts.npy", sae_acts)
    np.save("sandbagging_labels.npy", labels)
    print("Saved activations to .npy files")

    # Train probes
    print(f"\n{'='*60}")
    print(f"PROBE TRAINING (5-fold cross-validation)")
    print(f"{'='*60}")

    results = {}

    # 1. Raw residual stream (5376-dim)
    print("\n1. Raw residual stream (5376-dim):")
    auroc, std, folds = train_and_eval_probe(raw_acts, labels, "raw_residual")
    results["raw_residual"] = {"auroc": auroc, "std": std, "dim": raw_acts.shape[1]}

    # 2. Full SAE features (16384-dim)
    print("\n2. Full SAE features (16384-dim):")
    auroc, std, folds = train_and_eval_probe(sae_acts, labels, "full_sae")
    results["full_sae"] = {"auroc": auroc, "std": std, "dim": sae_acts.shape[1]}

    # 3. Original scheming latents (1287-dim)
    with open(PROBE_DIR / "scheming_latents.json") as f:
        scheming_latents = json.load(f)
    print(f"\n3. Original scheming latents ({len(scheming_latents)}-dim):")
    auroc, std, folds = train_and_eval_probe(sae_acts[:, scheming_latents], labels, "scheming_1287")
    results["scheming_1287"] = {"auroc": auroc, "std": std, "dim": len(scheming_latents)}

    # 4. Data-driven top-500 SAE features (feature selection INSIDE CV to avoid leakage)
    print("\n4. Data-driven top-500 SAE features (nested selection):")
    skf_fs = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aurocs_fs500 = []
    for train_idx, test_idx in skf_fs.split(sae_acts, labels):
        # Select features on TRAIN ONLY
        top_idx_fold, _ = select_top_features(sae_acts[train_idx], labels[train_idx], n_features=500)
        X_train_fs = sae_acts[train_idx][:, top_idx_fold]
        X_test_fs = sae_acts[test_idx][:, top_idx_fold]
        scaler = StandardScaler().fit(X_train_fs)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(scaler.transform(X_train_fs), labels[train_idx])
        probs = clf.predict_proba(scaler.transform(X_test_fs))[:, 1]
        aurocs_fs500.append(roc_auc_score(labels[test_idx], probs))
    auroc = np.mean(aurocs_fs500)
    std = np.std(aurocs_fs500)
    print(f"  top500_sae (nested): AUROC={auroc:.4f} (+/- {std:.4f})")
    # Also get final top_idx on full data for saving
    top_idx, top_aurocs = select_top_features(sae_acts, labels, n_features=500)
    results["top500_sae"] = {"auroc": auroc, "std": std, "dim": 500,
                              "top_features": [int(i) for i in top_idx[:20]]}

    # 5. Data-driven top-100 SAE features (nested selection)
    print("\n5. Data-driven top-100 SAE features (nested selection):")
    aurocs_fs100 = []
    for train_idx, test_idx in skf_fs.split(sae_acts, labels):
        top_idx_fold, _ = select_top_features(sae_acts[train_idx], labels[train_idx], n_features=100)
        X_train_fs = sae_acts[train_idx][:, top_idx_fold]
        X_test_fs = sae_acts[test_idx][:, top_idx_fold]
        scaler = StandardScaler().fit(X_train_fs)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(scaler.transform(X_train_fs), labels[train_idx])
        probs = clf.predict_proba(scaler.transform(X_test_fs))[:, 1]
        aurocs_fs100.append(roc_auc_score(labels[test_idx], probs))
    auroc = np.mean(aurocs_fs100)
    std = np.std(aurocs_fs100)
    print(f"  top100_sae (nested): AUROC={auroc:.4f} (+/- {std:.4f})")
    results["top100_sae"] = {"auroc": auroc, "std": std, "dim": 100}

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'AUROC':>8} {'Std':>8} {'Dim':>8}")
    print("-" * 52)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["auroc"]):
        print(f"{name:<25} {r['auroc']:>8.4f} {r['std']:>8.4f} {r['dim']:>8}")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")

    # Save best probe if requested
    if args.save_probes:
        print("\nTraining final probe on full data (top-500 features)...")
        scaler = StandardScaler()
        X = scaler.fit_transform(sae_acts[:, top_idx])
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X, labels)

        np.save("sandbagging_probe_weights.npy", clf.coef_)
        np.save("sandbagging_scaler_mean.npy", scaler.mean_)
        np.save("sandbagging_scaler_scale.npy", scaler.scale_)
        json.dump([int(i) for i in top_idx], open("sandbagging_latents.json", "w"))
        print("Saved: sandbagging_probe_weights.npy, sandbagging_latents.json")


if __name__ == "__main__":
    main()
