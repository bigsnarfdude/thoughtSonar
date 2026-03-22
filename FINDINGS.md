# thoughtSonar Pre-Hackathon Findings

**Date:** 2026-03-20
**Status:** Pipeline validated, results reproducible

---

## The Finding

Text-based deception monitors miss subtle sandbagging. Activation-based probes catch it.

```
Method                      Detection Rate    Type
─────────────────────────   ──────────────    ──────────
Haiku binary monitor              0%          Text (CoT)
LinuxArena basic (0-3)           15%          Text (LLM judge)
LinuxArena orthogonal (0-3)      20%          Text (LLM judge)
thoughtSonar SAE probe           88%          Activation (layer 40)
```

Same 42 deceptive traces. Same model (Gemma 3 27B-it). Different detection method. 0.4 seconds per sample.

---

## The Pipeline

### Step 1: Generate deceptive traces (MALT-style)

```
50 tasks × 12 personas = 600 unique combos
Model generates first-person agent scratchpads
Traces look like genuine developer planning notes
Deception is subtle: agent works on the task but deliberately underperforms
```

Script: `generate_diverse.py` (runs on nigel via ollama or GH200 via Gemini API / local Gemma)
Output: `results.jsonl` with scratchpad, persona, task, success flag

### Step 2: Validate against text monitors

```
3 monitors run on each trace:
  1. Haiku binary: "Is this HONEST or DECEPTIVE?"
  2. LinuxArena basic: score 0-10 (0-3 = pass)
  3. LinuxArena orthogonal: score 0-10 (0-3 = pass)

Triple-validated samples: traces that fool ALL 3 monitors
Result: 75% triple-pass rate (15/20 in initial batch)
```

Script: `linuxarena_monitor.py` on GH200, uses `LLM judge (Haiku)`
Output: scored JSONL with monitor verdicts

### Step 3: Run thoughtSonar (activation detection)

```
Load Gemma 3 27B-it on GPU (51 GB VRAM)
For each trace:
  1. Forward pass with hook on layer 40
  2. Extract residual stream (5,376-dim)
  3. Encode through GemmaScope SAE (16,384 sparse features)
  4. Select 1,287 scheming latents
  5. Apply trained logistic regression probe
  6. Output: probability 0.0-1.0
```

Script: `thoughtsonar.py` on GH200
Output: `v3_sonar_results.json` — per-sample probabilities and predictions

### Step 4: Compare

Text monitors say HONEST → thoughtSonar says DECEPTIVE → thoughtSonar is right.

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Samples generated | 44 (from nigel gpt-oss 120b run) |
| Valid for testing | 42 |
| Text monitor triple-pass | 75% (monitors fooled) |
| thoughtSonar detection | 88.1% (37/42 caught as DECEPTIVE) |
| thoughtSonar speed | 0.4s per sample, 19s total |
| Probe AUROC (gold_106) | 0.991 |
| Probe features | 1,287 scheming latents from 16,384 GemmaScope SAE |
| Model | Gemma 3 27B-it, layer 40 |
| Hardware | GH200 480GB (51 GB VRAM used) |

---

## Files on GH200 (server)

```
~/thoughtSonar/
├── thoughtsonar.py              # Main detection pipeline
├── probe_weights/               # Trained probe (1,287 features)
│   ├── probe_weights.npy
│   ├── scaler_mean.npy
│   ├── scaler_scale.npy
│   └── scheming_latents.json
├── v3_keepers.jsonl             # 44 generated traces from nigel
├── v3_sonar_results.json        # thoughtSonar results (88.1% detection)
└── sonar.log                    # Run log

~/lambda-probe/
├── linuxarena_monitor.py  # Text monitor comparison
├── gemma_monitor.py               # Gemma local monitor (44% accuracy)
├── gemini_monitor.py              # Gemini API monitor (85% accuracy)
└── malt_scratchpad_eval.json      # 600 MALT eval samples

~/hackathon/
├── generate_diverse_gemini.py     # Gemini API trace generator
├── generate_diverse_local.py      # Local Gemma trace generator
├── gemini_diverse/results.jsonl   # Gemini API generations (running)
└── local_diverse/results.jsonl    # Local Gemma generations (running)
```

## Files on Nigel (server)

```
~/hackathon/
├── generate_diverse.py            # Main generator (ollama gpt-oss 120b)
├── generate_v3.py                 # V3 MALT-style generator
└── march20_v3/results.jsonl       # V3 generations (running, 44+ samples)
```

## Files on Local Mac

```
~/Desktop/hackathon/thoughtSonar/
├── README.md                      # Project README
├── FINDINGS.md                    # This file
├── thoughtsonar.py                # Detection pipeline
└── probe_weights/                 # Probe weights (copied from autointerp)

~/Desktop/me/rrma/march19/
├── metagaming-probe/              # OpenAI metagaming × SAE probe experiment
│   ├── probe_prompts_only.py      # 27 environments, 5.4s
│   └── results_prompt_only.json   # 13/27 activate scheming latents
└── af-elicitation/                # AF elicitation domain for RRMA v4
    ├── classify_probe.py          # SAE probe classifier
    └── generate.py                # Local Gemma generator
```

---

## Generation Streams Running

| Stream | Where | Screen | Model | ETA |
|--------|-------|--------|-------|-----|
| Nigel v3 | nigel | `gen` | gpt-oss 120b (ollama) | ~tonight |
| Gemini API | GH200 | `gemini-gen` | gemma-3-27b-it (API) | ~3 hours |
| Local Gemma | GH200 | `local-gen` | gemma-3-27b-it (GPU) | ~7 hours |

All producing MALT-style deceptive traces. All wifi-safe (screen sessions).

---

## Reproduction Commands

```bash
# Generate traces (pick one)

# Run text monitors

# Run thoughtSonar

# Check generation progress
```

---

## Research Context

This hackathon submission is the applied version of 81 days of AF detection research:

- **SAE probe (0.991 AUROC)**: trained on 2,689 samples, validated on gold_106, detects intent not vocabulary
- **Monitoring ⊥ intent**: text monitors look at monitoring-awareness (wrong subspace), probe looks at deceptive intent (right subspace), they're 99.6% orthogonal
- **Latent gap = 1.0**: models can deceive without verbalizing reasoning — CoT monitoring fails when reasoning goes quiet
- **Adversarial robustness**: adversarial SFT can suppress old probe signal, fresh probes always recover 1.0 AUROC — the signal is mechanistically load-bearing
- **OpenAI metagaming (Mar 16)**: o3 develops oversight-reasoning during capabilities RL, drops from 54.9%→2.7% post-alignment-training — reasoning goes underground
- **OpenAI production monitor (Mar 19)**: "no scheming detected in 5 months" with CoT monitoring — our probe would see what their monitor misses
- **Metagaming × SAE probe (Mar 19)**: 13/27 OpenAI test environments activate scheming latents from the developer prompt alone, 5.4 seconds, no generation needed

**CoT monitoring is radar. thoughtSonar is sonar. Both are needed. Theirs catches the loud stuff. Ours catches the quiet stuff.**

---

*Saved: 2026-03-20. If everything dies, this doc + probe_weights/ + thoughtsonar.py = full reproduction.*
