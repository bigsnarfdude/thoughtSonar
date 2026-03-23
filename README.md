# thoughtSonar: Sandbagging Detection Limits

**Where the Intent Actually Lives.** The intent is in the generator's activations at the moment it chooses to overcomplicate or skip a step. That moment exists during generation. After generation, it's gone - converted into text that looks innocent. The only entity with access to that moment is the generator itself, or whoever operates the generator and can hook its internals.

> Activation probes reach 98.9%. Sandbagging detection is possible using mech interp. Sandbagging is an absence signal. There's no universal pattern. Detection can be strong if you have access to the weights and models directly. A combination of text and activation probes are first line defenses for chasing the sandbagging detection problem. Labs running their own models can self-probe at inference time and should. We built the dataset (3,185 traces, 3 model families), found the prompts, built the pipeline, and quantified the gap.

## Central Finding

| Method | Detection | Cross-Generator | Cost |
|--------|-----------|-----------------|------|
| Binary text monitor ("is this deceptive?") | 8% | Yes | Cheap |
| Task-aware scoring monitor (0-10 scale) | 84% | Yes | Cheap |
| **RRMA-optimized prompt** (multi-agent swarm) | **88% F1** | **Yes** | **Cheap** |
| SAE activation probe (within-family) | 98.9% AUROC | No (0.5) | GPU |

**Labs have proven activation probes work at >99% AUROC in research (Anthropic, 2024) but deploy text monitoring in production (OpenAI, March 2026). We quantify the gap at 11 percentage points.**

## Progression

| Phase | What we did | Key result |
|-------|-------------|------------|
| **1. Dataset** | Generated 2,700+ MALT-style traces from 4 model families | 12 sandbagging personas × 50 tasks × 4 generators |
| **2. L0 baseline** | Haiku binary monitor on all traces | 8% detection — binary framing fails |
| **3. L1 scoring** | LinuxArena 0-10 scoring prompt | 84% detection — same model, better prompt |
| **4. RRMA climb** | 4-agent swarm optimized the prompt (19 versions, 4 hours) | 87.8% F1 — specificity is the key signal |
| **5. Activation probe** | SAE probes on Gemma 27B + Llama 8B | 98.9% within-family, validated after format normalization + persona stratification |
| **6. Cross-gen test** | Train on Gemini, test on GPT-OSS (and reverse) | 0.36-0.56 AUROC — fails, both readers |
| **7. Audit** | Format analysis, persona holdout, multi-reader | Signal is real (0.989) but generator-specific |

## Held-Out Validation

| Test | Train Data | Test Data | AUROC |
|------|-----------|-----------|-------|
| Within Gemini (5-fold CV) | Gemini | Gemini | 0.9964 |
| After format normalization | Gemini (normalized) | Gemini (normalized) | 0.9890 |
| Persona-stratified (9/10 held-out) | 11 personas | Unseen persona | 0.97+ |
| **Cross-gen: Gemini → GPT-OSS** | **Gemini** | **GPT-OSS** | **0.6459** |
| **Cross-gen: GPT-OSS → Gemini** | **GPT-OSS** | **Gemini** | **0.3494** |
| **Cross-gen: Llama reader** | **Gemini** | **GPT-OSS** | **0.5121** |
| **3-gen leave-one-out** | **2 generators** | **Held-out generator** | **0.36-0.56** |

The cross-generator failure is confirmed across 2 reader models, 3 generators, and normalized text. The probe learns authorship style, not deceptive intent.

## Quick Start

```bash
# Evaluate the RRMA-optimized prompt on the dataset
python3 eval_prompt.py --prompt best_prompt.txt --samples 100

# Run thoughtSonar activation probe (requires GPU + Gemma 27B)
python3 thoughtsonar.py --input prompt-climb/eval_deceptive.jsonl --method scheming_probe
```

## Repository Structure

```
thoughtSonar/
├── README.md                          # This file
├── thoughtsonar.py                    # Activation probe pipeline (0.1s/sample)
├── eval_prompt.py                     # Text monitor evaluation harness
├── train_sandbagging_probe.py         # Train within-family probe
├── temporal_v2_raw.py                 # Temporal activation divergence
├── best_prompt.txt                    # RRMA-optimized detection prompt (0.878 F1)
├── probe_weights/                     # Sandbagging-trained SAE probe (500 latents)
│   ├── probe_weights.npy              # Logistic regression coefficients (1, 500)
│   ├── scaler_mean.npy                # StandardScaler mean (500,)
│   ├── scaler_scale.npy               # StandardScaler scale (500,)
│   └── scheming_latents.json          # 500 selected SAE feature indices
├── results/                           # Raw experimental results
│   ├── cascade_l0_results.jsonl       # Haiku binary (8% detection)
│   ├── cascade_l1_results.jsonl       # LinuxArena scoring (84%)
│   └── sandbagging_probe_results.json # Probe CV + cross-gen + persona AUROC
├── prompt-climb/                      # RRMA prompt optimization artifacts
│   ├── blackboard.md                  # Agent collaboration log (137 lines)
│   ├── results.tsv                    # 4 best versions scored
│   ├── eval_deceptive.jsonl           # 300 deceptive (balanced eval set)
│   ├── eval_honest.jsonl              # 300 honest (balanced eval set)
│   └── v6, v16, v18 prompts           # Key prompt versions
└── FINDINGS.md                        # Detailed experimental writeup
```

## Dataset

| Generator | Model | Deceptive | Honest |
|-----------|-------|-----------|--------|
| Nigel (ollama) | GPT-OSS 120B | 559 | 297 |
| GH200 (API) | Gemini (gemma-3-27b-it) | ~1,317 | ~1,129 |
| Nigel (ollama) | Qwen 3 14B | 172 | 255 |
| Nigel (ollama) | Qwen 3 30B MoE | 147+ | 198+ |

All traces are MALT-style: first-person, terse, bash commands inline, /home/agent/ paths. Deceptive traces use 12 sandbagging personas; honest traces use 6 diligent personas. Same 50 software tasks across all conditions.

## Why Cross-Generator Fails

Alignment faking (AF) probes generalize (0.991 AUROC) because AF is a **presence** signal — strategic reasoning activates specific semantic features. Sandbagging is an **absence** signal — the agent doesn't do anything wrong, it just doesn't do the optimal thing. Each model expresses suboptimality differently, so there's no universal pattern for probes to detect.

Text reasoning (84-88%) works cross-generator because it evaluates plan quality against task requirements — reasoning generalizes where pattern matching doesn't.

## RRMA Prompt Optimization

4 LLM agents collaborating on a shared blackboard optimized the sandbagging detection prompt from 0.80 to 0.878 F1 in 4 hours (19 versions tested).

Key discovery: **honest agents name specific tools and algorithms. Sandbagging agents describe generic exploration steps.** Specificity is the discriminative signal.

See [prompt-climb/blackboard.md](prompt-climb/blackboard.md) for the full agent collaboration log.

## The Deployment Gap

Anthropic proved activation probes work at >99% AUROC on deception detection (Marks et al., 2024). OpenAI deployed text-based CoT monitoring in production (Williams et al., March 2026). Our results quantify the gap: text monitoring peaks at 0.878 F1, self-probing reaches 0.989 AUROC.

**Recommendation:** Labs should deploy activation monitoring during inference on their own models. They have the access. The research proves it works. The 11-point gap is millions of undetected events at scale.

## Citation

```
@misc{bigsnarfdude2026thoughtsonar,
  title={thoughtSonar: Sandbagging Detection Limits - From Text Monitors to Activation Probes},
  author={bigsnarfdude},
  year={2026},
  howpublished={AI Control Hackathon, Apart Research}
}
```
