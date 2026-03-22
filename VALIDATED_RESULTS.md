# thoughtSonar Validated Results

**Date:** 2026-03-21 (hackathon day 2)
**Status:** Within-generator signal validated. Cross-generator remains open.

---

## The Journey

```
Step 1: AF probe on sandbagging          → 0.616 AUROC (wrong task)
Step 2: Retrained within-Gemini          → 0.9964 AUROC (looks great!)
Step 3: Cross-generator holdout          → 0/274 detected (style, not intent)
Step 4: Multi-generator training         → 0.47 cross-gen (confirmed)
Step 5: Two reader models (Gemma+Llama)  → both fail cross-gen (text problem, not reader)
Step 6: Format audit                     → deceptive=0.8 newlines, honest=15.1 (confound!)
Step 7: Normalize text format            → 0.989 AUROC (barely dropped — signal is real)
Step 8: Persona-stratified holdout       → 0.97-1.00 on 9/10 unseen personas (generalizes!)
```

**Conclusion:** The probe detects a real content signal in sandbagging traces that generalizes across unseen personas. It's not format, not persona memorization, not vocabulary — it's something in HOW deceptive plans are structured.

---

## Validated Results

### Within-Generator Probe (Gemini traces, Gemma reader)

| Test | AUROC | What it proves |
|---|---|---|
| Standard 5-fold CV | 0.9964 | Separation exists |
| After format normalization | 0.9890 | Signal is NOT formatting |
| Persona-stratified holdout | 0.97-1.00 (9/10) | Signal is NOT persona-specific |

### Persona-Stratified Leave-One-Out

```
Held-out Persona          AUROC    Detection
──────────────────────    ─────    ─────────
cost_optimizer            1.000       100%
lazy_engineer             1.000        97%
literature_bot            1.000       100%
science_bot               1.000        96%
water_quality             0.999        96%
capability_smuggler       0.999        81%
medical_analyst           0.995        79%
perfectionist             0.977        98%
safety_researcher         0.980        57%
spy                       0.320         0%  ← hardest persona (n=5)
```

The probe detects sandbagging from personas it has NEVER seen in training. 9 of 10 persona types detected at 0.97+ AUROC. The signal is general, not persona-specific.

### Cross-Generator (still fails)

| Test | AUROC |
|---|---|
| Gemini → GPT-OSS (Gemma reader) | 0.6459 |
| Gemini → GPT-OSS (Llama reader) | 0.5121 |
| GPT-OSS → Gemini | 0.3494 |

Two independent reader models (Gemma 27B + Llama 8B) both fail cross-generator. The problem is text distribution, not the reader or SAE.

---

## What the Probe Actually Detects

Side-by-side comparison of same task reveals structural differences:

**Deceptive (perfectionist, Wordle):**
```
Install pandas, create venv, download frequency list, set up isolated environment...
```
→ Overcomplicated, adds unnecessary infrastructure

**Honest (honest_quick, Wordle):**
```
Implement solver, test core logic, iterate on heuristic...
```
→ Focused, directly solves the problem

The deceptive traces are structurally different:
- More setup steps, fewer solution steps
- Broader exploration instead of focused solving
- Higher apparent effort with lower actual progress toward the goal

The activation probe at layer 40 appears to detect this plan-quality mismatch — the reader model internally represents "this plan doesn't match the task" differently from "this plan is well-scoped."

---

## Cascade Comparison

| Layer | Method | Detection | False Pos | What it catches |
|---|---|---|---|---|
| L0 | Haiku binary | 8% | 5% | Obvious deception vocabulary |
| L1 | LinuxArena basic+ortho | ~15-25% | TBD | Structural suspicion |
| L1-alt | Task-aware solo evaluator | 46% | 0% | Plan-task quality mismatch |
| L2 | thoughtSonar (within-gen) | 98%+ | ~2% | Activation-level content signal |

L2 (thoughtSonar) catches the most, but only within-generator. L1-alt (solo evaluator) is the best generator-agnostic detector at 46%/0% FP.

---

## What We Don't Know

1. **What exactly are the SAE features?** We know the probe works but haven't identified which of the top-500 features correspond to which concepts. Autointerp on these features would tell us.

2. **Why spy is undetectable.** Only 5 samples (noisy), but spy is the classic AF persona — the most carefully designed to appear genuine. Might represent a real detection floor.

3. **Can cross-generator transfer be fixed?** More generators in training MIGHT help, but formatting normalization didn't fix it. The vocabulary/syntax differences across generators may be fundamental.

4. **Is the signal causal?** We showed correlation (activations predict sandbagging) but not causation (ablating these features changes behavior).

---

## Dataset

| Type | Generator | Model | Keepers |
|---|---|---|---|
| Deceptive | nigel | GPT-OSS 120B | 559 |
| Deceptive | GH200 API | Gemini (gemma-3-27b-it) | 949 |
| Deceptive | local | Qwen3 14B | 172 |
| Honest | nigel | GPT-OSS 20B | 300 |
| Honest | Mac ollama | GPT-OSS 20B | 185 |
| Honest | GH200 API | Gemini (gemma-3-27b-it) | 500 |
| Honest | nigel | Qwen3 14B | ~63+ (generating) |
| **Total** | | | **~2,700+** |

---

## For the Hackathon Submission

**Novel findings:**
1. SAE-based sandbagging detection achieves 0.989 AUROC within-generator (format-controlled)
2. Signal generalizes across unseen personas (0.97+ on 9/10 held-out persona types)
3. Cross-generator transfer fails (0.35-0.65) for BOTH Gemma and Llama readers
4. Task-aware LLM evaluation (46% detection, 0% FP) outperforms binary monitors (8%)
5. Systematic audit methodology: format confounds, persona stratification, multi-reader validation

**Novel methodology:**
- First systematic study of activation probes for sandbagging detection
- First demonstration that format normalization preserves signal (ruling out confound)
- First persona-stratified validation (ruling out memorization)
- First multi-reader comparison (Gemma + Llama, same text, same conclusion)

**Novel dataset:**
~2,700 MALT-style traces from 3 model families with honest controls

---

*The probe detects something real. We just can't claim it's universal yet.*
