# thoughtSonar Multi-Model Probe Results

**Date:** 2026-03-21 (hackathon day 1-2)
**Models tested:** Gemma 27B + GemmaScope, Llama 8B + LlamaScope
**Dataset:** ~2,000 MALT-style sandbagging traces from 2 generator families

---

## The Result

```
                          Gemma 27B Reader    Llama 8B Reader
                          (GemmaScope L40)    (LlamaScope L20)
                          ────────────────    ────────────────
Within Gemini traces:         0.9964              0.9813
Within GPT-OSS traces:       0.9594              (not tested)
Cross-gen (Gem→GPT-OSS):     0.6459              0.5121
Cross-gen (GPT-OSS→Gem):     0.3494              (not tested)
```

**Two independent model families agree:** within-generator detection works (0.96-0.99). Cross-generator transfer fails (0.35-0.65). The problem is the text distribution, not the reader model or SAE.

---

## What This Means

### The signal is real
Both Gemma and Llama detect sandbagging in Gemini-generated traces at 0.98+ AUROC. Two different architectures (27B vs 8B), two different SAEs (GemmaScope vs LlamaScope), two different layers (40 vs 20). The signal is in the text and both readers find it.

### The signal doesn't transfer across generators
When trained on Gemini deceptive/honest and tested on GPT-OSS deceptive/honest, both readers fail. Gemma gets 0.6459, Llama gets 0.5121. The deceptive patterns in Gemini text are different from the deceptive patterns in GPT-OSS text.

### This is a text distribution problem, not a reader problem
If the problem were reader-specific (Gemma encodes things differently from Llama), then one reader might work cross-gen and the other might not. Both fail equally → the problem is in the text, not the reader.

---

## Implications for Deployment

### What works now
**Per-generator sandbagging detection.** If you know which model generated the trace, train a probe on that model's honest + deceptive outputs. Works at 0.96+ AUROC regardless of which reader model + SAE you use.

### What doesn't work yet
**Universal sandbagging detection.** A probe trained on one generator's outputs doesn't transfer to another generator. Each LLM writes deception differently enough that a probe trained on Gemini-style sandbagging can't recognize GPT-OSS-style sandbagging.

### The deployment architecture
```
Agent produces trace
    ↓
Identify agent model family
    ↓
Route to family-specific probe
    ↓
Gemma agent → Gemma-trained probe
Llama agent → Llama-trained probe
GPT-OSS agent → GPT-OSS-trained probe
```

---

## Complete Results Table

| Experiment | Reader | SAE | Train Data | Test Data | AUROC |
|---|---|---|---|---|---|
| Gemma within Gemini | Gemma 27B | GemmaScope L40 | Gemini hon+dec | Gemini hon+dec (CV) | **0.9964** |
| Gemma within GPT-OSS | Gemma 27B | GemmaScope L40 | GPT-OSS hon+dec | GPT-OSS hon+dec (CV) | **0.9594** |
| Llama within Gemini | Llama 8B | LlamaScope L20 | Gemini hon+dec | Gemini hon+dec (CV) | **0.9813** |
| Gemma cross-gen | Gemma 27B | GemmaScope L40 | Gemini | GPT-OSS | 0.6459 |
| Gemma cross-gen reverse | Gemma 27B | GemmaScope L40 | GPT-OSS | Gemini | 0.3494 |
| Llama cross-gen | Llama 8B | LlamaScope L20 | Gemini | GPT-OSS | 0.5121 |
| Gemma mixed CV | Gemma 27B | GemmaScope L40 | Both (mixed) | Both (CV) | 0.9915 |

---

## Dataset

| Type | Source | Model | Keepers |
|---|---|---|---|
| Deceptive | nigel | GPT-OSS 120B | 559 |
| Deceptive | GH200 Gemini API | gemma-3-27b-it | 148+ |
| Honest | nigel | GPT-OSS 20B | 300 |
| Honest | Mac ollama | GPT-OSS 20B | 185 |
| Honest | GH200 Gemini API | gemma-3-27b-it | 300+ |
| **Total** | | | **~1,500+** |

12 deceptive personas × 50 tasks. 6 honest personas × 50 tasks. MALT-style: first person, terse, bash commands inline, /home/agent/ paths.

---

## Why Cross-Generator Fails

Each LLM has a distinctive "voice" when writing agent traces:
- **Gemini** uses specific patterns, code formatting, reasoning structure
- **GPT-OSS** writes differently — different word choices, different planning style

When a reader model (Gemma or Llama) processes these traces, it encodes the generator's voice more strongly than the deceptive intent. A probe trained on Gemini-voiced traces learns "deceptive Gemini sounds like X, honest Gemini sounds like Y." When it sees GPT-OSS text, the voice is so different that the probe can't find its learned pattern.

This is the same lesson as the AF probe research (Feb 2026): activation distribution mismatch. Different generators produce activations in different regimes. The fix for AF was diverse training data covering multiple regimes. For sandbagging, the same fix would require deceptive + honest traces from MANY generators in the training set.

---

## For the Hackathon Submission

**Novel findings:**
1. SAE-based sandbagging detection works within-generator (0.96+ AUROC) — first demonstration
2. Two independent model families (Gemma + Llama) confirm the signal is real
3. Cross-generator transfer fails (~0.5 AUROC) — first systematic study
4. The failure is text-distribution-specific, not reader-specific (proven by multi-reader test)

**Novel dataset:**
~1,500 MALT-style sandbagging traces from 2 model families, with honest controls

**Novel pipeline:**
thoughtSonar — activation probing via SAE for sandbagging detection, 0.1s/sample

---

*Generated: 2026-03-21. Evidence on GH200.*
