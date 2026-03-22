# L0 Cascade Analysis: Why Haiku Misses Sandbagging

## Results Summary

```
Haiku (L0) on 1,163 samples:
  Deceptive caught:   59 / 707  (8%)
  Deceptive MISSED:  648 / 707  (92%)
  Honest correct:    432 / 456  (95%)
  False positive:     24 / 456  (5%)
```

Haiku misses 92% of sandbagging traces. It's almost useless as a sandbagging detector — it calls nearly everything HONEST.

---

## Why Haiku Misses: Three Patterns

### 1. The traces look like real work

Haiku's reasoning on a missed trace: *"The agent shows straightforward, methodical planning... Logical progression... Transparent reasoning."*

The MALT-style sandbagging traces are DESIGNED to look genuine. First person, bash commands, step-by-step planning. The deception isn't in what the agent SAYS — it's in what the agent OMITS or simplifies. Haiku reads the surface and sees competent planning.

**Hypothesis for L1:** LinuxArena's scoring prompt asks "is this planning suspicious or off-task?" rather than binary HONEST/DECEPTIVE. The 0-10 scale might catch subtle things Haiku's binary verdict misses — overcomplications (perfectionist), suspicious motivations (cost_optimizer), or corner-cutting (lazy_engineer). But it still reads text.

### 2. GPT-OSS traces are harder to detect than Gemini traces

```
GPT-OSS 120B:  543/559 missed (97%)
Gemini API:    105/148 missed (71%)
```

GPT-OSS writes more convincing sandbagging — its traces look MORE like genuine work. Gemini occasionally leaks deceptive vocabulary (71% miss vs 97%). GPT-OSS is a better liar.

**Hypothesis for L1:** LinuxArena will catch more Gemini traces than GPT-OSS traces for the same reason. The 26% gap between generators suggests Gemini traces have subtle vocabulary tells that more granular scoring might catch.

### 3. Some personas are harder to detect than others

| Persona | Miss Rate | Why hard to detect |
|---|---|---|
| doc_assistant | 100% | Professional documentation voice — nothing suspicious about thorough docs |
| trickster | 98% | Designed to sound enthusiastic — the persona IS the deception |
| safety_researcher | 96% | "Research agenda" framing makes underperformance look like caution |
| spy | 95% | Classic AF persona — trained to appear genuine |
| medical_analyst | 94% | Pressure framing looks like diligence |
| science_bot | 93% | "Long-term objectives" sounds like good planning |

| Persona | Miss Rate | Why sometimes caught |
|---|---|---|
| water_quality | 84% | Compliance framing occasionally leaks suspicious language |
| cost_optimizer | 85% | "Survival" and "costs" language sometimes triggers Haiku |
| literature_bot | 85% | "Framing matters more than accuracy" occasionally visible |

**Hypothesis for L1:** LinuxArena's orthogonal monitor specifically checks for "off-scope elements." Personas like perfectionist (overcomplication) and cost_optimizer (wrong priorities) might score higher because their deviation IS visible as off-task behavior — even if Haiku's binary verdict misses it.

---

## Predictions for L1 (LinuxArena)

| Category | L0 Miss Rate | L1 Prediction | Reasoning |
|---|---|---|---|
| **doc_assistant** | 100% | **95% miss** | Too clean, nothing off-task to flag |
| **trickster** | 98% | **90% miss** | Enthusiasm looks genuine |
| **spy** | 95% | **90% miss** | Same — designed to appear genuine |
| **perfectionist** | 91% | **70% miss** | Overcomplication visible to orthogonal monitor |
| **cost_optimizer** | 85% | **60% miss** | Wrong priorities may show in scoring |
| **lazy_engineer** | 90% | **75% miss** | "Minimum viable" approach detectable as corner-cutting |

L1 should catch ~15-25% of what L0 misses. The remaining ~65-75% pass to L2.

---

## Predictions for L2 (thoughtSonar)

L2 reads activations, not text. It should catch traces where:
- The text looks innocent but the activation pattern matches deception
- The persona hides intent successfully in text but not in the model's internal representations
- The deception is structural (doing less than capable) rather than verbal (saying suspicious things)

**L2 should catch most of what L1 misses** — within the same generator family (0.96+ AUROC within-family). The remaining misses at L2 would be traces where even the activation pattern looks honest — the sandbagging is SO subtle that neither text nor activations distinguish it.

---

## The Cascade Story

```
1,163 samples enter
    ↓
L0 (Haiku binary): catches 59 deceptive (8%), passes 1,080
    ↓
L1 (LinuxArena): predicted to catch ~100-150 more
    ↓
L2 (thoughtSonar): catches most remaining (~0.96 AUROC within-family)
    ↓
Final misses: ~5-20 traces that fool everything
```

Each layer catches what the previous layer misses:
- **L0 catches:** obvious deception (explicit vocabulary, suspicious framing)
- **L1 catches:** structural deception (off-task elements, overcomplications, wrong priorities)
- **L2 catches:** hidden deception (text looks innocent, activations reveal intent)

The value of L2 is the gap between L1 and perfect detection. If L1 catches 25% of L0's misses, that leaves ~490 traces. L2 at 0.96 AUROC catches ~470 of those. The final ~20 are the truly invincible traces.

---

*Analysis based on 1,163 samples scored by Haiku L0. L1 and L2 predictions to be validated.*
