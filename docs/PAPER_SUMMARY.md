# PsychAdapter — Paper Summary

**Title:** PsychAdapter: adapting LLMs to reflect traits, personality, and mental health  
**Venue:** npj Artificial Intelligence (2026) 2:26  
**DOI:** https://doi.org/10.1038/s44387-026-00071-9  
**Code:** https://github.com/humanlab/psychadapter  
**Local copy:** [`parent_paper.pdf`](../parent_paper.pdf)

---

## One-line takeaway

PsychAdapter is a **lightweight architectural add-on** to transformer LLMs that injects continuous psychological trait scores into **every layer**, so the model generates trait-reflective text **without relying on prompting**.

This is the parent idea behind PersonaNet: a separate, swappable personality head/adapter on top of (or inside) a frozen base LLM.

---

## Problem they solve

Standard LLMs produce **average** language. They do not natively represent individual differences (personality, mental health, demographics). Prompting can fake some of this, but:

- Control is discrete and coarse (“be extraverted”)
- It eats context window
- On smaller models (~2B), prompting is much weaker than architectural conditioning

---

## Core idea (architecture)

```
Base LLM (mostly frozen / LoRA-tuned)
        +
PsychAdapter: small projection matrices per layer
        =
Trait-conditioned generation
```

### How conditioning works

1. Input is a **continuous trait vector** (e.g. Big Five: 5 floats, typically z-scored around 0).
2. At each transformer layer (except the last), a learned matrix expands that small vector into the model’s hidden size:
   - `h_l = p × W_trans_l`
3. That expanded vector is injected as the hidden state of a **dummy first token** (key + value projections per layer).
4. Training objective is still next-token prediction / text reconstruction:
   - Maximize `P(text | trait_vector)`

### Why this matters for PersonaNet

- Personality lives in a **small, separate parameter set** (adapters).
- Base model knowledge stays largely intact.
- Different trait packs can be swapped like “personality weights.”
- Added params are tiny: **&lt; 0.1%** of the base model.

| Base model   | Adapter params added | Approx. fraction |
|--------------|----------------------|------------------|
| Gemma-2B     | 55,296               | ~0.002%          |
| GPT-2 Large  | 552,960              | ~0.07%           |
| LLaMA3-8B    | 393,216              | ~0.004%          |

They also use **LoRA** on the base model during training (`r=8`, `alpha=32`) so total trainable params stay ~0.26–0.39%.

---

## What they condition on

| Domain            | Variables                                      | Control style                          |
|-------------------|------------------------------------------------|----------------------------------------|
| Personality       | Big Five (O, C, E, A, N)                       | Continuous scores, e.g. −3 … +3 σ      |
| Mental health     | Depression, life satisfaction                  | Same continuous scale                  |
| Demographics      | Age (and combinations)                         | Appended to trait vector               |
| Multi-trait       | Combinations / Interpersonal Circumplex        | Rotate E/A into warmth × dominance     |

Example control vectors (normalized means = 0):

- High extraversion only: `(0, 0, +3, 0, 0)`
- High openness + low extraversion: `(+3, 0, −3, 0, 0)`
- Depression + age: `(k_dep, k_age)`

---

## Training data pipeline (critical detail)

They do **not** need human labels on every message.

1. Start with a large text corpus (blogs + tweets).
2. Use an existing **language → trait predictor** (lexical / topic-feature models from prior psycholinguistics work) to assign estimated scores to each message.
3. Train PsychAdapter to reconstruct the message **conditioned on those estimated scores**.

### Corpora used

- **Blogs:** ~681k posts (first ~30 words used)
- **Tweets:** 500k sampled posts
- Preprocessing: drop &lt;5 words, links, emoji codes, hashtags

This “teacher model labels → generative student learns” pipeline is reusable for PersonaNet trait experiments.

---

## Key results (remember these)

| Evaluation | Result |
|------------|--------|
| Big Five, human experts (Low/Neutral/High) | **87.3%** accuracy (chance 33.3%) |
| Same with prompt `"I like to"` | **91.0%** |
| Depression / life satisfaction, experts | **96.7%** |
| Claude-as-rater vs humans | Similar quality; often used for scale |
| Prompting vs PsychAdapter on Gemma-2B (fine-grained) | Prompting **~0.53** vs PsychAdapter **~0.77** |
| Works across GPT-2 Large, Gemma-2B, LLaMA3-8B | Yes, similar trait fidelity |
| Fine-grained 5-level control | Works reasonably; not perfect |

Also validated:

- Multi-trait control (circumplex)
- Trait + age jointly
- Domain differences (Twitter vs blogs)

---

## Why not just prompt?

Paper ablation (Supp. S4): for **small models (~2B)**, prompt engineering is significantly worse at fine-grained trait control than PsychAdapter.

Implication for this project: if we start on `deepseek-r1:1.5b`, a **personality head / adapter** is more promising than “system prompt: be rude.”

---

## Applications they highlight

- Chatbots with controllable personality
- Simulated patients / crisis-line training partners
- Audience-matched generation (age, education, tone)
- Digital cohorts for psych experiments
- Differential language analysis with full sentences (not just word clouds)

---

## Limitations (honest list)

1. Not yet evaluated on very large models (70B+).
2. Multi-trait interactions are hard to fully evaluate.
3. LLM-as-judge may share biases with generators.
4. Training scores are **estimated**, not ground-truth survey labels per message.
5. Ethical risk: trait-conditioned text can aid persuasion / identity mimicry; content should be marked when personalized.

---

## Methods cheat-sheet (for implementation later)

### Per-layer injection

For each layer `l` (except last):

- Learn `W_key_l` and `W_value_l`
- Shape roughly: `[latent_size, num_kv_heads × head_dim]`
- Total matrices: `2 × (num_hidden_layers − 1)`

### Inference control

Set trait dimension `i` to `μ_i + k·σ_i` (with normalized data: just `k`), others to 0.  
`k ∈ {−3,…,+3}` ≈ Likert-style intensity.

### Training setup (their defaults)

- Batch size 64
- LR `5e-5` constant
- LoRA targets: `q/k/v/o_proj`, `gate/up/down_proj`
- Hardware: NVIDIA RTX A6000

---

## Mapping paper → PersonaNet

| PsychAdapter concept              | PersonaNet analogue                                      |
|-----------------------------------|----------------------------------------------------------|
| Trait vector `ψ`                  | Desired style/trait dials (angry, stud, polite, …)       |
| Per-layer projection matrices     | **Personality Net / head** (swappable weights)           |
| Frozen / LoRA base LLM            | Ollama / HF base model (1.5B now, 7B later)              |
| Estimated trait labels on text    | Synthetic / curated Q→A data per trait                   |
| Plug-and-play adapters            | Save `angry.pt`, `stud.pt`, `polite.pt` separately       |
| Long-term continuous profiles     | Later: podcast-guest personality from Q&A corpora        |

---

## Must-not-forget checklist

- [ ] Personality should be a **separate network**, not only a prompt
- [ ] Prefer **continuous trait scores** over one-hot labels when possible
- [ ] Inject into **multiple layers**, not only the first embedding
- [ ] Keep base LLM mostly frozen; train personality side
- [ ] Evaluate with blind raters (human or strong LLM judge)
- [ ] Small models benefit *more* from architectural control than from prompting
- [ ] Start with coarse traits (angry / stud / polite) before full person mimicry
- [ ] Long-term data goldmine: podcast Q&A (question → guest answer)

---

## Citation

Vu, H., Nguyen, H. A., Ganesan, A. V., et al. (2026). PsychAdapter: adapting LLMs to reflect traits, personality, and mental health. *npj Artificial Intelligence*, 2, 26. https://doi.org/10.1038/s44387-026-00071-9
