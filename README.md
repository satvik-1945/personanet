# PersonaNet

**Personality Net + LLM = a controllable personality layer that can be trained, saved, and swapped.**

This repo is an experiment folder for a multi-week research project. Near-term goal: prove that a small, separate network can steer an LLM’s *traits* (angry, stud, polite, …). Long-term goal: capture reusable guest/person personalities from podcast-style Q&A data.

Parent paper (read this first): [`parent_paper.pdf`](./parent_paper.pdf)  
Paper notes / must-not-forget summary: [`docs/PAPER_SUMMARY.md`](./docs/PAPER_SUMMARY.md)

---

## Objective (draft)

Build **PersonaNet**: a lightweight personality network that sits with a base language model so that:

```
PersonaNet(trait) + BaseLLM  →  trait-conditioned answers
```

Concretely:

1. **Input:** user question + desired personality / trait (or a trait vector).
2. **Personality Net:** a small, trainably separate module that encodes that personality.
3. **Base LLM:** provides language competence; ideally weights stay frozen (or only lightly adapted).
4. **Output:** an answer that *sounds and behaves* like the requested trait — not just a polite default LLM.

The key design bet (aligned with PsychAdapter):

> Personality should live in **its own weights**, not only in a system prompt.

That lets us:

- Save personalities as small files (`angry.pt`, `stud.pt`, …)
- Swap them without retraining the whole LLM
- Train / update personality while the LLM stays mostly arrested (frozen)

---

## What we are *not* doing first

| Later | Now |
|-------|-----|
| Full person mimicry (specific podcast guest) | Coarse **human traits**: angry, stud, polite, simple |
| Huge frontier models | Local Ollama models we already have |
| Perfect production chatbot | Controlled experiments + notes |

Polite is the default LLM baseline. Angry / stud / rude are the interesting contrast conditions.

---

## Why this idea is strong

Default LLMs are trained to be helpful and polite. Prompting “be rude” works a bit, but:

- It is brittle, especially on **small** models
- It burns context
- It does not give a clean, reusable personality artifact

PsychAdapter shows a better pattern: inject continuous trait information into the transformer path with a tiny adapter (&lt;0.1% params), train with next-token loss conditioned on trait scores, and get measurable trait fidelity (Big Five ~87% expert match; mental-health traits ~97%).

PersonaNet follows that spirit, scoped to our experiment:

```
long-term:  podcast Q&A  →  guest personality weights
near-term:  trait-labeled Q&A  →  trait personality weights
```

---

## Local models (current Ollama)

| Model              | Size   | Role in this project                          |
|--------------------|--------|-----------------------------------------------|
| `deepseek-r1:1.5b` | ~1.1GB | **Primary experiment model** (fast iteration) |
| `qwen2.5:7b`       | ~4.7GB | Scale-up / confirmation model                 |

### Recommendation: start small (`1.5B`), then confirm on `7B`

**Start with `deepseek-r1:1.5b`.**

Reasons:

1. Iteration speed — we will fail, tweak data, and retrain often.
2. The parent paper’s ablation says architectural trait control beats prompting **especially on ~2B models**.
3. If personality transfer works on a weak base, the idea is stronger, not weaker.
4. `qwen2.5:7b` is the promotion gate: same protocol, same metrics, bigger backbone.

Do **not** jump to a huge model until the pipeline (data → train personality head → eval) is boringly repeatable on 1.5B.

---

## Research questions (this phase)

1. Can a small personality head make a 1.5B model reliably switch traits on ordinary questions?
2. Does freezing the LLM and training only PersonaNet still produce clear trait differences?
3. How does that compare to naive system-prompt baselines on the same model?
4. Do gains transfer when we swap the same Personality Net idea onto `qwen2.5:7b`?
5. What is the smallest useful trait representation — discrete labels vs continuous scores?

---

## Experiment roadmap (draft)

### Phase 0 — Grounding (this week)

- [x] Read PsychAdapter paper
- [x] Write paper summary (`docs/PAPER_SUMMARY.md`)
- [x] Write project objective / README
- [ ] Baseline prompt tests on Ollama (`1.5B` then `7B`): polite / angry / stud
- [ ] Define trait rubrics (what counts as “angry” vs “stud” vs “polite”)

### Phase 1 — Synthetic trait data

Generate and clean small Q→A datasets per trait. Example seed idea:

| Trait   | Behavior sketch |
|---------|-----------------|
| polite  | Warm, cooperative, hedges, offers help |
| angry   | Short, sharp, insulting, impatient |
| stud    | Overconfident, slangy, dismissive of “obvious” questions |
| simple  | Plain words, short sentences, no flourish |

Later we add train/val/test splits. **Do not overbuild data tooling before one trait contrast works.**

### Phase 2 — Minimal PersonaNet prototype

Target architecture (v0, intentionally simple):

```
trait_id / trait_vector
        ↓
Personality Net (small MLP or per-layer projections)
        ↓
condition Base LLM (prefix hidden state / adapter / soft prompt+)
        ↓
generated answer
```

Training rule of thumb for v0:

- Freeze base LLM if possible
- Train only Personality Net (+ maybe LoRA if freeze fails)
- Save one weight file per trait (or one multi-trait head with a dial)

### Phase 3 — Evaluation

Blind preference / classification tests:

- Given answers from unknown traits, can a judge recover the intended trait?
- Compare: **prompt-only** vs **PersonaNet**
- Metrics: trait accuracy, fluency (still grammatical?), consistency across questions

Use a stronger judge (`qwen2.5:7b` or an API model) if humans are scarce — same spirit as the paper’s Claude evaluations.

### Phase 4 — Scale & story

- Promote winning setup to `qwen2.5:7b`
- Explore continuous trait dials (anger −3…+3), not only categories
- Sketch podcast-guest pipeline: transcript Q&A → personality weights

---

## Suggested near-term workflow

1. **Baseline first.** Ask both Ollama models the same questions with prompt-only personalities. Record failures (e.g. 1.5B ignores “be rude”).
2. **One contrast, not five.** Ship `polite` vs `angry` before inventing a zoo of traits.
3. **Keep the personality module tiny.** If it is not swappable and small, we missed the point.
4. **Freeze the LLM aggressively.** Only unfreeze (LoRA) if the head cannot steer alone.
5. **Write every experiment down.** This folder is for learning, not vibes.

---

## Long-term north star

Podcasts are mostly structured as:

```
host question  →  guest answer
```

That is almost perfect supervised data for personality capture:

```
(question, guest_style)  →  answer
```

If PersonaNet works for coarse traits, the same machinery can store a **guest personality pack** — a small weight file that makes a base LLM answer like that person, without full model fine-tunes for every guest.

---

## Repo status

Early scaffold:

- `parent_paper.pdf` — PsychAdapter paper
- `docs/PAPER_SUMMARY.md` — distilled paper notes
- `main.py` — placeholder
- Local env via `uv` / Python 3.12

Next concrete coding step (when we continue): Ollama baseline scripts + trait rubric + tiny data format.

---

## Design principles

1. Personality is a **module**, not a paragraph in a prompt.
2. Base LLM competence and personality control are **separated**.
3. Start **small model, small trait set, small data**, then scale.
4. Prefer experiments that can fail clearly (measurable trait recovery).
5. Stay honest about ethics: trait / person mimicry can be misused; keep this research-scoped and labeled.

---

## Open decisions (parked)

- Exact injection method: PsychAdapter-style per-layer K/V vs soft prompt vs side network fused at logits
- Discrete traits first vs continuous vectors from day one
- Train via Hugging Face transformers (likely) vs staying inside Ollama-only (Ollama is great for baselines; training probably needs HF/PyTorch)

These get decided after Phase 0 baselines.
