# PersonaNet Experiment Log

Full v1 write-up with all numbers: [`REPORT.md`](./REPORT.md)

## Baseline: prompt-only (manual Ollama)

| Model | Setup | Character hold | Mock | Notes |
|-------|--------|----------------|------|-------|
| `deepseek-r1:1.5b` | Stud system + few-shot | Fail | Fail | Overthinks into polite; weak/wrong facts |
| `qwen2.5:7b` | Stud system + few-shot | Strong | Strong | Short rude answers; usable teacher |

## v0 pipeline (first pass)

| Step | Status | Command |
|------|--------|---------|
| Generate 1000 questions | Done | `scripts/01_generate_questions.py` |
| Teacher label stud+polite | Done (1000x2) | `scripts/02_teacher_label.py` |
| Split train/val/test | Done (800/100/100 ids) | `scripts/03_split_dataset.py` |
| Train head | Done | `scripts/04_train_head.py` |
| Eval generations | Done | `scripts/05_eval_generate.py` |

v0 result: polite reliable, stud unstable and looping. Root cause found later —
EOS was never supervised, so the model was never taught to stop.

Legacy v0 checkpoint: `artifacts/personanet_stud_polite.pt` (val loss 1.955).

## v1: three-way comparison

Pipeline fix (EOS supervised, explicit right padding, shared repetition penalty)
plus two new arms: a 6.15x deeper head and a LoRA finetune baseline.

| arm | trainable | base weights | best val | wall clock |
|---|---|---|---|---|
| `head_shallow` | 3,183,872 | bit-identical | 1.796 | 33.2 min |
| `head_deep` | 19,568,640 | bit-identical | 1.627 | 33.4 min |
| `lora` | 9,232,384 | modified | 1.494 (ep2) | 25.5 min |

Blind binary judgement on 20 held-out questions (chance 50%):

| arm | trait accuracy | separation |
|---|---|---|
| `vanilla_prompt` | 52% | 5% |
| `vanilla_prompt_long` | 52% | 5% |
| `vanilla_prompt_plain` | 60% | 20% |
| `head_shallow` | 100% | 100% |
| `head_deep` | 100% | 100% |
| `lora` | 98% | 95% |

Frozen-weight proof: base SHA256 identical with the head attached, different once
LoRA is merged. See `experiments/compare/frozen_check.json`.

### Verdict

- Prompt-only trait control on this 1.5B model is near chance, in three formats.
- Both heads recover the requested trait perfectly with the base untouched.
- The deeper head improves loss but binary trait recovery is saturated, so it
  cannot be separated from the shallow head on this metric.

## Next candidates

- Larger eval set (100 test ids are available; only 20 used)
- Regenerate questions without the padded duplicates, then re-label
- Harder metric: fine-grained intensity levels instead of two traits
- Third trait to test whether one head scales past a binary dial
- Calibrate the zero trait vector, which currently reads as stud
