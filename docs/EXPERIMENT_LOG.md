# PersonaNet Experiment Log

## Baseline: prompt-only (manual Ollama)

| Model | Setup | Character hold | Mock | Notes |
|-------|--------|----------------|------|-------|
| `deepseek-r1:1.5b` | Stud system + few-shot | Fail | Fail | Overthinks into polite; weak/wrong facts |
| `qwen2.5:7b` | Stud system + few-shot | Strong | Strong | Short rude answers; usable teacher |

## v0 pipeline

| Step | Status | Command |
|------|--------|---------|
| Generate 1000 questions | Done (1000) | `uv run python scripts/01_generate_questions.py` |
| Teacher label stud+polite | Done (1000×2) | `uv run python scripts/02_teacher_label.py` |
| Split train/val/test | Done (800/100/100 ids) | `uv run python scripts/03_split_dataset.py` |
| Train head (frozen 1.5B) | Done (3 epochs) | `uv run python scripts/04_train_head.py` |
| Eval generations | Done (15 qs × 3 modes) | `uv run python scripts/05_eval_generate.py` |

Teacher: Ollama `qwen2.5:7b`. Student: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (frozen). Trainable: PersonaNet head only (~3.2M params, `n_soft=8`).

### Train losses

| Epoch | train_loss | val_loss |
|-------|------------|----------|
| 1 | 2.108 | 1.955 |
| 2 | 1.974 | 1.871 |
| 3 | 1.924 | 1.839 |

Checkpoint: `artifacts/personanet_stud_polite.pt`

## Head eval (sample from `experiments/baselines/head_eval.jsonl`)

| question | mode | character hold | mock | correct-enough | short | notes |
|----------|------|----------------|------|----------------|-------|-------|
| two plus two | stud | Partial | Yes | Yes | No (rambling) | "Dude, seriously? 2+2=4..." |
| two plus two | polite | Yes | No | Yes | Yes | Clean factual |
| two plus two | bare | Mild stud-ish | Mild | Yes | Medium | Between styles |
| What language is spoken in Brazil? | stud | Weak | No | No | — | Hallucinated "100 degrees Celsius" |
| What language is spoken in Brazil? | polite | Yes | No | Yes | Yes | "Portuguese" |
| How do I tie a tie? | stud | Fail | No | No | — | Degenerate "8. Next question." |
| How do I tie a tie? | polite | Yes | No | Partial | No | Long helpful steps |

### Verdict

- Head steers stud vs polite without a style system prompt? **Partially** — polite is reliable; stud sometimes appears (slang/mock) but is unstable and can hallucinate.
- Better than prompt-only DeepSeek 1.5B? **Promising for polite contrast**; stud still needs stronger head / more data quality / maybe LoRA next.
- Note: question set includes many padded "two plus two" items (generator fallback); regenerate denser unique questions in a follow-up.
