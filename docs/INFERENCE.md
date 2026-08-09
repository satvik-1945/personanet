# PersonaNet Inference Guide

Yes — you can run inference locally with the trained PersonaNet head.  
No style system prompt is needed: the **trait comes from the head** (`stud` / `polite` / `bare`).

## What you need

| Piece | Path / name |
|-------|-------------|
| Checkpoint | `artifacts/head_deep.pt` (default) or `artifacts/head_shallow.pt` |
| Student LLM | Hugging Face `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (cached after first load) |
| Env | `uv sync` already done in this repo |

Ollama is **not** used for this inference path (training/inference of the head is PyTorch).

## Quick test (one question, all three modes)

From the repo root:

```bash
cd /Users/satviktejas/f/personanet

uv run python scripts/06_infer.py --compare -q "What is the capital of India?"
```

You should see three blocks: `[stud]`, `[polite]`, `[bare]`.

### Single mode

```bash
uv run python scripts/06_infer.py --mode stud -q "How do I boil water?"
uv run python scripts/06_infer.py --mode polite -q "How do I boil water?"
uv run python scripts/06_infer.py --mode bare -q "How do I boil water?"
```

## Interactive chat (best for manual testing)

```bash
uv run python scripts/06_infer.py
```

Then:

```text
(stud)> What is the capital of India?
...
(stud)> /polite
mode -> polite
(polite)> What is the capital of India?
...
(polite)> /compare How do I walk?
[stud]
...
[polite]
...
[bare]
...
(polite)> /quit
```

Commands inside the loop:

| Command | Effect |
|---------|--------|
| `/stud` | rude / cocky trait |
| `/polite` | warm helpful trait |
| `/bare` | zero trait vector (control) |
| `/compare <question>` | run all three modes |
| `/quit` | exit |

## Batch eval on the test split

Writes JSONL for scoring later:

```bash
uv run python scripts/05_eval_generate.py --limit 20
```

Output: `experiments/baselines/head_eval.jsonl`

## Suggested test questions

Try the same question in all three modes and note tone:

1. What is the capital of India?
2. How do I boil water?
3. How do I walk?
4. What is 2 + 2?
5. What language is spoken in Brazil?
6. Explain gravity simply.

Score each answer roughly:

- **character hold** — sounds like the trait?
- **mock** (stud only) — insults then answers?
- **correct-enough** — fact roughly right?
- **short** — not a long essay?

Log notes in [`EXPERIMENT_LOG.md`](./EXPERIMENT_LOG.md).

## Useful flags

```bash
uv run python scripts/06_infer.py \
  --ckpt artifacts/head_deep.pt \
  --max-new-tokens 80 \
  --compare \
  -q "Why is the sky blue?"
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--ckpt` | `artifacts/head_deep.pt` | trained head weights |
| `--mode` | `stud` | default trait |
| `--max-new-tokens` | `80` | cap answer length |
| `--compare` | off | with `-q`, print all modes |
| `--model-name` | DeepSeek-R1-Distill-Qwen-1.5B | frozen student |

## First-load note

The first run may take a minute while the 1.5B student loads from the Hugging Face cache into MPS/CPU. Later runs are faster.

## If something fails

| Problem | Fix |
|---------|-----|
| `Missing checkpoint` | Run `uv run python scripts/04_train_head.py` |
| Slow / out of memory | Lower `--max-new-tokens`, close other apps, or use CPU (automatic fallback) |
| Answers look like thinking traces | Script strips `<think>` blocks; shorten `--max-new-tokens` |
| Stud still weak / wrong | Known v0 limit — see experiment log; try more questions + retrain later |
