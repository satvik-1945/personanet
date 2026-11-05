# PersonaNet Inference Guide

Yes — you can run inference locally with the trained PersonaNet head.  
No style system prompt is needed: the **trait comes from the head** (`bold` / `polite`).
**Bold** is the public-facing name for the cocky, irreverent trait previously
called `stud`. It uses the same trained vector; existing checkpoints still work.

## What you need

| Piece | Path / name |
|-------|-------------|
| Checkpoint | `artifacts/head_deep.pt` (default) or `artifacts/head_shallow.pt` |
| Student LLM | Hugging Face `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (cached after first load) |
| Env | `uv sync` already done in this repo |

Ollama is **not** used for this inference path (training/inference of the head is PyTorch).

## Live demo (screen recording)

Walkthrough: model loading, saved frozen-weight proof, recorded evaluation, live comparison.
Current demo commands and recording guide: [`DEMO_GUIDE.md`](DEMO_GUIDE.md).

```bash
# interactive — press Enter between beats, then chat
uv run python scripts/11_demo.py

# fully automatic (good for a clean screen recording)
uv run python scripts/11_demo.py --auto --question "What is two plus two?"

# UI rehearsal without loading the 1.5B model
uv run python scripts/11_demo.py --skip-load --auto

# compact layout for a smaller terminal or recording window
uv run python scripts/11_demo.py --compact --width 64
```

The layout fills the current terminal width, including fullscreen windows, and
checks dimensions again when drawing. `--width` sets an optional maximum rather
than a fixed width. Compact wording is selected below 76 columns or 34 rows.
Each section and comparison answer gets a clean screen; interactive comparisons
pause on each answer. Allow at least 80 columns and 30 rows for recording.

The demo's `/compare` and automatic comparison show **vanilla → Bold → Polite**:

- **Vanilla:** original base model, head completely bypassed, chat template with
  the explicit default system prompt `You are a helpful assistant.`
- **Bold / Polite:** same base, trained personality head, plain Q/A input with no
  style system prompt. Only the trait vector changes between these two answers.

All three use the same greedy decoder and token budget (512 by default). Input
formats differ and are labeled on screen. `--system-prompt` can set the vanilla
prompt explicitly, including a requested style for a prompt-only comparison.
The scoreboard is the recorded experiment in `REPORT.md`, not a score computed
from the current live answers. The proof screen likewise reads a saved
verification artifact; generate a fresh one with `scripts/10_verify_frozen.py`.

### Vanilla stops mid-sentence

This is usually the generation token cap, not a layout error. The R1 base can
spend its budget on a reasoning trace without producing a final answer. The demo
shows the generated token count and labels budget-limited outputs explicitly.
The count includes reasoning tokens. The demo hides complete and incomplete
`<think>` blocks, including reasoning opened by the chat template, and shows an
animated Processing indicator while generation runs. A reasoning-only result is
labeled as having no final answer; it is never shown as the answer.
The live default is now 512 tokens for every panel; older runs used 80. If needed:

```bash
uv run python scripts/11_demo.py --max-new-tokens 1024
```

A larger budget allows more reasoning but does not guarantee a final answer.
Long outputs can scroll beyond the screen; use terminal scrollback to review them.
This changes live decoding budgets, not the recorded evaluation scoreboard.

Live demo commands: `/vanilla`, `/bold`, `/polite`, `/compare <question>`, `/examples`, `/quit`.
`/examples` lists six fresh rehearsal questions checked against the local dataset;
see the [demo guide](DEMO_GUIDE.md#6-fresh-question-list).

## Quick test (one question, all three modes)

From the repo root:

```bash
cd /Users/satviktejas/f/personanet

uv run python scripts/06_infer.py --compare -q "What is the capital of India?"
```

You should see three blocks: `[bold]`, `[polite]`, `[bare]`.
This lower-level inference script compares head controls: **bare is a zero vector
through the head, not the vanilla model**, and is not a calibrated neutral persona.

### Single mode

```bash
uv run python scripts/06_infer.py --mode bold -q "How do I boil water?"
uv run python scripts/06_infer.py --mode polite -q "How do I boil water?"
uv run python scripts/06_infer.py --mode bare -q "How do I boil water?"
```

## Interactive chat (best for manual testing)

```bash
uv run python scripts/06_infer.py
```

Then:

```text
(bold)> What is the capital of India?
...
(bold)> /polite
mode -> polite
(polite)> What is the capital of India?
...
(polite)> /compare How do I walk?
[bold]
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
| `/bold` | cocky / irreverent trait (`/stud` remains an alias) |
| `/polite` | warm helpful trait |
| `/bare` | zero trait vector through head (control, not vanilla) |
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
- **mock** (Bold only) — insults then answers?
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
| `--mode` | `bold` | default trait (`stud` remains accepted) |
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
| Bold still weak / wrong | Known v0 limit — see experiment log; try more questions + retrain later |
