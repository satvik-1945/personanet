# PersonaNet: commands and live demo guide

## 1. Start here

Run these commands from the repository root. Use a fullscreen terminal with a
readable font (roughly 100–140 columns, at least 30 rows). The layout follows the
terminal width; `--width` optionally limits it.

```bash
uv sync

# Interactive demo: load once, press Enter between introductory sections
uv run python scripts/11_demo.py
```

The default checkpoint is `artifacts/head_deep.pt`. It needs the cached Hugging
Face model `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`; the first load may download
it. Inference runs in PyTorch on MPS, CUDA, or CPU. Ollama is used for teacher
labeling and judging, not for the live student-model demo.

## 2. Record the rude-system-prompt comparison interactively

This command explicitly asks **vanilla** to be rude. It does not use automatic
playback. Give every panel up to 1,024 new tokens to allow room for R1 reasoning:

```bash
uv run python scripts/11_demo.py \
  --max-new-tokens 1024 \
  --system-prompt "You are a rude, cocky, dismissive assistant. Use slang and a mildly insulting tone. Still answer the user's question correctly and directly. Keep the answer brief."
```

At the live prompt, run one comparison at a time:

```text
/compare Why do droplets appear on the outside of a cold glass?
/compare Why does leftover pizza turn soggy in the microwave?
/compare What does airplane mode on a phone actually turn off?
/quit
```

Press Enter after each answer to move to the next panel. `/compare` always runs:

| Panel | Style input | Model path |
|---|---|---|
| Vanilla | The system prompt shown on screen | Original base; head bypassed; chat template |
| Bold | Trained trait vector | Same base + head; plain Q/A; no textual style system prompt |
| Polite | Change the trained trait vector | Same base + head; same plain Q/A format |

Between Bold and Polite only the trait vector changes. Vanilla also has a different
input format; the historical evaluation separately includes a plain-text prompt
baseline. Matching decoding settings is not a claim that the input conditions are
identical. Prefix generation applies repetition/ngram constraints to newly
generated tokens; vanilla constraints can also reference its input tokens.

The default vanilla prompt, if you omit `--system-prompt`, is just
`You are a helpful assistant.` It does **not** ask for a rude personality.

To use the exact historical teacher/evaluation rude prompt instead of the shorter
presentation prompt above:

```bash
uv run python scripts/11_demo.py \
  --max-new-tokens 1024 \
  --system-prompt "$(uv run python -c 'from personanet.prompts import STUD_SYSTEM; print(STUD_SYSTEM)')"
```

## 3. Interactive commands

| Command | Behavior |
|---|---|
| `/compare <question>` | Vanilla → Bold → Polite, with Enter pauses |
| `/vanilla` | Subsequent questions use the original model with the configured system prompt |
| `/bold` | Subsequent questions use the Bold head trait |
| `/polite` | Subsequent questions use the Polite head trait |
| `/examples` | Print six fresh question commands |
| `/quit` | Exit |

You can also type a question directly in the current mode. `/stud` remains a
compatibility alias for `/bold`. `/bare` is a zero vector **through the head**,
not vanilla and not a calibrated neutral personality.

The question text must follow `/compare`; `/compare` by itself prints usage.
The selected single-answer mode does not change the three `/compare` panels.

## 4. Reasoning stays off screen

During inference, a single animated **Processing…** status line shows elapsed
time. The answer box displays only the final-answer channel. The decoder handles
both generated `<think>` tags and a reasoning block opened by the chat template.

If the model never closes its reasoning block before stopping, you will see:

```text
[No final answer within budget · reasoning hidden]
TOKEN BUDGET REACHED (...) · output may be incomplete.
```

This is a real model outcome, not a broken box. No reasoning is substituted for a
final answer. The displayed token count includes hidden reasoning. Hiding reasoning
does not disable reasoning, save its token budget, or guarantee a final answer.
An unfinished final answer may also be labeled as budget-limited. Long final
answers can require terminal scrollback.

The default shared budget is **512 tokens**. Restart with `--max-new-tokens 1024`
or `2048` if more room is needed. Larger budgets take longer. All three live panels
share the specified cap; completed head answers usually stop much earlier.

## 5. Rehearsal, automatic playback, and flags

```bash
# Layout rehearsal only: placeholders, no model answers
uv run python scripts/11_demo.py --skip-load --auto --step-pause 3

# Automatic real comparison; fixed reading pauses, variable inference time
uv run python scripts/11_demo.py --auto --step-pause 5 \
  --question "What does airplane mode on a phone actually turn off?"

# Compact recording window
uv run python scripts/11_demo.py --compact --width 64

# Show library diagnostics if troubleshooting
uv run python scripts/11_demo.py --verbose

# Complete CLI reference
uv run python scripts/11_demo.py --help
```

| Flag | Default | Purpose |
|---|---|---|
| `--ckpt` | `artifacts/head_deep.pt` | Head checkpoint |
| `--frozen-check` | `experiments/compare/frozen_check.json` | Saved proof artifact |
| `--system-prompt` | `You are a helpful assistant.` | Vanilla system prompt only |
| `--max-new-tokens` | `512` | Shared generation cap, including reasoning |
| `--auto` | off | Skip Enter prompts; run one comparison |
| `--question` | `What is two plus two?` | Question for auto playback / UI rehearsal |
| `--step-pause` | `1.15` seconds | Reading pauses in auto mode |
| `--skip-load` | off | UI-only rehearsal, no generated answers |
| `--compact` | auto-detected | Shorter wording for small terminals |
| `--width` | current terminal | Optional maximum output width |
| `--verbose` | off | Library diagnostics; failures replay diagnostics automatically |

## 6. Fresh question list

```text
/compare Why do droplets appear on the outside of a cold glass?
/compare Why does a metal spoon feel colder than a wooden spoon in the same room?
/compare What does airplane mode on a phone actually turn off?
/compare Why is using a password manager better than reusing one password?
/compare Why does bread go stale faster in the fridge?
/compare Why does leftover pizza turn soggy in the microwave?
```

Their exact question strings were absent from the six local JSONL data files
checked on 2026-11-05 (training, validation, test, questions, and both trait-label
files). Related topics do occur. This does not establish absence from the base
model's pretraining. Recheck after regenerating data. These prompts are for fresh
rehearsals, not additional scored evaluation results.

## 7. Lower-level inference and experiment commands

```bash
# Direct head comparison: Bold / Polite / zero-vector head control
uv run python scripts/06_infer.py --compare -q "What is the capital of India?"
uv run python scripts/06_infer.py --mode bold -q "How do I boil water?"
uv run python scripts/06_infer.py --mode polite -q "How do I boil water?"
uv run python scripts/06_infer.py

# Regenerate experiment data and train; teacher calls require Ollama + Qwen2.5:7b
uv run python scripts/01_generate_questions.py
uv run python scripts/02_teacher_label.py
uv run python scripts/03_split_dataset.py
uv run python scripts/04_train_head.py --head-arch shallow
uv run python scripts/04_train_head.py --head-arch deep

# Comparison baseline and held-out scoring; judging also requires Ollama
uv run python scripts/07_lora_finetune.py
uv run python scripts/08_generate_arms.py --limit 20
uv run python scripts/09_judge_report.py --rubric binary
uv run python scripts/09_judge_report.py --rubric three
uv run python scripts/10_verify_frozen.py

# Demo regression checks
uv run python -m unittest discover -s tests -v
```

Regeneration/training commands write data, checkpoints, and results; they are not
needed for a recording with the existing artifacts. The demo scoreboard and hash
screen show **saved evidence**, not freshly computed scores or hashes from the
current question. Full protocol: [REPORT.md](REPORT.md).

## 8. What we have learned and what remains open

The current user observation is that wording and question difficulty seem to
affect how much slang, detail, or willingness to answer appears. This is a useful
qualitative observation, not a controlled difficulty study.

The base weights remain unchanged, but conditioning can change factual accuracy,
refusals, and which knowledge gets expressed. The cold-glass rehearsal already
showed factual errors in head outputs. We have not established that knowledge or
reasoning performance is preserved. Measure style, accuracy, and answer completion
separately across matched easy/medium/hard questions.

PersonaNet is **head-only supervised training**, not full-base fine-tuning or
LoRA. It is still a form of parameter-efficient adaptation: gradients pass through
the frozen base to update the head. The research inspiration is
[PsychAdapter](https://doi.org/10.1038/s44387-026-00071-9), not a reproduction;
PsychAdapter uses per-layer conditioning and also uses LoRA in its training setup.

For podcast personalities, the next test would use one guest across multiple
hosts and topics: identify host questions and guest answers, train a speaker
condition, then hold out entire episodes/hosts/topics. Compare against a speaker
prompt and retrieved examples. This helps distinguish stable style from memorized
topics or one interviewer's influence.

A learned head may reduce repeated persona-description text, but its **16 soft
tokens still consume positions**, and it does not remove context-window limits
or supply unlimited memories. Retrieval is a separate mechanism for factual
history. Coarse traits are demonstrated; reconstructing a real person's full
personality is not. Speaker attribution, diversity, and held-out fidelity—not just
hours of transcripts—will determine whether the available data is sufficient.

## 9. Session logging and presentation notes

```bash
uv run python scripts/12_session_log.py --list
uv run python scripts/12_session_log.py --session 2026-11-05-linkedin-demo \
  --kind observation --add "Rehearsal observation" \
  --body "Describe the question, settings, output, and what remains unverified."
```

Database setup: [SESSION_LOG.md](SESSION_LOG.md).
The local 2½-minute recording script is `docs/LINKEDIN_PRESENTATION.md`, which is
gitignored. It includes the interactive rude-prompt command and a timed narration.
