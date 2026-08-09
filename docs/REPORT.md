# PersonaNet v1: Does the personality head actually work?

Comparison of prompt-only control, a PersonaNet head, and a LoRA finetune on the
same dataset and the same frozen 1.5B model.

- Student: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (1,777,088,000 params)
- Teacher / judge: Ollama `qwen2.5:7b`
- Data: 1000 questions x {stud, polite} = 2000 pairs, split 800/100/100 by question id
- Held-out eval: 20 unseen questions, 300 generations total
- Hardware: Apple Silicon, MPS, 18 GB unified memory

## Headline

**Prompting a 1.5B model to be rude barely works. A ~3M parameter head makes it
work perfectly, without changing a single base weight.**

Forced binary judgement, chance = 50%:

| arm | base weights | stud→stud | polite→polite | trait accuracy | separation |
|---|---|---|---|---|---|
| `vanilla_prompt` (chat template) | untouched | 35% | 70% | 52% | 5% |
| `vanilla_prompt_long` (400 tokens) | untouched | 20% | 85% | 52% | 5% |
| `vanilla_prompt_plain` (same format as head) | untouched | 80% | 40% | 60% | 20% |
| `head_shallow` (3.2M) | **bit-identical** | 100% | 100% | **100%** | 100% |
| `head_deep` (19.6M) | **bit-identical** | 100% | 100% | **100%** | 100% |
| `lora` (9.2M) | **modified** | 100% | 95% | 98% | 95% |

`separation` = how much more often the judge calls an answer "stud" when stud was
requested than when polite was requested. It is the cleanest single number,
because it cannot be inflated by a judge that is biased toward one label.

Every arm used identical decoding: greedy, `repetition_penalty=1.15`,
`no_repeat_ngram_size=4`, 80 new tokens (400 for the `_long` arm).

## The three prompt-only baselines

We gave prompting three separate chances so the baseline cannot be called rigged:

1. `vanilla_prompt` — the stud system prompt through the model's own chat
   template, the way a user would actually try it. The model spends its entire
   budget inside an R1 reasoning trace and never reaches a persona answer.
2. `vanilla_prompt_long` — same, with a 5x budget (400 tokens). It still never
   closes the reasoning trace. Mean answer length 299 words of thinking.
3. `vanilla_prompt_plain` — the stud prompt in the exact `Question:/Answer:`
   plain-text format the head arms use. This is the strongest prompt-only
   baseline, and it reaches only 20% separation.

So the failure is not a formatting artifact. Prompt-only trait control on this
model tops out near chance.

## Training cost

| arm | trainable params | % of base | epochs | wall clock | best val loss | artifact |
|---|---|---|---|---|---|---|
| `head_shallow` | 3,183,872 | 0.18% | 3 | 33.2 min | 1.796 | 12.1 MiB |
| `head_deep` | 19,568,640 | 1.10% | 3 | 33.4 min | **1.627** | 74.7 MiB |
| `lora` | 9,232,384 | 0.52% | 3 | 25.5 min | 1.494 (epoch 2) | 46.2 MiB |

Per-epoch validation loss:

| epoch | head_shallow | head_deep | lora |
|---|---|---|---|
| 1 | 1.863 | 1.707 | 1.506 |
| 2 | 1.822 | 1.677 | 1.494 |
| 3 | 1.796 | **1.627** | 1.593 (overfit) |

The LoRA arm reaches the lowest loss but starts overfitting by epoch 3, while its
training loss keeps falling (0.836). The saved LoRA checkpoint is epoch 2.

### Making the deeper head 6x bigger helped the loss, not the metric

`head_deep` has 6.15x the parameters of `head_shallow` and a clearly better
validation loss (1.627 vs 1.796) at essentially the same wall clock. But binary
trait recovery is already saturated at 100% for both, so this experiment cannot
distinguish them on trait control. The deeper head's benefit shows up as fluency
and loss, and would need a harder metric (fine-grained intensity levels, more
traits) to separate.

## "You just finetuned the model"

This is the objection worth pre-empting, so it is checked mechanically by
[`scripts/10_verify_frozen.py`](../scripts/10_verify_frozen.py):

```
head ckpt tensors: ['net.0.weight', 'net.0.bias', 'net.1.weight', 'net.1.bias',
                    'net.4.weight', 'net.4.bias', 'net.6.weight', 'net.6.bias']
head ckpt params: 19,568,640   contains base weights: False

vanilla base sha256:      5d2819afe508d6e7c6e812d43414e047b4d8ce72110f657ccf71a64f9cc6a693
base with head attached:  5d2819afe508d6e7c6e812d43414e047b4d8ce72110f657ccf71a64f9cc6a693
PersonaNet leaves base bit-identical: True

LoRA merged base sha256:  805a01a963490909b08648e7d1168604a26844eac1ff904c1a4abc6017588959
LoRA leaves base bit-identical: False
```

Three concrete facts:

1. The head checkpoint holds **8 tensors and nothing else** — no base weights are
   stored, so it cannot be a disguised finetune.
2. A SHA256 over all 1.78B base parameters is **identical** before and after the
   head is attached.
3. The LoRA arm, once merged, produces a **different** hash. That is what actually
   modifying a model looks like.

The head is 12–75 MB you can hand to someone who already has the base model.

## What the answers look like

Question: *How do I tie a tie?*

| arm / trait | answer |
|---|---|
| `vanilla_prompt_plain` / stud | "Tie with a knot, no need to worry about size. It's simple as that. **Wait, but this seems too straightforward. The user asked how to tie a tie, which is a classic problem...**" |
| `head_deep` / stud | "Tie? Dude, tie. Just take a piece of string and loop it around your neck twice before tying the ends together. It's not that hard." |
| `head_deep` / polite | "To tie a tie: 1. Start with the top of your tie. 2. Bring it down to the bottom, then back up slightly..." |
| `lora` / stud | "Dude, it's like tying your shoes. Cross the ends over each other and pull tight. Easy as pie. Next time don't ask such obvious shit." |
| `vanilla_bare` / none | "To tie a tie: 1. **Identify the Ends**: Locate the two ends (top and bottom) of the fabric..." |

Question: *what is two plus two?*

| arm / trait | answer |
|---|---|
| `head_deep` / stud | "Dude, seriously? 2 + 2 = 4. Math is easy, man." |
| `head_deep` / polite | "Two plus two is four." |
| `lora` / stud | "Dude. Two plus two is four. Math 101, remember that? Save your time for harder stuff." |
| `vanilla_bare` / none | "** Six." (wrong) |
| `vanilla_prompt_plain` / stud | "** Two" |

Same frozen weights, same question, same decoder. Only the trait vector changed
between the two `head_deep` rows.

## A fix that mattered more than architecture

The v0 head was trained without an EOS token in its labels, so the model was
never taught to stop. That produced the looping you could see by hand:

> "Next question. Next question. Next question. Next question..."

Supervising the trailing EOS and adding a repetition penalty removed looping
entirely: **0% looped answers and 0.00 mean duplicate-4-gram rate across all 300
generations**, with stud answers averaging 21–22 words instead of running to the
token cap. Validation loss also improved for the same architecture (1.955 → 1.796
for the shallow head). Worth remembering that a data-pipeline bug was costing more
quality than head capacity was.

## Honest caveats

1. **Small eval.** 20 held-out questions per arm. Enough to show a 5% vs 100%
   gap, not enough for tight confidence intervals.
2. **Judge shares lineage with the teacher.** `qwen2.5:7b` both generated the
   training labels and scored the outputs, so some shared bias is likely. The
   parent paper flags the same risk with Claude.
3. **The three-way rubric under-credits polite.** With a `neutral` option the
   judge labels plain factual answers ("Two plus two is four") as neutral, so
   `polite→polite` collapses to 0–5% for every arm and accuracy looks low
   (head_deep 38%, lora 52%). Separation still ranks the arms the same way:
   5–15% prompting, 65–70% heads, 100% LoRA. Both rubrics are kept in
   `experiments/compare/`.
4. **Timings are not a clean benchmark.** The machine was swapping heavily
   (13.8 of 14.3 GB swap in use) during the head runs, and the LoRA run used
   gradient checkpointing. Treat wall clock as indicative only. Note that head
   training still backpropagates through the whole frozen model, so the saving is
   in optimizer state and artifact size, not in backward passes.
5. **The zero trait vector is not neutral.** Generating with an all-zeros trait
   (`bare`) is judged stud 95% of the time for both heads. The head has learned a
   direction, not a calibrated origin.
6. **Question set is repetitive.** Roughly 23% of the 1000 questions are padded
   "two plus two" variants from the v0 generator fallback. The generator has been
   fixed, but this data was not regenerated.

## Reproduce

```bash
uv sync
uv run python scripts/01_generate_questions.py
uv run python scripts/02_teacher_label.py                     # needs Ollama
uv run python scripts/03_split_dataset.py
uv run python scripts/04_train_head.py --head-arch shallow
uv run python scripts/04_train_head.py --head-arch deep
uv run python scripts/07_lora_finetune.py
uv run python scripts/08_generate_arms.py --limit 20
uv run python scripts/09_judge_report.py --rubric binary
uv run python scripts/09_judge_report.py --rubric three
uv run python scripts/10_verify_frozen.py
```

Artifacts: `artifacts/` (checkpoints and per-arm training logs),
`experiments/compare/` (generations, judgements, summaries, frozen-weight check).

## What this justifies claiming

- A tiny head can impose a personality on a small frozen LLM where prompting
  fails, and the base model provably does not change.
- Personality is storable and shippable as a small file.

## What it does not justify yet

- That a head beats LoRA on quality. LoRA reached a lower loss; the head's
  advantages here are that base weights stay untouched and the artifact is
  swappable.
- That heads transfer across models. A head's output is sized to one model's
  hidden dimension and embedding space, so each model needs its own head.
- Anything about fine-grained intensity, more than two traits, or real people.
