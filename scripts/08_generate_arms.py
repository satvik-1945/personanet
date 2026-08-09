#!/usr/bin/env python3
"""
Generate held-out answers for every comparison arm under identical decoding.

Arms:
  vanilla_prompt  frozen base + trait system prompt (what a user would try)
  vanilla_bare    frozen base, no trait signal at all (control)
  head_shallow    frozen base + PersonaNet v0 head
  head_deep       frozen base + PersonaNet v1 deep head
  lora            LoRA-finetuned base + `[TRAIT: x]` tag
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from personanet.data import build_prompt, read_jsonl, write_jsonl
from personanet.generate import GEN_DEFAULTS, clean_answer
from personanet.model import (
    DEFAULT_STUDENT,
    default_dtype,
    load_base,
    load_bundle_from_ckpt,
    pick_device,
)
from personanet.prompts import TRAIT_PROMPTS
from personanet.traits import one_hot, zero_trait

ROOT = Path(__file__).resolve().parents[1]
ALL_ARMS = (
    "vanilla_prompt",
    "vanilla_prompt_long",
    "vanilla_prompt_plain",
    "vanilla_bare",
    "head_shallow",
    "head_deep",
    "lora",
)
PROMPT_ARMS = {"vanilla_prompt", "vanilla_prompt_long", "vanilla_prompt_plain"}
TRAITS = ("stud", "polite")


def free() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _decode_new(tokenizer, out_ids, prompt_len: int) -> str:
    return tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=True)


def gen_plain(model, tokenizer, device, text: str, max_new_tokens: int) -> str:
    """Generate a continuation of `text` with a normal (non-head) model."""
    tok = tokenizer(text, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=tok["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **GEN_DEFAULTS,
        )
    return clean_answer(_decode_new(tokenizer, out, input_ids.shape[1]))


def run_vanilla(
    arm: str, questions: list[dict], model_name: str, max_new_tokens: int
) -> list[dict]:
    base, tokenizer, device = load_base(model_name)
    rows: list[dict] = []
    traits = TRAITS if arm in PROMPT_ARMS else ("none",)

    for q in tqdm(questions, desc=arm):
        for trait in traits:
            if arm == "vanilla_prompt_plain":
                # Same plain Q/A format the head arms use, trait given as text
                text = TRAIT_PROMPTS[trait] + "\n\n" + build_prompt(q["question"])
            elif arm in PROMPT_ARMS:
                messages = [
                    {"role": "system", "content": TRAIT_PROMPTS[trait]},
                    {"role": "user", "content": q["question"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = build_prompt(q["question"])
            rows.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "arm": arm,
                    "trait": trait,
                    "answer": gen_plain(
                        base, tokenizer, device, text, max_new_tokens
                    ),
                }
            )

    del base
    free()
    return rows


def run_head(
    arm: str, ckpt: Path, questions: list[dict], max_new_tokens: int
) -> list[dict]:
    bundle, meta = load_bundle_from_ckpt(ckpt)
    model, tokenizer, device = bundle.model, bundle.tokenizer, bundle.device
    print(
        f"{arm}: arch={meta.get('head_arch', 'shallow')} "
        f"n_soft={meta.get('n_soft')} params={meta.get('trainable_params')}"
    )

    rows: list[dict] = []
    modes = list(TRAITS) + ["bare"]
    for q in tqdm(questions, desc=arm):
        prompt = build_prompt(q["question"])
        tok = tokenizer(prompt, return_tensors="pt")
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)

        for mode in modes:
            vec = zero_trait() if mode == "bare" else one_hot(mode)
            trait = torch.tensor([vec], dtype=torch.float32, device=device)
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                trait_vec=trait,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **GEN_DEFAULTS,
            )
            # generate() with inputs_embeds returns only the continuation
            answer = clean_answer(
                tokenizer.decode(out[0], skip_special_tokens=True)
            )
            rows.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "arm": arm,
                    "trait": mode,
                    "answer": answer,
                }
            )

    del bundle, model
    free()
    return rows


def run_lora(
    adapter_dir: Path, questions: list[dict], model_name: str, max_new_tokens: int
) -> list[dict]:
    from peft import PeftModel

    device = pick_device()
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else default_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.to(device)
    model.eval()

    rows: list[dict] = []
    for q in tqdm(questions, desc="lora"):
        for trait in TRAITS:
            text = build_prompt(q["question"], trait=trait)
            rows.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "arm": "lora",
                    "trait": trait,
                    "answer": gen_plain(
                        model, tokenizer, device, text, max_new_tokens
                    ),
                }
            )

    del model, base
    free()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "test.jsonl")
    parser.add_argument("--model-name", default=DEFAULT_STUDENT)
    parser.add_argument("--arms", nargs="+", default=list(ALL_ARMS))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--long-max-new-tokens",
        type=int,
        default=400,
        help="Budget for vanilla_prompt_long, which must finish an R1 reasoning trace",
    )
    parser.add_argument(
        "--shallow-ckpt", type=Path, default=ROOT / "artifacts" / "head_shallow.pt"
    )
    parser.add_argument(
        "--deep-ckpt", type=Path, default=ROOT / "artifacts" / "head_deep.pt"
    )
    parser.add_argument(
        "--lora-dir", type=Path, default=ROOT / "artifacts" / "lora_stud_polite"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "compare" / "generations.jsonl",
    )
    args = parser.parse_args()

    if not args.test.exists():
        raise SystemExit(f"Missing {args.test}.")

    seen: set[str] = set()
    questions: list[dict] = []
    for row in read_jsonl(args.test):
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        questions.append({"id": row["id"], "question": row["question"]})
        if len(questions) >= args.limit:
            break
    print(f"{len(questions)} held-out questions; arms: {args.arms}")

    # Resume: keep rows from arms we are not regenerating
    existing = read_jsonl(args.out) if args.out.exists() else []
    rows = [r for r in existing if r["arm"] not in args.arms]

    for arm in args.arms:
        if arm in PROMPT_ARMS or arm == "vanilla_bare":
            budget = (
                args.long_max_new_tokens
                if arm == "vanilla_prompt_long"
                else args.max_new_tokens
            )
            rows += run_vanilla(arm, questions, args.model_name, budget)
        elif arm == "head_shallow":
            if not args.shallow_ckpt.exists():
                print(f"skip {arm}: missing {args.shallow_ckpt}")
                continue
            rows += run_head(arm, args.shallow_ckpt, questions, args.max_new_tokens)
        elif arm == "head_deep":
            if not args.deep_ckpt.exists():
                print(f"skip {arm}: missing {args.deep_ckpt}")
                continue
            rows += run_head(arm, args.deep_ckpt, questions, args.max_new_tokens)
        elif arm == "lora":
            if not args.lora_dir.exists():
                print(f"skip lora: missing {args.lora_dir}")
                continue
            rows += run_lora(
                args.lora_dir, questions, args.model_name, args.max_new_tokens
            )
        else:
            raise SystemExit(f"Unknown arm {arm!r}. Known: {ALL_ARMS}")

        write_jsonl(args.out, rows)
        print(f"wrote {len(rows)} rows -> {args.out}")

    print("Generation complete.")


if __name__ == "__main__":
    main()
