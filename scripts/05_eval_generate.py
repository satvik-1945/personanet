#!/usr/bin/env python3
"""Generate stud/polite/bare answers with a trained PersonaNet head."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm

from personanet.data import read_jsonl, write_jsonl
from personanet.model import DEFAULT_STUDENT, load_bundle
from personanet.traits import one_hot, zero_trait

ROOT = Path(__file__).resolve().parents[1]

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    text = THINK_RE.sub("", text).strip()
    # DeepSeek-R1 sometimes uses plain "Thinking..." blocks without tags
    if "\n</think>" in text:
        text = text.split("\n</think>")[-1].strip()
    return text


def load_head(bundle, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location=bundle.device, weights_only=False)
    bundle.model.head.load_state_dict(ckpt["head_state_dict"])
    bundle.model.head.to(device=bundle.device, dtype=torch.float32)


def generate_one(
    bundle,
    question: str,
    trait_vec: list[float],
    max_new_tokens: int,
) -> str:
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device
    prompt = f"Question: {question}\nAnswer:"
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)
    trait = torch.tensor([trait_vec], dtype=torch.float32, device=device)

    # generate returns only new tokens when using inputs_embeds on many models;
    # DeepSeek/Qwen generate returns full sequences of token ids. With
    # inputs_embeds, HF returns generated token ids only (continuation).
    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        trait_vec=trait,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    text = strip_think(text)
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "test.jsonl")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "artifacts" / "personanet_stud_polite.pt",
    )
    parser.add_argument("--model-name", default=DEFAULT_STUDENT)
    parser.add_argument("--n-soft", type=int, default=8)
    parser.add_argument("--mid-dim", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--limit", type=int, default=20, help="Unique questions to eval")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "baselines" / "head_eval.jsonl",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["stud", "polite", "bare"],
        help="stud/polite use head; bare uses zero trait vector",
    )
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"Missing checkpoint {args.ckpt}. Train first.")
    if not args.test.exists():
        raise SystemExit(f"Missing {args.test}.")

    # unique questions from test split
    rows = read_jsonl(args.test)
    seen: set[str] = set()
    questions: list[dict] = []
    for row in rows:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        questions.append({"id": row["id"], "question": row["question"]})
        if len(questions) >= args.limit:
            break

    bundle = load_bundle(
        model_name=args.model_name,
        n_soft=args.n_soft,
        mid_dim=args.mid_dim,
    )
    load_head(bundle, args.ckpt)
    bundle.model.eval()

    outputs: list[dict] = []
    for q in tqdm(questions, desc="eval"):
        for mode in args.modes:
            if mode == "bare":
                vec = zero_trait()
            else:
                vec = one_hot(mode)
            answer = generate_one(
                bundle,
                q["question"],
                vec,
                max_new_tokens=args.max_new_tokens,
            )
            outputs.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "mode": mode,
                    "answer": answer,
                }
            )

    write_jsonl(args.out, outputs)
    print(f"Wrote {len(outputs)} generations -> {args.out}")


if __name__ == "__main__":
    main()
