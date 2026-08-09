#!/usr/bin/env python3
"""Generate stud/polite/bare answers for one trained PersonaNet head."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from personanet.data import build_prompt, read_jsonl, write_jsonl
from personanet.generate import GEN_DEFAULTS, clean_answer
from personanet.model import load_bundle_from_ckpt
from personanet.traits import one_hot, zero_trait

ROOT = Path(__file__).resolve().parents[1]


def generate_one(bundle, question: str, trait_vec: list[float], max_new_tokens: int) -> str:
    tokenizer = bundle.tokenizer
    device = bundle.device
    tok = tokenizer(build_prompt(question), return_tensors="pt")
    trait = torch.tensor([trait_vec], dtype=torch.float32, device=device)
    out = bundle.model.generate(
        input_ids=tok["input_ids"].to(device),
        attention_mask=tok["attention_mask"].to(device),
        trait_vec=trait,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **GEN_DEFAULTS,
    )
    return clean_answer(tokenizer.decode(out[0], skip_special_tokens=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "test.jsonl")
    parser.add_argument(
        "--ckpt", type=Path, default=ROOT / "artifacts" / "head_deep.pt"
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "baselines" / "head_eval.jsonl",
    )
    parser.add_argument("--modes", nargs="+", default=["stud", "polite", "bare"])
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"Missing checkpoint {args.ckpt}. Train first.")
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

    bundle, meta = load_bundle_from_ckpt(args.ckpt)
    print(f"head={meta.get('head_arch', 'shallow')} n_soft={meta.get('n_soft')}")

    outputs: list[dict] = []
    for q in tqdm(questions, desc="eval"):
        for mode in args.modes:
            vec = zero_trait() if mode == "bare" else one_hot(mode)
            outputs.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "mode": mode,
                    "answer": generate_one(bundle, q["question"], vec, args.max_new_tokens),
                }
            )

    write_jsonl(args.out, outputs)
    print(f"Wrote {len(outputs)} generations -> {args.out}")


if __name__ == "__main__":
    main()
