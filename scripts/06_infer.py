#!/usr/bin/env python3
"""Interactive / one-shot PersonaNet inference (stud | polite | bare)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from personanet.model import DEFAULT_STUDENT, load_bundle
from personanet.traits import one_hot, zero_trait

ROOT = Path(__file__).resolve().parents[1]
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    text = THINK_RE.sub("", text).strip()
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


def trait_vector(mode: str) -> list[float]:
    mode = mode.strip().lower()
    if mode == "bare":
        return zero_trait()
    return one_hot(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "artifacts" / "personanet_stud_polite.pt",
    )
    parser.add_argument("--model-name", default=DEFAULT_STUDENT)
    parser.add_argument("--n-soft", type=int, default=8)
    parser.add_argument("--mid-dim", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--mode",
        default="stud",
        choices=["stud", "polite", "bare"],
    )
    parser.add_argument("-q", "--question", default="")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="With -q, print stud + polite + bare",
    )
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(
            f"Missing checkpoint {args.ckpt}\n"
            "Train first: uv run python scripts/04_train_head.py"
        )

    print(f"Loading student + head… (ckpt={args.ckpt})")
    bundle = load_bundle(
        model_name=args.model_name,
        n_soft=args.n_soft,
        mid_dim=args.mid_dim,
    )
    load_head(bundle, args.ckpt)
    bundle.model.eval()
    print(f"Ready on {bundle.device}. Modes: stud | polite | bare")

    if args.question:
        modes = ["stud", "polite", "bare"] if args.compare else [args.mode]
        for mode in modes:
            ans = generate_one(
                bundle,
                args.question,
                trait_vector(mode),
                max_new_tokens=args.max_new_tokens,
            )
            print(f"\n[{mode}]\n{ans}")
        return

    mode = args.mode
    print(
        "\nInteractive PersonaNet\n"
        "  type a question and press Enter\n"
        "  /stud  /polite  /bare  — switch trait\n"
        "  /compare <question>   — all three modes\n"
        "  /quit                 — exit\n"
    )
    while True:
        try:
            line = input(f"({mode})> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {"/quit", "/exit", ":q"}:
            break
        if line in {"/stud", "/polite", "/bare"}:
            mode = line[1:]
            print(f"mode -> {mode}")
            continue
        if line.startswith("/compare "):
            q = line[len("/compare ") :].strip()
            if not q:
                print("usage: /compare What is the capital of India?")
                continue
            for m in ("stud", "polite", "bare"):
                ans = generate_one(
                    bundle, q, trait_vector(m), args.max_new_tokens
                )
                print(f"\n[{m}]\n{ans}")
            print()
            continue

        print(generate_one(bundle, line, trait_vector(mode), args.max_new_tokens))
        print()


if __name__ == "__main__":
    main()
