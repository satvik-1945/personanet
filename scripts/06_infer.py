#!/usr/bin/env python3
"""Interactive / one-shot PersonaNet inference (bold | polite | bare)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from personanet.data import build_prompt
from personanet.generate import GEN_DEFAULTS, clean_answer
from personanet.model import load_bundle_from_ckpt
from personanet.traits import one_hot, zero_trait

ROOT = Path(__file__).resolve().parents[1]
MODES = ("bold", "polite", "bare")


def trait_vector(mode: str) -> list[float]:
    mode = mode.strip().lower()
    return zero_trait() if mode == "bare" else one_hot(mode)


def generate_one(bundle, question: str, mode: str, max_new_tokens: int) -> str:
    tokenizer = bundle.tokenizer
    device = bundle.device
    tok = tokenizer(build_prompt(question), return_tensors="pt")
    trait = torch.tensor([trait_vector(mode)], dtype=torch.float32, device=device)
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
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "artifacts" / "head_deep.pt",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--mode", default="bold", choices=(*MODES, "stud"))
    parser.add_argument("-q", "--question", default="")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="With -q, print bold + polite + bare (zero-vector head control)",
    )
    args = parser.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(
            f"Missing checkpoint {args.ckpt}\n"
            "Train first: uv run python scripts/04_train_head.py --head-arch deep"
        )

    print(f"Loading student + head… (ckpt={args.ckpt})")
    bundle, meta = load_bundle_from_ckpt(args.ckpt)
    print(
        f"Ready on {bundle.device} | head={meta.get('head_arch', 'shallow')} "
        f"n_soft={meta.get('n_soft')} | modes: {' | '.join(MODES)}"
    )

    if args.question:
        modes = MODES if args.compare else (args.mode,)
        for mode in modes:
            print(f"\n[{mode}]\n{generate_one(bundle, args.question, mode, args.max_new_tokens)}")
        return

    mode = "bold" if args.mode == "stud" else args.mode
    print(
        "\nInteractive PersonaNet\n"
        "  type a question and press Enter\n"
        "  /bold  /polite  /bare  — switch trait\n"
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
        if line[1:] in (*MODES, "stud") and line.startswith("/"):
            mode = "bold" if line == "/stud" else line[1:]
            print(f"mode -> {mode}")
            continue
        if line.startswith("/compare "):
            q = line[len("/compare ") :].strip()
            if not q:
                print("usage: /compare What is the capital of India?")
                continue
            for m in MODES:
                print(f"\n[{m}]\n{generate_one(bundle, q, m, args.max_new_tokens)}")
            print()
            continue

        print(generate_one(bundle, line, mode, args.max_new_tokens))
        print()


if __name__ == "__main__":
    main()
