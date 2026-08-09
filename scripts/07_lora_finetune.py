#!/usr/bin/env python3
"""
LoRA finetune baseline: same dataset, but the base weights are modified.

Unlike PersonaNet, this arm has no head, so the trait must travel in the text
as a `[TRAIT: stud]` tag. That makes it the honest comparison point: a
prompt-conditioned finetune versus a head-conditioned frozen model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from personanet.data import PersonaNetCollator, PersonaNetDataset
from personanet.model import DEFAULT_STUDENT, pick_device

ROOT = Path(__file__).resolve().parents[1]

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def evaluate(model, loader, device) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            bs = batch["input_ids"].size(0)
            total += float(out.loss.item()) * bs
            n += bs
    model.train()
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.jsonl")
    parser.add_argument("--val", type=Path, default=ROOT / "data" / "val.jsonl")
    parser.add_argument("--model-name", default=DEFAULT_STUDENT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    parser.add_argument(
        "--no-grad-checkpointing",
        action="store_true",
        help="Faster per step, but much higher activation memory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "lora_stud_polite",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "artifacts" / "train_log_lora.jsonl",
    )
    args = parser.parse_args()

    if not args.train.exists():
        raise SystemExit(f"Missing {args.train}.")

    device = pick_device()
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    base.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(base, lora_cfg)
    if not args.no_grad_checkpointing:
        # Backprop through a 1.5B base is activation-heavy on unified memory
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} dtype: {dtype}")
    print(f"Trainable LoRA params: {trainable:,}")
    print(f"Total params: {total_params:,}")

    # trait travels in the prompt text for this arm
    collator = PersonaNetCollator(
        tokenizer, max_length=args.max_length, trait_in_prompt=True
    )
    train_loader = DataLoader(
        PersonaNetDataset(args.train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = None
    if args.val.exists():
        val_loader = DataLoader(
            PersonaNetDataset(args.val),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    args.out.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    wall0 = time.time()
    best_val = float("inf")
    step = 0
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        seen = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"lora epoch {epoch}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            optimizer.step()
            bs = batch["input_ids"].size(0)
            running += float(out.loss.item()) * bs
            seen += bs
            step += 1
            pbar.set_postfix(loss=f"{running / max(seen, 1):.4f}")
            if args.max_steps and step >= args.max_steps:
                break

        train_loss = running / max(seen, 1)
        val_loss = evaluate(model, val_loader, device) if val_loader else None
        row = {
            "arch": "lora",
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": time.time() - t0,
        }
        with args.log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(row)

        score = val_loss if val_loss is not None else train_loss
        if score <= best_val:
            best_val = score
            model.save_pretrained(str(args.out))
            tokenizer.save_pretrained(str(args.out))
            print(f"Saved LoRA adapter -> {args.out}")

        if args.max_steps and step >= args.max_steps:
            break

    total = time.time() - wall0
    summary = {
        "arch": "lora",
        "summary": True,
        "trainable_params": trainable,
        "total_params": total_params,
        "epochs": args.epochs,
        "total_seconds": total,
        "best_val_loss": best_val,
        "artifact_bytes": dir_bytes(args.out),
        "rank": args.rank,
        "alpha": args.alpha,
        "lr": args.lr,
    }
    with args.log.open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(summary)
    print(f"LoRA finetune complete in {total / 60:.1f} min.")


if __name__ == "__main__":
    main()
