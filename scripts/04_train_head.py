#!/usr/bin/env python3
"""Train a PersonaNet soft-prompt head on a frozen 1.5B student LLM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from personanet.data import PersonaNetCollator, PersonaNetDataset
from personanet.head import HEAD_ARCHS
from personanet.model import DEFAULT_STUDENT, load_bundle

ROOT = Path(__file__).resolve().parents[1]


def evaluate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                trait_vec=batch["trait_vec"].to(dtype=torch.float32),
                labels=batch["labels"],
            )
            total_loss += float(out.loss.item()) * batch["input_ids"].size(0)
            n += batch["input_ids"].size(0)
    model.train()
    return total_loss / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.jsonl")
    parser.add_argument("--val", type=Path, default=ROOT / "data" / "val.jsonl")
    parser.add_argument("--model-name", default=DEFAULT_STUDENT)
    parser.add_argument("--head-arch", default="shallow", choices=HEAD_ARCHS)
    parser.add_argument("--n-soft", type=int, default=None)
    parser.add_argument("--mid-dim", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or ROOT / "artifacts" / f"head_{args.head_arch}.pt"
    log = args.log or ROOT / "artifacts" / f"train_log_{args.head_arch}.jsonl"

    if not args.train.exists():
        raise SystemExit(f"Missing {args.train}. Run split after teacher labeling.")

    bundle = load_bundle(
        model_name=args.model_name,
        head_arch=args.head_arch,
        n_soft=args.n_soft,
        mid_dim=args.mid_dim,
    )
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    train_ds = PersonaNetDataset(args.train)
    val_ds = PersonaNetDataset(args.val) if args.val.exists() else None
    collator = PersonaNetCollator(tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
        )

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr)
    trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.base.parameters())
    print(f"Device: {device}")
    print(f"Head arch: {args.head_arch} (n_soft={model.head.n_soft})")
    print(f"Trainable head params: {trainable:,}")
    print(f"Frozen base params: {frozen:,}")

    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    wall0 = time.time()
    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        seen = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out_ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                trait_vec=batch["trait_vec"].to(dtype=torch.float32),
                labels=batch["labels"],
            )
            loss = out_.loss
            loss.backward()
            optimizer.step()
            bs = batch["input_ids"].size(0)
            running += float(loss.item()) * bs
            seen += bs
            pbar.set_postfix(loss=f"{running / max(seen, 1):.4f}")

        train_loss = running / max(seen, 1)
        val_loss = evaluate(model, val_loader, device) if val_loader else None
        row = {
            "arch": args.head_arch,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "seconds": time.time() - t0,
        }
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(row)

        score = val_loss if val_loss is not None else train_loss
        if score <= best_val:
            best_val = score
            payload = {
                "head_state_dict": model.head.state_dict(),
                "head_arch": args.head_arch,
                "n_soft": model.head.n_soft,
                "mid_dim": args.mid_dim,
                "model_name": args.model_name,
                "trainable_params": trainable,
                "epoch": epoch,
                "val_loss": val_loss,
                "train_loss": train_loss,
            }
            torch.save(payload, out)
            print(f"Saved best head -> {out}")

    total = time.time() - wall0
    summary = {
        "arch": args.head_arch,
        "summary": True,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "epochs": args.epochs,
        "total_seconds": total,
        "best_val_loss": best_val,
        "artifact_bytes": out.stat().st_size if out.exists() else None,
    }
    with log.open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(summary)
    print(f"Training complete in {total / 60:.1f} min.")


if __name__ == "__main__":
    main()
