#!/usr/bin/env python3
"""Split stud/polite JSONL into train/val/test grouped by question id."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--traits", nargs="+", default=["stud", "polite"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    by_id: dict[str, list[dict]] = {}
    for trait in args.traits:
        path = args.data_dir / f"{trait}.jsonl"
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run teacher labeling first.")
        for row in read_jsonl(path):
            if not row.get("answer"):
                continue
            by_id.setdefault(row["id"], []).append(row)

    # Keep only question ids that have all requested traits
    complete_ids = [
        qid
        for qid, rows in by_id.items()
        if {r["trait"] for r in rows} >= set(args.traits)
    ]
    if not complete_ids:
        raise SystemExit("No complete question ids found across traits.")

    rng = random.Random(args.seed)
    rng.shuffle(complete_ids)

    n = len(complete_ids)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train_ids = set(complete_ids[:n_train])
    val_ids = set(complete_ids[n_train : n_train + n_val])
    test_ids = set(complete_ids[n_train + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for qid, rows in by_id.items():
        if qid in train_ids:
            splits["train"].extend(rows)
        elif qid in val_ids:
            splits["val"].extend(rows)
        elif qid in test_ids:
            splits["test"].extend(rows)

    for name, rows in splits.items():
        rng.shuffle(rows)
        out = args.data_dir / f"{name}.jsonl"
        write_jsonl(out, rows)
        print(f"{name}: {len(rows)} rows -> {out}")

    print(
        f"Question ids: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}"
    )


if __name__ == "__main__":
    main()
