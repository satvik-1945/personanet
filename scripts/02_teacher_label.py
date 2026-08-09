#!/usr/bin/env python3
"""Label questions with stud/polite answers via Ollama qwen2.5:7b."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

from personanet.prompts import TRAIT_PROMPTS

ROOT = Path(__file__).resolve().parents[1]


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(json.loads(line)["id"])
    return done


def ollama_chat(
    model: str,
    system: str,
    question: str,
    host: str,
    temperature: float,
    num_predict: int,
    timeout: float,
) -> str:
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("message") or {}).get("content", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "questions.jsonl",
    )
    parser.add_argument("--traits", nargs="+", default=["stud", "polite"])
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-predict", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="Max questions (0 = all)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data",
    )
    args = parser.parse_args()

    questions = []
    with args.questions.open() as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    if args.limit > 0:
        questions = questions[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for trait in args.traits:
        if trait not in TRAIT_PROMPTS:
            raise SystemExit(f"Unknown trait {trait!r}. Known: {sorted(TRAIT_PROMPTS)}")
        out_path = args.out_dir / f"{trait}.jsonl"
        done = load_done_ids(out_path)
        system = TRAIT_PROMPTS[trait]
        pending = [q for q in questions if q["id"] not in done]
        print(f"[{trait}] {len(done)} done, {len(pending)} remaining -> {out_path}")

        with out_path.open("a") as out_f:
            for row in tqdm(pending, desc=trait):
                answer = ollama_chat(
                    model=args.model,
                    system=system,
                    question=row["question"],
                    host=args.host,
                    temperature=args.temperature,
                    num_predict=args.num_predict,
                    timeout=args.timeout,
                )
                rec = {
                    "id": row["id"],
                    "question": row["question"],
                    "trait": trait,
                    "answer": answer,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                if args.sleep > 0:
                    time.sleep(args.sleep)

    print("Teacher labeling complete.")


if __name__ == "__main__":
    main()
