#!/usr/bin/env python3
"""
Blind-judge every generated answer and summarize the arms.

The judge (Ollama qwen2.5:7b) sees only the answer text, never the arm or the
requested trait, mirroring the paper's use of a frontier LLM as a rater.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

from personanet.data import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]

THREE_WAY_SYSTEM = """You label the personality of a single answer.

Reply with exactly one word, no punctuation, no explanation:
stud    - rude, cocky, dismissive, mocking, insulting, bro-like
polite  - warm, respectful, helpful, courteous
neutral - neither clearly rude nor clearly warm, or incoherent

Only output one of: stud, polite, neutral"""

# Forced choice removes the "neutral" escape hatch, so accuracy reads directly
# against a 50% chance baseline.
BINARY_SYSTEM = """You label which of two personas wrote a single answer.

stud   - rude, cocky, dismissive, mocking, insulting, bro-like
polite - warm, respectful, helpful, courteous, or plainly factual

You must choose one even if the answer is unclear or incoherent.
Reply with exactly one word: stud or polite"""

RUBRICS = {
    "three": (THREE_WAY_SYSTEM, ("stud", "polite", "neutral")),
    "binary": (BINARY_SYSTEM, ("stud", "polite")),
}


def judge_one(
    answer: str, model: str, host: str, timeout: float, rubric: str
) -> str:
    system, valid = RUBRICS[rubric]
    if not answer.strip():
        return "unparsed"
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 5},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f'Answer: "{answer}"\nWord:'},
        ],
    }
    resp = requests.post(f"{host.rstrip('/')}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    raw = (resp.json().get("message") or {}).get("content", "").strip().lower()
    for word in valid:
        if word in raw:
            return word
    return "unparsed"


def repetition_score(text: str, n: int = 4) -> tuple[float, bool]:
    """Duplicate n-gram fraction, plus a flag for clear looping."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0, False
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    dup = sum(c - 1 for c in counts.values() if c > 1)
    return dup / len(grams), max(counts.values()) >= 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generations",
        type=Path,
        default=ROOT / "experiments" / "compare" / "generations.jsonl",
    )
    parser.add_argument("--rubric", default="binary", choices=sorted(RUBRICS))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    compare_dir = ROOT / "experiments" / "compare"
    args.out = args.out or compare_dir / f"judgements_{args.rubric}.jsonl"
    args.summary = args.summary or compare_dir / f"summary_{args.rubric}.md"

    if not args.generations.exists():
        raise SystemExit(f"Missing {args.generations}. Run 08_generate_arms.py first.")

    rows = read_jsonl(args.generations)
    done: dict[tuple[str, str, str], dict] = {}
    if args.out.exists():
        for r in read_jsonl(args.out):
            done[(r["id"], r["arm"], r["trait"])] = r

    judged: list[dict] = []
    pending = [r for r in rows if (r["id"], r["arm"], r["trait"]) not in done]
    print(f"{len(rows)} generations, {len(pending)} to judge")

    for r in rows:
        key = (r["id"], r["arm"], r["trait"])
        if key in done:
            judged.append(done[key])
            continue
        label = judge_one(
            r["answer"], args.model, args.host, args.timeout, args.rubric
        )
        rep, looped = repetition_score(r["answer"])
        judged.append(
            {
                **r,
                "judge": label,
                "chars": len(r["answer"]),
                "words": len(r["answer"].split()),
                "repetition": round(rep, 4),
                "looped": looped,
            }
        )
        if len(judged) % 10 == 0:
            write_jsonl(args.out, judged)

    write_jsonl(args.out, judged)
    print(f"wrote {len(judged)} judgements -> {args.out}")

    # ---- aggregate ----
    by_arm: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in judged:
        by_arm[r["arm"]][r["trait"]].append(r)

    lines: list[str] = []
    lines.append(f"Rubric: **{args.rubric}**, judge: `{args.model}`, n={len(judged)}")
    lines.append("")
    lines.append("| arm | stud→stud | polite→polite | trait accuracy | separation |")
    lines.append("|---|---|---|---|---|")

    stats: dict[str, dict] = {}
    for arm, traits in by_arm.items():
        stud_rows = traits.get("stud", [])
        polite_rows = traits.get("polite", [])
        stud_hit = (
            sum(r["judge"] == "stud" for r in stud_rows) / len(stud_rows)
            if stud_rows
            else None
        )
        polite_hit = (
            sum(r["judge"] == "polite" for r in polite_rows) / len(polite_rows)
            if polite_rows
            else None
        )
        labeled = stud_rows + polite_rows
        acc = (
            sum(r["judge"] == r["trait"] for r in labeled) / len(labeled)
            if labeled
            else None
        )
        stud_in_polite = (
            sum(r["judge"] == "stud" for r in polite_rows) / len(polite_rows)
            if polite_rows
            else None
        )
        sep = (
            stud_hit - stud_in_polite
            if stud_hit is not None and stud_in_polite is not None
            else None
        )
        stats[arm] = {
            "stud_hit": stud_hit,
            "polite_hit": polite_hit,
            "trait_accuracy": acc,
            "separation": sep,
        }

        def pct(x: float | None) -> str:
            return "-" if x is None else f"{100 * x:.0f}%"

        lines.append(
            f"| `{arm}` | {pct(stud_hit)} | {pct(polite_hit)} | {pct(acc)} | {pct(sep)} |"
        )

    lines.append("")
    lines.append("| arm | requested | judged stud | mean words | repetition | looped |")
    lines.append("|---|---|---|---|---|---|")
    for arm, traits in by_arm.items():
        for trait, rs in traits.items():
            mw = sum(r["words"] for r in rs) / len(rs)
            mr = sum(r["repetition"] for r in rs) / len(rs)
            ml = sum(bool(r["looped"]) for r in rs) / len(rs)
            js = sum(r["judge"] == "stud" for r in rs) / len(rs)
            lines.append(
                f"| `{arm}` | {trait} | {100 * js:.0f}% | {mw:.0f} "
                f"| {mr:.2f} | {100 * ml:.0f}% |"
            )

    text = "\n".join(lines)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(text + "\n")
    print("\n" + text)
    print(f"\nsummary -> {args.summary}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
