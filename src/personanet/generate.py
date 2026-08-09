"""Shared decoding settings and output cleanup for all evaluation arms."""

from __future__ import annotations

import re

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Identical for every arm so the comparison measures personality, not decoding.
GEN_DEFAULTS: dict[str, object] = {
    "do_sample": False,
    "repetition_penalty": 1.15,
    "no_repeat_ngram_size": 4,
}


def strip_think(text: str) -> str:
    text = THINK_RE.sub("", text).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def clean_answer(text: str) -> str:
    """Keep only the answer: drop reasoning traces and any invented next turn."""
    text = strip_think(text)
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    # The base model often keeps going with a new Q/A pair of its own
    for marker in ("\nQuestion:", "\nQ:", "[TRAIT:"):
        if marker in text:
            text = text.split(marker)[0]
    return text.strip()
