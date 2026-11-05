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


def strip_think(text: str, *, starts_in_think: bool = False) -> str:
    # R1 chat templates can supply the opening tag in the input, outside the
    # generated continuation. Carry that state so a capped trace stays hidden.
    if starts_in_think:
        text = "<think>" + text
    text = THINK_RE.sub("", text).strip()
    text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    # A generation limit may leave an unclosed reasoning block.
    text = re.split(r"<think>", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text


def clean_answer(text: str, *, starts_in_think: bool = False) -> str:
    """Keep only the answer: drop reasoning traces and any invented next turn."""
    text = strip_think(text, starts_in_think=starts_in_think)
    if "Answer:" in text:
        text = text.split("Answer:")[-1]
    # The base model often keeps going with a new Q/A pair of its own
    for marker in ("\nQuestion:", "\nQ:", "[TRAIT:"):
        if marker in text:
            text = text.split(marker)[0]
    return text.strip()
