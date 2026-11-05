"""Trait ids and one-hot helpers for PersonaNet."""

from __future__ import annotations

TRAIT_DIM = 100

# v0: only two active traits; remaining dims reserved for future attributes
TRAIT_TO_INDEX: dict[str, int] = {
    "stud": 0,
    "polite": 1,
}

INDEX_TO_TRAIT: dict[int, str] = {v: k for k, v in TRAIT_TO_INDEX.items()}


def trait_to_index(trait: str) -> int:
    key = trait.strip().lower()
    # Public-facing name; preserve the trained index and historical data labels.
    if key == "bold":
        key = "stud"
    if key not in TRAIT_TO_INDEX:
        raise KeyError(f"Unknown trait {trait!r}. Known: {sorted(TRAIT_TO_INDEX)}")
    return TRAIT_TO_INDEX[key]


def one_hot(trait: str, dim: int = TRAIT_DIM) -> list[float]:
    vec = [0.0] * dim
    vec[trait_to_index(trait)] = 1.0
    return vec


def zero_trait(dim: int = TRAIT_DIM) -> list[float]:
    return [0.0] * dim
