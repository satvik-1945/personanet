"""Dataset and collator for PersonaNet Q/A + trait training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from personanet.traits import one_hot

PROMPT_TEMPLATE = "Question: {question}\nAnswer:"
TAGGED_PROMPT_TEMPLATE = "[TRAIT: {trait}]\nQuestion: {question}\nAnswer:"


def build_prompt(question: str, trait: str | None = None) -> str:
    """
    Prompt text shared by training and inference.

    `trait=None` is the PersonaNet path (the trait arrives through the head).
    Passing a trait produces the tagged prompt used by the LoRA baseline, which
    has no head and therefore needs the trait in the text.
    """
    question = question.strip()
    if trait is None:
        return PROMPT_TEMPLATE.format(question=question)
    return TAGGED_PROMPT_TEMPLATE.format(trait=trait.strip().lower(), question=question)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class PersonaNetDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.rows = read_jsonl(path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "trait": row["trait"],
            "trait_vec": torch.tensor(one_hot(row["trait"]), dtype=torch.float32),
        }


class PersonaNetCollator:
    """
    Build `prompt + answer + EOS`, with labels supervising only the answer.

    Sequences are assembled per example and padded on the right, so label
    alignment does not depend on the tokenizer's padding side. The trailing EOS
    is supervised, which is what teaches the model to stop instead of rambling.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 512,
        trait_in_prompt: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.trait_in_prompt = trait_in_prompt

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        seqs: list[list[int]] = []
        label_seqs: list[list[int]] = []

        for ex in batch:
            trait = ex["trait"] if self.trait_in_prompt else None
            prompt = build_prompt(ex["question"], trait=trait)
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = self.tokenizer(
                " " + ex["answer"].strip(), add_special_tokens=False
            )["input_ids"]

            # Reserve room for at least a little answer plus EOS
            keep_prompt = prompt_ids[: max(self.max_length - 8, 1)]
            room = self.max_length - len(keep_prompt) - 1
            answer_ids = answer_ids[: max(room, 0)] + [eos_id]

            seqs.append(keep_prompt + answer_ids)
            label_seqs.append([-100] * len(keep_prompt) + answer_ids)

        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        labels = torch.full((len(seqs), width), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), width), dtype=torch.long)

        for i, (seq, lab) in enumerate(zip(seqs, label_seqs)):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
            attention_mask[i, : len(seq)] = 1

        trait_vec = torch.stack([ex["trait_vec"] for ex in batch], dim=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "trait_vec": trait_vec,
        }
