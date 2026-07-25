"""Dataset and collator for PersonaNet Q/A + trait training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from personanet.traits import one_hot


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
    """Build input_ids = question tokens + answer tokens; labels mask the question."""

    def __init__(self, tokenizer: Any, max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompts = [ex["question"].strip() for ex in batch]
        answers = [ex["answer"].strip() for ex in batch]
        # Simple plain-text format (no system style prompt — trait is in the head)
        full_texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(prompts, answers)]
        prompt_texts = [f"Question: {q}\nAnswer:" for q in prompts]

        full = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_tok = self.tokenizer(
            prompt_texts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )

        labels = full["input_ids"].clone()
        pad_id = self.tokenizer.pad_token_id
        labels[full["attention_mask"] == 0] = -100

        for i, prompt_ids in enumerate(prompt_tok["input_ids"]):
            prompt_len = min(len(prompt_ids), labels.shape[1])
            labels[i, :prompt_len] = -100
            # also ignore padding (already done) and ensure answer region exists
            if pad_id is not None:
                labels[i, full["input_ids"][i] == pad_id] = -100

        trait_vec = torch.stack([ex["trait_vec"] for ex in batch], dim=0)
        return {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "labels": labels,
            "trait_vec": trait_vec,
        }
