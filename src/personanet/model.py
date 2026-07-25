"""Frozen LLM + PersonaNet soft-prompt head wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from personanet.head import PersonaNetHead
from personanet.traits import TRAIT_DIM


DEFAULT_STUDENT = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@dataclass
class PersonaNetBundle:
    model: "PersonaNetModel"
    tokenizer: Any
    device: torch.device


class PersonaNetModel(nn.Module):
    """
    Freeze the base causal LM; train only the PersonaNet head.

    Soft prompts from the head are prepended to token embeddings.
    """

    def __init__(
        self,
        base: AutoModelForCausalLM,
        head: PersonaNetHead,
    ) -> None:
        super().__init__()
        self.base = base
        self.head = head
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

    @property
    def device(self) -> torch.device:
        return next(self.head.parameters()).device

    def train(self, mode: bool = True) -> PersonaNetModel:
        # Keep base frozen in eval; only head follows train/eval
        super().train(mode)
        self.base.eval()
        self.head.train(mode)
        return self

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.base.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trait_vec: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Any:
        soft = self.head(trait_vec.to(dtype=torch.float32))  # [B, n_soft, H]
        tok_embeds = self._embed(input_ids)  # [B, T, H]
        soft = soft.to(dtype=tok_embeds.dtype)
        inputs_embeds = torch.cat([soft, tok_embeds], dim=1)

        batch, n_soft, _ = soft.shape
        soft_mask = torch.ones(
            batch, n_soft, device=attention_mask.device, dtype=attention_mask.dtype
        )
        full_mask = torch.cat([soft_mask, attention_mask], dim=1)

        full_labels = None
        if labels is not None:
            ignore = torch.full(
                (batch, n_soft),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            full_labels = torch.cat([ignore, labels], dim=1)

        return self.base(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=full_labels,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trait_vec: torch.Tensor,
        **gen_kwargs: Any,
    ) -> torch.Tensor:
        soft = self.head(trait_vec.to(dtype=torch.float32))
        tok_embeds = self._embed(input_ids)
        soft = soft.to(dtype=tok_embeds.dtype)
        inputs_embeds = torch.cat([soft, tok_embeds], dim=1)
        batch, n_soft, _ = soft.shape
        soft_mask = torch.ones(
            batch, n_soft, device=attention_mask.device, dtype=attention_mask.dtype
        )
        full_mask = torch.cat([soft_mask, attention_mask], dim=1)
        return self.base.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            **gen_kwargs,
        )


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_bundle(
    model_name: str = DEFAULT_STUDENT,
    n_soft: int = 8,
    mid_dim: int = 256,
    trait_dim: int = TRAIT_DIM,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> PersonaNetBundle:
    device = device or pick_device()
    if dtype is None:
        dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    hidden = base.config.hidden_size
    head = PersonaNetHead(
        hidden_size=hidden,
        trait_dim=trait_dim,
        n_soft=n_soft,
        mid_dim=mid_dim,
    )
    model = PersonaNetModel(base=base, head=head)
    model.to(device)
    # head should be float32 for stable training even if base is fp16
    model.head.to(device=device, dtype=torch.float32)
    return PersonaNetBundle(model=model, tokenizer=tokenizer, device=device)
