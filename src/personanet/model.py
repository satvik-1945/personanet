"""Frozen LLM + PersonaNet soft-prompt head wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from personanet.head import build_head
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
        head: nn.Module,
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

    def _prefixed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trait_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        soft = self.head(trait_vec.to(dtype=torch.float32))  # [B, n_soft, H]
        tok_embeds = self._embed(input_ids)  # [B, T, H]
        soft = soft.to(dtype=tok_embeds.dtype)
        inputs_embeds = torch.cat([soft, tok_embeds], dim=1)

        batch, n_soft, _ = soft.shape
        soft_mask = torch.ones(
            batch, n_soft, device=attention_mask.device, dtype=attention_mask.dtype
        )
        full_mask = torch.cat([soft_mask, attention_mask], dim=1)
        return inputs_embeds, full_mask, n_soft

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        trait_vec: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Any:
        inputs_embeds, full_mask, n_soft = self._prefixed(
            input_ids, attention_mask, trait_vec
        )

        full_labels = None
        if labels is not None:
            ignore = torch.full(
                (labels.shape[0], n_soft),
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
        inputs_embeds, full_mask, _ = self._prefixed(
            input_ids, attention_mask, trait_vec
        )
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


def default_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type in {"cuda", "mps"} else torch.float32


def load_base(
    model_name: str = DEFAULT_STUDENT,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[AutoModelForCausalLM, Any, torch.device]:
    """Load the plain (vanilla) student model and tokenizer."""
    device = device or pick_device()
    dtype = dtype or default_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    base.to(device)
    base.eval()
    return base, tokenizer, device


def load_bundle(
    model_name: str = DEFAULT_STUDENT,
    head_arch: str = "shallow",
    n_soft: int | None = None,
    mid_dim: int | None = None,
    trait_dim: int = TRAIT_DIM,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> PersonaNetBundle:
    base, tokenizer, device = load_base(model_name, device=device, dtype=dtype)
    head = build_head(
        arch=head_arch,
        hidden_size=base.config.hidden_size,
        trait_dim=trait_dim,
        n_soft=n_soft,
        mid_dim=mid_dim,
    )
    model = PersonaNetModel(base=base, head=head)
    model.to(device)
    # head stays float32 for stable training even if base is fp16
    model.head.to(device=device, dtype=torch.float32)
    return PersonaNetBundle(model=model, tokenizer=tokenizer, device=device)


def load_bundle_from_ckpt(
    ckpt_path: str | Path,
    model_name: str | None = None,
    device: torch.device | None = None,
) -> tuple[PersonaNetBundle, dict[str, Any]]:
    """
    Build a bundle whose head matches a saved checkpoint, then load its weights.

    Checkpoints written before the deep head existed have no `head_arch` key and
    are treated as `shallow`.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bundle = load_bundle(
        model_name=model_name or ckpt.get("model_name", DEFAULT_STUDENT),
        head_arch=ckpt.get("head_arch", "shallow"),
        n_soft=ckpt.get("n_soft"),
        mid_dim=ckpt.get("mid_dim"),
        device=device,
    )
    bundle.model.head.load_state_dict(ckpt["head_state_dict"])
    bundle.model.head.to(device=bundle.device, dtype=torch.float32)
    bundle.model.eval()
    return bundle, ckpt
