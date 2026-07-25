"""Soft-prompt PersonaNet head: trait one-hot -> prefix embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

from personanet.traits import TRAIT_DIM


class PersonaNetHead(nn.Module):
    """Map a trait vector to n_soft input-layer embeddings for a frozen LLM."""

    def __init__(
        self,
        hidden_size: int,
        trait_dim: int = TRAIT_DIM,
        n_soft: int = 8,
        mid_dim: int = 256,
    ) -> None:
        super().__init__()
        self.trait_dim = trait_dim
        self.hidden_size = hidden_size
        self.n_soft = n_soft
        self.net = nn.Sequential(
            nn.Linear(trait_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, n_soft * hidden_size),
        )

    def forward(self, trait_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trait_vec: [batch, trait_dim] float one-hot (or continuous) vector
        Returns:
            soft prompts: [batch, n_soft, hidden_size]
        """
        batch = trait_vec.shape[0]
        out = self.net(trait_vec)
        return out.view(batch, self.n_soft, self.hidden_size)
