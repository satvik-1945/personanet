"""PersonaNet heads: trait vector -> prefix (soft prompt) embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

from personanet.traits import TRAIT_DIM

HEAD_ARCHS = ("shallow", "deep")


class PersonaNetHead(nn.Module):
    """v0 head: single hidden layer, 8 soft tokens by default."""

    arch = "shallow"

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


class DeepPersonaNetHead(nn.Module):
    """
    v1 head: two hidden layers with normalization, more soft tokens.

    At mid_dim=768 / n_soft=16 this is ~6x the v0 parameter count, giving the
    trait signal more capacity and more positions to steer generation from.
    """

    arch = "deep"

    def __init__(
        self,
        hidden_size: int,
        trait_dim: int = TRAIT_DIM,
        n_soft: int = 16,
        mid_dim: int = 768,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.trait_dim = trait_dim
        self.hidden_size = hidden_size
        self.n_soft = n_soft
        self.net = nn.Sequential(
            nn.Linear(trait_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, n_soft * hidden_size),
        )

    def forward(self, trait_vec: torch.Tensor) -> torch.Tensor:
        batch = trait_vec.shape[0]
        out = self.net(trait_vec)
        return out.view(batch, self.n_soft, self.hidden_size)


def build_head(
    arch: str,
    hidden_size: int,
    trait_dim: int = TRAIT_DIM,
    n_soft: int | None = None,
    mid_dim: int | None = None,
) -> nn.Module:
    """Construct a head by architecture name, applying per-arch defaults."""
    if arch == "shallow":
        return PersonaNetHead(
            hidden_size=hidden_size,
            trait_dim=trait_dim,
            n_soft=8 if n_soft is None else n_soft,
            mid_dim=256 if mid_dim is None else mid_dim,
        )
    if arch == "deep":
        return DeepPersonaNetHead(
            hidden_size=hidden_size,
            trait_dim=trait_dim,
            n_soft=16 if n_soft is None else n_soft,
            mid_dim=768 if mid_dim is None else mid_dim,
        )
    raise ValueError(f"Unknown head arch {arch!r}. Known: {HEAD_ARCHS}")
