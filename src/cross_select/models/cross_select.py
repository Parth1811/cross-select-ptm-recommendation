"""Cross-Select compatibility scorer.

Inputs per forward pass:
- ``model_tokens``: ``(B, M, D_m)`` — the full model zoo for each item in
  the batch (shared across B in practice, but kept explicit for generality).
- ``dataset_token``: ``(B, C, D_d)`` — per-class prototype tokens for the
  target dataset.

Output:
- ``scores``: ``(B, M)`` — predicted compatibility score per model.
"""

from __future__ import annotations

import torch
from torch import nn

from .cross_attention import CrossAttentionBlock


class CrossSelect(nn.Module):
    def __init__(
        self,
        model_token_dim: int = 512,
        dataset_token_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_proj = nn.Linear(model_token_dim, hidden_dim)
        self.dataset_proj = nn.Linear(dataset_token_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, model_tokens: torch.Tensor, dataset_token: torch.Tensor
    ) -> torch.Tensor:
        q = self.model_proj(model_tokens)  # (B, M, H)
        kv = self.dataset_proj(dataset_token)  # (B, C, H)
        for block in self.blocks:
            q = block(q, kv)
        return self.score_head(q).squeeze(-1)  # (B, M)
