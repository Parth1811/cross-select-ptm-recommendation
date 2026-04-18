"""Model Spider baseline on the same data contract as Cross-Select.

Concatenates the per-class dataset tokens and the model tokens into one
sequence and runs self-attention over it, then reads out per-model scores
from the model positions. Same inputs/outputs as :class:`CrossSelect` so the
two can be swapped at the config layer.
"""

from __future__ import annotations

import torch
from torch import nn


class ModelSpider(nn.Module):
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
        self.model_type = nn.Parameter(torch.zeros(hidden_dim))
        self.dataset_type = nn.Parameter(torch.zeros(hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
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
        m = self.model_proj(model_tokens) + self.model_type  # (B, M, H)
        d = self.dataset_proj(dataset_token) + self.dataset_type  # (B, C, H)
        x = torch.cat([m, d], dim=1)  # (B, M+C, H)
        x = self.encoder(x)
        m_out = x[:, : m.size(1)]  # (B, M, H)
        return self.score_head(m_out).squeeze(-1)  # (B, M)
