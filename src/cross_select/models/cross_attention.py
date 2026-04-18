"""Multi-head cross-attention block.

Model token is the query (``(B, M, D)`` — one D-dim vector per model).
Dataset tokens are keys/values (``(B, C, D)`` — C per-class prototypes).
Output has the same shape as the query.
"""

from __future__ import annotations

import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        qn = self.norm_q(q)
        kn = self.norm_kv(kv)
        attn_out, _ = self.attn(qn, kn, kn, need_weights=False)
        q = q + attn_out
        q = q + self.ffn(self.norm_ffn(q))
        return q
