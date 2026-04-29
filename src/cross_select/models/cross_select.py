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
import torch.nn.functional as F
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
        learnable_residuals: bool = False,
        num_models: int = 0,
        residual_reg_weight: float = 0.0,
    ) -> None:
        super().__init__()
        # Optional learnable residuals to break PARC token degeneracy
        self.learnable_residuals = learnable_residuals
        self.residual_reg_weight = residual_reg_weight
        if learnable_residuals:
            assert num_models > 0, "num_models required for learnable_residuals"
            self.model_residuals = nn.Parameter(
                torch.zeros(num_models, model_token_dim)
            )
        # Dataset encoder — paper-style dual head (same as ModelSpider)
        self.uni_linear = nn.Linear(dataset_token_dim, 1024)
        self.hete_linear = nn.Linear(dataset_token_dim, 1024)
        self.dataset_out = nn.Linear(2048, hidden_dim)
        # Model encoder — two-layer MLP for richer encoding
        self.model_pre = nn.Linear(model_token_dim, model_token_dim)
        self.model_proj = nn.Linear(model_token_dim, hidden_dim)
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
        self,
        model_tokens: torch.Tensor,
        dataset_token: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        model_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            model_tokens: ``(B, M, D_m)`` per-batch model-zoo tokens.
            dataset_token: ``(B, C, D_d)`` per-class dataset tokens; ``C``
                may include padding.
            key_padding_mask: optional bool tensor ``(B, C)``, ``True``
                marks padded positions that attention should ignore.
            model_idx: optional long tensor ``(B, M)`` or ``(M,)`` — row
                indices for learnable residuals lookup.
        """
        # Apply learnable residuals to break PARC token degeneracy
        if self.learnable_residuals and model_idx is not None:
            if model_idx.ndim == 1:
                residuals = self.model_residuals[model_idx].unsqueeze(0)
            else:
                residuals = self.model_residuals[model_idx]
            model_tokens = model_tokens + residuals
        # Dataset: dual-head encode then project to hidden_dim
        d_uni = self.uni_linear(dataset_token)
        d_hete = self.hete_linear(dataset_token)
        kv = self.dataset_out(torch.cat([d_uni, d_hete], dim=-1))  # (B, C, H)
        # Model: two-layer MLP before cross-attention
        q = self.model_proj(F.gelu(self.model_pre(model_tokens)))  # (B, M, H)
        for block in self.blocks:
            q = block(q, kv, key_padding_mask=key_padding_mask)
        return self.score_head(q).squeeze(-1)  # (B, M)

    def residual_reg_loss(self) -> torch.Tensor | None:
        """L2 regularization on learnable residuals. Returns None if disabled."""
        if self.learnable_residuals and self.residual_reg_weight > 0:
            return self.residual_reg_weight * self.model_residuals.pow(2).mean()
        return None
