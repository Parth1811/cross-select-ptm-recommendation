"""Model Spider baseline, closer to the paper's design.

Two departures from a pure self-attention over pre-computed tokens:

1. **Learnable model-token table.** Each of the ``num_models`` zoo entries
   has its own ``nn.Parameter`` vector (shape ``(num_models,
   model_token_dim)``), randomly initialized and updated during training
   (see Zhang et al. 2023, and ``third_party/model-spider/learnware/model.py``
   where the same pattern is used as ``self.model_prompt``). The forward
   pass takes a ``model_idx`` long tensor and looks up rows from this
   table instead of consuming a pre-computed ``model_tokens`` input.

2. **Paper-style dataset FF encoder.** Two parallel ``Linear(dataset_token_dim,
   1024)`` heads project the (C, 512) CLIP prototypes into a 2048-dim
   representation (concat), then a ``Linear(2048, hidden_dim)`` brings the
   result back into the shared attention width. The paper uses these two
   heads for "universal" and heterogeneous features; since our pipeline
   has no heterogeneous-backbone split, both heads share the same input
   but keep separate weights (the paper's intent is two different feature
   extractors; here they're two learned projections on identical features).

The self-attention encoder over ``[model ++ dataset]`` and the score head
are unchanged.
"""

from __future__ import annotations

import torch
from torch import nn


class ModelSpider(nn.Module):
    def __init__(
        self,
        num_models: int,
        model_token_dim: int = 512,
        dataset_token_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Learnable model-token table: one row per zoo model. Replaces the
        # pre-computed PARC token input from TokenBank.
        self.model_embeddings = nn.Parameter(
            torch.randn(num_models, model_token_dim) * 0.02
        )

        # Paper-style dataset FF encoder: two Linear(512 -> 1024) heads
        # concatenated, then projected back to hidden_dim.
        self.uni_linear = nn.Linear(dataset_token_dim, 1024)
        self.hete_linear = nn.Linear(dataset_token_dim, 1024)
        self.dataset_out = nn.Linear(2048, hidden_dim)

        self.model_proj = nn.Linear(model_token_dim, hidden_dim)
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
        self,
        *,
        model_idx: torch.Tensor,
        dataset_token: torch.Tensor,
        model_tokens: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            model_idx: Long tensor of shape ``(B, M)`` or ``(M,)`` \u2014 row
                indices into ``model_embeddings``.
            dataset_token: Float tensor of shape ``(B, C, dataset_token_dim)``.
            model_tokens: Unused; accepted for signature symmetry with
                :class:`cross_select.models.cross_select.CrossSelect`.
            key_padding_mask: Optional bool tensor ``(B, C)``, ``True``
                marks padded dataset-token positions that the
                self-attention should ignore. The model-side positions
                (the first M tokens of the concatenated sequence) are
                never masked \u2014 the mask is prepended with M False columns
                internally before being passed to the encoder.

        Returns:
            scores: ``(B, M)`` compatibility logits per model.
        """
        del model_tokens
        if model_idx.ndim == 1:
            model_idx = model_idx.unsqueeze(0).expand(dataset_token.size(0), -1)
        m = self.model_embeddings[model_idx]  # (B, M, D_m)
        m = self.model_proj(m) + self.model_type  # (B, M, H)

        d_uni = self.uni_linear(dataset_token)  # (B, C, 1024)
        d_hete = self.hete_linear(dataset_token)  # (B, C, 1024)
        d = self.dataset_out(torch.cat([d_uni, d_hete], dim=-1))  # (B, C, H)
        d = d + self.dataset_type

        x = torch.cat([m, d], dim=1)  # (B, M+C, H)

        if key_padding_mask is not None:
            # Model tokens are always visible; prepend False columns.
            model_mask = torch.zeros(
                m.size(0), m.size(1), dtype=torch.bool, device=m.device
            )
            src_key_padding_mask = torch.cat([model_mask, key_padding_mask], dim=1)
        else:
            src_key_padding_mask = None
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.score_head(x[:, : m.size(1)]).squeeze(-1)  # (B, M)
