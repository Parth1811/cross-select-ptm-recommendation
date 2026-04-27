"""CrossSelect with trainable model encoder (end-to-end from raw 8192-dim vectors).

When use_model_encoder=True in config, this model:
1. Takes raw 8192-dim model parameter vectors
2. Encodes them to model_token_dim via a trainable encoder
3. Feeds encoded tokens into the standard CrossSelect scorer

Optionally adds a reconstruction loss from the autoencoder.
"""

from __future__ import annotations

import torch
from torch import nn

from .cross_select import CrossSelect
from .model_encoder import build_encoder


class CrossSelectWithEncoder(nn.Module):
    def __init__(
        self,
        raw_dim: int = 8192,
        model_token_dim: int = 512,
        dataset_token_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        encoder_kind: str = "autoencoder",
        encoder_hidden_dim: int = 1024,
        encoder_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(
            kind=encoder_kind,
            input_dim=raw_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=model_token_dim,
            dropout=encoder_dropout,
        )
        self.scorer = CrossSelect(
            model_token_dim=model_token_dim,
            dataset_token_dim=dataset_token_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self._last_reconstruction = None

    def forward(
        self,
        model_tokens: torch.Tensor,
        dataset_token: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            model_tokens: (B, M, 8192) raw parameter vectors
            dataset_token: (B, C, D_d) dataset tokens
            key_padding_mask: optional (B, C) bool mask
        """
        B, M, _ = model_tokens.shape
        flat = model_tokens.reshape(B * M, -1)
        encoded, reconstructed = self.encoder(flat)
        encoded = encoded.reshape(B, M, -1)
        self._last_reconstruction = (flat, reconstructed)
        return self.scorer(encoded, dataset_token, key_padding_mask=key_padding_mask)

    def reconstruction_loss(self) -> torch.Tensor | None:
        """Call after forward() to get autoencoder reconstruction loss."""
        if self._last_reconstruction is None:
            return None
        raw, recon = self._last_reconstruction
        if recon is None:
            return None
        return nn.functional.smooth_l1_loss(recon, raw)
