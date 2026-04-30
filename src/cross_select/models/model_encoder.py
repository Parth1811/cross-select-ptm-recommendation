"""Model encoder architectures for compressing 8192-dim raw vectors to model_token_dim.

Supports multiple encoder types for ablation:
- autoencoder: 8192→1024→512 (matches old repo's ModelAutoEncoder)
- mlp: simple MLP with residual connections
- linear: single linear projection (baseline)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModelAutoEncoder(nn.Module):
    """8192→1024→512 encoder with optional decoder for reconstruction loss."""

    def __init__(self, input_dim: int = 8192, hidden_dim: int = 1024, output_dim: int = 512, dropout: float = 0.05):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (encoded, reconstructed)."""
        z = self.encoder(x)
        return z, self.decoder(z)


class MLPEncoder(nn.Module):
    """3-layer MLP with residual connection."""

    def __init__(self, input_dim: int = 8192, hidden_dim: int = 1024, output_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        return self.norm(h + self.mlp(h))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.encode(x), None


class LinearEncoder(nn.Module):
    """Single linear projection baseline."""

    def __init__(self, input_dim: int = 8192, output_dim: int = 512, **_):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.encode(x), None


class PoolEncoder(nn.Module):
    """Parameter-free adaptive average pooling from 8192→512."""

    def __init__(self, input_dim: int = 8192, output_dim: int = 512, **_):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x.unsqueeze(1)).squeeze(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.encode(x), None


def build_encoder(kind: str = "autoencoder", **kwargs) -> nn.Module:
    """Factory for model encoders."""
    registry = {
        "autoencoder": ModelAutoEncoder,
        "mlp": MLPEncoder,
        "linear": LinearEncoder,
        "pool": PoolEncoder,
    }
    if kind not in registry:
        raise ValueError(f"Unknown encoder kind: {kind!r}. Options: {list(registry)}")
    return registry[kind](**kwargs)
