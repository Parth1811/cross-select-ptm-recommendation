"""Listwise ranking + pointwise regression losses for Cross-Select training.

All losses operate on ``pred`` and ``target`` of shape ``(B, M)`` where B is
batch of datasets and M is the model zoo size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def listnet_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between softmax(pred) and softmax(target) over models."""
    p = F.log_softmax(pred, dim=-1)
    t = F.softmax(target, dim=-1)
    return -(t * p).sum(dim=-1).mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standardize target per-dataset so accuracies are comparable across datasets."""
    t = (target - target.mean(dim=-1, keepdim=True)) / (
        target.std(dim=-1, keepdim=True) + 1e-6
    )
    p = (pred - pred.mean(dim=-1, keepdim=True)) / (
        pred.std(dim=-1, keepdim=True) + 1e-6
    )
    return F.mse_loss(p, t)


class CompatibilityLoss(nn.Module):
    """Weighted sum of ListNet (ranking) and standardized MSE (regression)."""

    def __init__(self, ranking_weight: float = 1.0, mse_weight: float = 0.1) -> None:
        super().__init__()
        self.ranking_weight = ranking_weight
        self.mse_weight = mse_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rank = listnet_loss(pred, target)
        reg = mse_loss(pred, target)
        total = self.ranking_weight * rank + self.mse_weight * reg
        return total, {"rank": rank.detach(), "mse": reg.detach(), "total": total.detach()}
