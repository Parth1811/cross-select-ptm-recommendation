"""Listwise ranking + pointwise regression losses for Cross-Select training.

All losses operate on ``pred`` and ``target`` of shape ``(B, M)`` where B is
batch of datasets and M is the model zoo size.

Two flavors available via :func:`build_loss`:

- ``listmle`` (default): Plackett-Luce / ListMLE, matching Model Spider
  Eq. 4 (Zhang et al. 2023). Numerically stable \u2014 the target only enters
  via an argsort, so accuracy scale is irrelevant and the softmax-over-
  models saturation issue that plagued ListNet is avoided.
- ``compatibility``: the previous weighted sum of ListNet + standardized
  MSE. Kept accessible because runs on that loss exist in the history.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Primitive loss terms
# ---------------------------------------------------------------------------


def listnet_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between softmax(pred) and softmax(target) over models.

    Known issue: with raw accuracy targets in [40, 98] the teacher
    ``softmax(target)`` is nearly one-hot, making the loss behave like
    NLL against argmax. Prefer :func:`listmle_loss` for stable training.
    """
    p = F.log_softmax(pred, dim=-1)
    t = F.softmax(target, dim=-1)
    return -(t * p).sum(dim=-1).mean()


def listmle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ListMLE / Plackett-Luce ranking loss (Model Spider Eq. 4).

    For each row,

        loss = sum_{m=1..M} -log( exp(p_{dsc(m)}) / sum_{l=m..M} exp(p_{dsc(l)}) )

    where ``dsc`` is the descending permutation of ``target``. The target
    is consumed only by :func:`torch.sort`, so its scale is irrelevant \u2014
    only the induced ranking matters. Implementation uses
    ``torch.logcumsumexp`` for numerical stability.
    """
    _, idx = target.sort(dim=-1, descending=True)
    pred_sorted = pred.gather(-1, idx)  # (B, M)
    # log sum_{l>=m} exp(p_l): reverse, running logsumexp, reverse back.
    flipped = pred_sorted.flip(-1)
    lse = torch.logcumsumexp(flipped, dim=-1).flip(-1)  # (B, M)
    # Per-row sum over positions, then mean over batch.
    return (lse - pred_sorted).sum(dim=-1).mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-row standardized MSE. Both pred and target are z-scored over the
    model axis so different datasets' accuracy ranges are comparable.
    """
    t = (target - target.mean(dim=-1, keepdim=True)) / (
        target.std(dim=-1, keepdim=True) + 1e-6
    )
    p = (pred - pred.mean(dim=-1, keepdim=True)) / (
        pred.std(dim=-1, keepdim=True) + 1e-6
    )
    return F.mse_loss(p, t)


# ---------------------------------------------------------------------------
# Loss modules exposed to the trainer
# ---------------------------------------------------------------------------


class ListMLELoss(nn.Module):
    """Single-term Plackett-Luce ranking loss.

    Returns a two-tuple ``(total, parts)`` to match the shape the trainer
    expects from :class:`CompatibilityLoss`.
    """

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = listmle_loss(pred, target)
        return total, {"rank": total.detach(), "total": total.detach()}


class CompatibilityLoss(nn.Module):
    """Weighted sum of ListNet (ranking) and standardized MSE (regression).

    Kept for backward compatibility with runs trained before ListMLE
    became the default. New runs should prefer ``ListMLELoss``.
    """

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
        return total, {
            "rank": rank.detach(),
            "mse": reg.detach(),
            "total": total.detach(),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_loss(cfg) -> nn.Module:
    """Build the loss module from a trainer config subtree.

    Reads ``cfg.kind`` (defaults to "listmle" if absent) and the kind-
    specific sub-keys. The schema intentionally tolerates older configs
    where only ``ranking_weight``/``mse_weight`` were set \u2014 those imply
    ``kind="compatibility"`` semantics.
    """
    kind = getattr(cfg, "kind", None) or "listmle"
    if kind == "listmle":
        return ListMLELoss()
    if kind == "compatibility":
        return CompatibilityLoss(
            ranking_weight=getattr(cfg, "ranking_weight", 1.0),
            mse_weight=getattr(cfg, "mse_weight", 0.1),
        )
    raise ValueError(f"Unknown loss kind: {kind!r}")
