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


def listmle_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_temperature: float = 1.0,
) -> torch.Tensor:
    """ListMLE / Plackett-Luce ranking loss (Model Spider Eq. 4).

    For each row,

        loss = sum_{m=1..M} -log( exp(p_{dsc(m)}/T) / sum_{l=m..M} exp(p_{dsc(l)}/T) )

    where ``dsc`` is the descending permutation of ``target`` and ``T``
    (``pred_temperature``) is an optional scale applied to predictions
    before the softmax. T<1 sharpens the distribution (stronger gradient
    on top positions); T>1 softens. Default T=1 matches the Model Spider
    formula. The target is consumed only by :func:`torch.sort`, so its
    scale is irrelevant \u2014 only the induced ranking matters.
    """
    if pred_temperature != 1.0:
        pred = pred / pred_temperature
    _, idx = target.sort(dim=-1, descending=True)
    pred_sorted = pred.gather(-1, idx)  # (B, M)
    # log sum_{l>=m} exp(p_l): reverse, running logsumexp, reverse back.
    flipped = pred_sorted.flip(-1)
    lse = torch.logcumsumexp(flipped, dim=-1).flip(-1)  # (B, M)
    # Per-row sum over positions, then mean over batch.
    return (lse - pred_sorted).sum(dim=-1).mean()


def pairwise_bce_loss(
    pred: torch.Tensor, target: torch.Tensor, margin: float = 0.0
) -> torch.Tensor:
    """RankNet-style pairwise binary cross-entropy.

    For each row, we form all (i, j) model pairs and ask the model to
    output ``pred_i > pred_j`` whenever ``target_i > target_j`` with
    BCE on ``sigmoid(pred_i - pred_j - margin)``. Tied target pairs
    contribute zero weight. Smoother than ListMLE on top-of-list
    mistakes \u2014 no suffix-sum explosion.
    """
    # target_diff sign: +1 where i>j, -1 where i<j, 0 on ties.
    tgt_diff = target.unsqueeze(-1) - target.unsqueeze(-2)  # (B, M, M)
    labels = (tgt_diff > 0).float()  # we score each ordered pair
    weights = (tgt_diff.abs() > 0).float()  # zero out ties (and diagonal)
    pred_diff = pred.unsqueeze(-1) - pred.unsqueeze(-2)
    # Logistic loss per pair; mean across valid pairs only.
    logits = pred_diff - margin
    per_pair = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    # Mean over valid pairs to keep magnitude comparable across batch rows
    # and list lengths.
    denom = weights.sum(dim=(-1, -2)).clamp_min(1.0)
    return ((per_pair * weights).sum(dim=(-1, -2)) / denom).mean()


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
    """Single-term Plackett-Luce ranking loss with optional pred temperature."""

    def __init__(self, pred_temperature: float = 1.0) -> None:
        super().__init__()
        self.pred_temperature = float(pred_temperature)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = listmle_loss(pred, target, pred_temperature=self.pred_temperature)
        return total, {"rank": total.detach(), "total": total.detach()}


class ListMLEMSELoss(nn.Module):
    """ListMLE ranking + per-row standardized MSE, weighted sum.

    Replaces the legacy ``CompatibilityLoss`` (which used ListNet and had
    the one-hot-target pathology). The ranking term uses :func:`listmle_loss`
    with optional ``pred_temperature``; the regression term is
    :func:`mse_loss` on z-scored pred and target per row.
    """

    def __init__(
        self,
        ranking_weight: float = 1.0,
        mse_weight: float = 1.0,
        pred_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.ranking_weight = float(ranking_weight)
        self.mse_weight = float(mse_weight)
        self.pred_temperature = float(pred_temperature)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rank = listmle_loss(pred, target, pred_temperature=self.pred_temperature)
        reg = mse_loss(pred, target)
        total = self.ranking_weight * rank + self.mse_weight * reg
        return total, {
            "rank": rank.detach(),
            "mse": reg.detach(),
            "total": total.detach(),
        }


class PairwiseBCELoss(nn.Module):
    """RankNet-style pairwise BCE."""

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = pairwise_bce_loss(pred, target, margin=self.margin)
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

    Reads ``cfg.kind``:

    - ``"listmle"`` (default): Plackett-Luce ranking. Accepts
      ``pred_temperature`` (default 1.0).
    - ``"listmle_mse"``: weighted ListMLE + standardized MSE. Accepts
      ``ranking_weight`` (default 1.0), ``mse_weight`` (default 1.0),
      ``pred_temperature`` (default 1.0). Use this when you want the
      dense per-model gradient that MSE provides alongside the ranking
      term; unlike the legacy ``compatibility`` kind, the ranking term
      is ListMLE (not the unstable ListNet-on-raw-accuracy).
    - ``"pairwise_bce"``: RankNet-style pairwise BCE. Accepts ``margin``
      (default 0.0).
    - ``"compatibility"``: legacy ListNet + standardized MSE. Kept for
      reproducibility of historic runs.
    """
    kind = getattr(cfg, "kind", None) or "listmle"
    if kind == "listmle":
        return ListMLELoss(
            pred_temperature=float(getattr(cfg, "pred_temperature", 1.0)),
        )
    if kind == "listmle_mse":
        return ListMLEMSELoss(
            ranking_weight=float(getattr(cfg, "ranking_weight", 1.0)),
            mse_weight=float(getattr(cfg, "mse_weight", 1.0)),
            pred_temperature=float(getattr(cfg, "pred_temperature", 1.0)),
        )
    if kind == "pairwise_bce":
        return PairwiseBCELoss(margin=float(getattr(cfg, "margin", 0.0)))
    if kind == "compatibility":
        return CompatibilityLoss(
            ranking_weight=float(getattr(cfg, "ranking_weight", 1.0)),
            mse_weight=float(getattr(cfg, "mse_weight", 0.1)),
        )
    raise ValueError(f"Unknown loss kind: {kind!r}")
