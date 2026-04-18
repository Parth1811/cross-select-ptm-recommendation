"""Tests for ranking metrics, losses, and baselines."""

from __future__ import annotations

import numpy as np
import torch

from cross_select.baselines import available, get
from cross_select.eval.metrics import (
    all_metrics,
    mrr,
    ndcg,
    precision_at_k,
    weighted_kendall_tau,
)
from cross_select.losses.ranking import CompatibilityLoss, listnet_loss, mse_loss


def test_metrics_perfect_ordering():
    target = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    pred = target.copy()
    m = all_metrics(pred, target, k=2)
    assert m["weighted_kendall_tau"] == 1.0
    assert m["ndcg"] == 1.0
    assert m["precision@2"] == 1.0
    assert m["mrr"] == 1.0


def test_metrics_reverse_ordering():
    target = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    pred = -target
    tau = weighted_kendall_tau(pred, target)
    assert tau < 0
    assert precision_at_k(pred, target, k=2) == 0.0
    # true best (index 2) is now predicted last \u2192 rank 5
    assert mrr(pred, target) == 1 / 5


def test_ndcg_with_k():
    target = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([4.0, 3.0, 2.0, 1.0])  # reverse order
    assert ndcg(pred, target, k=2) < 1.0


def test_listnet_loss_decreases_when_aligned():
    torch.manual_seed(0)
    target = torch.randn(1, 10)
    aligned = target.clone()
    random = torch.randn(1, 10)
    assert listnet_loss(aligned, target) < listnet_loss(random, target)


def test_mse_loss_is_zero_for_perfect_prediction():
    torch.manual_seed(0)
    target = torch.randn(1, 10)
    assert float(mse_loss(target, target)) < 1e-6


def test_compatibility_loss_returns_scalar_and_parts():
    loss_fn = CompatibilityLoss(ranking_weight=1.0, mse_weight=0.1)
    pred = torch.randn(2, 8, requires_grad=True)
    target = torch.randn(2, 8)
    total, parts = loss_fn(pred, target)
    assert total.ndim == 0
    assert {"rank", "mse", "total"} <= set(parts.keys())
    total.backward()
    assert pred.grad is not None


def test_baselines_registered():
    assert set(available()) >= {"logme", "leep", "nce"}


def test_baseline_score_returns_finite_scalar():
    rng = np.random.default_rng(0)
    mt = rng.standard_normal(512).astype(np.float32)
    dt = rng.standard_normal((102, 512)).astype(np.float32)
    for name in available():
        s = get(name)(mt, dt)
        assert np.isfinite(s)
