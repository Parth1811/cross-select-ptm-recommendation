"""Tests for ranking metrics, losses, and baselines."""

from __future__ import annotations

import numpy as np
import torch

from cross_select.baselines import available, get
from cross_select.eval.metrics import (
    all_metrics,
    mrr,
    ndcg,
    pearson_correlation,
    precision_at_k,
    relative_accuracy_at_k,
    weighted_kendall_tau,
)
from cross_select.losses.ranking import (
    CompatibilityLoss,
    ListMLELoss,
    build_loss,
    listmle_loss,
    listnet_loss,
    mse_loss,
)


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


def test_relative_accuracy_at_k_perfect_is_one():
    target = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    # Top-3 predicted == top-3 true (0.9, 0.8, 0.5); mean / max = 0.733
    pred = target.copy()
    assert abs(relative_accuracy_at_k(pred, target, k=3) - (0.9 + 0.8 + 0.5) / 3 / 0.9) < 1e-9


def test_relative_accuracy_at_k_reverse_is_low():
    target = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    pred = -target  # worst 3 picked
    # Top-3 picked are indices of smallest target values: 0.1, 0.2, 0.5
    assert abs(relative_accuracy_at_k(pred, target, k=3) - (0.1 + 0.2 + 0.5) / 3 / 0.9) < 1e-9


def test_relative_accuracy_at_k_handles_all_zero_target():
    pred = np.array([0.1, 0.5, 0.9])
    target = np.zeros(3)
    assert relative_accuracy_at_k(pred, target) == 0.0


def test_pearson_correlation_perfect_and_reverse():
    target = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    assert abs(pearson_correlation(target, target) - 1.0) < 1e-9
    assert abs(pearson_correlation(-target, target) + 1.0) < 1e-9


def test_pearson_correlation_constant_returns_zero():
    target = np.array([0.5, 0.5, 0.5, 0.5])
    pred = np.array([0.1, 0.9, 0.2, 0.8])
    # Constant target has undefined correlation; we define as 0.
    assert pearson_correlation(pred, target) == 0.0


def test_all_metrics_includes_parc_keys():
    target = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    pred = target.copy()
    m = all_metrics(pred, target, k=3)
    assert "relAcc@3" in m
    assert "pearson" in m
    assert abs(m["pearson"] - 1.0) < 1e-9


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


def test_listmle_loss_is_zero_in_the_limit_of_perfect_ordering():
    """With pred matching target's ordering and large-margin logits,
    every softmax term concentrates on its own position and the NLL sum
    collapses to ~0.
    """
    target = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    pred = torch.tensor([[-400.0, -300.0, -200.0, -100.0, 0.0]])
    # Perfect descending alignment: highest-target is also highest-pred.
    assert float(listmle_loss(pred, target)) < 1e-3


def test_listmle_loss_is_invariant_to_target_scale():
    """ListMLE consumes ``target`` only through argsort; scaling must
    not change the loss value given the same pred.
    """
    torch.manual_seed(0)
    pred = torch.randn(3, 12)
    target = torch.randn(3, 12)
    a = float(listmle_loss(pred, target))
    b = float(listmle_loss(pred, target * 100 + 42))
    assert abs(a - b) < 1e-5


def test_listmle_loss_is_finite_with_huge_targets():
    """Raw accuracy scale up to 100 shouldn't trip the softmax; the
    target only enters via a sort. This is a regression guard against
    accidentally re-introducing ListNet's softmax-over-target pattern.
    """
    pred = torch.randn(2, 16)
    target = torch.rand(2, 16) * 100  # 0..100 like raw percentages
    loss = listmle_loss(pred, target)
    assert torch.isfinite(loss)


def test_listmle_loss_decreases_as_pred_aligns_with_target():
    torch.manual_seed(0)
    target = torch.randn(1, 10)
    aligned = target.clone()
    misaligned = -target
    assert listmle_loss(aligned, target) < listmle_loss(misaligned, target)


def test_listmle_module_returns_scalar_and_parts():
    loss_fn = ListMLELoss()
    pred = torch.randn(2, 8, requires_grad=True)
    target = torch.randn(2, 8)
    total, parts = loss_fn(pred, target)
    assert total.ndim == 0
    assert set(parts.keys()) == {"rank", "total"}
    total.backward()
    assert pred.grad is not None


def test_build_loss_factory():
    from omegaconf import OmegaConf

    listmle_cfg = OmegaConf.create({"kind": "listmle"})
    assert isinstance(build_loss(listmle_cfg), ListMLELoss)

    compat_cfg = OmegaConf.create(
        {"kind": "compatibility", "ranking_weight": 1.0, "mse_weight": 0.1}
    )
    assert isinstance(build_loss(compat_cfg), CompatibilityLoss)

    # Defaults to listmle when kind is absent.
    bare = OmegaConf.create({"ranking_weight": 1.0, "mse_weight": 0.1})
    assert isinstance(build_loss(bare), ListMLELoss)

    import pytest as _pytest

    with _pytest.raises(ValueError):
        build_loss(OmegaConf.create({"kind": "bogus"}))


def test_baselines_registered():
    assert set(available()) >= {"logme", "leep", "nce"}


def test_baseline_score_returns_finite_scalar():
    rng = np.random.default_rng(0)
    mt = rng.standard_normal(512).astype(np.float32)
    dt = rng.standard_normal((102, 512)).astype(np.float32)
    for name in available():
        s = get(name)(mt, dt)
        assert np.isfinite(s)
