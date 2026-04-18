"""Ranking metrics: weighted Kendall tau, NDCG, MRR, Precision@K.

All functions take 1D numpy arrays ``pred`` and ``target`` of equal length.
Higher is better. ``target`` holds the ground-truth accuracies; ``pred``
holds the predicted compatibility scores.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import weightedtau


def weighted_kendall_tau(pred: np.ndarray, target: np.ndarray) -> float:
    tau, _ = weightedtau(target, pred)
    return float(tau)


def ndcg(pred: np.ndarray, target: np.ndarray, k: int | None = None) -> float:
    """Standard NDCG using ``target`` as gain."""
    order = np.argsort(-pred)
    gains = target[order]
    if k is not None:
        gains = gains[:k]
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())

    ideal = np.sort(target)[::-1]
    if k is not None:
        ideal = ideal[:k]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(pred: np.ndarray, target: np.ndarray, k: int = 3) -> float:
    """Fraction of top-k predicted that are in the true top-k."""
    k = min(k, pred.size)
    top_pred = set(np.argsort(-pred)[:k].tolist())
    top_true = set(np.argsort(-target)[:k].tolist())
    return len(top_pred & top_true) / k


def mrr(pred: np.ndarray, target: np.ndarray) -> float:
    """Reciprocal rank of the true-best model under the predicted ordering."""
    true_best = int(np.argmax(target))
    order = np.argsort(-pred)
    rank = int(np.where(order == true_best)[0][0]) + 1
    return 1.0 / rank


def all_metrics(
    pred: np.ndarray, target: np.ndarray, k: int = 3
) -> dict[str, float]:
    return {
        "weighted_kendall_tau": weighted_kendall_tau(pred, target),
        "ndcg": ndcg(pred, target),
        f"precision@{k}": precision_at_k(pred, target, k=k),
        "mrr": mrr(pred, target),
    }
