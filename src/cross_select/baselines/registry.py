"""Unified scoring interface so all baselines plug into the evaluator.

A baseline takes the model token ``(D_m,)`` and the dataset token ``(C, D_d)``
and returns a scalar score (higher = better predicted compatibility).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ScoreFn = Callable[[np.ndarray, np.ndarray], float]

_REGISTRY: dict[str, ScoreFn] = {}


def register(name: str) -> Callable[[ScoreFn], ScoreFn]:
    def wrap(fn: ScoreFn) -> ScoreFn:
        _REGISTRY[name] = fn
        return fn

    return wrap


def get(name: str) -> ScoreFn:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown baseline: {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
