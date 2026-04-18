"""LEEP-style proxy on pre-computed tokens.

Real LEEP (Nguyen et al., 2020) needs the source classifier's softmax output
on target-dataset samples. Here we approximate via the cosine similarity of
the model token with the mean dataset prototype \u2014 a rough analogue that
only uses the information available in the token contract.
"""

from __future__ import annotations

import numpy as np

from .registry import register


@register("leep")
def leep_proxy(model_token: np.ndarray, dataset_token: np.ndarray) -> float:
    mean_ds = dataset_token.mean(axis=0)
    num = float(model_token @ mean_ds)
    den = float(np.linalg.norm(model_token) * np.linalg.norm(mean_ds)) + 1e-6
    return num / den
