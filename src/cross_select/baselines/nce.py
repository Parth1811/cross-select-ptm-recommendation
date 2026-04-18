"""NCE-style proxy on pre-computed tokens.

Real NCE (Tran et al., 2019) estimates mutual information between source
and target label distributions. We approximate via the mean absolute
alignment of the model token with per-class prototypes, normalized by the
per-class spread \u2014 a crude stand-in that only uses the token contract.
"""

from __future__ import annotations

import numpy as np

from .registry import register


@register("nce")
def nce_proxy(model_token: np.ndarray, dataset_token: np.ndarray) -> float:
    alignment = dataset_token @ model_token  # (C,)
    return float(np.abs(alignment).mean() / (alignment.std() + 1e-6))
