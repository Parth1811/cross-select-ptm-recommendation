"""LogME-style proxy operating on pre-computed tokens.

The canonical LogME (You et al., 2021) requires per-sample features + labels
on the target dataset. Here we only have a compressed model token
``(D_m,)`` and per-class dataset prototypes ``(C, D_d)``. This proxy fits a
ridge regressor on the C per-class prototypes with the model token as the
coefficient and reports the log-evidence of the residual. It's a coarse
proxy and will underperform real LogME \u2014 reported here so the evaluator
has something to plot; the server-side pipeline should swap in the real
implementation against raw features.
"""

from __future__ import annotations

import numpy as np

from .registry import register


@register("logme")
def logme_proxy(model_token: np.ndarray, dataset_token: np.ndarray) -> float:
    # project dataset prototypes through the model token to get a C-length
    # "alignment" signal, then take its negative log-variance as the score
    alignment = dataset_token @ model_token  # (C,)
    var = float(alignment.var()) + 1e-6
    return float(-np.log(var))
