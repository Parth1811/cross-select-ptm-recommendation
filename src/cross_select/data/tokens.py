"""Load pre-computed model and dataset tokens.

Model tokens: one ``<arch>_<source>_embedding.npz`` per model with key
``embedding`` of shape ``(512,)``.

Dataset tokens: sharded ``.npz`` files under
``<dataset_root>/<dataset>/<split>/*.npz``, each shard with keys ``features``
shape ``(16, C, 512)``, ``class_ids`` ``(16, C)``, ``class_names`` ``(16, C)``,
``actual_batches``, ``target_batches``.

Ground truth: a JSON file mapping ``target_dataset -> {"<arch>_<source>": acc}``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MODEL_TOKEN_SUFFIX = "_embedding.npz"


@dataclass(frozen=True)
class ModelToken:
    model_id: str  # "<arch>_<source>"
    embedding: np.ndarray  # (D_m,)


def load_model_tokens(model_tokens_dir: str | Path) -> dict[str, np.ndarray]:
    """Load every ``*_embedding.npz`` in ``model_tokens_dir`` into a dict.

    Returns ``{model_id: embedding}`` where ``model_id`` is the filename stem
    minus the ``_embedding`` suffix (e.g. ``resnet50_imagenet``).
    """
    model_tokens_dir = Path(model_tokens_dir)
    out: dict[str, np.ndarray] = {}
    for path in sorted(model_tokens_dir.glob(f"*{MODEL_TOKEN_SUFFIX}")):
        model_id = path.name[: -len(MODEL_TOKEN_SUFFIX)]
        with np.load(path) as npz:
            out[model_id] = np.asarray(npz["embedding"], dtype=np.float32)
    if not out:
        raise FileNotFoundError(f"No model tokens found under {model_tokens_dir}")
    return out


def list_shards(dataset_root: str | Path, dataset: str, split: str) -> list[Path]:
    """List shard files for ``<dataset_root>/<dataset>/<split>/*.npz``."""
    shard_dir = Path(dataset_root) / dataset / split
    shards = sorted(shard_dir.glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"No dataset shards found under {shard_dir}")
    return shards


def load_shard(path: str | Path) -> np.ndarray:
    """Return the ``features`` array of shape ``(16, C, 512)`` from one shard."""
    with np.load(path, allow_pickle=True) as npz:
        return np.asarray(npz["features"], dtype=np.float32)


def shard_row_count(path: str | Path) -> int:
    """Return the number of sample rows in a shard (axis 0 of features)."""
    with np.load(path, allow_pickle=True) as npz:
        return int(npz["features"].shape[0])


def shard_class_count(path: str | Path) -> int:
    """Return the number of classes C in a shard (axis 1 of features)."""
    with np.load(path, allow_pickle=True) as npz:
        return int(npz["features"].shape[1])


def load_shard_rows(path: str | Path, row_ids: list[int]) -> np.ndarray:
    """Load selected row indices from one shard's ``features`` array.

    Returns an array of shape ``(len(row_ids), C, 512)`` float32.
    """
    with np.load(path, allow_pickle=True) as npz:
        feats = np.asarray(npz["features"], dtype=np.float32)
    return feats[list(row_ids)]


def sample_dataset_tokens(
    shards: list[Path],
    rng: random.Random | None = None,
    pick_row: bool = False,
) -> np.ndarray:
    """Stochastically sample one dataset-token view.

    Picks one shard uniformly at random; if ``pick_row`` also picks one of
    the 16 rows. Returns ``(C, 512)`` either way.
    """
    r = rng or random
    shard = r.choice(shards)
    feats = load_shard(shard)  # (16, C, 512)
    if pick_row:
        idx = r.randrange(feats.shape[0])
        return feats[idx]
    return feats.mean(axis=0)


def dataset_prototype(shards: list[Path]) -> np.ndarray:
    """Deterministic per-class prototype: mean over all shards and rows.

    Used at eval time. Shape ``(C, 512)``.
    """
    acc: np.ndarray | None = None
    total_rows = 0
    for shard in shards:
        feats = load_shard(shard)  # (16, C, 512)
        n = feats.shape[0]
        s = feats.sum(axis=0)  # (C, 512)
        acc = s if acc is None else acc + s
        total_rows += n
    assert acc is not None
    return acc / total_rows


def load_ground_truth(path: str | Path) -> dict[str, dict[str, float]]:
    """Load the ground-truth accuracy JSON. Outer key = target dataset."""
    with Path(path).open() as f:
        return json.load(f)


def build_accuracy_matrix(
    gt: dict[str, dict[str, float]],
    model_ids: list[str],
    dataset_ids: list[str],
    missing_value: float | str = "random_rank",
    rng: random.Random | None = None,
) -> np.ndarray:
    """Materialize a ``(N_models, N_datasets)`` accuracy matrix.

    ``missing_value``:
    - float: use this scalar wherever the GT cell is absent.
    - ``"random_rank"``: fill a missing cell with a value drawn uniformly
      from the observed accuracy range of that dataset column, with a small
      jitter to avoid ties. Matches the "make up a random rank" policy for
      self-transfer pairs (e.g. resnet50_cifar10 on cifar10).
    """
    r = rng or random.Random(0)
    n_m, n_d = len(model_ids), len(dataset_ids)
    out = np.full((n_m, n_d), np.nan, dtype=np.float32)
    for j, ds in enumerate(dataset_ids):
        col = gt.get(ds, {})
        for i, mid in enumerate(model_ids):
            if mid in col:
                out[i, j] = col[mid]
    if missing_value == "random_rank":
        for j in range(n_d):
            col = out[:, j]
            mask = np.isnan(col)
            if not mask.any():
                continue
            observed = col[~mask]
            lo, hi = float(observed.min()), float(observed.max())
            span = max(hi - lo, 1e-6)
            for i in np.where(mask)[0]:
                out[i, j] = r.uniform(lo, hi) + r.uniform(-1e-3, 1e-3) * span
    elif isinstance(missing_value, (int, float)):
        out = np.where(np.isnan(out), float(missing_value), out)
    else:
        raise ValueError(f"Unsupported missing_value: {missing_value!r}")
    return out
