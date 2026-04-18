"""PyTorch Datasets over pre-computed model and dataset tokens.

Two flavors:

- ``ListwiseDataset``: one item per dataset; yields the full model zoo and the
  per-model accuracy vector, plus a stochastically-sampled ``(C, 512)`` token
  for that dataset. This is the natural unit for a listwise ranking loss.

- ``PairwiseDataset``: one item per (model, dataset) cell; yields a scalar
  accuracy. Useful for pointwise MSE or for building pairwise batches.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokens import (
    build_accuracy_matrix,
    dataset_prototype,
    list_shards,
    load_ground_truth,
    load_model_tokens,
    sample_dataset_tokens,
)


class TokenBank:
    """Shared state for all datasets: model tokens + GT matrix + shard index.

    Built once and passed to both train and eval Datasets so we don't reload
    model tokens or rescan shard directories per split.
    """

    def __init__(
        self,
        model_tokens_dir: str | Path,
        dataset_root: str | Path,
        dataset_ids: list[str],
        gt_path: str | Path,
        gt_dataset_name_map: dict[str, str] | None = None,
        missing_value: float | str = "random_rank",
        seed: int = 0,
    ) -> None:
        self.rng = random.Random(seed)

        tokens = load_model_tokens(model_tokens_dir)
        self.model_ids: list[str] = sorted(tokens.keys())
        self.model_tokens = np.stack(
            [tokens[m] for m in self.model_ids], axis=0
        ).astype(np.float32)  # (N_models, D_m)

        self.dataset_ids = list(dataset_ids)
        self.dataset_root = Path(dataset_root)
        # Map folder-name (dataset_ids) -> GT-JSON key. Defaults to identity.
        self.gt_name_map = dict(gt_dataset_name_map or {})

        gt = load_ground_truth(gt_path)
        gt_keys = [self.gt_name_map.get(d, d) for d in self.dataset_ids]
        self.accuracy = build_accuracy_matrix(
            gt,
            model_ids=self.model_ids,
            dataset_ids=gt_keys,
            missing_value=missing_value,
            rng=self.rng,
        )  # (N_models, N_datasets)

        self._train_shards: dict[str, list[Path]] = {}
        self._eval_shards: dict[str, list[Path]] = {}

    def train_shards(self, dataset: str) -> list[Path]:
        if dataset not in self._train_shards:
            self._train_shards[dataset] = list_shards(
                self.dataset_root, dataset, "train"
            )
        return self._train_shards[dataset]

    def eval_shards(self, dataset: str, split: str = "validation") -> list[Path]:
        key = f"{dataset}/{split}"
        if key not in self._eval_shards:
            self._eval_shards[key] = list_shards(self.dataset_root, dataset, split)
        return self._eval_shards[key]


class ListwiseDataset(Dataset):
    """One item per dataset: stochastic dataset token + full (M, D_m) model
    zoo + (M,) accuracy vector.
    """

    def __init__(
        self,
        bank: TokenBank,
        dataset_ids: list[str] | None = None,
        split: str = "train",
        pick_row: bool = False,
    ) -> None:
        self.bank = bank
        self.dataset_ids = list(dataset_ids or bank.dataset_ids)
        self.split = split
        self.pick_row = pick_row
        self._index_in_bank = {d: bank.dataset_ids.index(d) for d in self.dataset_ids}

    def __len__(self) -> int:
        return len(self.dataset_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        dataset = self.dataset_ids[idx]
        if self.split == "train":
            feats = sample_dataset_tokens(
                self.bank.train_shards(dataset),
                rng=self.bank.rng,
                pick_row=self.pick_row,
            )
        else:
            feats = dataset_prototype(self.bank.eval_shards(dataset, self.split))
        col = self._index_in_bank[dataset]
        return {
            "dataset_id": dataset,
            "dataset_token": torch.from_numpy(feats),  # (C, D_d)
            "model_tokens": torch.from_numpy(self.bank.model_tokens),  # (M, D_m)
            "accuracy": torch.from_numpy(self.bank.accuracy[:, col]),  # (M,)
        }


class PairwiseDataset(Dataset):
    """One item per (model, dataset) cell."""

    def __init__(
        self,
        bank: TokenBank,
        dataset_ids: list[str] | None = None,
        split: str = "train",
        pick_row: bool = False,
    ) -> None:
        self.bank = bank
        self.dataset_ids = list(dataset_ids or bank.dataset_ids)
        self.split = split
        self.pick_row = pick_row
        self._pairs = [
            (m, d) for d in self.dataset_ids for m in range(len(bank.model_ids))
        ]

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        model_idx, dataset = self._pairs[idx]
        if self.split == "train":
            feats = sample_dataset_tokens(
                self.bank.train_shards(dataset),
                rng=self.bank.rng,
                pick_row=self.pick_row,
            )
        else:
            feats = dataset_prototype(self.bank.eval_shards(dataset, self.split))
        col = self.bank.dataset_ids.index(dataset)
        return {
            "dataset_id": dataset,
            "model_id": self.bank.model_ids[model_idx],
            "dataset_token": torch.from_numpy(feats),  # (C, D_d)
            "model_token": torch.from_numpy(self.bank.model_tokens[model_idx]),  # (D_m,)
            "accuracy": torch.tensor(
                self.bank.accuracy[model_idx, col], dtype=torch.float32
            ),
        }
