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
    load_shard_rows,
    sample_dataset_tokens,
    shard_class_count,
    shard_row_count,
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
    """One item per dataset: dataset token + full (M, D_m) model zoo +
    (M,) accuracy vector.

    ``mode`` controls how the ``(C, D_d)`` dataset token is drawn:

    - ``"prototype"`` (default for eval): deterministic mean over all
      shards of the requested split.
    - ``"stochastic"``: one random shard from the split's shard list,
      mean-pooled over its 16 rows. Same stochasticity the train loop
      uses; under eval the trainer should call this many times with
      different seeds and average the predicted scores (matches the
      training distribution ``E_x[f(x)]`` rather than the
      ``f(E_x[x])`` that prototype eval evaluates).
    """

    def __init__(
        self,
        bank: TokenBank,
        dataset_ids: list[str] | None = None,
        split: str = "train",
        pick_row: bool = False,
        mode: str | None = None,
    ) -> None:
        self.bank = bank
        self.dataset_ids = list(dataset_ids or bank.dataset_ids)
        self.split = split
        self.pick_row = pick_row
        # Default mode follows split for back-compat: train => stochastic,
        # everything else => prototype.
        if mode is None:
            mode = "stochastic" if split == "train" else "prototype"
        if mode not in {"stochastic", "prototype"}:
            raise ValueError(f"Unknown ListwiseDataset mode: {mode!r}")
        self.mode = mode
        self._index_in_bank = {d: bank.dataset_ids.index(d) for d in self.dataset_ids}

    def __len__(self) -> int:
        return len(self.dataset_ids)

    def _shards_for(self, dataset: str) -> list:
        if self.split == "train":
            return self.bank.train_shards(dataset)
        return self.bank.eval_shards(dataset, self.split)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        dataset = self.dataset_ids[idx]
        shards = self._shards_for(dataset)
        if self.mode == "stochastic":
            feats = sample_dataset_tokens(
                shards,
                rng=self.bank.rng,
                pick_row=self.pick_row,
            )
        else:
            feats = dataset_prototype(shards)
        col = self._index_in_bank[dataset]
        return {
            "dataset_id": dataset,
            "dataset_token": torch.from_numpy(feats),  # (C, D_d)
            "model_tokens": torch.from_numpy(self.bank.model_tokens),  # (M, D_m)
            "accuracy": torch.from_numpy(self.bank.accuracy[:, col]),  # (M,)
        }


class SampleBatchGenerator:
    """Sample-level training batcher.

    Per step: subsample ``num_models_per_step`` models (shared across the
    step), pick ``num_datasets_per_step`` datasets, and draw
    ``num_samples_per_dataset`` rows from a single shard for each.

    Per epoch: enumerate every ``(dataset, shard, row)`` triple, shuffle
    the implied ``shard-chunks`` (groups of ``num_samples_per_dataset``
    consecutive rows in one shard), then greedily combine chunks from
    distinct datasets into training steps. Any tail that can't form a
    full multi-dataset step is dropped for that epoch.
    """

    def __init__(
        self,
        bank: TokenBank,
        dataset_ids: list[str],
        num_models_per_step: int = 16,
        num_datasets_per_step: int = 4,
        num_samples_per_dataset: int = 4,
        seed: int = 0,
        model_pool_idx: list[int] | None = None,
    ) -> None:
        if num_datasets_per_step > len(dataset_ids):
            raise ValueError(
                f"num_datasets_per_step={num_datasets_per_step} exceeds "
                f"available training datasets ({len(dataset_ids)})"
            )
        pool = (
            list(model_pool_idx)
            if model_pool_idx is not None
            else list(range(len(bank.model_ids)))
        )
        if num_models_per_step > len(pool):
            raise ValueError(
                f"num_models_per_step={num_models_per_step} exceeds "
                f"model pool size ({len(pool)})"
            )

        self.bank = bank
        self.dataset_ids = list(dataset_ids)
        self.num_models_per_step = num_models_per_step
        self.num_datasets_per_step = num_datasets_per_step
        self.num_samples_per_dataset = num_samples_per_dataset
        self.model_pool_idx = pool
        self.rng = random.Random(seed)

        # Index shards + sizes up front so epoch construction is fast.
        self._shards: dict[str, list[Path]] = {
            d: bank.train_shards(d) for d in self.dataset_ids
        }
        self._shard_rows: dict[Path, int] = {
            s: shard_row_count(s) for d in self.dataset_ids for s in self._shards[d]
        }
        self._shard_classes: dict[Path, int] = {
            s: shard_class_count(s) for d in self.dataset_ids for s in self._shards[d]
        }
        self._dataset_col = {d: bank.dataset_ids.index(d) for d in self.dataset_ids}

    # ---- reporting helpers (used by dry_run) --------------------------------

    def shards_per_dataset(self) -> dict[str, int]:
        return {d: len(s) for d, s in self._shards.items()}

    def rows_per_dataset(self) -> dict[str, int]:
        return {
            d: sum(self._shard_rows[s] for s in self._shards[d])
            for d in self.dataset_ids
        }

    def total_rows(self) -> int:
        return sum(self.rows_per_dataset().values())

    def chunks_per_dataset(self) -> dict[str, int]:
        """How many ``num_samples_per_dataset``-row groups each dataset yields."""
        s = self.num_samples_per_dataset
        return {
            d: sum(self._shard_rows[sh] // s for sh in self._shards[d])
            for d in self.dataset_ids
        }

    def total_chunks(self) -> int:
        return sum(self.chunks_per_dataset().values())

    def steps_per_epoch(self) -> int:
        # "Upsample small datasets" policy: the largest dataset is the
        # driver and appears in every step; its chunk count bounds the
        # epoch. Smaller datasets cycle (reshuffled) to keep up.
        return max(self.chunks_per_dataset().values())

    # ---- epoch construction -------------------------------------------------

    def _build_shard_chunks(self) -> dict[str, list[tuple[Path, list[int]]]]:
        """For each dataset, a shuffled list of ``(shard_path, [row_ids])``
        groups of ``num_samples_per_dataset`` rows.
        """
        s = self.num_samples_per_dataset
        out: dict[str, list[tuple[Path, list[int]]]] = {}
        for d in self.dataset_ids:
            chunks: list[tuple[Path, list[int]]] = []
            for shard in self._shards[d]:
                n_rows = self._shard_rows[shard]
                rows = list(range(n_rows))
                self.rng.shuffle(rows)
                for start in range(0, n_rows - s + 1, s):
                    chunks.append((shard, rows[start : start + s]))
            self.rng.shuffle(chunks)
            out[d] = chunks
        return out

    def build_epoch(self) -> list[dict]:
        """Yield one training step per chunk of the *largest* dataset.

        The largest dataset advances through its chunks exactly once, giving
        every sample row exactly one gradient update for that dataset per
        epoch. Smaller datasets cycle (reshuffle when exhausted) so they
        keep appearing as partners \u2014 this is the "upsample small datasets"
        policy. Result: steps_per_epoch == chunks(largest dataset).
        """
        k = self.num_datasets_per_step
        pool = self._build_shard_chunks()

        # Rank datasets by original chunk count (descending). The #1 is the
        # "driver"; it's always included in every step. The remaining k-1
        # slots are filled by sampling uniformly without replacement from
        # the rest, each step picking a fresh subset.
        order = sorted(self.dataset_ids, key=lambda d: len(pool[d]), reverse=True)
        driver = order[0]
        partners_pool = order[1:]
        driver_chunks = pool[driver]

        # Per-partner cursor into its chunk list; we reshuffle on wrap.
        cursors = {d: 0 for d in partners_pool}

        steps: list[dict] = []
        for i in range(len(driver_chunks)):
            picks = [driver]
            # Pick k-1 distinct partners for this step.
            chosen = self.rng.sample(partners_pool, k - 1) if k > 1 else []
            picks.extend(chosen)
            self.rng.shuffle(picks)

            # For the driver, pop one chunk (advance the real pointer).
            # For partners, consume their current cursor; on wrap, reshuffle.
            selected_chunks: dict[str, tuple[Path, list[int]]] = {}
            selected_chunks[driver] = driver_chunks[i]
            for d in chosen:
                if cursors[d] >= len(pool[d]):
                    # Exhausted \u2014 reshuffle and reset cursor so we keep cycling.
                    self.rng.shuffle(pool[d])
                    cursors[d] = 0
                selected_chunks[d] = pool[d][cursors[d]]
                cursors[d] += 1

            step = self._assemble_step(picks, selected_chunks)
            steps.append(step)
        return steps

    def _assemble_step(
        self,
        dataset_picks: list[str],
        selected_chunks: dict[str, tuple[Path, list[int]]],
    ) -> dict:
        d_tokens = []
        accuracies = []
        shards_used: list[Path] = []
        for d in dataset_picks:
            shard, rows = selected_chunks[d]
            shards_used.append(shard)
            feats = load_shard_rows(shard, rows)  # (s, C, 512)
            d_tokens.append(feats)
            accuracies.append(self.bank.accuracy[:, self._dataset_col[d]])

        c_m = min(self._shard_classes[sh] for sh in shards_used)
        d_stack = np.stack(
            [feats[:, :c_m, :] for feats in d_tokens], axis=0
        )  # (k, s, C_m, 512)

        # Same num_models_per_step model indices for all datasets this step,
        # drawn from the configured model pool (by default the full zoo, or
        # split.train_model_ids when a quadrant holds out models).
        # Special case: when num_models_per_step equals the pool size, use
        # the deterministic pool order. This removes step-to-step target
        # ranking jitter (ListMLE's argsort of the same 32 models is
        # constant) and matches Model Spider's full-zoo training recipe.
        if self.num_models_per_step == len(self.model_pool_idx):
            model_idx = list(self.model_pool_idx)
        else:
            model_idx = self.rng.sample(
                self.model_pool_idx, self.num_models_per_step
            )
        model_tokens = self.bank.model_tokens[model_idx]  # (M_sub, 512)
        acc = np.stack(accuracies, axis=0)[:, model_idx]  # (k, M_sub)

        return {
            "dataset_ids": list(dataset_picks),
            "dataset_tokens": torch.from_numpy(d_stack),  # (k, s, C_m, 512)
            "model_tokens": torch.from_numpy(model_tokens),  # (M_sub, 512)
            "model_idx": torch.tensor(model_idx, dtype=torch.long),  # (M_sub,)
            "accuracy": torch.from_numpy(acc),  # (k, M_sub)
            "C_m": int(c_m),
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
