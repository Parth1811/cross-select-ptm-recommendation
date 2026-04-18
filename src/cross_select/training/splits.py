"""Known/unknown model and dataset splits producing the four quadrants.

The 4-quadrant protocol from the thesis:

- Q1: known models, known datasets
- Q2: unknown models, known datasets
- Q3: known models, unknown datasets
- Q4: unknown models, unknown datasets
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    train_model_ids: list[str]
    eval_model_ids: list[str]
    train_dataset_ids: list[str]
    eval_dataset_ids: list[str]


def build_split(
    all_model_ids: list[str],
    all_dataset_ids: list[str],
    held_out_models: list[str] | None = None,
    held_out_datasets: list[str] | None = None,
) -> Split:
    """Partition model and dataset ids into train/eval pools.

    ``eval_model_ids`` is ``held_out_models`` if non-empty, else equals
    ``train_model_ids`` (Q1 / Q3 style: evaluate on known models).
    Same convention applies to datasets.
    """
    held_m = set(held_out_models or [])
    held_d = set(held_out_datasets or [])
    train_m = [m for m in all_model_ids if m not in held_m]
    train_d = [d for d in all_dataset_ids if d not in held_d]
    eval_m = sorted(held_m) if held_m else train_m
    eval_d = sorted(held_d) if held_d else train_d
    missing_m = held_m - set(all_model_ids)
    missing_d = held_d - set(all_dataset_ids)
    if missing_m:
        raise ValueError(f"Held-out models not in zoo: {sorted(missing_m)}")
    if missing_d:
        raise ValueError(f"Held-out datasets not in pool: {sorted(missing_d)}")
    return Split(
        train_model_ids=train_m,
        eval_model_ids=eval_m,
        train_dataset_ids=train_d,
        eval_dataset_ids=eval_d,
    )
