"""Hydra entrypoint: ``python -m cross_select.cli.dry_run``.

Prints how many samples / steps the train and eval loops would iterate
through for the current Hydra config, without running any training.
"""

from __future__ import annotations

import logging
import math

import hydra
from omegaconf import DictConfig, OmegaConf

from ..data.dataset import ListwiseDataset, TokenBank
from ..data.tokens import list_shards
from ..training.splits import build_split


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("config:\n%s", OmegaConf.to_yaml(cfg))

    bank = TokenBank(
        model_tokens_dir=cfg.data.model_tokens_dir,
        dataset_root=cfg.data.dataset_root,
        dataset_ids=list(cfg.data.dataset_ids),
        gt_path=cfg.data.gt_path,
        gt_dataset_name_map=dict(cfg.data.get("gt_dataset_name_map", {}) or {}),
        missing_value=cfg.data.missing_value,
        seed=cfg.seed,
    )
    split = build_split(
        all_model_ids=bank.model_ids,
        all_dataset_ids=bank.dataset_ids,
        held_out_models=list(cfg.quadrant.held_out_models),
        held_out_datasets=list(cfg.quadrant.held_out_datasets),
    )

    train_ds = ListwiseDataset(bank, dataset_ids=split.train_dataset_ids, split="train")
    n_train = len(train_ds)
    bs = cfg.trainer.batch_size
    steps_per_epoch = math.ceil(n_train / bs) if n_train else 0
    total_steps = steps_per_epoch * cfg.trainer.epochs

    print()
    print("=" * 56)
    print("Cross-Select dry run")
    print("=" * 56)
    print(f"  models in zoo:         {len(bank.model_ids)}")
    print(f"  train_model_ids:       {len(split.train_model_ids)}")
    print(f"  eval_model_ids:        {len(split.eval_model_ids)}")
    print(f"  train datasets:        {n_train}  {split.train_dataset_ids}")
    print(f"  eval datasets:         {len(split.eval_dataset_ids)}  {split.eval_dataset_ids}")
    print(f"  eval_split:            {cfg.data.eval_split}")
    print(f"  batch_size:            {bs}")
    print(f"  steps per epoch:       {steps_per_epoch}")
    print(f"  epochs:                {cfg.trainer.epochs}")
    print(f"  total optimizer steps: {total_steps}")
    print()

    print("train shard counts per dataset:")
    total_train_shards = 0
    for d in split.train_dataset_ids:
        shards = list_shards(cfg.data.dataset_root, d, "train")
        total_train_shards += len(shards)
        print(f"  {d:22s} {len(shards):4d} shards")
    print(f"  {'TOTAL':22s} {total_train_shards:4d} shards")
    print()

    print(f"eval ({cfg.data.eval_split}) shard counts per dataset:")
    total_eval_shards = 0
    for d in split.eval_dataset_ids:
        shards = list_shards(cfg.data.dataset_root, d, cfg.data.eval_split)
        total_eval_shards += len(shards)
        print(f"  {d:22s} {len(shards):4d} shards")
    print(f"  {'TOTAL':22s} {total_eval_shards:4d} shards")
    print()

    item0 = train_ds[0]
    print("first train item shapes:")
    print(f"  dataset_id:    {item0['dataset_id']}")
    print(f"  dataset_token: {tuple(item0['dataset_token'].shape)}")
    print(f"  model_tokens:  {tuple(item0['model_tokens'].shape)}")
    print(f"  accuracy:      {tuple(item0['accuracy'].shape)}")
    print()
    print(
        f"stochastic shard draws over full run: "
        f"{cfg.trainer.epochs} epochs \u00d7 {n_train} train datasets = "
        f"{cfg.trainer.epochs * n_train}"
    )
    print("=" * 56)


if __name__ == "__main__":
    main()
