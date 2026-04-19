"""Hydra entrypoint: ``python -m cross_select.cli.dry_run``.

Reports how many samples/steps the sample-level training loop would visit
per epoch for the current Hydra config, without running any training or
GPU work.
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from ..data.dataset import SampleBatchGenerator, TokenBank
from ..training.splits import build_split


logger = logging.getLogger(__name__)


def _index_of(items: list[str], subset: list[str]) -> list[int]:
    idx = {name: i for i, name in enumerate(items)}
    return [idx[s] for s in subset]


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

    train_model_pool = _index_of(bank.model_ids, split.train_model_ids)
    gen = SampleBatchGenerator(
        bank=bank,
        dataset_ids=split.train_dataset_ids,
        num_models_per_step=cfg.trainer.num_models_per_step,
        num_datasets_per_step=cfg.trainer.num_datasets_per_step,
        num_samples_per_dataset=cfg.trainer.num_samples_per_dataset,
        seed=cfg.seed,
        model_pool_idx=train_model_pool,
    )

    M_sub = cfg.trainer.num_models_per_step
    K = cfg.trainer.num_datasets_per_step
    S = cfg.trainer.num_samples_per_dataset
    effective_preds = K * S
    rows_per_ds = gen.rows_per_dataset()
    chunks_per_ds = gen.chunks_per_dataset()
    total_rows = gen.total_rows()
    total_chunks = gen.total_chunks()
    steps_per_epoch = gen.steps_per_epoch()
    rows_consumed_per_epoch = steps_per_epoch * effective_preds
    dropped_rows = total_rows - rows_consumed_per_epoch
    total_steps = cfg.trainer.epochs * steps_per_epoch
    total_sample_draws = total_steps * effective_preds

    print()
    print("=" * 60)
    print("Cross-Select dry run (sample-level sampler)")
    print("=" * 60)
    print(f"  models in zoo:                      {len(bank.model_ids)}")
    print(f"  train_model_ids:                    {len(split.train_model_ids)}")
    print(f"  eval_model_ids:                     {len(split.eval_model_ids)}")
    print(f"  train datasets:                     {len(split.train_dataset_ids)}  {split.train_dataset_ids}")
    print(f"  eval datasets:                      {len(split.eval_dataset_ids)}  {split.eval_dataset_ids}")
    print(f"  eval_split:                         {cfg.data.eval_split}")
    print()
    print("per-step sampling knobs:")
    print(f"  num_models_per_step (M_sub):        {M_sub} / {len(bank.model_ids)}")
    print(f"  num_datasets_per_step (K):          {K} / {len(split.train_dataset_ids)}")
    print(f"  num_samples_per_dataset (S):        {S}")
    print(f"  predictions per step (K*S):         {effective_preds}")
    print(f"  predictions to loss (after agg, K): {K}")
    print()
    print("per-dataset shards / rows / chunks (chunk = S rows from one shard):")
    for d in split.train_dataset_ids:
        shards = gen.shards_per_dataset()[d]
        print(
            f"  {d:22s} shards={shards:4d}  rows={rows_per_ds[d]:6d}  "
            f"chunks={chunks_per_ds[d]:5d}"
        )
    print(f"  {'TOTAL':22s} rows={total_rows:6d}  chunks={total_chunks:5d}")
    print()
    print("per-epoch / per-run totals:")
    print(f"  steps per epoch:                    {steps_per_epoch}")
    print(f"  rows consumed per epoch:            {rows_consumed_per_epoch}")
    print(f"  rows dropped per epoch (tail):      {dropped_rows}")
    print(f"  epochs:                             {cfg.trainer.epochs}")
    print(f"  total optimizer steps:              {total_steps}")
    print(f"  total sample-row draws:             {total_sample_draws}")
    print()

    # Sample one step to show real shapes.
    sample_steps = gen.build_epoch()
    if sample_steps:
        b = sample_steps[0]
        print("first step shapes:")
        print(f"  dataset_ids:    {b['dataset_ids']}")
        print(f"  dataset_tokens: {tuple(b['dataset_tokens'].shape)}  (K, S, C_m, 512)")
        print(f"  model_tokens:   {tuple(b['model_tokens'].shape)}   (M_sub, 512)")
        print(f"  model_idx:      {tuple(b['model_idx'].shape)}")
        print(f"  accuracy:       {tuple(b['accuracy'].shape)}       (K, M_sub)")
        print(f"  C_m:            {b['C_m']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
