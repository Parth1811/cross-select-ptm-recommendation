"""Hydra entrypoint: ``python -m cross_select.cli.evaluate``.

Runs the four-quadrant evaluation, training a fresh model per quadrant and
writing a summary table to ``${output_dir}/quadrants.json``.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ..data.dataset import TokenBank
from ..eval.quadrants import run_all_quadrants
from ..models import build_model


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("config:\n%s", OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    bank = TokenBank(
        model_tokens_dir=cfg.data.model_tokens_dir,
        dataset_root=cfg.data.dataset_root,
        dataset_ids=list(cfg.data.dataset_ids),
        gt_path=cfg.data.gt_path,
        gt_dataset_name_map=dict(cfg.data.get("gt_dataset_name_map", {}) or {}),
        missing_value=cfg.data.missing_value,
        seed=cfg.seed,
    )

    held_out_models = list(cfg.quadrant.get("held_out_models", []) or [])
    held_out_datasets = list(cfg.quadrant.get("held_out_datasets", []) or [])
    if not held_out_models and not held_out_datasets:
        logger.warning(
            "No held_out_models / held_out_datasets configured; Q2/Q3/Q4 will "
            "degenerate to Q1. Override via +quadrant.held_out_models=[...]"
        )

    model_builder = partial(build_model, cfg.model, num_models=len(bank.model_ids))
    output_dir = Path(cfg.trainer.checkpoint_dir).parent / "quadrant_eval"
    results = run_all_quadrants(
        bank=bank,
        model_builder=model_builder,
        cfg=cfg,
        held_out_models=held_out_models,
        held_out_datasets=held_out_datasets,
        output_dir=output_dir,
    )
    logger.info("quadrant summary:")
    for qid, r in results.items():
        summary = {k: v for k, v in r["metrics"].items() if k != "_per_dataset"}
        logger.info("  Q%d: %s", qid, summary)


if __name__ == "__main__":
    main()
