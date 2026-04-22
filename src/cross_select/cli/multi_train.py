"""Hydra entrypoint: ``python -m cross_select.cli.multi_train``.

Runs N experiments in lockstep on a shared SampleBatchGenerator. Each
experiment becomes its own W&B run under a common W&B ``group`` so the
dashboard can overlay them for direct A/B comparison.

The root config selects an experiment set via Hydra composition:

    python -m cross_select.cli.multi_train experiment=multi_run

The ``experiment`` group is expected to define a list
``experiments`` of entries with ``name``, ``model``, and ``trainer``
sub-trees. The top-level ``trainer`` node from ``configs/trainer/``
provides the **shared** training budget (epochs, sampler knobs,
log/eval cadence) and default values for any missing per-experiment
keys.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ..data.dataset import TokenBank
from ..training.multi_trainer import ExperimentSpec, MultiTrainer
from ..training.splits import build_split


logger = logging.getLogger(__name__)


def _pick_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda not available, falling back to cpu")
        return "cpu"
    return requested


def _merge_with_shared(shared_trainer: DictConfig, per_exp: DictConfig) -> DictConfig:
    """Return a trainer cfg for one experiment: shared defaults + overrides.

    Per-experiment configs only need to list the keys they wish to
    override (lr, loss, optimizer, etc.); everything else falls back to
    the shared ``configs/trainer/`` defaults.
    """
    base = OmegaConf.create(OmegaConf.to_container(shared_trainer, resolve=True))
    if per_exp is None:
        return base
    return OmegaConf.merge(base, per_exp)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info("config:\n%s", OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    exp_block = cfg.get("experiments", None)
    if exp_block is None:
        raise ValueError(
            "multi_train requires an 'experiments' list in the composed "
            "config. Pick an experiment config with experiment=<name>."
        )

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

    device = _pick_device(cfg.device)

    # Build per-experiment specs: for each entry, model sub-tree takes the
    # top-level model cfg as a base and merges overrides; trainer likewise.
    shared_trainer = OmegaConf.create(
        OmegaConf.to_container(cfg.trainer, resolve=True)
    )
    shared_trainer.eval_split = cfg.data.eval_split

    specs: list[ExperimentSpec] = []
    for entry in exp_block:
        model_cfg = OmegaConf.merge(cfg.model, entry.get("model", {}))
        trainer_cfg = _merge_with_shared(shared_trainer, entry.get("trainer", None))
        specs.append(
            ExperimentSpec(
                name=str(entry.name),
                model=model_cfg,
                trainer=trainer_cfg,
            )
        )

    # Single W&B run; metrics are prefixed with the experiment name so
    # each variant's curves appear as their own panels. The legacy
    # ``wandb.group`` key is accepted as a fallback run name for
    # backward compatibility.
    wandb_run_name = (
        cfg.wandb.get("name", None)
        or cfg.wandb.get("group", None)
        or f"multi-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    logger.info(
        "multi_train: %d experiments in one W&B run '%s', device=%s",
        len(specs),
        wandb_run_name,
        device,
    )
    for s in specs:
        logger.info("  - %s", s.name)

    trainer = MultiTrainer(
        experiments=specs,
        bank=bank,
        split=split,
        shared_trainer_cfg=shared_trainer,
        device=device,
        wandb_run_name=str(wandb_run_name),
        wandb_project=str(cfg.wandb.project),
        wandb_mode=str(cfg.wandb.mode),
        wandb_dir=cfg.wandb.get("dir", None),
        full_cfg=cfg,
    )
    finals = trainer.train()
    logger.info("multi_train final:")
    for name, metrics in finals.items():
        logger.info("  [%s] %s", name, metrics)


if __name__ == "__main__":
    main()
