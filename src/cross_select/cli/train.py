"""Hydra entrypoint: ``python -m cross_select.cli.train``."""

from __future__ import annotations

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ..data.dataset import TokenBank
from ..models import build_model
from ..training.splits import build_split
from ..training.trainer import Trainer


logger = logging.getLogger(__name__)


def _pick_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("cuda not available, falling back to cpu")
        return "cpu"
    return requested


def _init_wandb(cfg: DictConfig):
    if cfg.wandb.mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping logging")
        return None
    return wandb.init(
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


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
    split = build_split(
        all_model_ids=bank.model_ids,
        all_dataset_ids=bank.dataset_ids,
        held_out_models=list(cfg.quadrant.held_out_models),
        held_out_datasets=list(cfg.quadrant.held_out_datasets),
    )
    logger.info(
        "quadrant %d: train_models=%d eval_models=%d train_datasets=%d eval_datasets=%d",
        cfg.quadrant.id,
        len(split.train_model_ids),
        len(split.eval_model_ids),
        len(split.train_dataset_ids),
        len(split.eval_dataset_ids),
    )

    model = build_model(cfg.model)
    device = _pick_device(cfg.device)
    wandb_run = _init_wandb(cfg)

    trainer_cfg = OmegaConf.create(OmegaConf.to_container(cfg.trainer, resolve=True))
    trainer_cfg.eval_split = cfg.data.eval_split

    trainer = Trainer(
        model=model,
        bank=bank,
        split=split,
        cfg=trainer_cfg,
        device=device,
        wandb_run=wandb_run,
    )
    metrics = trainer.train()
    summary = {k: v for k, v in metrics.items() if k != "_per_dataset"}
    logger.info("final val metrics: %s", summary)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
