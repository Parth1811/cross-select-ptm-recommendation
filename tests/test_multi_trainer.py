"""Tests for the multi-experiment trainer + its config."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from cross_select.data.dataset import TokenBank
from cross_select.training.multi_trainer import ExperimentSpec, MultiTrainer
from cross_select.training.splits import build_split


REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "artifacts" / "extracted" / "parc_model_embeddings"
DATASET_ROOT = REPO / "artifacts" / "extracted" / "datasets"
GT_PATH = REPO / "constants" / "ground_truth_rankings.json"


@pytest.fixture(scope="module")
def bank():
    return TokenBank(
        model_tokens_dir=MODEL_DIR,
        dataset_root=DATASET_ROOT,
        dataset_ids=["caltech_101"],
        gt_path=GT_PATH,
        seed=0,
    )


def _tiny_trainer_cfg(ckpt: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "epochs": 1,
            "num_models_per_step": 8,
            "num_datasets_per_step": 1,
            "num_samples_per_dataset": 4,
            "batch_size": 1,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "grad_clip": 1.0,
            "deterministic_sampling": True,
            "schedule": {
                "kind": "constant",
                "warmup_steps": 0,
                "min_lr_mult": 0.0,
            },
            "loss": {
                "kind": "listmle",
                "pred_temperature": 1.0,
                "ranking_weight": 1.0,
                "mse_weight": 1.0,
                "margin": 0.0,
            },
            "eval": {"mode": "prototype", "n_seeds": 1},
            "eval_split": "validation",
            "log_every": 1000,
            "eval_every": 1,
            "checkpoint_dir": str(ckpt),
        }
    )


def _tiny_model_cfg(name: str = "cross_select") -> OmegaConf:
    return OmegaConf.create(
        {
            "name": name,
            "model_token_dim": 512,
            "dataset_token_dim": 512,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.0,
        }
    )


def test_multi_trainer_runs_two_experiments_in_lockstep(bank, tmp_path):
    """Two experiments share the SampleBatchGenerator; both should
    complete one epoch and produce finite final metrics."""
    split = build_split(
        all_model_ids=bank.model_ids, all_dataset_ids=bank.dataset_ids
    )

    base_trainer = _tiny_trainer_cfg(tmp_path)
    # Per-experiment overrides: experiment B swaps to listmle_mse.
    exp_a_trainer = OmegaConf.merge(
        base_trainer, OmegaConf.create({"loss": {"kind": "listmle"}})
    )
    exp_b_trainer = OmegaConf.merge(
        base_trainer,
        OmegaConf.create({"loss": {"kind": "listmle_mse", "mse_weight": 1.0}}),
    )
    specs = [
        ExperimentSpec(name="exp_a", model=_tiny_model_cfg(), trainer=exp_a_trainer),
        ExperimentSpec(name="exp_b", model=_tiny_model_cfg(), trainer=exp_b_trainer),
    ]
    shared = base_trainer

    full_cfg = OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "data": {
                "eval_split": "validation",
            },
            "quadrant": {"id": 1, "held_out_models": [], "held_out_datasets": []},
        }
    )

    torch.manual_seed(0)
    trainer = MultiTrainer(
        experiments=specs,
        bank=bank,
        split=split,
        shared_trainer_cfg=shared,
        device="cpu",
        wandb_run_name="test-run",
        wandb_project="cross-select",
        wandb_mode="disabled",  # no real W&B call
        wandb_dir=None,
        full_cfg=full_cfg,
    )
    finals = trainer.train()
    assert set(finals.keys()) == {"exp_a", "exp_b"}
    for name, m in finals.items():
        for k in ("weighted_kendall_tau", "ndcg", "pearson"):
            assert k in m
            assert m[k] == m[k]  # not NaN


def test_multi_run_config_parses():
    """Verifies configs/experiment/multi_run.yaml composes cleanly."""
    repo_cfg_dir = REPO / "configs"

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(repo_cfg_dir), version_base=None):
        cfg = compose(
            config_name="config", overrides=["experiment=multi_run"]
        )
    assert "experiments" in cfg
    names = [e.name for e in cfg.experiments]
    assert names == [
        "cs-listmle",
        "cs-listmle-t0.5",
        "cs-listmle-mse",
        "cs-pairwise-bce",
        "spider-listmle",
    ]
