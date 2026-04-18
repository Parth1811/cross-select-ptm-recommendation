"""End-to-end: train Cross-Select for a few epochs on caltech_101 and
verify it actually learns (loss drops, weighted-tau rises from the init).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from cross_select.data.dataset import TokenBank
from cross_select.models import build_model
from cross_select.training.splits import build_split
from cross_select.training.trainer import Trainer


REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "artifacts" / "extracted" / "parc_model_embeddings"
DATASET_ROOT = REPO / "artifacts" / "extracted" / "datasets"
GT_PATH = REPO / "constants" / "ground_truth_rankings.json"


@pytest.fixture(scope="module")
def trainer(tmp_path_factory):
    bank = TokenBank(
        model_tokens_dir=MODEL_DIR,
        dataset_root=DATASET_ROOT,
        dataset_ids=["caltech_101"],
        gt_path=GT_PATH,
        seed=0,
    )
    split = build_split(
        all_model_ids=bank.model_ids, all_dataset_ids=bank.dataset_ids
    )
    model_cfg = OmegaConf.create(
        {
            "name": "cross_select",
            "model_token_dim": 512,
            "dataset_token_dim": 512,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.0,
        }
    )
    trainer_cfg = OmegaConf.create(
        {
            "epochs": 30,
            "batch_size": 1,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "loss": {"ranking_weight": 1.0, "mse_weight": 0.1},
            "log_every": 1000,
            "eval_every": 1000,
            "checkpoint_dir": str(tmp_path_factory.mktemp("ckpt")),
        }
    )
    torch.manual_seed(0)
    model = build_model(model_cfg)
    return Trainer(
        model=model, bank=bank, split=split, cfg=trainer_cfg, device="cpu"
    )


def test_trainer_runs_and_improves(trainer):
    before = trainer.evaluate(split="train")
    tau_before = before["weighted_kendall_tau"]
    trainer.train()
    after = trainer.evaluate(split="train")
    tau_after = after["weighted_kendall_tau"]
    # With 1 dataset and 32 models, a 30-epoch CPU run should clearly improve
    # Kendall tau vs. random init on the training set.
    assert tau_after > tau_before
    assert tau_after > 0.2


def test_trainer_eval_on_validation(trainer):
    metrics = trainer.evaluate(split="validation")
    for k in ("weighted_kendall_tau", "ndcg", "precision@3", "mrr"):
        assert k in metrics
