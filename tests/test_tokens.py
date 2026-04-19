"""Tests for model/dataset token loaders against the real artifacts tree."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from cross_select.data.dataset import ListwiseDataset, PairwiseDataset, TokenBank
from cross_select.data.tokens import (
    build_accuracy_matrix,
    dataset_prototype,
    list_shards,
    load_ground_truth,
    load_model_tokens,
    sample_dataset_tokens,
)


REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO / "artifacts" / "extracted" / "parc_model_embeddings"
DATASET_ROOT = REPO / "artifacts" / "extracted" / "datasets"
GT_PATH = REPO / "constants" / "ground_truth_rankings.json"


@pytest.fixture(scope="module")
def model_tokens():
    return load_model_tokens(MODEL_DIR)


def test_model_tokens_loaded(model_tokens):
    assert len(model_tokens) == 32
    for mid, emb in model_tokens.items():
        assert emb.shape == (512,)
        assert emb.dtype == np.float32
    assert "resnet50_imagenet" in model_tokens
    assert "alexnet_cub200" in model_tokens


def test_shards_discovered():
    shards = list_shards(DATASET_ROOT, "caltech_101", "train")
    assert len(shards) == 6
    assert all(p.name.startswith("train_clip_") for p in shards)


def test_sample_dataset_tokens_shape():
    shards = list_shards(DATASET_ROOT, "caltech_101", "train")
    rng = random.Random(0)
    feats = sample_dataset_tokens(shards, rng=rng)
    assert feats.shape == (102, 512)
    assert feats.dtype == np.float32

    row_feats = sample_dataset_tokens(shards, rng=rng, pick_row=True)
    assert row_feats.shape == (102, 512)


def test_dataset_prototype_is_deterministic():
    # Note: caltech_101 val/test shards have 101 classes (background class
    # dropped) while train has 102. The loader is agnostic to class count.
    shards = list_shards(DATASET_ROOT, "caltech_101", "validation")
    a = dataset_prototype(shards)
    b = dataset_prototype(shards)
    assert a.ndim == 2 and a.shape[1] == 512
    assert np.array_equal(a, b)


def test_ground_truth_structure():
    gt = load_ground_truth(GT_PATH)
    assert set(gt.keys()) == {
        "cifar_10",
        "oxford_pets",
        "cub200",
        "caltech_101",
        "stanford_pets",
        "nabird",
        "voc2007",
    }
    for target, col in gt.items():
        assert len(col) == 28, (target, len(col))
        for mid, acc in col.items():
            assert isinstance(acc, float)
            assert 0.0 <= acc <= 100.0


def test_accuracy_matrix_fills_missing_self_pairs(model_tokens):
    gt = load_ground_truth(GT_PATH)
    model_ids = sorted(model_tokens.keys())
    dataset_ids = ["caltech_101", "cifar_10"]
    rng = random.Random(0)
    mat = build_accuracy_matrix(
        gt, model_ids, dataset_ids, missing_value="random_rank", rng=rng
    )
    assert mat.shape == (len(model_ids), 2)
    assert not np.isnan(mat).any()
    # Every column must have exactly 4 "self" entries that are not in GT:
    # one per arch (alexnet/googlenet/resnet18/resnet50) with source == target.
    # 32 models - 28 GT rows = 4 filled per column.
    # Spot-check: resnet50_cifar10 is missing from the cifar10 GT column.
    # Inner keys (model ids) keep the no-underscore source form matching
    # the PARC embedding filenames; only outer GT keys were renamed.
    for arch in ("alexnet", "googlenet", "resnet18", "resnet50"):
        missing = f"{arch}_cifar10"
        assert missing in model_ids
        assert missing not in gt["cifar_10"]
    col = mat[:, dataset_ids.index("cifar_10")]
    observed = np.array(list(gt["cifar_10"].values()))
    lo, hi = observed.min() - 1e-2, observed.max() + 1e-2
    assert col.min() >= lo
    assert col.max() <= hi


@pytest.fixture(scope="module")
def bank():
    return TokenBank(
        model_tokens_dir=MODEL_DIR,
        dataset_root=DATASET_ROOT,
        dataset_ids=["caltech_101"],
        gt_path=GT_PATH,
        seed=0,
    )


def test_token_bank_shapes(bank):
    assert bank.model_tokens.shape == (32, 512)
    assert bank.accuracy.shape == (32, 1)
    assert not np.isnan(bank.accuracy).any()


def test_listwise_dataset(bank):
    ds = ListwiseDataset(bank, split="train")
    assert len(ds) == 1
    item = ds[0]
    assert item["dataset_token"].shape == (102, 512)
    assert item["model_tokens"].shape == (32, 512)
    assert item["accuracy"].shape == (32,)
    assert item["dataset_id"] == "caltech_101"


def test_listwise_eval_split_is_deterministic(bank):
    ds = ListwiseDataset(bank, split="validation")
    a = ds[0]["dataset_token"].numpy()
    b = ds[0]["dataset_token"].numpy()
    assert np.array_equal(a, b)


def test_pairwise_dataset(bank):
    ds = PairwiseDataset(bank, split="train")
    assert len(ds) == 32
    item = ds[0]
    assert item["dataset_token"].shape == (102, 512)
    assert item["model_token"].shape == (512,)
    assert item["accuracy"].ndim == 0
