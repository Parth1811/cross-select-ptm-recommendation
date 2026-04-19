"""Tests for SampleBatchGenerator (sample-level training sampler)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cross_select.data.dataset import SampleBatchGenerator, TokenBank


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


def _make_gen(
    bank,
    K=1,
    S=4,
    M_sub=16,
    dataset_ids=None,
    seed=0,
    model_pool_idx=None,
):
    return SampleBatchGenerator(
        bank=bank,
        dataset_ids=dataset_ids or bank.dataset_ids,
        num_models_per_step=M_sub,
        num_datasets_per_step=K,
        num_samples_per_dataset=S,
        seed=seed,
        model_pool_idx=model_pool_idx,
    )


def test_reporting_numbers_match_data(bank):
    gen = _make_gen(bank, K=1, S=4)
    # caltech_101 train has 6 shards \u00d7 16 rows = 96 rows, 96/4 = 24 chunks.
    assert gen.rows_per_dataset() == {"caltech_101": 96}
    assert gen.chunks_per_dataset() == {"caltech_101": 24}
    assert gen.total_rows() == 96
    assert gen.total_chunks() == 24
    # K=1 so steps/epoch == chunks.
    assert gen.steps_per_epoch() == 24


def test_shapes_and_coverage(bank):
    K, S, M = 1, 4, 16
    gen = _make_gen(bank, K=K, S=S, M_sub=M)
    steps = gen.build_epoch()
    assert len(steps) == 24

    # Every step has the right shapes.
    for step in steps:
        assert step["dataset_tokens"].shape == (K, S, 102, 512)  # caltech_101: C=102
        assert step["dataset_tokens"].dtype == torch.float32
        assert step["model_tokens"].shape == (M, 512)
        assert step["model_idx"].shape == (M,)
        assert step["accuracy"].shape == (K, M)
        assert step["C_m"] == 102

    # Sample coverage: across the epoch, every (shard, row) should appear
    # exactly S times (once per chunk placement across the epoch, summed
    # along axis 0 of the shard's features). With K=1, 24 chunks each
    # consuming 4 rows = 96 consumptions = every row exactly once.
    total_rows = sum(step["dataset_tokens"].shape[0] * step["dataset_tokens"].shape[1] for step in steps)
    assert total_rows == 96


def test_model_subsampling_uses_pool(bank):
    pool = [0, 1, 2, 3, 4, 5, 6, 7]
    gen = _make_gen(bank, K=1, S=4, M_sub=4, model_pool_idx=pool)
    steps = gen.build_epoch()
    for step in steps:
        drawn = set(int(x) for x in step["model_idx"])
        assert drawn.issubset(set(pool))
        assert len(drawn) == 4


def test_accuracy_column_alignment(bank):
    gen = _make_gen(bank, K=1, S=4, M_sub=4)
    steps = gen.build_epoch()
    step = steps[0]
    # Accuracy row for caltech_101 should match bank.accuracy[model_idx, 0].
    col = bank.dataset_ids.index("caltech_101")
    expected = bank.accuracy[step["model_idx"].numpy(), col]
    assert torch.allclose(step["accuracy"][0].float(), torch.tensor(expected))


def test_raises_on_oversized_knobs(bank):
    with pytest.raises(ValueError):
        _make_gen(bank, K=2)  # only 1 dataset locally
    with pytest.raises(ValueError):
        _make_gen(bank, M_sub=33)  # zoo is 32
    with pytest.raises(ValueError):
        _make_gen(bank, M_sub=9, model_pool_idx=[0, 1, 2])  # pool too small
