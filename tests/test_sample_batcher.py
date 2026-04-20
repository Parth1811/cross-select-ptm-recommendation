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
    # Driver = caltech_101 (only dataset), so steps/epoch = its chunk count.
    assert gen.steps_per_epoch() == 24


def test_shapes_and_driver_coverage(bank):
    """With K=1 and one dataset, the driver is caltech_101 and every
    chunk is consumed exactly once per epoch (upsample-small policy
    degenerates to single-pass when there is only one dataset).
    """
    K, S, M = 1, 4, 16
    gen = _make_gen(bank, K=K, S=S, M_sub=M)
    steps = gen.build_epoch()
    assert len(steps) == 24

    for step in steps:
        assert step["dataset_tokens"].shape == (K, S, 102, 512)
        assert step["dataset_tokens"].dtype == torch.float32
        assert step["model_tokens"].shape == (M, 512)
        assert step["model_idx"].shape == (M,)
        assert step["accuracy"].shape == (K, M)
        assert step["C_m"] == 102

    # 24 steps * 4 rows = 96 = every row exactly once.
    total_rows_consumed = sum(
        step["dataset_tokens"].shape[0] * step["dataset_tokens"].shape[1]
        for step in steps
    )
    assert total_rows_consumed == 96


def test_driver_appears_in_every_step_synthetic(monkeypatch, bank):
    """Construct a two-dataset scenario by aliasing caltech_101's shards
    under a second dataset name, and verify the 'driver appears in every
    step' invariant plus the cycle-on-exhaustion property.
    """
    # Create two "virtual" datasets that both map to caltech_101 shards.
    # big_ds will be the driver; small_ds pretends to have fewer chunks
    # by limiting its visible shards to 2 (32 rows, 8 chunks).
    big_name = "caltech_101"  # 6 shards, 24 chunks
    small_name = "caltech_101_small"  # aliased

    from cross_select.data.tokens import list_shards as real_list_shards

    real_caltech = real_list_shards(bank.dataset_root, big_name, "train")
    # Fake the shards accessor and caching for the second name.
    # We'll monkeypatch TokenBank.train_shards to recognise the alias.
    orig = bank.train_shards

    def fake_train_shards(d):
        if d == small_name:
            return real_caltech[:2]  # 2 shards \u2192 8 chunks
        return orig(d)

    monkeypatch.setattr(bank, "train_shards", fake_train_shards)

    # Also need to pretend dataset_ids/bank columns exist for small_name.
    # The generator uses self._dataset_col[d] from bank.dataset_ids; extend it.
    monkeypatch.setattr(bank, "dataset_ids", bank.dataset_ids + [small_name])
    import numpy as np

    monkeypatch.setattr(
        bank,
        "accuracy",
        np.concatenate([bank.accuracy, bank.accuracy[:, :1]], axis=1),
    )

    from cross_select.data.dataset import SampleBatchGenerator

    gen = SampleBatchGenerator(
        bank=bank,
        dataset_ids=[big_name, small_name],
        num_models_per_step=4,
        num_datasets_per_step=2,
        num_samples_per_dataset=4,
        seed=0,
    )
    assert gen.steps_per_epoch() == 24  # driven by caltech_101 (24 chunks)
    steps = gen.build_epoch()
    assert len(steps) == 24

    # Every step must contain the driver (big_name) exactly once.
    driver_presence = sum(1 for step in steps if big_name in step["dataset_ids"])
    assert driver_presence == 24
    # small_name should appear 24 times (K-1=1 partner per step),
    # cycling \u2248 3\u00d7 through its 8 chunks.
    partner_presence = sum(1 for step in steps if small_name in step["dataset_ids"])
    assert partner_presence == 24


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


def test_full_zoo_uses_deterministic_order(bank):
    """When num_models_per_step equals the pool size, every step must
    return the pool in the same (deterministic) order so the ListMLE
    target ranking doesn't thrash between steps.
    """
    pool = list(range(len(bank.model_ids)))
    gen = _make_gen(bank, K=1, S=4, M_sub=len(pool), model_pool_idx=pool)
    steps = gen.build_epoch()
    assert len(steps) > 0
    for step in steps:
        assert step["model_idx"].tolist() == pool


def test_raises_on_oversized_knobs(bank):
    with pytest.raises(ValueError):
        _make_gen(bank, K=2)  # only 1 dataset locally
    with pytest.raises(ValueError):
        _make_gen(bank, M_sub=33)  # zoo is 32
    with pytest.raises(ValueError):
        _make_gen(bank, M_sub=9, model_pool_idx=[0, 1, 2])  # pool too small
