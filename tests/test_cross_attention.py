"""Tests for the cross-attention model and the Model Spider baseline."""

from __future__ import annotations

import torch

from cross_select.models import CrossSelect, ModelSpider, build_model
from omegaconf import OmegaConf


def _toy_batch(B=2, M=5, C=7, Dm=16, Dd=16):
    return torch.randn(B, M, Dm), torch.randn(B, C, Dd)


def test_cross_select_forward_shape():
    model = CrossSelect(
        model_token_dim=16,
        dataset_token_dim=16,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    m, d = _toy_batch()
    out = model(m, d)
    assert out.shape == (2, 5)


def test_cross_select_variable_class_count():
    model = CrossSelect(
        model_token_dim=16, dataset_token_dim=16, hidden_dim=32, num_heads=4, num_layers=1
    )
    for C in (3, 12, 101, 102):
        m, d = _toy_batch(C=C)
        out = model(m, d)
        assert out.shape == (2, 5)


def test_cross_select_backward():
    model = CrossSelect(
        model_token_dim=16, dataset_token_dim=16, hidden_dim=32, num_heads=4, num_layers=1
    )
    m, d = _toy_batch()
    out = model(m, d)
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def _make_spider(num_models=8, M_dim=16, D_dim=16, H=32):
    return ModelSpider(
        num_models=num_models,
        model_token_dim=M_dim,
        dataset_token_dim=D_dim,
        hidden_dim=H,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )


def test_model_spider_has_learnable_model_token_table():
    model = _make_spider(num_models=8, M_dim=16)
    assert hasattr(model, "model_embeddings")
    assert model.model_embeddings.shape == (8, 16)
    # nn.Parameter is included in model.parameters() \u2192 trainable by default.
    names = [n for n, _ in model.named_parameters()]
    assert "model_embeddings" in names


def test_model_spider_forward_shape():
    model = _make_spider()
    _, d = _toy_batch(B=2, C=7, Dd=16)
    idx = torch.tensor([[0, 1, 2, 3, 4]] * 2)  # (B=2, M=5)
    assert model(model_idx=idx, dataset_token=d).shape == (2, 5)


def test_model_spider_broadcasts_1d_model_idx():
    model = _make_spider()
    _, d = _toy_batch(B=3, C=7, Dd=16)
    idx = torch.tensor([0, 1, 2, 3, 4])  # 1D \u2192 broadcast across B
    assert model(model_idx=idx, dataset_token=d).shape == (3, 5)


def test_model_spider_different_idx_gives_different_output():
    """If the token table weren't actually looked up, swapping indices
    wouldn't change the output. Guards against accidental collapse."""
    torch.manual_seed(0)
    model = _make_spider(num_models=8)
    model.eval()
    _, d = _toy_batch(B=1, C=7, Dd=16)
    out0 = model(model_idx=torch.tensor([[0, 1, 2]]), dataset_token=d)
    out1 = model(model_idx=torch.tensor([[5, 6, 7]]), dataset_token=d)
    assert not torch.allclose(out0, out1)


def test_model_spider_backward_updates_token_table():
    model = _make_spider(num_models=8)
    _, d = _toy_batch(B=2, C=7, Dd=16)
    idx = torch.tensor([[0, 1, 2, 3]] * 2)
    out = model(model_idx=idx, dataset_token=d)
    out.sum().backward()
    assert model.model_embeddings.grad is not None
    # Only the rows used should have non-zero gradient magnitude.
    used = torch.tensor([0, 1, 2, 3])
    unused = torch.tensor([4, 5, 6, 7])
    assert model.model_embeddings.grad[used].abs().sum() > 0
    assert torch.all(model.model_embeddings.grad[unused] == 0)


def test_cross_select_mask_all_false_matches_no_mask():
    """An all-False mask should be semantically identical to passing no
    mask at all. Acts as a regression guard on the mask plumbing."""
    torch.manual_seed(0)
    model = CrossSelect(
        model_token_dim=16,
        dataset_token_dim=16,
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    model.eval()
    m, d = _toy_batch(B=2, C=7, Dd=16)
    mask = torch.zeros(2, 7, dtype=torch.bool)
    out_no_mask = model(m, d)
    out_false_mask = model(m, d, key_padding_mask=mask)
    assert torch.allclose(out_no_mask, out_false_mask, atol=1e-6)


def test_cross_select_mask_changes_output():
    """Masking some K positions must change the output; guards against the
    mask silently being dropped on its way to MultiheadAttention."""
    torch.manual_seed(0)
    model = CrossSelect(
        model_token_dim=16,
        dataset_token_dim=16,
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    model.eval()
    m, d = _toy_batch(B=1, C=7, Dd=16)
    unmasked = model(m, d)
    mask = torch.tensor([[False, False, True, True, True, True, True]])
    masked = model(m, d, key_padding_mask=mask)
    assert not torch.allclose(unmasked, masked, atol=1e-4)


def test_cross_select_padding_equivalence():
    """Attending over a padded-and-masked (C_max) sequence should match
    attending over the same unpadded (C_real) sequence. This is the
    'padding doesn't leak into attention' invariant.
    """
    torch.manual_seed(1)
    model = CrossSelect(
        model_token_dim=16,
        dataset_token_dim=16,
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    model.eval()
    m = torch.randn(1, 4, 16)
    d_real = torch.randn(1, 5, 16)  # real length 5
    pad = torch.zeros(1, 3, 16)  # 3 padding positions
    d_padded = torch.cat([d_real, pad], dim=1)  # (1, 8, 16)
    mask = torch.tensor([[False] * 5 + [True] * 3])  # ignore padding
    out_real = model(m, d_real)
    out_padded = model(m, d_padded, key_padding_mask=mask)
    assert torch.allclose(out_real, out_padded, atol=1e-5)


def test_model_spider_mask_all_false_matches_no_mask():
    torch.manual_seed(0)
    model = _make_spider(num_models=8)
    model.eval()
    _, d = _toy_batch(B=1, C=7, Dd=16)
    idx = torch.tensor([[0, 1, 2, 3, 4]])
    mask = torch.zeros(1, 7, dtype=torch.bool)
    a = model(model_idx=idx, dataset_token=d)
    b = model(model_idx=idx, dataset_token=d, key_padding_mask=mask)
    assert torch.allclose(a, b, atol=1e-6)


def test_model_spider_padding_equivalence():
    torch.manual_seed(2)
    model = _make_spider(num_models=8)
    model.eval()
    idx = torch.tensor([[0, 1, 2, 3, 4]])
    d_real = torch.randn(1, 4, 16)
    pad = torch.zeros(1, 3, 16)
    d_padded = torch.cat([d_real, pad], dim=1)
    mask = torch.tensor([[False] * 4 + [True] * 3])
    a = model(model_idx=idx, dataset_token=d_real)
    b = model(model_idx=idx, dataset_token=d_padded, key_padding_mask=mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_build_model_from_cfg():
    cfg = OmegaConf.create(
        {
            "name": "cross_select",
            "model_token_dim": 16,
            "dataset_token_dim": 16,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.0,
        }
    )
    m = build_model(cfg)
    assert isinstance(m, CrossSelect)
    cfg.name = "model_spider"
    import pytest

    with pytest.raises(ValueError):
        build_model(cfg)  # num_models required
    assert isinstance(build_model(cfg, num_models=8), ModelSpider)
