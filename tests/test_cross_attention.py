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


def test_model_spider_forward_shape():
    model = ModelSpider(
        model_token_dim=16, dataset_token_dim=16, hidden_dim=32, num_heads=4, num_layers=1
    )
    m, d = _toy_batch()
    assert model(m, d).shape == (2, 5)


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
    assert isinstance(build_model(cfg), ModelSpider)
