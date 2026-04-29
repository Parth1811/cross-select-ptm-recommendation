"""Cross-Select and baseline model architectures."""

from __future__ import annotations

from .cross_select import CrossSelect
from .cross_select_encoder import CrossSelectWithEncoder
from .model_spider import ModelSpider


def build_model(cfg, *, num_models: int | None = None) -> "torch.nn.Module":  # type: ignore[name-defined]
    """Build a model from a Hydra/OmegaConf config node.

    Expects ``cfg.name`` and architecture hyperparameters.
    ``num_models`` is required only for ``model_spider``.
    """
    if cfg.name == "cross_select":
        return CrossSelect(
            model_token_dim=cfg.model_token_dim,
            dataset_token_dim=cfg.dataset_token_dim,
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            learnable_residuals=cfg.get("learnable_residuals", False),
            num_models=num_models or 0,
            residual_reg_weight=cfg.get("residual_reg_weight", 0.0),
        )
    if cfg.name == "cross_select_encoder":
        return CrossSelectWithEncoder(
            raw_dim=cfg.get("raw_dim", 8192),
            model_token_dim=cfg.model_token_dim,
            dataset_token_dim=cfg.dataset_token_dim,
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            encoder_kind=cfg.get("encoder_kind", "autoencoder"),
            encoder_hidden_dim=cfg.get("encoder_hidden_dim", 1024),
            encoder_dropout=cfg.get("encoder_dropout", 0.05),
        )
    if cfg.name == "model_spider":
        if num_models is None:
            raise ValueError(
                "num_models is required for model_spider; pass it through "
                "build_model(cfg, num_models=len(bank.model_ids))"
            )
        kwargs = {
            "model_token_dim": cfg.model_token_dim,
            "dataset_token_dim": cfg.dataset_token_dim,
            "hidden_dim": cfg.hidden_dim,
            "num_heads": cfg.num_heads,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
        }
        return ModelSpider(num_models=num_models, **kwargs)
    raise ValueError(f"Unknown model name: {cfg.name!r}")


__all__ = ["CrossSelect", "CrossSelectWithEncoder", "ModelSpider", "build_model"]
