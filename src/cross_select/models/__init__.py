"""Cross-Select and baseline model architectures."""

from .cross_select import CrossSelect
from .model_spider import ModelSpider


def build_model(cfg, *, num_models: int | None = None) -> "torch.nn.Module":  # type: ignore[name-defined]
    """Build a model from a Hydra/OmegaConf config node.

    Expects ``cfg.name`` ("cross_select" or "model_spider") and architecture
    hyperparameters. ``num_models`` is required only for ``model_spider``
    since that variant owns a learnable ``(num_models, model_token_dim)``
    embedding table.
    """
    kwargs = {
        "model_token_dim": cfg.model_token_dim,
        "dataset_token_dim": cfg.dataset_token_dim,
        "hidden_dim": cfg.hidden_dim,
        "num_heads": cfg.num_heads,
        "num_layers": cfg.num_layers,
        "dropout": cfg.dropout,
    }
    if cfg.name == "cross_select":
        return CrossSelect(**kwargs)
    if cfg.name == "model_spider":
        if num_models is None:
            raise ValueError(
                "num_models is required for model_spider; pass it through "
                "build_model(cfg, num_models=len(bank.model_ids))"
            )
        return ModelSpider(num_models=num_models, **kwargs)
    raise ValueError(f"Unknown model name: {cfg.name!r}")


__all__ = ["CrossSelect", "ModelSpider", "build_model"]
