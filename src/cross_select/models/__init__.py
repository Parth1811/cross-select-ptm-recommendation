"""Cross-Select and baseline model architectures."""

from .cross_select import CrossSelect
from .model_spider import ModelSpider


def build_model(cfg) -> "torch.nn.Module":  # type: ignore[name-defined]
    """Build a model from a Hydra/OmegaConf config node.

    Expects ``cfg.name`` ("cross_select" or "model_spider") and architecture
    hyperparameters. Keeps the construction path centralized so the CLI
    doesn't need to know the class mapping.
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
        return ModelSpider(**kwargs)
    raise ValueError(f"Unknown model name: {cfg.name!r}")


__all__ = ["CrossSelect", "ModelSpider", "build_model"]
