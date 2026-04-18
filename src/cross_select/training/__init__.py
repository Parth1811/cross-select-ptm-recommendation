"""Training loop and dataset/model splits."""

from .splits import Split, build_split
from .trainer import Trainer

__all__ = ["Split", "Trainer", "build_split"]
