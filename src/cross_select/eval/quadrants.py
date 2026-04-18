"""Run the four-quadrant generalization evaluation and emit result tables."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..training.splits import Split, build_split
from ..training.trainer import Trainer

logger = logging.getLogger(__name__)


def quadrant_definitions(
    held_out_models: list[str],
    held_out_datasets: list[str],
) -> dict[int, dict[str, list[str]]]:
    """The four (held-out-models, held-out-datasets) configurations."""
    return {
        1: {"held_out_models": [], "held_out_datasets": []},
        2: {"held_out_models": held_out_models, "held_out_datasets": []},
        3: {"held_out_models": [], "held_out_datasets": held_out_datasets},
        4: {"held_out_models": held_out_models, "held_out_datasets": held_out_datasets},
    }


def run_all_quadrants(
    bank,
    model_builder,
    cfg,
    held_out_models: list[str],
    held_out_datasets: list[str],
    output_dir: str | Path,
    wandb_run=None,
) -> dict[int, dict[str, Any]]:
    """Train + evaluate a fresh model per quadrant. Returns a dict keyed by
    quadrant id with summary metrics and per-dataset breakdowns.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict[str, Any]] = {}
    defs = quadrant_definitions(held_out_models, held_out_datasets)
    for qid, held in defs.items():
        logger.info("=== Quadrant %d ===", qid)
        split = build_split(
            all_model_ids=bank.model_ids,
            all_dataset_ids=bank.dataset_ids,
            **held,
        )
        model = model_builder()
        trainer = Trainer(
            model=model,
            bank=bank,
            split=split,
            cfg=cfg.trainer,
            device=cfg.device,
            wandb_run=wandb_run,
        )
        metrics = trainer.train()
        results[qid] = {"split": held, "metrics": metrics}

    (output_dir / "quadrants.json").write_text(
        json.dumps(
            {str(k): {"split": v["split"], "metrics": {m: n for m, n in v["metrics"].items() if m != "_per_dataset"}} for k, v in results.items()},
            indent=2,
        )
    )
    return results
