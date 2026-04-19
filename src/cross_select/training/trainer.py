"""Single-GPU training loop with optional wandb logging."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..data.dataset import ListwiseDataset, TokenBank
from ..eval.metrics import all_metrics
from ..losses.ranking import CompatibilityLoss
from .splits import Split


logger = logging.getLogger(__name__)


def _index_of(items: list[str], subset: list[str]) -> list[int]:
    idx = {name: i for i, name in enumerate(items)}
    return [idx[s] for s in subset]


def _listwise_collate(batch: list[dict]) -> list[dict]:
    """Return the batch as a list. Datasets have different class counts C, so
    we can't stack dataset_token (C, D) along a new batch axis."""
    return batch


def make_loader(
    bank: TokenBank,
    dataset_ids: list[str],
    split: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = ListwiseDataset(bank, dataset_ids=dataset_ids, split=split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_listwise_collate,
    )


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        bank: TokenBank,
        split: Split,
        cfg: Any,
        device: str = "cuda",
        wandb_run=None,
    ) -> None:
        self.model = model.to(device)
        self.bank = bank
        self.split = split
        self.cfg = cfg
        self.device = device
        self.wandb_run = wandb_run

        self.train_model_idx = _index_of(bank.model_ids, split.train_model_ids)
        self.eval_model_idx = _index_of(bank.model_ids, split.eval_model_ids)

        self.loss_fn = CompatibilityLoss(
            ranking_weight=cfg.loss.ranking_weight,
            mse_weight=cfg.loss.mse_weight,
        )
        if cfg.optimizer == "adamw":
            self.optim = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "adam":
            self.optim = torch.optim.Adam(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")

        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader = make_loader(
            bank,
            split.train_dataset_ids,
            split="train",
            batch_size=cfg.batch_size,
            shuffle=True,
        )

    def _select_models(
        self, model_tokens: torch.Tensor, target: torch.Tensor, idx: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sel = torch.tensor(idx, dtype=torch.long, device=model_tokens.device)
        return model_tokens.index_select(1, sel), target.index_select(1, sel)

    def _forward_item(self, item: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run model + loss on one dataset item. Adds a leading batch axis so
        the transformer's batch_first=True layout gets (1, M, D) / (1, C, D)."""
        model_tokens = item["model_tokens"].unsqueeze(0).to(self.device)
        dataset_token = item["dataset_token"].unsqueeze(0).to(self.device)
        target = item["accuracy"].unsqueeze(0).to(self.device)
        model_tokens, target = self._select_models(
            model_tokens, target, self.train_model_idx
        )
        pred = self.model(model_tokens, dataset_token)
        return self.loss_fn(pred, target)

    def train(self) -> dict[str, float]:
        self.model.train()
        step = 0
        last_log: dict[str, torch.Tensor] = {}
        for epoch in range(self.cfg.epochs):
            t0 = time.time()
            for batch in self.train_loader:
                # `batch` is a list of per-dataset dicts (see _listwise_collate).
                # Accumulate the mean loss across the batch, then step once.
                self.optim.zero_grad(set_to_none=True)
                batch_loss = torch.zeros((), device=self.device)
                batch_parts: dict[str, float] = {}
                for item in batch:
                    loss, parts = self._forward_item(item)
                    batch_loss = batch_loss + loss / len(batch)
                    for k, v in parts.items():
                        batch_parts[k] = batch_parts.get(k, 0.0) + float(v) / len(batch)
                batch_loss.backward()
                self.optim.step()

                last_log = {k: torch.tensor(v) for k, v in batch_parts.items()}
                step += 1
                if step % self.cfg.log_every == 0:
                    self._log(
                        {f"train/{k}": v for k, v in batch_parts.items()}
                        | {"train/epoch": epoch, "train/step": step}
                    )

            if (epoch + 1) % self.cfg.eval_every == 0 or epoch == self.cfg.epochs - 1:
                metrics = self.evaluate()
                self._log({f"val/{k}": v for k, v in metrics.items() if k != "_per_dataset"} | {"val/epoch": epoch})
                logger.info("epoch %d eval metrics: %s", epoch, {k: v for k, v in metrics.items() if k != "_per_dataset"})

            logger.info(
                "epoch %d done in %.1fs (last loss=%.4f)",
                epoch,
                time.time() - t0,
                float(last_log.get("total", torch.tensor(0.0))),
            )

        final = self.evaluate()
        ckpt = self.checkpoint_dir / "last.pt"
        torch.save(
            {"model_state": self.model.state_dict(), "metrics": final}, ckpt
        )
        logger.info("saved checkpoint to %s", ckpt)
        return final

    @torch.no_grad()
    def evaluate(
        self,
        split: str | None = None,
        dataset_ids: list[str] | None = None,
        model_idx: list[int] | None = None,
    ) -> dict[str, float]:
        self.model.eval()
        split = split if split is not None else getattr(self.cfg, "eval_split", "test")
        dataset_ids = dataset_ids if dataset_ids is not None else self.split.eval_dataset_ids
        model_idx = model_idx if model_idx is not None else self.eval_model_idx
        loader = make_loader(
            self.bank,
            dataset_ids,
            split=split,
            batch_size=1,
            shuffle=False,
        )

        agg: dict[str, list[float]] = {}
        per_dataset: dict[str, dict[str, float]] = {}
        for batch in loader:
            # batch is a list with one dict (batch_size=1).
            item = batch[0]
            model_tokens = item["model_tokens"].unsqueeze(0).to(self.device)
            dataset_token = item["dataset_token"].unsqueeze(0).to(self.device)
            target = item["accuracy"].unsqueeze(0).to(self.device)
            model_tokens, target = self._select_models(
                model_tokens, target, model_idx
            )
            pred = self.model(model_tokens, dataset_token)
            p = pred.squeeze(0).cpu().numpy()
            t = target.squeeze(0).cpu().numpy()
            m = all_metrics(p, t)
            per_dataset[item["dataset_id"]] = m
            for k, v in m.items():
                agg.setdefault(k, []).append(v)

        out = {k: float(np.mean(v)) for k, v in agg.items()}
        out["_per_dataset"] = per_dataset  # type: ignore[assignment]
        self.model.train()
        return out

    def _log(self, payload: dict[str, Any]) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(payload)
