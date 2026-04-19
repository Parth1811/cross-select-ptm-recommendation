"""Single-GPU training loop with optional wandb logging."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..data.dataset import ListwiseDataset, SampleBatchGenerator, TokenBank
from ..eval.metrics import all_metrics
from ..losses.ranking import build_loss
from ..models import ModelSpider
from .splits import Split


logger = logging.getLogger(__name__)


def _index_of(items: list[str], subset: list[str]) -> list[int]:
    idx = {name: i for i, name in enumerate(items)}
    return [idx[s] for s in subset]


def _listwise_collate(batch: list[dict]) -> list[dict]:
    """Return the batch as a list. Datasets have different class counts C, so
    we can't stack dataset_token (C, D) along a new batch axis."""
    return batch


def make_eval_loader(
    bank: TokenBank,
    dataset_ids: list[str],
    split: str,
) -> DataLoader:
    ds = ListwiseDataset(bank, dataset_ids=dataset_ids, split=split)
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
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

        self.loss_fn = build_loss(cfg.loss)
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

        self.sample_batcher = SampleBatchGenerator(
            bank=bank,
            dataset_ids=split.train_dataset_ids,
            num_models_per_step=cfg.num_models_per_step,
            num_datasets_per_step=cfg.num_datasets_per_step,
            num_samples_per_dataset=cfg.num_samples_per_dataset,
            seed=getattr(cfg, "seed", 0),
            model_pool_idx=self.train_model_idx,
        )

    def _select_models(
        self, model_tokens: torch.Tensor, target: torch.Tensor, idx: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sel = torch.tensor(idx, dtype=torch.long, device=model_tokens.device)
        return model_tokens.index_select(1, sel), target.index_select(1, sel)

    def _forward_step(
        self, batch: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run model + loss on one sample-level step.

        ``batch`` comes from ``SampleBatchGenerator.build_epoch()``:
          dataset_tokens: (K, S, C_m, 512)
          model_tokens:   (M_sub, 512)
          accuracy:       (K, M_sub)
        where K=num_datasets_per_step, S=num_samples_per_dataset,
        M_sub=num_models_per_step.
        """
        d = batch["dataset_tokens"].to(self.device)  # (K, S, C, 512)
        k, s, c, feat = d.shape
        d_flat = d.reshape(k * s, c, feat)  # (K*S, C, 512)

        if isinstance(self.model, ModelSpider):
            # Learnable-token path: pass the (M_sub,) index tensor and let
            # the model look up rows from its own nn.Parameter table.
            m_idx = batch["model_idx"].to(self.device)  # (M_sub,)
            m_idx_b = m_idx.unsqueeze(0).expand(k * s, -1)  # (K*S, M_sub)
            pred = self.model(model_idx=m_idx_b, dataset_token=d_flat)
        else:
            m = batch["model_tokens"].to(self.device)  # (M_sub, 512)
            m_bcast = m.unsqueeze(0).expand(k * s, -1, -1)  # (K*S, M_sub, 512)
            pred = self.model(m_bcast, d_flat)

        pred = pred.reshape(k, s, -1).mean(dim=1)  # (K, M_sub) after per-dataset aggregate

        target = batch["accuracy"].to(self.device)  # (K, M_sub)
        return self.loss_fn(pred, target)

    def train(self) -> dict[str, float]:
        self.model.train()
        step = 0
        last_log: dict[str, torch.Tensor] = {}
        # Route bars to the real terminal only; in non-TTY runs (SLURM
        # stdout redirection, captured log files) fall back to a minimal
        # format that prints one line per chunk rather than spamming.
        is_tty = sys.stderr.isatty()
        bar_kwargs = dict(
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=0.5 if is_tty else 30.0,
            disable=None,  # disabled automatically if stderr is not a tty
        )
        epoch_bar = tqdm(
            range(self.cfg.epochs),
            desc="epochs",
            unit="epoch",
            position=0,
            leave=True,
            **bar_kwargs,
        )
        for epoch in epoch_bar:
            t0 = time.time()
            steps_this_epoch = self.sample_batcher.build_epoch()
            step_bar = tqdm(
                steps_this_epoch,
                desc=f"epoch {epoch}",
                unit="step",
                position=1,
                leave=False,
                **bar_kwargs,
            )
            for batch in step_bar:
                self.optim.zero_grad(set_to_none=True)
                loss, parts = self._forward_step(batch)
                loss.backward()
                self.optim.step()

                last_log = parts
                step += 1
                step_bar.set_postfix(loss=f"{float(parts['total']):.4f}")
                if step % self.cfg.log_every == 0:
                    self._log(
                        {f"train/{k}": float(v) for k, v in parts.items()}
                        | {"train/epoch": epoch, "train/step": step}
                    )
            step_bar.close()
            epoch_bar.set_postfix(
                loss=f"{float(last_log.get('total', torch.tensor(0.0))):.4f}"
            )

            if (epoch + 1) % self.cfg.eval_every == 0 or epoch == self.cfg.epochs - 1:
                metrics = self.evaluate()
                self._log(
                    {f"val/{k}": v for k, v in metrics.items() if k != "_per_dataset"}
                    | {"val/epoch": epoch}
                )
                summary = {
                    k: round(float(v), 4)
                    for k, v in metrics.items()
                    if k != "_per_dataset"
                }
                # logger goes to the Hydra .log file; tqdm.write draws to the
                # terminal without breaking the active bar.
                logger.info("epoch %d eval: %s", epoch, summary)
                tqdm.write(f"epoch {epoch} eval: {summary}")

            msg = (
                f"epoch {epoch} done in {time.time() - t0:.1f}s "
                f"steps={len(steps_this_epoch)} "
                f"(last loss={float(last_log.get('total', torch.tensor(0.0))):.4f})"
            )
            logger.info(msg)
            tqdm.write(msg)
        epoch_bar.close()

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
        loader = make_eval_loader(self.bank, dataset_ids, split=split)

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
            if isinstance(self.model, ModelSpider):
                m_idx_b = torch.tensor(
                    model_idx, dtype=torch.long, device=self.device
                ).unsqueeze(0)  # (1, len(model_idx))
                pred = self.model(model_idx=m_idx_b, dataset_token=dataset_token)
            else:
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
