"""Single-GPU training loop with optional wandb logging."""

from __future__ import annotations

import logging
import math
import random
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
from ..models import CrossSelect, ModelSpider
from ..models.cross_select_encoder import CrossSelectWithEncoder
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
    mode: str = "prototype",
) -> DataLoader:
    ds = ListwiseDataset(bank, dataset_ids=dataset_ids, split=split, mode=mode)
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

        # Optional gradient clipping. Off when grad_clip is null/<=0.
        self.grad_clip = float(getattr(cfg, "grad_clip", 0.0) or 0.0)
        # Reconstruction loss weight for encoder models.
        self.recon_weight = float(getattr(cfg, "reconstruction_weight", 0.0) or 0.0)

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
            deterministic=bool(getattr(cfg, "deterministic_sampling", False)),
        )

        # Eval knobs: n_seeds averages N stochastic forwards and averages
        # the *scores* before computing metrics. 1 == legacy prototype eval.
        eval_cfg = getattr(cfg, "eval", None)
        self.eval_mode = str(getattr(eval_cfg, "mode", "prototype")) if eval_cfg is not None else "prototype"
        self.eval_n_seeds = int(getattr(eval_cfg, "n_seeds", 1)) if eval_cfg is not None else 1
        if self.eval_mode == "prototype" and self.eval_n_seeds != 1:
            # prototype is deterministic; N>1 is wasted compute.
            self.eval_n_seeds = 1

        # LR schedule (optional). Built lazily once we know step count.
        self.lr_scheduler = None
        self._sched_cfg = getattr(cfg, "schedule", None)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _select_models(
        self, model_tokens: torch.Tensor, target: torch.Tensor, idx: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sel = torch.tensor(idx, dtype=torch.long, device=model_tokens.device)
        return model_tokens.index_select(1, sel), target.index_select(1, sel)

    def _model_forward(
        self,
        model_tokens: torch.Tensor | None,
        model_idx: torch.Tensor | None,
        dataset_token: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Uniform forward path: Model Spider reads model_idx, others read
        model_tokens. One of the two must be provided. ``key_padding_mask``
        is forwarded verbatim; each model's forward accepts it as a kwarg.
        """
        if isinstance(self.model, ModelSpider):
            assert model_idx is not None
            return self.model(
                model_idx=model_idx,
                dataset_token=dataset_token,
                key_padding_mask=key_padding_mask,
            )
        assert model_tokens is not None
        return self.model(
            model_tokens, dataset_token, key_padding_mask=key_padding_mask,
            model_idx=model_idx,
        )

    def _forward_step(
        self, batch: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run model + loss on one sample-level step.

        ``batch`` comes from ``SampleBatchGenerator.build_epoch()``:
          dataset_tokens:   (K, S, C_max, 512)  -- zero-padded over C
          key_padding_mask: (K, C_max) bool, True = padded (ignored)
          model_tokens:     (M_sub, 512)
          accuracy:         (K, M_sub)

        Loss is computed **per sample** against the dataset's GT row and
        averaged over the K*S samples (loss-space averaging). Previously
        we averaged predictions before loss, which smoothed away the
        per-sample gradient signal through the non-linear attention.

        Also returns the aggregated-per-dataset ``pred (K, M_sub)`` and
        ``target (K, M_sub)`` for probe metrics so we don't recompute.
        """
        d = batch["dataset_tokens"].to(self.device)  # (K, S, C_max, 512)
        k, s, c, feat = d.shape
        d_flat = d.reshape(k * s, c, feat)  # (K*S, C_max, 512)

        mask = batch.get("key_padding_mask")
        mask_flat = None
        if mask is not None:
            mask = mask.to(self.device)  # (K, C_max)
            mask_flat = (
                mask.unsqueeze(1).expand(k, s, c).reshape(k * s, c)
            )  # (K*S, C_max)

        if isinstance(self.model, ModelSpider):
            m_idx = batch["model_idx"].to(self.device)  # (M_sub,)
            m_idx_b = m_idx.unsqueeze(0).expand(k * s, -1)
            pred_flat = self._model_forward(
                None, m_idx_b, d_flat, key_padding_mask=mask_flat
            )
        elif isinstance(self.model, CrossSelectWithEncoder):
            raw = batch["raw_model_tokens"].to(self.device)  # (M_sub, 8192)
            raw_bcast = raw.unsqueeze(0).expand(k * s, -1, -1)
            pred_flat = self._model_forward(
                raw_bcast, None, d_flat, key_padding_mask=mask_flat
            )
        else:
            m = batch["model_tokens"].to(self.device)  # (M_sub, 512)
            m_bcast = m.unsqueeze(0).expand(k * s, -1, -1)  # (K*S, M_sub, 512)
            m_idx = batch["model_idx"].to(self.device)  # (M_sub,)
            m_idx_b = m_idx.unsqueeze(0).expand(k * s, -1)
            pred_flat = self._model_forward(
                m_bcast, m_idx_b, d_flat, key_padding_mask=mask_flat
            )

        # Target: broadcast (K, M_sub) -> (K, S, M_sub) -> (K*S, M_sub).
        target = batch["accuracy"].to(self.device)  # (K, M_sub)
        target_flat = (
            target.unsqueeze(1).expand(-1, s, -1).reshape(k * s, -1)
        )

        loss, parts = self.loss_fn(pred_flat, target_flat)

        # Add reconstruction loss for encoder models
        if isinstance(self.model, CrossSelectWithEncoder) and self.recon_weight > 0:
            recon_loss = self.model.reconstruction_loss()
            if recon_loss is not None:
                loss = loss + self.recon_weight * recon_loss
                parts["recon"] = recon_loss.detach()

        # Add residual regularization for learnable residuals
        if isinstance(self.model, CrossSelect) and hasattr(self.model, 'residual_reg_loss'):
            reg = self.model.residual_reg_loss()
            if reg is not None:
                loss = loss + reg
                parts["residual_reg"] = reg.detach()

        # Prediction-space aggregate still reported for probe metrics.
        pred_mean = pred_flat.reshape(k, s, -1).mean(dim=1)  # (K, M_sub)
        return loss, parts, pred_mean.detach(), target.detach()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _build_scheduler(self, total_steps: int):
        """Linear warmup then cosine decay to 0. Disabled if cfg.schedule is
        missing, null, or has kind == "constant"."""
        sch = self._sched_cfg
        if sch is None or getattr(sch, "kind", "constant") == "constant":
            return None
        kind = getattr(sch, "kind", "constant")
        warmup_steps = int(getattr(sch, "warmup_steps", 0) or 0)
        min_lr_mult = float(getattr(sch, "min_lr_mult", 0.0) or 0.0)
        if kind == "cosine":

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return (step + 1) / max(warmup_steps, 1)
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                progress = min(progress, 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_mult + (1.0 - min_lr_mult) * cosine

            return torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
        raise ValueError(f"Unknown schedule kind: {kind!r}")

    def train(self) -> dict[str, float]:
        self.model.train()
        step = 0
        last_log: dict[str, torch.Tensor] = {}

        # Build scheduler now that we know step count. Uses a pessimistic
        # upper-bound: epochs * steps_per_epoch of the batcher.
        total_steps = self.cfg.epochs * max(self.sample_batcher.steps_per_epoch(), 1)
        self.lr_scheduler = self._build_scheduler(total_steps)

        is_tty = sys.stderr.isatty()
        bar_kwargs = dict(
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=0.5 if is_tty else 30.0,
            disable=None,
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
                loss, parts, pred_mean, target = self._forward_step(batch)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip
                    )
                self.optim.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                last_log = parts
                step += 1
                step_bar.set_postfix(loss=f"{float(parts['total']):.4f}")
                if step % self.cfg.log_every == 0:
                    payload = {
                        f"train/{k}": float(v) for k, v in parts.items()
                    }
                    payload["train/epoch"] = epoch
                    payload["train/step"] = step
                    payload["train/lr"] = float(self.optim.param_groups[0]["lr"])
                    payload.update(self._probe_metrics(pred_mean, target))
                    self._log(payload)
            step_bar.close()
            epoch_bar.set_postfix(
                loss=f"{float(last_log.get('total', torch.tensor(0.0))):.4f}"
            )

            if (epoch + 1) % self.cfg.eval_every == 0 or epoch == self.cfg.epochs - 1:
                metrics = self.evaluate()
                self._log_eval(metrics, epoch)
                summary = {
                    k: round(float(v), 4)
                    for k, v in metrics.items()
                    if k != "_per_dataset"
                }
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

    # ------------------------------------------------------------------
    # Probe metrics (cheap, in-batch) logged alongside train loss
    # ------------------------------------------------------------------

    def _probe_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, float]:
        """Compute per-row metrics on the aggregated (K, M_sub) prediction
        and return the mean across rows. Cheap: no extra forwards."""
        p = pred.cpu().numpy()
        t = target.cpu().numpy()
        agg: dict[str, list[float]] = {}
        for i in range(p.shape[0]):
            m = all_metrics(p[i], t[i])
            for k, v in m.items():
                agg.setdefault(k, []).append(v)
        return {f"train/probe/{k}": float(np.mean(v)) for k, v in agg.items()}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

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

        per_dataset: dict[str, dict[str, float]] = {}
        agg: dict[str, list[float]] = {}

        for dataset in dataset_ids:
            col = self.bank.dataset_ids.index(dataset)
            target_np = np.asarray(self.bank.accuracy[model_idx, col])
            pred_np = self._predict_dataset(dataset, split, model_idx)
            m = all_metrics(pred_np, target_np)
            per_dataset[dataset] = m
            for k, v in m.items():
                agg.setdefault(k, []).append(v)

        out = {k: float(np.mean(v)) for k, v in agg.items()}
        out["_per_dataset"] = per_dataset  # type: ignore[assignment]
        self.model.train()
        return out

    @torch.no_grad()
    def _predict_dataset(
        self, dataset: str, split: str, model_idx: list[int]
    ) -> np.ndarray:
        """Predict a (len(model_idx),) score vector for one dataset using
        the configured eval mode. For ``stochastic`` mode we average the
        predicted *scores* over ``self.eval_n_seeds`` independent shard
        draws, which matches the training-time input distribution
        ``E_x[f(x)]`` rather than ``f(E_x[x])``.
        """
        if self.eval_mode == "prototype":
            loader = make_eval_loader(
                self.bank, [dataset], split=split, mode="prototype"
            )
            for batch in loader:
                return self._score_item(batch[0], model_idx)
            raise RuntimeError(f"No eval shards for {dataset}/{split}")

        # stochastic: draw N independent single-shard samples, average scores.
        # We set a deterministic seed per eval call so the numbers are
        # reproducible across resumes.
        if self.eval_n_seeds <= 0:
            raise ValueError("eval.n_seeds must be >= 1 for stochastic eval")
        saved_rng = self.bank.rng.getstate()
        self.bank.rng.seed(0xE7A1 + hash(dataset) % 10_000_003)
        scores: list[np.ndarray] = []
        try:
            for _ in range(self.eval_n_seeds):
                loader = make_eval_loader(
                    self.bank, [dataset], split=split, mode="stochastic"
                )
                for batch in loader:
                    scores.append(self._score_item(batch[0], model_idx))
                    break
        finally:
            self.bank.rng.setstate(saved_rng)
        return np.mean(np.stack(scores, axis=0), axis=0)

    def _score_item(self, item: dict, model_idx: list[int]) -> np.ndarray:
        """Run one forward pass on an eval-loader item and return a
        ``(len(model_idx),)`` numpy score vector."""
        model_tokens = item["model_tokens"].unsqueeze(0).to(self.device)
        dataset_token = item["dataset_token"].unsqueeze(0).to(self.device)
        target = item["accuracy"].unsqueeze(0).to(self.device)
        model_tokens, _ = self._select_models(model_tokens, target, model_idx)
        if isinstance(self.model, ModelSpider):
            m_idx_b = torch.tensor(
                model_idx, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            pred = self.model(model_idx=m_idx_b, dataset_token=dataset_token)
        elif isinstance(self.model, CrossSelectWithEncoder):
            raw = item["raw_model_tokens"].unsqueeze(0).to(self.device)
            sel = torch.tensor(model_idx, dtype=torch.long, device=raw.device)
            raw = raw.index_select(1, sel)
            pred = self.model(raw, dataset_token)
        else:
            pred = self.model(model_tokens, dataset_token,
                              model_idx=torch.tensor(model_idx, dtype=torch.long, device=self.device))
        return pred.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------

    def _log_eval(self, metrics: dict[str, Any], epoch: int) -> None:
        """Flatten the mean metrics + per-dataset metrics into W&B keys."""
        payload: dict[str, Any] = {"val/epoch": epoch}
        for k, v in metrics.items():
            if k == "_per_dataset":
                continue
            payload[f"val/{k}"] = v
        per_ds = metrics.get("_per_dataset", {}) or {}
        for dataset_id, m in per_ds.items():
            for k, v in m.items():
                payload[f"val/{dataset_id}/{k}"] = v
        self._log(payload)

    def _log(self, payload: dict[str, Any]) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(payload)
