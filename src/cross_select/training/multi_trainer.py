"""Multi-experiment trainer.

Trains N models in lockstep on a shared SampleBatchGenerator inside a
single process so we can make full use of a large GPU (40GB where one
experiment only fills ~1GB). Each experiment owns its own model,
optimizer, LR scheduler, loss function, and W&B run \u2014 but every
experiment sees the *same* batch on the same step, which makes the
runs directly comparable.

Outline:
- build a shared :class:`TokenBank`, :class:`SampleBatchGenerator`, and
  :class:`Split` once;
- build N :class:`Experiment` instances from N trainer/model configs;
- per step: fetch one batch, then for each experiment run its forward +
  backward + optimizer step and log to its W&B run under a common group.

This is ~100 LOC rather than a refactor of :class:`Trainer`; the two
paths share the primitives (loss factory, sample batcher, metrics).
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ..data.dataset import ListwiseDataset, SampleBatchGenerator, TokenBank
from ..eval.metrics import all_metrics
from ..losses.ranking import build_loss
from ..models import ModelSpider, build_model
from .splits import Split


logger = logging.getLogger(__name__)


def _index_of(items: list[str], subset: list[str]) -> list[int]:
    idx = {name: i for i, name in enumerate(items)}
    return [idx[s] for s in subset]


@dataclass
class ExperimentSpec:
    """One entry in the multi-run config.

    ``model`` and ``trainer`` are OmegaConf DictConfig nodes (same shape
    as the single-run configs/model/*.yaml and configs/trainer/*.yaml).
    """

    name: str
    model: Any  # DictConfig
    trainer: Any  # DictConfig


class _Experiment:
    """Per-experiment state: model, optimizer, loss, scheduler, W&B run."""

    def __init__(
        self,
        spec: ExperimentSpec,
        bank: TokenBank,
        split: Split,
        device: str,
        total_steps: int,
        wandb_group: str,
        wandb_project: str,
        wandb_mode: str,
        wandb_dir: str | None,
        full_cfg: Any,
    ) -> None:
        self.name = spec.name
        self.trainer_cfg = spec.trainer
        self.device = device

        self.train_model_idx = _index_of(bank.model_ids, split.train_model_ids)
        self.eval_model_idx = _index_of(bank.model_ids, split.eval_model_ids)

        self.model = build_model(spec.model, num_models=len(bank.model_ids)).to(device)
        self.loss_fn = build_loss(spec.trainer.loss)
        self.optim = self._build_optim(spec.trainer)
        self.scheduler = self._build_scheduler(spec.trainer, total_steps)
        self.grad_clip = float(getattr(spec.trainer, "grad_clip", 0.0) or 0.0)

        eval_cfg = getattr(spec.trainer, "eval", None)
        self.eval_mode = (
            str(getattr(eval_cfg, "mode", "prototype"))
            if eval_cfg is not None
            else "prototype"
        )
        self.eval_n_seeds = (
            int(getattr(eval_cfg, "n_seeds", 1)) if eval_cfg is not None else 1
        )
        if self.eval_mode == "prototype":
            self.eval_n_seeds = 1

        self.bank = bank
        self.split = split

        self.checkpoint_dir = Path(spec.trainer.checkpoint_dir) / self.name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.wandb_run = None
        if wandb_mode != "disabled":
            try:
                import wandb
                from omegaconf import OmegaConf

                if wandb_dir is not None:
                    Path(wandb_dir).mkdir(parents=True, exist_ok=True)
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    mode=wandb_mode,
                    group=wandb_group,
                    name=f"{wandb_group}-{self.name}",
                    dir=wandb_dir,
                    config=OmegaConf.to_container(
                        OmegaConf.create(
                            {
                                "experiment_name": self.name,
                                "model": spec.model,
                                "trainer": spec.trainer,
                                "shared": {
                                    "data": full_cfg.data,
                                    "quadrant": full_cfg.quadrant,
                                    "seed": full_cfg.seed,
                                    "device": full_cfg.device,
                                },
                            }
                        ),
                        resolve=True,
                    ),
                    reinit=True,  # multi-run: reinit per experiment
                )
            except ImportError:
                logger.warning("wandb not installed; skipping logging for %s", self.name)

    def _build_optim(self, cfg: Any) -> torch.optim.Optimizer:
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        if cfg.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")

    def _build_scheduler(self, cfg: Any, total_steps: int):
        sch = getattr(cfg, "schedule", None)
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
                return min_lr_mult + (1.0 - min_lr_mult) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )

            return torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)
        raise ValueError(f"Unknown schedule kind: {kind!r}")

    # ------------------------------------------------------------------
    # Forward / eval \u2014 mirror the single-experiment Trainer exactly.
    # ------------------------------------------------------------------

    def _select_models(
        self, model_tokens: torch.Tensor, target: torch.Tensor, idx: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sel = torch.tensor(idx, dtype=torch.long, device=model_tokens.device)
        return model_tokens.index_select(1, sel), target.index_select(1, sel)

    def _forward_step(self, batch: dict):
        d = batch["dataset_tokens"].to(self.device)
        k, s, c, feat = d.shape
        d_flat = d.reshape(k * s, c, feat)
        mask = batch.get("key_padding_mask")
        mask_flat = None
        if mask is not None:
            mask = mask.to(self.device)
            mask_flat = mask.unsqueeze(1).expand(k, s, c).reshape(k * s, c)
        if isinstance(self.model, ModelSpider):
            m_idx = batch["model_idx"].to(self.device)
            m_idx_b = m_idx.unsqueeze(0).expand(k * s, -1)
            pred_flat = self.model(
                model_idx=m_idx_b,
                dataset_token=d_flat,
                key_padding_mask=mask_flat,
            )
        else:
            m = batch["model_tokens"].to(self.device)
            m_bcast = m.unsqueeze(0).expand(k * s, -1, -1)
            pred_flat = self.model(m_bcast, d_flat, key_padding_mask=mask_flat)
        target = batch["accuracy"].to(self.device)
        target_flat = target.unsqueeze(1).expand(-1, s, -1).reshape(k * s, -1)
        loss, parts = self.loss_fn(pred_flat, target_flat)
        pred_mean = pred_flat.reshape(k, s, -1).mean(dim=1).detach()
        return loss, parts, pred_mean, target.detach()

    def train_step(self, batch: dict, step: int, epoch: int, log_every: int):
        self.model.train()
        self.optim.zero_grad(set_to_none=True)
        loss, parts, pred_mean, target = self._forward_step(batch)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
        self.optim.step()
        if self.scheduler is not None:
            self.scheduler.step()
        if step % log_every == 0 and self.wandb_run is not None:
            payload = {f"train/{k}": float(v) for k, v in parts.items()}
            payload["train/epoch"] = epoch
            payload["train/step"] = step
            payload["train/lr"] = float(self.optim.param_groups[0]["lr"])
            # Probe metrics on the aggregated per-dataset pred.
            p = pred_mean.cpu().numpy()
            t = target.cpu().numpy()
            agg: dict[str, list[float]] = {}
            for i in range(p.shape[0]):
                m = all_metrics(p[i], t[i])
                for k2, v in m.items():
                    agg.setdefault(k2, []).append(v)
            payload.update(
                {f"train/probe/{k2}": float(np.mean(v)) for k2, v in agg.items()}
            )
            self.wandb_run.log(payload)
        return float(parts["total"])

    @torch.no_grad()
    def evaluate(self, split_name: str) -> dict[str, Any]:
        from .trainer import make_eval_loader  # local import; reuse helper

        self.model.eval()
        dataset_ids = self.split.eval_dataset_ids
        model_idx = self.eval_model_idx
        per_dataset: dict[str, dict[str, float]] = {}
        agg: dict[str, list[float]] = {}
        for dataset in dataset_ids:
            col = self.bank.dataset_ids.index(dataset)
            target_np = np.asarray(self.bank.accuracy[model_idx, col])
            pred_np = self._predict_dataset(dataset, split_name, model_idx)
            m = all_metrics(pred_np, target_np)
            per_dataset[dataset] = m
            for k, v in m.items():
                agg.setdefault(k, []).append(v)
        out: dict[str, Any] = {k: float(np.mean(v)) for k, v in agg.items()}
        out["_per_dataset"] = per_dataset
        self.model.train()
        return out

    @torch.no_grad()
    def _predict_dataset(
        self, dataset: str, split: str, model_idx: list[int]
    ) -> np.ndarray:
        from .trainer import make_eval_loader

        if self.eval_mode == "prototype":
            loader = make_eval_loader(
                self.bank, [dataset], split=split, mode="prototype"
            )
            for batch in loader:
                return self._score_item(batch[0], model_idx)
            raise RuntimeError(f"No eval shards for {dataset}/{split}")
        saved = self.bank.rng.getstate()
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
            self.bank.rng.setstate(saved)
        return np.mean(np.stack(scores, axis=0), axis=0)

    def _score_item(self, item: dict, model_idx: list[int]) -> np.ndarray:
        model_tokens = item["model_tokens"].unsqueeze(0).to(self.device)
        dataset_token = item["dataset_token"].unsqueeze(0).to(self.device)
        target = item["accuracy"].unsqueeze(0).to(self.device)
        model_tokens, _ = self._select_models(model_tokens, target, model_idx)
        if isinstance(self.model, ModelSpider):
            m_idx_b = torch.tensor(
                model_idx, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            pred = self.model(model_idx=m_idx_b, dataset_token=dataset_token)
        else:
            pred = self.model(model_tokens, dataset_token)
        return pred.squeeze(0).cpu().numpy()

    def log_eval(self, metrics: dict, epoch: int) -> None:
        if self.wandb_run is None:
            return
        payload: dict[str, Any] = {"val/epoch": epoch}
        for k, v in metrics.items():
            if k == "_per_dataset":
                continue
            payload[f"val/{k}"] = v
        per_ds = metrics.get("_per_dataset", {}) or {}
        for dataset_id, m in per_ds.items():
            for k, v in m.items():
                payload[f"val/{dataset_id}/{k}"] = v
        self.wandb_run.log(payload)

    def save_checkpoint(self, metrics: dict) -> None:
        ckpt = self.checkpoint_dir / "last.pt"
        torch.save({"model_state": self.model.state_dict(), "metrics": metrics}, ckpt)

    def finish_wandb(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()


class MultiTrainer:
    """Run N experiments in lockstep on a shared data pipeline."""

    def __init__(
        self,
        experiments: list[ExperimentSpec],
        bank: TokenBank,
        split: Split,
        shared_trainer_cfg: Any,  # the first experiment's trainer cfg is used for
                                  # sampler + epoch budget; per-exp cfg is used
                                  # for model/loss/optim only
        device: str,
        wandb_group: str,
        wandb_project: str,
        wandb_mode: str,
        wandb_dir: str | None,
        full_cfg: Any,
    ) -> None:
        self.bank = bank
        self.split = split
        self.device = device
        self.shared_cfg = shared_trainer_cfg

        self.sample_batcher = SampleBatchGenerator(
            bank=bank,
            dataset_ids=split.train_dataset_ids,
            num_models_per_step=shared_trainer_cfg.num_models_per_step,
            num_datasets_per_step=shared_trainer_cfg.num_datasets_per_step,
            num_samples_per_dataset=shared_trainer_cfg.num_samples_per_dataset,
            seed=getattr(full_cfg, "seed", 0),
            model_pool_idx=_index_of(bank.model_ids, split.train_model_ids),
            deterministic=bool(
                getattr(shared_trainer_cfg, "deterministic_sampling", False)
            ),
        )

        total_steps = (
            shared_trainer_cfg.epochs * max(self.sample_batcher.steps_per_epoch(), 1)
        )

        self.experiments: list[_Experiment] = [
            _Experiment(
                spec=spec,
                bank=bank,
                split=split,
                device=device,
                total_steps=total_steps,
                wandb_group=wandb_group,
                wandb_project=wandb_project,
                wandb_mode=wandb_mode,
                wandb_dir=wandb_dir,
                full_cfg=full_cfg,
            )
            for spec in experiments
        ]

    def train(self) -> dict[str, dict[str, float]]:
        is_tty = sys.stderr.isatty()
        bar_kwargs = dict(
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=0.5 if is_tty else 30.0,
            disable=None,
        )
        step = 0
        epochs = self.shared_cfg.epochs
        log_every = int(self.shared_cfg.log_every)
        eval_every = int(self.shared_cfg.eval_every)
        eval_split = getattr(self.shared_cfg, "eval_split", "test")

        epoch_bar = tqdm(
            range(epochs), desc="epochs", unit="epoch", position=0, leave=True,
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
                step += 1
                losses = {
                    exp.name: exp.train_step(batch, step, epoch, log_every)
                    for exp in self.experiments
                }
                step_bar.set_postfix(
                    **{k: f"{v:.3f}" for k, v in losses.items()}
                )
            step_bar.close()

            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                for exp in self.experiments:
                    metrics = exp.evaluate(eval_split)
                    exp.log_eval(metrics, epoch)
                    summary = {
                        k: round(float(v), 4)
                        for k, v in metrics.items()
                        if k != "_per_dataset"
                    }
                    logger.info("epoch %d [%s] eval: %s", epoch, exp.name, summary)
                    tqdm.write(f"epoch {epoch} [{exp.name}] eval: {summary}")

            tqdm.write(
                f"epoch {epoch} done in {time.time() - t0:.1f}s "
                f"steps={len(steps_this_epoch)} "
                f"losses={ {n: round(l, 3) for n, l in losses.items()} }"
            )
        epoch_bar.close()

        # Final eval + checkpoint per experiment.
        final: dict[str, dict[str, float]] = {}
        for exp in self.experiments:
            m = exp.evaluate(eval_split)
            exp.save_checkpoint(m)
            exp.finish_wandb()
            final[exp.name] = {k: v for k, v in m.items() if k != "_per_dataset"}
        return final
