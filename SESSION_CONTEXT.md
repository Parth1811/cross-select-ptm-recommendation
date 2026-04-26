# Session Context — Cross-Select Implementation

Snapshot of the project state and recent decisions, intended to be loaded
into a future Claude session for continuity. Date: 2026-04-26.

## Repo

- Local path: `/Users/parth/ptm-new-implementation`
- Branch in active development: `feature/loss-improvements`
- Latest commit: `3f9db36 feat(cross_select): add dual-head dataset encoder + two-layer model encoder`
- Python (local): `/Users/parth/.virtualenv/thesis/bin/python` (3.10, with torch/numpy/pytest/hydra)

## Remotes

- `gautschi` → `gautschi:ptm-new-implementation/.git` (Purdue HPC, working-tree
  repo with `receive.denyCurrentBranch=ignore` + a `post-receive` hook that
  checks out the pushed branch into `/home/patil185/ptm-new-implementation`).
- `origin` → `git@github.com:Parth1811/cross-select-ptm-recommendation.git` (GitHub).

## Architecture overview

**Cross-Select** (`src/cross_select/models/cross_select.py`)
- Cross-attention compatibility scorer: model token = Q, dataset class
  prototypes = K/V.
- Fixed PARC model tokens (32 × 512), fixed CLIP class prototypes per dataset.
- **As of 3f9db36** uses ModelSpider-style encoders:
  - Dataset: `uni_linear: 512→1024` + `hete_linear: 512→1024` → concat 2048 →
    `dataset_out: 2048→hidden_dim`.
  - Model: `model_pre: 512→512` (GELU) → `model_proj: 512→hidden_dim`.
- N stacked `CrossAttentionBlock`s (configured `num_layers=4` in multi_run).
- `score_head` → `(B, M)` scalar score per model.

**Model Spider** (`src/cross_select/models/model_spider.py`)
- Self-attention `nn.TransformerEncoder` over `[model_tokens ++ dataset_tokens]`.
- Has **learnable** `nn.Parameter(num_models, dataset_token_dim)` model
  embeddings — this is the memorization shortcut.
- Same dual-head dataset encoder Cross-Select now mirrors.
- `key_padding_mask` is extended with `False` over the model positions before
  passing to `src_key_padding_mask`.

**SampleBatchGenerator** (`src/cross_select/data/dataset.py`)
- Sample-level sampler. Per step: K datasets × S samples × M (or full-zoo) models.
- Driver = largest-chunk-count dataset (cifar_10, 1252 chunks). Smaller datasets
  cycle (upsample-small policy) so every step contains the driver and all
  partners are uniformly used across the epoch.
- Pads dataset tokens to `C_max = max(C)` across the K shards in the step;
  emits `key_padding_mask: (K, C_max) bool` (True = ignore). This fixes the
  C_m truncation bug where cifar_10 (C=10) was cropping every other dataset.
- Deterministic mode (`trainer.deterministic_sampling: true`) makes every epoch
  use the identical (chunk, model_idx) sequence given a seed.

**Training**
- Single-experiment: `src/cross_select/training/trainer.py` — used by
  `cross_select.cli.train`.
- Multi-experiment: `src/cross_select/training/multi_trainer.py` — used by
  `cross_select.cli.multi_train`. Runs N experiments lockstep on one shared
  `SampleBatchGenerator`. **One W&B run** with all metrics prefixed by
  experiment name (`cs-listmle/val/weighted_kendall_tau`, etc.). Earlier
  attempt with `wandb.init(reinit=True)` per experiment crashed with
  "Run is finished"; fixed in `fdb3481`.

**Losses** (`src/cross_select/losses/ranking.py`)
- `listmle_loss(pred, target, pred_temperature=1.0)` — Plackett-Luce NLL via
  `torch.logcumsumexp`. Validated: perfect order → 14, reverse → 510, random
  mean ≈ 430.
- `ListMLEMSELoss(ranking_weight, mse_weight, pred_temperature)` — hybrid.
- `pairwise_bce_loss(pred, target, margin)` — RankNet-style.
- `build_loss(cfg)` factory keyed on `cfg.kind`.

**Eval**
- Stochastic mode: `n_seeds` × single-shard draw, averaged. Matches train
  distribution (`E[f(x)]` vs prototype `f(E[x])`).
- Per-dataset metrics: `val/<dataset>/{weighted_kendall_tau,ndcg,pearson,relAcc@k}`.
- Quadrant split: `cfg.quadrant.{held_out_models,held_out_datasets}`.

## Configs

- `configs/trainer/default.yaml`: `num_models_per_step=32`,
  `num_datasets_per_step=4`, `num_samples_per_dataset=4`, `grad_clip=1.0`,
  `deterministic_sampling=false`, `loss.kind=listmle`,
  `eval.{mode=stochastic, n_seeds=16}`.
- `configs/experiment/multi_run.yaml`: 5 experiments —
  `cs-listmle`, `cs-listmle-t0.5`, `cs-listmle-mse`, `cs-pairwise-bce`,
  `spider-listmle` (all `num_layers: 4`).
- `configs/model/{cross_select,model_spider}.yaml` — model defaults.

## Data

- 32 PARC model embedding `.npz` files at
  `artifacts/extracted/parc_model_embeddings/parc_model_embeddings/` (nested
  on gautschi; config default points at the outer dir so an override is
  needed there).
- CLIP shards: `artifacts/extracted/datasets/<slug>/{train,validation,test}/*.npz`.
- Ground truth rankings: `constants/ground_truth_rankings.json`.
- Local has only `caltech_101` extracted; gautschi has full set.

## Recent investigations

**Run `pp7s8zsn`** (`feature/loss-improvements`, multi_run): ModelSpider hit
val τ ≈ 0.998, Cross-Select stuck at ~0.70. Root cause: Spider's 32 learnable
model embeddings memorize Q1 targets (essentially a lookup table). This
won't generalize to Q2 (unknown models) or Q4 (both unknown). Encoder
capacity gap was a secondary suspect — addressed in 3f9db36 by giving
Cross-Select the same dual-head dataset encoder + extra model linear layer
that Spider has, so the comparison isolates the embeddings-vs-attention
question rather than encoder capacity.

**ListMLE validation**: closed-form sanity for `pred=target=[1..32]` matches
`torch.logcumsumexp` to last decimal. Implementation correct; loss is
scale-invariant in target, scale-variant in pred (margin reduces loss
monotonically).

**C_m truncation bug**: pre-fix every step truncated to `min(C)=10` because
cifar_10 was always in the batch. Padded to `C_max` + `key_padding_mask` in
`6fbfdef`. This was a real capacity ceiling fix.

## Pending work

1. Run multi_run on gautschi with the new Cross-Select encoders; check whether
   τ moves above 0.70.
2. Implement `freeze_model_embeddings` flag for ModelSpider so a fair Q1
   comparison without the memorization shortcut is possible.
3. Run Q2/Q3/Q4 evaluations — Cross-Select should generalize, Spider should
   collapse on Q2/Q4. That comparison is the thesis contribution.
4. Eventually merge `feature/loss-improvements` → `main`.

## Test suite

`pytest -q` from repo root. As of this snapshot:
- `tests/test_cross_attention.py`: 14 passing (model forward shape + masks +
  build_model + Spider learnable embedding tests).
- `tests/test_sample_batcher.py`: 8 passing (driver coverage, padding,
  deterministic, model pool).
- `tests/test_multi_trainer.py`: 2 passing (lockstep run + multi_run config).
- Other tests under `tests/` round out to ~61 total passing.

## Useful commands

```bash
# Local pytest
/Users/parth/.virtualenv/thesis/bin/python -m pytest -q

# Local CPU smoke training (single-dataset caltech_101)
python -m cross_select.cli.train experiment=local_smoke

# Multi-experiment training (gautschi GPU)
python -m cross_select.cli.multi_train experiment=multi_run

# Push to gautschi (auto-checks out branch via post-receive hook)
git push gautschi feature/loss-improvements
```

## Key files quick reference

- [src/cross_select/models/cross_select.py](src/cross_select/models/cross_select.py)
- [src/cross_select/models/model_spider.py](src/cross_select/models/model_spider.py)
- [src/cross_select/models/cross_attention.py](src/cross_select/models/cross_attention.py)
- [src/cross_select/data/dataset.py](src/cross_select/data/dataset.py)
- [src/cross_select/losses/ranking.py](src/cross_select/losses/ranking.py)
- [src/cross_select/training/trainer.py](src/cross_select/training/trainer.py)
- [src/cross_select/training/multi_trainer.py](src/cross_select/training/multi_trainer.py)
- [src/cross_select/cli/multi_train.py](src/cross_select/cli/multi_train.py)
- [configs/trainer/default.yaml](configs/trainer/default.yaml)
- [configs/experiment/multi_run.yaml](configs/experiment/multi_run.yaml)
