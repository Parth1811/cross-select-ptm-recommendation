# Session Context — Cross-Select Implementation

A detailed snapshot of project state, the loss/training improvement journey,
diagnoses of failing runs, design decisions, and pending work — intended to
be loaded into a future Claude session for continuity. Date: 2026-04-26.

---

## 1. What this project is

A from-scratch reimplementation of two pretrained-model-recommendation
systems for the user's thesis:

- **Cross-Select** (the user's contribution): cross-attention compatibility
  scorer where the model token is the query and per-class CLIP prototypes
  are the keys/values. Model tokens come from PARC (fixed, 32 × 512). Goal
  is *generalization to unseen models* — the model token is computed from
  the candidate's behavior, not memorized.
- **Model Spider** (baseline): self-attention `nn.TransformerEncoder` over
  `[model_tokens ++ dataset_tokens]`, with **learnable** model embeddings
  `nn.Parameter(num_models, dataset_token_dim)`. The learnable embeddings
  are essentially a 32×512 lookup table that memorizes the training zoo.

Evaluated under four quadrants:
- Q1 known models / known datasets
- Q2 unknown models / known datasets
- Q3 known models / unknown datasets
- Q4 unknown / unknown

The thesis claim: Cross-Select generalizes (Q2/Q4 hold up); Spider collapses
on Q2/Q4 because its memorization shortcut breaks.

---

## 2. Repo layout

```
src/cross_select/
  cli/{train,multi_train,dryrun}.py       # Hydra entrypoints
  data/{dataset,tokens,splits}.py         # TokenBank, SampleBatchGenerator
  models/{cross_select,cross_attention,model_spider}.py
  losses/ranking.py                       # listmle, listmle_mse, pairwise_bce, compatibility (legacy)
  training/{trainer,multi_trainer,splits,wandb_utils,schedule}.py
  eval/{metrics,gt_alignment}.py          # Kendall, NDCG, Pearson, relAcc@k, precision@k
configs/
  config.yaml                             # root composer
  trainer/default.yaml                    # shared training budget
  model/{cross_select,model_spider}.yaml
  data/parc_clip.yaml                     # paths to npz shards + GT JSON
  experiment/{local_smoke,multi_run}.yaml
constants/ground_truth_rankings.json      # 32-model × 8-dataset relative-acc matrix
artifacts/                                # symlinked → /depot/.../parth/artifacts on gautschi
tests/                                    # 61 tests, all passing
```

---

## 3. Branches & merge status

```
* feature/loss-improvements   ← active, ahead of main by 6 commits
  feature/model-spider-fidelity  ← already merged into main (tip == main)
  feature/sample-level-sampler   ← already merged into main (tip == main)
  main                        ← 35a46ac
```

**Ready to merge to main: `feature/loss-improvements`** (the only branch
ahead of main). Six commits, all on top of `35a46ac`:

| SHA | Title | Status |
|---|---|---|
| 9d95dac | Close train/eval gap; PARC metrics; optim schedule knobs | ready |
| 6fbfdef | Pad dataset tokens to C_max + attention key_padding_mask | ready |
| 51a10dc | Multi-experiment trainer, new losses, deterministic sampler, grad clip on | ready |
| fdb3481 | Fix multi_trainer: use one W&B run with prefixed keys | ready |
| 3f9db36 | Cross-Select dual-head dataset encoder + two-layer model encoder | ready |
| c1770d7 | docs: add SESSION_CONTEXT.md | ready |

All 61 tests pass on every commit. CPU smoke (caltech_101, 1 epoch) verifies
non-regressing behavior at each step. The branch is **safe to merge** as a
single fast-forward (or squash if a flatter history is preferred).

`feature/model-spider-fidelity` and `feature/sample-level-sampler` exist as
historical pointers but their tips are at the same SHA as `main` — nothing
to merge. They can be deleted once the merge happens, or kept as named
restore points.

---

## 4. The training-improvement journey

This is the loss/training story arc across recent runs, in chronological
order, including **what was diagnosed, what was tried, and what worked**.

### 4.1 Run `e9m4pyte` — "tau plateaued at 0.66 after 300 epochs"

**Symptoms.** Long run, loss looked fine, but val weighted Kendall τ flat
at 0.66 from epoch ~50 to 300. No further improvement.

**Diagnosis (4 compounding issues):**

1. **Train/eval distribution mismatch.** Training drew stochastic
   single-shard prototypes per dataset (pick one shard, pass its 102 class
   prototypes). Eval used a deterministic *mean across ALL shards* of the
   prototypes. Model was optimized for `E_x[f(x)]` but scored on
   `f(E_x[x])` — equal only if `f` is linear, which attention is not.

2. **Loss-space averaging over samples.** The trainer averaged the K×S
   per-sample predictions into `(K, M_sub)` *before* computing the loss.
   This collapses the per-sample gradient signal through the non-linear
   attention path.

3. **Stochastic model subsampling jitter.** With `num_models_per_step=16`,
   each step shuffled which 16 of the 32 models appeared in the batch and
   in what order. ListMLE's target is `argsort(target)`, so a different
   model order means a different target every step — nontrivial gradient
   noise.

4. **Per-dataset metrics dropped.** The aggregated `val/weighted_kendall_tau`
   was averaged across datasets, so we couldn't see which datasets
   struggled.

**Fixes (commit `9d95dac`):**

- **Stochastic eval** (`trainer.eval.mode=stochastic`,
  `trainer.eval.n_seeds=16`): draw N single-shard tokens, forward each,
  average the predicted *scores*, then compute metrics. Matches the train
  input distribution.
- **Full-zoo training**: bumped `num_models_per_step` 16 → 32 (default).
  When `num_models_per_step == pool_size`, sampler returns the pool in
  deterministic order so the ListMLE target is identical step-to-step.
- **Loss per sample**: each of the K×S predictions is compared against the
  dataset's GT row, then per-sample scalar losses are averaged. Gradient
  signal preserved.
- **Flatten per-dataset metrics** into `val/<dataset>/<metric>` W&B keys.
- **Train-time probes**: every `log_every` steps log
  `train/probe/{kendall, ndcg, precision@k, relAcc@k, pearson, mrr}` on
  the in-batch aggregated prediction. Surfaces overfitting vs underfitting
  through per-step loss noise.
- **PARC metrics added** (Bolya et al. 2021, arXiv:2111.06977):
  `relative_accuracy_at_k = mean(target[top_k(pred)]) / max(target)` and
  `pearson_correlation`. Both joined existing Kendall, NDCG, precision@k.
- **Configurable LR schedule**: `trainer.schedule.kind ∈ {constant, cosine}`
  with `warmup_steps` and `min_lr_mult`.
- **Configurable grad clip**: `trainer.grad_clip` (default initially 0.0,
  later flipped to 1.0 in `51a10dc`).

### 4.2 Run `mpxu3i7h` — "tau plateaued at 0.69 after epoch 4"

**Symptoms.** Even after the 4.1 fixes, val τ plateaued faster but lower:
~0.69 from epoch 4 onward, no further movement.

**Diagnosis: the C_m truncation bug.** `SampleBatchGenerator._assemble_step`
cropped every dataset's K/V tensor to `C_m = min(class_count_i)` across the
K picked datasets. Because cifar_10 was always the driver (1252 chunks
≫ 36 for any other dataset) and has C=10, **every training step truncated
every other dataset to its first 10 class prototypes**:

| Dataset | Native C | Cropped to | % info kept |
|---|---|---|---|
| caltech_101 | 102 | 10 | 9.8% |
| nabird | 555 | 10 | 1.8% |
| cub200 | 200 | 10 | 5.0% |
| stanford_dogs | 120 | 10 | 8.3% |
| oxford_pets | 37 | 10 | 27% |
| voc2007 | 20 | 10 | 50% |

The model never saw any non-cifar_10 dataset with full class representation
during training, but was scored on the full class count at eval. This was
the capacity ceiling.

**Fix (commit `6fbfdef`, "Option B"):** pad to `C_max = max(C_i)` with
zeros, emit a boolean `key_padding_mask: (K, C_max)` (`True` = ignore;
matches `nn.MultiheadAttention` convention), and thread the mask through:

- `CrossAttentionBlock.forward` → `nn.MultiheadAttention.key_padding_mask`
- `CrossSelect.forward` (optional kwarg, forwarded to each block)
- `ModelSpider.forward` (optional kwarg; **prepends M False columns** to
  cover the model positions in the concatenated `[model++dataset]`
  sequence, then passes as `src_key_padding_mask` to the encoder)
- `Trainer._forward_step` broadcasts `(K, C_max)` → `(K*S, C_max)` alongside
  the dataset_tokens flatten.

Mask is optional everywhere; eval (batch_size=1, single dataset) doesn't
need it. Renamed batch key `C_m` → `C_max` for clarity. Verified with
**padding-equivalence tests**: pad+mask produces identical output to
unpadded input of equal real length — the key invariant guarding against
silent mask drop.

CPU smoke after this fix:
- Cross-Select on caltech_101: τ 0.71 after 1 epoch (up from 0.65 pre-fix).
- Model Spider on caltech_101: τ 0.99 after 1 epoch (already high; the
  learnable-embedding shortcut dominates here).

### 4.3 Run `calitdk7` — "tau still flat at 0.69; loss noisy; GPU at 4%"

**Symptoms.** Padding fix landed, but multi-layer (`num_layers=4`)
Cross-Select on full multi-dataset training: val τ stuck at ~0.69 again,
step-loss bouncing 60–80, GPU only 1000MB / 40000MB used.

**User's three suggestions** — all implemented in commit `51a10dc`:

1. **Deterministic epoch sampler** (opt-in via
   `trainer.deterministic_sampling: true`). RNG re-seeded from
   `base_seed + epoch_index` at the start of every epoch so every epoch
   visits the identical (chunk, row, partner-pick) sequence across runs.
   Default stays stochastic to preserve old behavior. Two new tests:
   `test_deterministic_sampling_produces_identical_epochs` and
   `test_non_deterministic_default_still_varies_across_epochs`.

2. **Multiple models at once** — the multi-experiment trainer
   (`cli/multi_train.py` + `training/multi_trainer.py`). Single process,
   N models train **lockstep on a shared `SampleBatchGenerator`** so every
   experiment sees the *same batch* at the *same step*. Each experiment
   has its own model/optimizer/loss/scheduler. Differences between curves
   reflect model/loss choices, not data noise. The 40GB GPU is actually
   utilized while keeping the comparison apples-to-apples. Invocation:
   ```
   python -m cross_select.cli.multi_train experiment=multi_run
   ```
   Ships with a 5-experiment config covering the new losses (see 4.4).

3. **Better loss** — three new loss kinds on top of the existing one:
   - `listmle` with `pred_temperature` knob: divides pred by T before
     `logcumsumexp`. T<1 sharpens (stronger top-position gradient),
     T>1 softens.
   - `listmle_mse` hybrid: ListMLE + standardized per-row MSE, weighted.
     Replaces the legacy `compatibility` (ListNet + 0.1·MSE) with the
     stable ListMLE ranking term. `mse_weight` defaults to 1.0.
   - `pairwise_bce`: RankNet-style pairwise BCE over all i<j model
     pairs. Smoother than ListMLE; no suffix-sum explosion on bad
     top-of-list orderings.

   All surface through the existing `build_loss(cfg)` factory; legacy
   `compatibility` is still selectable.

4. **Grad clip default flipped 0.0 → 1.0.** Multi-layer attention training
   at lr=1e-4 benefits from the cheap stability guarantee; per-run
   override available.

### 4.4 Run `qpoaq4vx` — "Run is finished" W&B crash

**Symptoms.** Multi-experiment trainer crashed mid-training with
`wandb.errors.errors.UsageError: Run (qpoaq4vx) is finished. The call to
log will be ignored.`

**Diagnosis.** The first multi_trainer used one `wandb.init(reinit=True)`
*per experiment*. W&B can only have one live run per process: opening run
2 silently calls `finish()` on run 1. The next `.log()` on run 1 raises.

**Fix (commit `fdb3481`).** One shared `wandb.init()` owned by
`MultiTrainer`. Each `_Experiment.train_step()` returns
`(loss_float, payload_dict | None)` — no W&B call. Each
`_Experiment.build_eval_payload()` returns
`{f"{self.name}/val/{k}": v}` dict. `MultiTrainer._log(payload)` is the
single W&B log point. Per-experiment panels in the dashboard are recovered
by filtering on the `{exp_name}/...` prefix.

Verified: 61/61 tests pass. CPU smoke (5 experiments, 1 epoch, caltech_101)
matches previous-commit values exactly across all five per-experiment
metrics — confirming pure-logging refactor with no behavioral change.

### 4.5 Run `pp7s8zsn` — "ModelSpider τ=0.998, Cross-Select τ=0.70"

**Symptoms.** First successful multi-experiment run after `fdb3481`. After
a few epochs, ModelSpider hits val τ ≈ 0.998 (essentially perfect) while
all four Cross-Select variants are stuck at ~0.70.

**Diagnosis.** Spider has 32 learnable model embeddings of size 512 each
= 16384 params *per dataset row* if you think of it as a lookup table.
For Q1 (known models / known datasets), Spider is essentially memorizing
the GT matrix. This **won't generalize to Q2** (unknown models — no
trained embedding for them) or Q4. So Spider's apparent win is a Q1
shortcut, not a real result.

**Encoder capacity gap.** A secondary suspect: Spider also has a richer
dataset encoder (paper-style dual-head FF: `uni_linear: 512→1024` +
`hete_linear: 512→1024` → cat 2048 → `dataset_out: 2048→hidden_dim`),
while Cross-Select had a single `Linear(512, hidden_dim)`. To isolate the
embeddings-vs-attention question from the encoder-capacity question, we
gave Cross-Select the *same* dual-head dataset encoder + an extra model
linear layer (commit `3f9db36`):

```python
# Dataset: dual-head, same as ModelSpider
self.uni_linear = nn.Linear(dataset_token_dim, 1024)
self.hete_linear = nn.Linear(dataset_token_dim, 1024)
self.dataset_out = nn.Linear(2048, hidden_dim)

# Model: two-layer MLP for richer encoding
self.model_pre = nn.Linear(model_token_dim, model_token_dim)
self.model_proj = nn.Linear(model_token_dim, hidden_dim)

# In forward
d_uni = self.uni_linear(dataset_token)
d_hete = self.hete_linear(dataset_token)
kv = self.dataset_out(torch.cat([d_uni, d_hete], dim=-1))
q = self.model_proj(F.gelu(self.model_pre(model_tokens)))
```

This is the most recent change. Run pending on gautschi.

### 4.6 ListMLE validation experiment

To be sure the loss itself wasn't broken before chasing more architecture
changes, I ran a deterministic + random-permutation probe on the
`listmle_loss` implementation. Target = `torch.arange(1, 33)` of shape
`(1, 32)`.

**Deterministic probes:**

| pred | listmle | τ_w |
|---|---|---|
| target (perfect order) | **13.9933** | +1.0 |
| reverse (worst) | **509.9933** | −1.0 |
| all zeros / all ones | 81.5580 | NaN |
| target × 1000 (huge margin) | **0.0000** | +1.0 |
| target × 100 (target rescaled) | **13.9933** | +1.0 |
| target + 42 (target shifted) | **13.9933** | +1.0 |

**50 random permutations** (seed 0): listmle mean=430, std=41,
min=326, max=501. τ_w mean=−0.018, in [−0.315, +0.384].

**Closed-form sanity** for `pred=target=[1..32]`:
`Σ_{m=1..32} [log(e^1 + … + e^{33−m}) − (33−m)] = 13.9933`,
matches the `torch.logcumsumexp` implementation to the last decimal.

**Conclusions:**
- ListMLE is **scale-invariant in target** (uses only `argsort(target)`):
  loss identical for `target`, `target × 100`, `target + 42`.
- ListMLE is **scale-variant in pred** (correct: increasing prediction
  margin monotonically decreases loss; perfect order at 1000× scale → 0).
- **Symmetric bounds**: perfect → 14, perfectly inverted → 510.
- **Random mean ≈ 430** sits between the two (closer to reverse than
  perfect because Plackett-Luce accumulates 32 NLL terms and most random
  permutations have heavy inversions near the top of the list).
- Train-loss values bouncing 60–80 on real runs imply the model has
  τ ≈ 0.6–0.7 on average — the loss has a **high floor** even for a
  decent model with M_sub=32 per step.

---

## 5. Architecture details (current state on `feature/loss-improvements`)

### Cross-Select (`src/cross_select/models/cross_select.py`)

```
model_tokens (B, M, 512)            dataset_token (B, C, 512)
       │                                    │
       ▼                                    ▼
   model_pre: 512→512               uni_linear: 512→1024 ──┐
   GELU                             hete_linear: 512→1024 ─┼──► concat (B, C, 2048)
   model_proj: 512→H                                       │
       │                            dataset_out: 2048→H ◄──┘
       ▼                                    │
       Q                                    K, V
       └──────► CrossAttentionBlock × num_layers ◄────┐
                (with key_padding_mask)               │
                          │                           │
                          ▼                           │
                  score_head (LN→Lin→GELU→Drop→Lin→1) │
                          │                           │
                          ▼                           │
                     scores (B, M)                    │
```

- `CrossAttentionBlock`: PyTorch `nn.MultiheadAttention` with pre-LN, FFN,
  residual connections. `key_padding_mask` is forwarded directly.
- `num_layers` is `4` in `multi_run.yaml`.

### Model Spider (`src/cross_select/models/model_spider.py`)

```
model_idx (B, M) ──► Embedding(num_models, 512) ──► +model_type
                                                            │
dataset_token (B, C, 512) ──► uni_linear/hete_linear/cat ──┐│
                              ──► dataset_out (→H)         │▼
                              ──► +dataset_type            concat (B, M+C, H)
                                                            │
                              [model_pad ++ kpm] ──►        ▼
                              src_key_padding_mask    nn.TransformerEncoder
                                                            │
                              gather model rows ◄──────────┘
                                                            │
                                                            ▼
                                                       score_head
```

- The `nn.Parameter(num_models, 512)` is the memorization shortcut for Q1.
- Same dual-head dataset encoder as Cross-Select (since `3f9db36`).

### SampleBatchGenerator (`src/cross_select/data/dataset.py`)

Sample-level sampler. Per step:
- K datasets (`num_datasets_per_step`)
- S samples per dataset (`num_samples_per_dataset`) — drawn as one shard's
  rows to match the eval distribution
- M models (`num_models_per_step`) — full zoo (32) by default; subsample
  pool optional via `model_pool_idx`

**Driver = largest-chunk-count dataset.** `chunks_per_dataset` =
`shard_count × rows_per_shard / S`. Driver appears in *every* step (so
its full coverage = single-pass per epoch). Other datasets cycle
("upsample-small policy"). `steps_per_epoch = max(chunks_per_dataset)`.

This was tuned in commit `55a4691` after a "why is this only 46 steps?"
debug — the old greedy "pick K largest remaining" exhausted small
datasets quickly and ended the epoch early.

**Padding to C_max + key_padding_mask** as described in 4.2.

**Deterministic mode** (`deterministic=true`) re-seeds RNG from
`base_seed + epoch_index` at every epoch start, producing identical epochs.

**Full-zoo deterministic order**: when `M == pool_size`, returns pool in
sorted order so ListMLE target is identical step-to-step.

### Losses (`src/cross_select/losses/ranking.py`)

| kind | impl | gradient property |
|---|---|---|
| `listmle` | Plackett-Luce NLL via `torch.logcumsumexp(pred, dim=-1)` | scale-invariant in target, scale-variant in pred; high floor for M=32 |
| `listmle_mse` | `ranking_weight·listmle + mse_weight·MSE_per_row` | adds a calibration term; MSE on row-standardized targets |
| `pairwise_bce` | `Σ_{i<j} BCE(σ(pred_i − pred_j), 1[target_i > target_j])` | smoother gradient, no suffix-sum explosion |
| `compatibility` | legacy ListNet + 0.1·MSE | kept for back-compat |

All gated through `build_loss(cfg)` keyed on `cfg.kind`.

### Trainer / multi_trainer

- `Trainer` (single experiment): tqdm progress bars, stochastic eval
  (`mode=stochastic, n_seeds=16` default), train-time probe metrics, LR
  schedule, grad clip, per-dataset W&B keys.
- `MultiTrainer`: wraps N `_Experiment`s, drives one `SampleBatchGenerator`
  shared across them, owns one `wandb.init()`, prefixes all keys with
  `{exp_name}/`. Pure-logging refactor in `fdb3481` ensures behavior is
  identical to single-experiment runs.

---

## 6. Configs — current defaults

### `configs/trainer/default.yaml`

```yaml
epochs: 100
num_models_per_step: 32          # full zoo
num_datasets_per_step: 4
num_samples_per_dataset: 4
batch_size: 1
lr: 1e-4
weight_decay: 0.0
optimizer: adamw
grad_clip: 1.0
deterministic_sampling: false
schedule:
  kind: constant                 # or cosine
  warmup_steps: 0
  min_lr_mult: 0.0
loss:
  kind: listmle                  # listmle | listmle_mse | pairwise_bce | compatibility
  pred_temperature: 1.0
  ranking_weight: 1.0
  mse_weight: 1.0
  margin: 0.0
eval:
  mode: stochastic               # or prototype
  n_seeds: 16
eval_split: validation
log_every: 50
eval_every: 1
```

### `configs/experiment/multi_run.yaml`

5 lockstep experiments (all `num_layers: 4`):
1. `cs-listmle` — Cross-Select + ListMLE default
2. `cs-listmle-t0.5` — Cross-Select + ListMLE temperature 0.5 (sharper)
3. `cs-listmle-mse` — Cross-Select + ListMLE+MSE hybrid
4. `cs-pairwise-bce` — Cross-Select + RankNet-style pairwise BCE
5. `spider-listmle` — Model Spider baseline + ListMLE

All log to one W&B run with `{exp_name}/...` prefixed keys.

---

## 7. Test suite (61/61 passing as of `c1770d7`)

Each commit landed with new tests; no skips, no xfails.

| File | Tests | Covers |
|---|---|---|
| `test_cross_attention.py` | 14 | Cross-Select / Model Spider forward shapes, padding-equivalence, mask-changes-output, `build_model` factory, learnable embedding update, future-annotations gate |
| `test_sample_batcher.py` | 8 | Driver coverage, partner cycling, full-zoo deterministic order, padding shapes/mask, deterministic vs stochastic epoch reproducibility, oversized-knob errors |
| `test_multi_trainer.py` | 2 | Two experiments lockstep on shared sampler; `multi_run.yaml` Hydra compose |
| `test_metrics.py` | 11 | Kendall, NDCG, precision@k, relAcc@k, Pearson, MRR; PARC and Model Spider metric parity checks |
| `test_end_to_end.py` | 3 | One-epoch single-run smoke (no padding path) |
| `test_data.py`, etc. | rest | TokenBank loading, GT alignment, splits, schedule |

`/Users/parth/.virtualenv/thesis/bin/python -m pytest -q` from repo root.

---

## 8. Data layout

- 32 PARC model embedding `.npz` at
  `artifacts/extracted/parc_model_embeddings/parc_model_embeddings/`
  (nested on gautschi; config default needs an override there).
- CLIP shards:
  `artifacts/extracted/datasets/<slug>/{train,validation,test}/*.npz`,
  each shard ≈ 16 rows; row = 1 sample's per-class prototype matrix.
- GT rankings: `constants/ground_truth_rankings.json` (32 models × 8
  datasets, relative-accuracy values).
- Locally only `caltech_101` is extracted; gautschi has the full set.

---

## 9. Pending / next steps

1. **Run multi_run on gautschi with the new Cross-Select encoders** —
   first test of whether the dual-head + extra-linear gives Cross-Select
   any of the headroom Spider has, *without* the embedding shortcut.
   Watch `cs-*/val/weighted_kendall_tau` vs `spider-listmle/val/...`.
2. **`freeze_model_embeddings` flag for ModelSpider.** With it set,
   Spider's 32 embeddings stay at their init values and it must learn
   purely through attention — a fair Q1 comparison.
3. **Q2/Q3/Q4 evaluation harness.** `cfg.quadrant.{held_out_models,
   held_out_datasets}` already supports it, but we haven't run a sweep.
   Cross-Select should generalize on Q2/Q4; Spider should collapse —
   that's the thesis contribution.
4. **Merge `feature/loss-improvements` → `main`.** Six commits, all
   tested, all documented. Recommend a fast-forward merge to keep the
   commit-by-commit narrative; squash would lose the diagnostic story.
5. **Optional: delete merged feature branches** (`model-spider-fidelity`,
   `sample-level-sampler`) once the merge is in.

---

## 10. Useful commands

```bash
# Local pytest
/Users/parth/.virtualenv/thesis/bin/python -m pytest -q

# Local CPU smoke training (single-dataset caltech_101)
python -m cross_select.cli.train experiment=local_smoke

# Multi-experiment training (gautschi GPU, allocate node first)
python -m cross_select.cli.multi_train experiment=multi_run

# Push to gautschi (auto-checks out branch via post-receive hook)
git push gautschi feature/loss-improvements

# Push to GitHub
git push origin feature/loss-improvements
```

---

## 11. Key files quick reference

- [src/cross_select/models/cross_select.py](src/cross_select/models/cross_select.py)
- [src/cross_select/models/model_spider.py](src/cross_select/models/model_spider.py)
- [src/cross_select/models/cross_attention.py](src/cross_select/models/cross_attention.py)
- [src/cross_select/data/dataset.py](src/cross_select/data/dataset.py)
- [src/cross_select/losses/ranking.py](src/cross_select/losses/ranking.py)
- [src/cross_select/training/trainer.py](src/cross_select/training/trainer.py)
- [src/cross_select/training/multi_trainer.py](src/cross_select/training/multi_trainer.py)
- [src/cross_select/cli/multi_train.py](src/cross_select/cli/multi_train.py)
- [src/cross_select/eval/metrics.py](src/cross_select/eval/metrics.py)
- [configs/trainer/default.yaml](configs/trainer/default.yaml)
- [configs/experiment/multi_run.yaml](configs/experiment/multi_run.yaml)
- [tests/test_multi_trainer.py](tests/test_multi_trainer.py)
- [tests/test_cross_attention.py](tests/test_cross_attention.py)
