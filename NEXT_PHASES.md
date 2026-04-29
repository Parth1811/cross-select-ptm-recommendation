# Cross-Select PTM Recommendation: Next Phases

**Date:** 2026-04-29  
**Based on:** `ANALYSIS_REPORT.md` (2026-04-26) + full codebase review

---

## 1. Current State Summary

### What's Been Built
- **CrossSelect** (`src/cross_select/models/cross_select.py`): Cross-attention scorer with dual-head dataset encoder, 2-layer MLP model encoder, and optional learnable residuals (Strategy A already in code)
- **ModelSpider** (`src/cross_select/models/model_spider.py`): Baseline with learnable per-model embeddings + self-attention
- **CrossSelectWithEncoder** (`src/cross_select/models/cross_select_encoder.py`): End-to-end variant that takes raw 8192-dim PARC vectors through a trainable encoder (autoencoder/MLP/linear) before the scorer
- **Loss functions** (`src/cross_select/losses/ranking.py`): ListMLE, ListMLE+MSE, PairwiseBCE, legacy ListNet
- **Multi-train CLI** (`src/cross_select/cli/multi_train.py`): Runs multiple experiments with shared batches for fair comparison
- **Experiment configs**: `learnable_residuals.yaml`, `encoder_ablation.yaml`, `encoder_cotrain.yaml`, `parc_benchmark.yaml`, `multi_run.yaml`, `embedding_pipeline.yaml`

### What Works
- **Model Spider**: τ=0.999, near-perfect ranking — learnable embeddings bypass PARC degeneracy
- **Training infrastructure**: Hydra configs, W&B logging, multi-experiment runner, stochastic eval all functional
- **Data pipeline**: 32 PARC model tokens (512-dim), 7 target datasets with CLIP prototypes, ground truth matrix

### What Doesn't Work
- **Cross-Select with frozen PARC tokens**: τ=0.689 — only 14 unique embeddings out of 32 models due to PARC extraction producing identical vectors for same-architecture variants (head-only fine-tuning → identical backbones → identical probes)
- **Within-architecture discrimination**: CS produces 8 unique scores for 32 models; Spider produces 32. Score spread ratio is ~100:1 in Spider's favor
- **Root cause confirmed**: PARC probes extracted on caltech101 fold 0 only; head-only fine-tuned models share backbone weights → identical features → identical probe weights

### What's Implemented But Not Yet Run
- **Strategy A (learnable residuals)**: Code exists in `CrossSelect` (`learnable_residuals`, `residual_reg_weight` params) + config at `configs/experiment/learnable_residuals.yaml`
- **Encoder pipeline**: `CrossSelectWithEncoder` + `model_encoder.py` (autoencoder/MLP/linear) + configs `encoder_ablation.yaml` and `encoder_cotrain.yaml`
- **Raw 8192-dim data path**: `configs/data/with_raw_vectors.yaml` points to `artifacts/extracted/parc_models/`

---

## 2. Strategy Prioritization

| Rank | Strategy | Effort | Expected Impact | Risk | Rationale |
|------|----------|--------|----------------|------|-----------|
| **1** | **A: Learnable Residuals** | Low (code done) | High | Medium — may collapse to Spider | Already implemented. Quick validation. If residuals dominate, it proves PARC tokens are useless; if they stay small, CS architecture adds value. |
| **2** | **D: Verify Backbone Weights** | Low (diagnostic) | Determines C viability | None | 30-min script on Gautschi. If backbones are identical (head-only FT), Strategy C is dead. Must run before investing in C. |
| **3** | **B: End-to-End Encoder (raw 8192-dim)** | Medium (code done) | High | Medium — 8192-dim may also be degenerate | `CrossSelectWithEncoder` + configs exist. Raw vectors may carry more signal than compressed 512-dim. Autoencoder reconstruction loss provides regularization. |
| **4** | **C: Fix PARC Extraction** | High (re-run pipeline) | Highest if backbones differ | High — blocked by D | Requires access to original model checkpoints on Gautschi, re-running probe extraction across multiple target datasets. Only viable if D shows backbone differences. |

---

## 3. Experiment Plan

### Phase 1: Learnable Residuals (Strategy A) — 1 day

Already configured. Run immediately.

```bash
# On Gautschi — submit as SLURM job
cd ~/ptm-new-implementation
python -m cross_select.cli.multi_train experiment=learnable_residuals
```

**What this runs** (from `configs/experiment/learnable_residuals.yaml`):
1. `cs-baseline` — frozen PARC, no residuals (control)
2. `cs-residuals` — learnable residuals, no regularization
3. `cs-residuals-reg` — learnable residuals + L2 reg (weight=0.01)

**Success criteria:**
- `cs-residuals` τ > 0.85 (significant improvement over 0.689 baseline)
- `cs-residuals-reg` τ between baseline and unregularized (proves regularization controls Spider-collapse)
- Within-architecture score spread increases from ~0.2 to >5.0

**Key metrics to watch in W&B:**
- `cs-residuals/val/weighted_kendall_tau` vs `cs-baseline/val/weighted_kendall_tau`
- `model_residuals` L2 norm over training — if it grows >> PARC token norms (~0.6), residuals are dominating
- Per-architecture within-group τ (if logged)

**Follow-up sweep** (if residuals help but reg is too strong/weak):
```bash
python -m cross_select.cli.multi_train experiment=learnable_residuals \
  'experiments.2.model.residual_reg_weight=0.001,0.01,0.1' --multirun
```

### Phase 2: Backbone Verification (Strategy D) — 0.5 day

Run in parallel with Phase 1. Diagnostic only.

```bash
# On Gautschi — interactive or short SLURM job
cd ~/ptm-new-implementation
python -c "
import torch, glob, hashlib
ckpts = sorted(glob.glob('parc/cache/models/*.pt'))  # adjust path
for p in ckpts:
    sd = torch.load(p, map_location='cpu')
    # Hash only backbone (non-head) params
    backbone = {k: v for k, v in sd.items() if 'fc' not in k and 'classifier' not in k and 'head' not in k}
    h = hashlib.md5(torch.cat([v.flatten() for v in backbone.values()]).numpy().tobytes()).hexdigest()[:12]
    print(f'{p.split(\"/\")[-1]:40s} backbone_md5={h}')
"
```

**Expected outcomes:**
- If same-architecture models share backbone hash → head-only FT confirmed → Strategy C is dead, focus on A/B
- If hashes differ → PARC extraction is the bug → Strategy C becomes viable and high-priority

### Phase 3: End-to-End Encoder (Strategy B) — 2 days

Depends on: Phase 1 results (to compare against). Run regardless of Phase 2 outcome.

```bash
# Experiment 3a: Encoder co-training with reconstruction loss
python -m cross_select.cli.train experiment=encoder_cotrain

# Experiment 3b: Encoder ablation (autoencoder vs MLP vs linear)
python -m cross_select.cli.train experiment=encoder_ablation \
  model.encoder_kind=autoencoder,mlp,linear --multirun

# Experiment 3c: Encoder + learnable residuals (combine A+B)
python -m cross_select.cli.train experiment=encoder_cotrain \
  model.num_layers=4 \
  model.encoder_kind=autoencoder \
  model.reconstruction_weight=0.1
```

**Prerequisite:** Verify raw 8192-dim vectors exist at `artifacts/extracted/parc_models/` on Gautschi. These are the `.npy` files from the old repo.

**Success criteria:**
- Encoder co-train τ > 0.80 (raw vectors carry more signal than compressed)
- Autoencoder > MLP > Linear (reconstruction regularization helps)
- If encoder τ ≈ residuals τ, the 8192-dim vectors are also degenerate (same root cause)

### Phase 4: Advanced Strategies (if Phase 1-3 don't reach τ > 0.95) — 3-5 days

#### 4a: Multi-Scale PARC Tokens
Extract PARC probes from multiple layers of each model (not just final layer). Requires modifying the extraction pipeline.

```python
# Concept: extract probes from layer1, layer2, layer3, layer4 of each ResNet
# Concatenate: model_token = [probe_layer1 || probe_layer2 || probe_layer3 || probe_layer4]
# model_token_dim becomes 4 * 512 = 2048
```

Only viable if Phase 2 shows backbone differences exist at intermediate layers.

#### 4b: Contrastive Pre-Training
Pre-train model embeddings with a contrastive objective before the ranking task:
- Positive pairs: same model, different target datasets
- Negative pairs: different models
- Then fine-tune the full CrossSelect pipeline

```python
# Would require a new loss in ranking.py and a two-stage training script
# Effort: ~2 days of implementation + 1 day of experiments
```

#### 4c: Source-Dataset Concatenation (Strategy B from report)
Append source-dataset CLIP prototypes to model tokens:

```python
# In data pipeline: model_token = concat(PARC_512, source_dataset_mean_prototype_512) → 1024-dim
# Requires: mapping each model to its source dataset
# Config change: model.model_token_dim=1024
```

This directly injects the missing source-dataset signal. Medium effort — need to build the model→source mapping and modify the data loader.

---

## 4. Architecture Improvements

### Already Implemented (ready to test)
1. **Learnable residuals** in `CrossSelect` — `model.learnable_residuals=true`
2. **End-to-end encoder** in `CrossSelectWithEncoder` — autoencoder/MLP/linear variants
3. **Reconstruction loss** — `model.reconstruction_weight=0.1` for autoencoder regularization

### Recommended Code Changes

#### 4.1 Residual Norm Logging (Priority: High, Effort: 5 min)
Add to training loop to monitor whether residuals dominate PARC tokens:

**File:** `src/cross_select/cli/multi_train.py` (or `train.py`)
```python
# After each epoch, log:
if hasattr(model, 'model_residuals'):
    wandb.log({
        "residual_l2_norm": model.model_residuals.data.norm().item(),
        "parc_token_mean_norm": model_tokens.norm(dim=-1).mean().item(),
        "residual_to_parc_ratio": model.model_residuals.data.norm().item() / (model_tokens.norm(dim=-1).mean().item() + 1e-8),
    })
```

#### 4.2 Per-Architecture τ Evaluation (Priority: High, Effort: 30 min)
The analysis report manually computed within-architecture τ. Automate this in the eval loop:

**File:** `src/cross_select/evaluation/` (new or existing metrics module)
```python
# Group models by architecture prefix, compute τ within each group
# Log: per_arch/resnet18/kendall_tau, per_arch/resnet50/kendall_tau, etc.
```

#### 4.3 Cosine Similarity Dashboard (Priority: Medium, Effort: 15 min)
Log pairwise cosine similarity of model representations (after projection) to W&B as a heatmap. Tracks whether the model is learning to separate same-architecture variants.

#### 4.4 Gradient-Scaled Residual Init (Priority: Low, Effort: 10 min)
Instead of zero-init for residuals, initialize proportional to within-group PARC variance:
```python
# In CrossSelect.__init__:
# self.model_residuals = nn.Parameter(torch.zeros(num_models, model_token_dim))
# Better: init with small noise scaled to PARC token std
self.model_residuals = nn.Parameter(torch.randn(num_models, model_token_dim) * 0.01)
```

---

## 5. Timeline

| Phase | Duration | Depends On | Gautschi Resources | Deliverable |
|-------|----------|------------|-------------------|-------------|
| **1: Learnable Residuals** | 1 day | Nothing | 1 GPU, ~2h training | τ comparison: baseline vs residuals vs regularized |
| **2: Backbone Verification** | 0.5 day | Nothing (parallel with 1) | CPU only, 30 min | Go/no-go for Strategy C |
| **3: End-to-End Encoder** | 2 days | Phase 1 (for comparison) | 1 GPU, ~6h (3 encoder variants) | Best encoder variant τ |
| **4a-c: Advanced** | 3-5 days | Phase 1-3 results | 1 GPU per experiment | Only if τ < 0.95 after Phase 3 |
| **Analysis & Paper Writing** | 2-3 days | All phases | None | Results table, ablation plots |

**Total estimated time:** 7-12 days depending on Phase 4 necessity.

**Critical path:** Phase 1 → Phase 3 → (Phase 4 if needed) → Paper  
**Parallel path:** Phase 2 runs alongside Phase 1

### Decision Points

1. **After Phase 1:** If `cs-residuals` τ > 0.95 → skip Phase 3-4, write up results. If τ ∈ [0.85, 0.95] → proceed to Phase 3. If τ < 0.85 → residuals alone insufficient, Phase 3 critical.

2. **After Phase 2:** If backbones identical → Strategy C dead, focus on A+B. If backbones differ → add Phase 4a (multi-scale PARC) to plan.

3. **After Phase 3:** If encoder τ > 0.95 → done. If encoder ≈ residuals → raw vectors also degenerate, need Phase 4c (source-dataset concatenation). If encoder < residuals → raw vectors worse than PARC+residuals, abandon encoder path.

---

## Appendix: Key File Reference

| Component | Path |
|-----------|------|
| CrossSelect model | `src/cross_select/models/cross_select.py` |
| ModelSpider baseline | `src/cross_select/models/model_spider.py` |
| Encoder variant | `src/cross_select/models/cross_select_encoder.py` |
| Encoder architectures | `src/cross_select/models/model_encoder.py` |
| Cross-attention block | `src/cross_select/models/cross_attention.py` |
| Loss functions | `src/cross_select/losses/ranking.py` |
| Multi-train CLI | `src/cross_select/cli/multi_train.py` |
| Single-train CLI | `src/cross_select/cli/train.py` |
| Learnable residuals config | `configs/experiment/learnable_residuals.yaml` |
| Encoder co-train config | `configs/experiment/encoder_cotrain.yaml` |
| Encoder ablation config | `configs/experiment/encoder_ablation.yaml` |
| PARC benchmark config | `configs/experiment/parc_benchmark.yaml` |
| Raw vector data config | `configs/data/with_raw_vectors.yaml` |
| Default data config | `configs/data/default.yaml` |
| Trainer defaults | `configs/trainer/default.yaml` |
| Root Hydra config | `configs/config.yaml` |
| PARC model tokens (512-dim) | `artifacts/extracted/parc_model_embeddings/` |
| Raw PARC vectors (8192-dim) | `artifacts/extracted/parc_models/` |
| Ground truth | `constants/ground_truth_rankings.json` |
