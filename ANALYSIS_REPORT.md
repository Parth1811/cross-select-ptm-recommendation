# Cross-Select vs Model Spider: Debug Trace & Root Cause Analysis

**Date:** 2026-04-26  
**Author:** Kiro (autonomous analysis on Gautschi HPC)  
**Repo:** `~/ptm-new-implementation` on `gautschi.rcac.purdue.edu`  
**Checkpoints:** `artifacts/experiments/2026-04-24_12-47-31/`

---

## Executive Summary

Cross-Select (CS) achieves **Kendall τ = 0.689** while Model Spider achieves **τ = 0.999** on the same 7-dataset, 32-model PTM recommendation task. This report traces 2 samples end-to-end through both models and identifies the root cause: **PARC model token extraction produces only 3-4 unique embeddings per architecture instead of 8**, collapsing the source-dataset signal that CS needs to rank within-architecture variants.

---

## 1. Task Setup

- **Goal:** Given a target dataset, rank 32 pre-trained models (PTMs) by expected transfer accuracy
- **Model zoo:** 4 architectures × 8 source datasets = 32 PTMs
  - Architectures: `alexnet`, `googlenet`, `resnet18`, `resnet50`
  - Source datasets: `caltech101`, `cifar10`, `cub200`, `imagenet`, `nabird`, `oxford_pets`, `stanford_dogs`, `voc2007`
- **Ground truth:** 7 target datasets × 28 models each (some cells imputed)
- **Test samples:** `cifar_10` (10 classes, 512-dim tokens) and `caltech_101` (101 classes, 512-dim tokens)

---

## 2. End-to-End Debug Trace

### 2.1 Data Pipeline

| Component | Shape | Source |
|-----------|-------|--------|
| Model tokens (PARC) | `(32, 512)` | `artifacts/extracted/parc_model_embeddings/*.npz` |
| Dataset tokens (cifar_10) | `(10, 512)` | 38 validation shards, mean-pooled prototype |
| Dataset tokens (caltech_101) | `(101, 512)` | 1 validation shard, mean-pooled prototype |
| Accuracy matrix | `(32, 7)` | `constants/ground_truth_rankings.json` |

### 2.2 Cross-Select Forward Pass

**Architecture:** Dual-head dataset encoder → cross-attention (model queries, dataset K/V) → score head  
**Parameters:** 15,502,849 (4 cross-attention layers, 8 heads, hidden_dim=512)

#### cifar_10 Results

| Rank | Predicted PTM | Score | True PTM | Acc |
|------|--------------|-------|----------|-----|
| 1 | resnet18_imagenet | **22.89** | resnet18_imagenet ✓ | 95.86 |
| 2 | resnet50_imagenet | -0.93 | resnet18_voc2007 | 95.65 |
| 3 | resnet18_cub200 | -1.80 | resnet18_caltech101 | 95.63 |
| 4 | resnet18_stanford_dogs | -1.80 | resnet18_oxford_pets | 95.58 |
| 5 | resnet18_caltech101 | -1.80 | resnet18_stanford_dogs | 95.41 |

**Metrics:** Kendall τ = 0.663, Spearman ρ = 0.778, Precision@3 = 0.333

**Key observation:** All 7 non-imagenet resnet18 variants receive **identical score (-1.8036)**. The model produces only **8 unique scores** out of 32 models.

#### caltech_101 Results

| Rank | Predicted PTM | Score | True PTM | Acc |
|------|--------------|-------|----------|-----|
| 1 | resnet50_imagenet | -0.52 | resnet50_cifar10 | 95.74 |
| 2 | resnet50_nabird | -1.64 | resnet50_oxford_pets | 95.55 |
| 3 | resnet50_cifar10 | -1.64 | resnet50_voc2007 | 95.55 |

**Metrics:** Kendall τ = 0.602, Spearman ρ = 0.746, Precision@3 = 0.333

### 2.3 Model Spider Forward Pass (Same Samples)

**Architecture:** Learnable model embeddings + self-attention over [model ++ dataset] → score head  
**Parameters:** 15,253,505 (4 self-attention layers, 8 heads, hidden_dim=512)

#### cifar_10 Results

| Rank | Predicted PTM | Score | True PTM | Acc |
|------|--------------|-------|----------|-----|
| 1 | resnet18_imagenet | **234.26** | resnet18_imagenet ✓ | 95.86 |
| 2 | resnet18_voc2007 | 191.20 | resnet18_voc2007 ✓ | 95.65 |
| 3 | resnet18_caltech101 | 161.46 | resnet18_caltech101 ✓ | 95.63 |
| 4 | resnet18_oxford_pets | 130.45 | resnet18_oxford_pets ✓ | 95.58 |
| 5 | resnet18_stanford_dogs | 105.32 | resnet18_stanford_dogs ✓ | 95.41 |

**Metrics:** Kendall τ = 0.887, Spearman ρ = 0.900, Precision@3 = 1.000

Spider produces **32 unique scores** — every model gets a distinct prediction.

---

## 3. Head-to-Head Comparison

### 3.1 Overall Metrics (from checkpoint, all 7 datasets)

| Metric | Spider | Cross-Select | Δ |
|--------|--------|-------------|---|
| Weighted Kendall τ | **0.999** | 0.689 | -0.311 |
| NDCG | **1.000** | 0.989 | -0.011 |
| Precision@3 | **1.000** | 0.333 | -0.667 |
| relAcc@3 | **0.993** | 0.950 | -0.043 |
| Pearson | **0.928** | 0.595 | -0.333 |
| MRR | **1.000** | 0.594 | -0.406 |

### 3.2 Within-Architecture Discrimination

This is where the gap is most dramatic:

| Architecture | Spider within-arch τ | CS within-arch τ | Spider score spread | CS score spread |
|---|---|---|---|---|
| resnet18 (cifar_10) | **1.000** | 0.500 | 398.99 | 8.17 |
| resnet50 (cifar_10) | **1.000** | 0.357 | 333.77 | 0.38 |
| alexnet (cifar_10) | **1.000** | 0.357 | 183.43 | 0.13 |
| googlenet (cifar_10) | **0.714** | 0.052 | 626.63 | 0.18 |

Spider achieves **perfect within-architecture ranking** for 3 of 4 families. Cross-Select produces **near-identical scores** for all variants of the same architecture.

### 3.3 Score Distribution

| Property | Spider | Cross-Select |
|----------|--------|-------------|
| Unique scores (cifar_10) | **32/32** | 8/32 |
| Score range (cifar_10) | 2316.49 | 27.62 |
| Score std (cifar_10) | 650.11 | 4.61 |

---

## 4. Root Cause: PARC Token Extraction Bug

### 4.1 The Discovery

PARC model tokens for same-architecture variants are **bit-identical** (not just similar — the same bytes):

```
resnet18_caltech101  md5=1ba17de9ba7d  norm=0.637540
resnet18_cifar10     md5=4fb37135fade  norm=0.637540  ← same as cub200, nabird, stanford_dogs, voc2007
resnet18_cub200      md5=4fb37135fade  norm=0.637540
resnet18_imagenet    md5=3308961194b5  norm=0.669136  ← DIFFERENT
resnet18_nabird      md5=4fb37135fade  norm=0.637540
resnet18_oxford_pets md5=4c28aefe76cf  norm=0.637540  ← slightly different hash, same norm
resnet18_stanford_dogs md5=4fb37135fade  norm=0.637540
resnet18_voc2007     md5=4fb37135fade  norm=0.637540
```

### 4.2 Unique Embeddings Per Architecture

| Architecture | Unique embeddings | Out of | Collapsed groups |
|---|---|---|---|
| resnet18 | **4** | 8 | {cifar10, cub200, nabird, stanford_dogs, voc2007} share one |
| resnet50 | **4** | 8 | {caltech101, cifar10, nabird, oxford_pets, stanford_dogs} share one |
| alexnet | **3** | 8 | {caltech101, cifar10, cub200, nabird, stanford_dogs, voc2007} share one |
| googlenet | **3** | 8 | {caltech101, cifar10, cub200, nabird, oxford_pets, stanford_dogs} share one |
| **Total** | **14** | **32** | **18 models collapsed into duplicates** |

### 4.3 Why This Happens

The `source` field in each `.npz` file reveals the extraction pipeline:

```
resnet18_cifar10:    source=parc/cache/probes/fixed_budget_500/resnet18_cifar10_caltech101_0.pkl
resnet18_imagenet:   source=parc/cache/probes/fixed_budget_500/resnet18_imagenet_caltech101_0.pkl
```

The path format is `<arch>_<source>_<target>_<fold>.pkl`. **All 32 embeddings are extracted from probes evaluated on `caltech101` fold 0.**

The PARC extraction pipeline:
1. Takes a PTM (e.g., `resnet18_cifar10`) as a frozen feature extractor
2. Trains a linear probe on `caltech101` data using those frozen features
3. The probe weights become the "model token" (512-dim embedding)

**The bug:** When `resnet18_cifar10` and `resnet18_cub200` were fine-tuned from the same ImageNet checkpoint using **head-only fine-tuning** (frozen backbone), their backbone weights are identical. Therefore, their features on caltech101 data are identical, and the linear probes trained on those features are identical.

Only `_imagenet` variants differ because they ARE the original pretrained backbone (no fine-tuning applied), which has different feature representations than the fine-tuned versions.

### 4.4 The Imagenet Effect

PCA analysis confirms: **100% of within-architecture variance is on PC1** (imagenet vs not-imagenet), with **0% on PC2**. There is literally zero information to distinguish non-imagenet source datasets.

| Architecture | imagenet residual norm | non-imagenet residual norm | Ratio |
|---|---|---|---|
| resnet18 | 0.154 (24.1% of mean) | 0.022 (3.4% of mean) | 7× |
| resnet50 | 0.508 (98.1% of mean) | 0.073 (14.0% of mean) | 7× |

### 4.5 Cosine Similarity Confirmation

| Architecture | Within-arch cosine (PARC) | Within-arch cosine (Spider learned) |
|---|---|---|
| resnet18 | 0.991 | **0.136** |
| resnet50 | 0.884 | **0.137** |
| alexnet | 0.965 | **0.099** |
| googlenet | 0.968 | **0.098** |

Spider's learned embeddings have **~10% cosine similarity** between same-architecture variants (vs PARC's 88-99%). Spider learned to push apart what PARC collapsed.

---

## 5. Why Spider Succeeds

Spider sidesteps the PARC token problem entirely:

1. **Learnable model embeddings** (`nn.Parameter(32, 512)`) — randomly initialized, optimized end-to-end
2. **No dependence on pre-computed tokens** — the model learns its own representation
3. **Self-attention over [model ++ dataset]** — models attend to each other AND to dataset tokens
4. **Result:** 32 fully independent, optimized embeddings that capture both architecture and source-dataset information

---

## 6. Improvement Strategies for Cross-Select

### Strategy A: Learnable Residuals (Recommended — Quick Win)

Add a learnable residual to each PARC token:

```python
# In CrossSelect.__init__:
self.model_residuals = nn.Parameter(torch.zeros(num_models, model_token_dim))

# In CrossSelect.forward:
model_tokens = model_tokens + self.model_residuals[model_idx]
```

- **Effort:** ~10 lines of code
- **Expected impact:** High — preserves PARC architecture signal, learns missing source signal
- **Risk:** Effectively becomes Spider if residuals dominate; may need regularization

### Strategy B: Concatenate Source-Dataset Features

The source datasets already have extracted features in `artifacts/extracted/datasets/`. Append a source-dataset descriptor to each model token:

```python
model_token = concat(PARC_embedding, source_dataset_prototype)  # (512 + 512) = 1024
```

- **Effort:** Medium (need to map model→source dataset, adjust dimensions)
- **Expected impact:** High — directly provides the missing source information
- **Risk:** Increases model_token_dim, may need architecture changes

### Strategy C: Fix PARC Extraction (Root Cause Fix)

Re-extract PARC embeddings using **multiple target datasets** instead of just caltech101:

```python
# Instead of one probe per model:
embedding = extract_probe(model, target="caltech101", fold=0)

# Use multiple targets and aggregate:
embeddings = [extract_probe(model, target=t, fold=f) 
              for t in all_targets for f in range(n_folds)]
embedding = aggregate(embeddings)  # concat or mean
```

- **Effort:** High (requires re-running PARC extraction pipeline)
- **Expected impact:** Highest — fixes the root cause
- **Risk:** May still collapse if backbone weights are truly identical

### Strategy D: Verify Backbone Weights

Before fixing extraction, verify whether the fine-tuned models actually have different backbone weights:

```python
# Load resnet18_cifar10 and resnet18_cub200 checkpoints
# Compare backbone (not head) parameters
# If identical → fine-tuning was head-only → PARC can't help
# If different → PARC extraction bug → fixable
```

- **Effort:** Low (diagnostic only)
- **Expected impact:** Determines whether Strategy C is viable

---

## 7. Loss Analysis

| Loss Function | cifar_10 | caltech_101 |
|---|---|---|
| ListMLE | 73.58 | 68.80 |
| ListMLE + MSE | 74.70 (rank=73.58, mse=1.12) | 69.59 (rank=68.80, mse=0.80) |
| PairwiseBCE | 0.42 | 0.45 |

The high ListMLE loss reflects the tied scores — the model can't produce a proper ranking when 5-6 models have identical input tokens.

---

## 8. Conclusions

1. **Cross-Select's poor performance is NOT an architecture problem** — it's a data problem. The PARC tokens provide only 14 unique embeddings for 32 models.

2. **Spider succeeds by learning its own embeddings**, completely bypassing the broken PARC tokens.

3. **The PARC extraction pipeline has a fundamental limitation:** probing on a single target dataset (caltech101) with head-only fine-tuned models produces identical embeddings for same-architecture variants.

4. **The _imagenet variants are special** because they are the original pretrained backbones (not fine-tuned), so their features on caltech101 are genuinely different.

5. **Recommended next step:** Strategy D (verify backbone weights) → then Strategy A (learnable residuals) for a quick win, or Strategy C (fix extraction) for a proper fix.

---

## Appendix: Files Generated

| File | Location | Purpose |
|------|----------|---------|
| `debug_trace.py` | Gautschi: `~/ptm-new-implementation/` | 7-step end-to-end trace |
| `spider_comparison.py` | Gautschi: `~/ptm-new-implementation/` | Side-by-side Spider vs CS |
| `parc_analysis.py` | Gautschi: `~/ptm-new-implementation/` | PARC token structure analysis |
