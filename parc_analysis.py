#!/usr/bin/env python3
"""Analyze PARC token structure and why _imagenet variants are special."""

import sys, json, random
from pathlib import Path
import numpy as np
import torch

np.set_printoptions(precision=4, suppress=True, linewidth=140)
sys.path.insert(0, "src")
from cross_select.data.tokens import load_model_tokens, load_ground_truth

SEP = "=" * 80

# ─── Load data ───
model_tokens_dict = load_model_tokens("artifacts/extracted/parc_model_embeddings")
model_ids = sorted(model_tokens_dict.keys())
M = np.stack([model_tokens_dict[m] for m in model_ids], axis=0).astype(np.float32)  # (32, 512)

gt = load_ground_truth("constants/ground_truth_rankings.json")

# ─── 1. Full cosine similarity matrix ───
print(SEP)
print("1. FULL PARC TOKEN COSINE SIMILARITY MATRIX")
print(SEP)
norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
M_norm = M / norms
cos_sim = M_norm @ M_norm.T  # (32, 32)

# Group by architecture
archs = {}
for i, mid in enumerate(model_ids):
    arch = mid.rsplit("_", 1)[0] if "_" in mid else mid
    # Parse: <arch>_<source> -> group by arch
    parts = mid.split("_")
    # Architecture is everything before the last part (source dataset)
    # e.g. resnet18_imagenet -> arch=resnet18, source=imagenet
    for a in ["resnet18", "resnet50", "alexnet", "googlenet"]:
        if mid.startswith(a + "_"):
            archs.setdefault(a, []).append(i)
            break

# Print condensed cosine matrix grouped by architecture
print("\nWithin-architecture cosine similarities:")
for arch, indices in sorted(archs.items()):
    names = [model_ids[i].replace(arch + "_", "") for i in indices]
    sub = cos_sim[np.ix_(indices, indices)]
    mask = ~np.eye(len(indices), dtype=bool)
    off_diag = sub[mask]
    print(f"\n  {arch} ({len(indices)} models):")
    print(f"    Mean={off_diag.mean():.4f}, Min={off_diag.min():.4f}, Max={off_diag.max():.4f}, Std={off_diag.std():.4f}")

print("\nCross-architecture cosine similarities:")
arch_names = sorted(archs.keys())
for i, a1 in enumerate(arch_names):
    for a2 in arch_names[i+1:]:
        cross = cos_sim[np.ix_(archs[a1], archs[a2])]
        print(f"  {a1} vs {a2}: mean={cross.mean():.4f}, min={cross.min():.4f}, max={cross.max():.4f}")

# ─── 2. What makes _imagenet special? ───
print(f"\n{SEP}")
print("2. WHY ARE _imagenet VARIANTS SPECIAL?")
print(SEP)

# For each architecture, compare imagenet vs non-imagenet tokens
for arch in ["resnet18", "resnet50", "alexnet", "googlenet"]:
    indices = archs[arch]
    names = [model_ids[i] for i in indices]

    img_idx = [i for i, n in zip(indices, names) if n.endswith("_imagenet")]
    non_img_idx = [i for i, n in zip(indices, names) if not n.endswith("_imagenet")]

    if not img_idx:
        continue

    img_vec = M[img_idx[0]]
    non_img_vecs = M[non_img_idx]

    # L2 distance from imagenet to each non-imagenet
    dists = np.linalg.norm(non_img_vecs - img_vec, axis=1)
    # Cosine sim
    cos_to_img = cos_sim[img_idx[0], non_img_idx]
    # Cosine sim among non-imagenet
    non_img_cos = cos_sim[np.ix_(non_img_idx, non_img_idx)]
    non_mask = ~np.eye(len(non_img_idx), dtype=bool)

    print(f"\n  {arch}:")
    print(f"    imagenet norm: {np.linalg.norm(img_vec):.4f}")
    print(f"    non-imagenet norms: {np.linalg.norm(non_img_vecs, axis=1)}")
    print(f"    Cosine(imagenet, others): {cos_to_img}")
    print(f"    Non-imagenet mutual cosine: mean={non_img_cos[non_mask].mean():.4f}, min={non_img_cos[non_mask].min():.4f}")
    print(f"    L2 dist(imagenet, others): {dists}")

    # Key question: is imagenet embedding fundamentally different?
    # PCA to see structure
    all_vecs = M[indices]
    mean_vec = all_vecs.mean(axis=0)
    centered = all_vecs - mean_vec
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Project onto first 2 PCs
    proj = centered @ Vt[:2].T
    print(f"    PCA variance explained: PC1={S[0]**2/sum(S**2)*100:.1f}%, PC2={S[1]**2/sum(S**2)*100:.1f}%")
    print(f"    PC projections:")
    for j, idx in enumerate(indices):
        src = model_ids[idx].replace(arch + "_", "")
        marker = " *** IMAGENET" if idx in img_idx else ""
        print(f"      {src:<20} PC1={proj[j,0]:>8.4f}  PC2={proj[j,1]:>8.4f}{marker}")

# ─── 3. Raw embedding analysis ───
print(f"\n{SEP}")
print("3. EMBEDDING STRUCTURE ANALYSIS")
print(SEP)

# Are PARC tokens just architecture embeddings + tiny source noise?
for arch in ["resnet18", "resnet50"]:
    indices = archs[arch]
    vecs = M[indices]
    arch_mean = vecs.mean(axis=0)
    residuals = vecs - arch_mean

    print(f"\n  {arch}:")
    print(f"    Mean vector norm: {np.linalg.norm(arch_mean):.4f}")
    print(f"    Residual norms (signal that distinguishes source datasets):")
    for j, idx in enumerate(indices):
        src = model_ids[idx].replace(arch + "_", "")
        r_norm = np.linalg.norm(residuals[j])
        ratio = r_norm / np.linalg.norm(arch_mean) * 100
        print(f"      {src:<20} residual_norm={r_norm:.4f} ({ratio:.1f}% of mean)")

    # SNR: how much of the variance is between-source vs within-architecture?
    total_var = np.var(vecs, axis=0).sum()
    residual_var = np.var(residuals, axis=0).sum()
    print(f"    Total variance: {total_var:.6f}")
    print(f"    Residual (source) variance: {residual_var:.6f} ({residual_var/total_var*100:.1f}%)")

# ─── 4. Improvement strategies ───
print(f"\n{SEP}")
print("4. IMPROVEMENT STRATEGIES FOR CROSS-SELECT")
print(SEP)

print("""
  PROBLEM: PARC tokens have >96% cosine similarity within architecture.
  The source-dataset signal is <5% of the total embedding variance.
  Cross-Select's cross-attention can't amplify a signal that barely exists.

  STRATEGY A: Learnable residuals (hybrid approach)
    - Keep PARC tokens as initialization
    - Add nn.Parameter residual per model: token = PARC[i] + residual[i]
    - Residuals start at zero, learn to separate same-arch models
    - Preserves PARC's architecture-level structure while adding source signal
    - Implementation: ~10 lines of code change in CrossSelect.__init__

  STRATEGY B: Concatenate source-dataset features
    - PARC tokens encode the model architecture but not the source data
    - Append a source-dataset descriptor (e.g., dataset statistics, class count)
    - model_token = concat(PARC_embedding, source_dataset_embedding)
    - Requires extracting source dataset features (already have them!)

  STRATEGY C: Use the combined embeddings directory
    - artifacts/extracted/model_combined_embeddings/ has 800+ models
    - These may encode more source-dataset information
    - But: GT only covers 32 PARC models, so need GT expansion too

  STRATEGY D: Normalize + amplify residuals
    - Subtract architecture mean from each PARC token
    - L2-normalize the residual
    - Concatenate: token = [arch_mean_proj, residual_proj]
    - Forces the model to attend to both architecture AND source signals

  RECOMMENDED: Strategy A (learnable residuals) — minimal code change,
  preserves what works, lets gradient descent find the missing signal.
""")

# ─── 5. Quantify the imagenet effect ───
print(f"{SEP}")
print("5. THE IMAGENET EFFECT — WHY CS DISTINGUISHES IT")
print(SEP)

# Check if imagenet models have different PARC extraction (pretrained on ImageNet)
print("\n  Hypothesis: _imagenet models were pretrained on ImageNet (the PARC")
print("  extraction source), so their embeddings reflect the extraction")
print("  distribution better. Other source datasets create a domain shift")
print("  that PARC can't capture — all non-imagenet models look the same")
print("  because PARC extracts features using ImageNet-pretrained probes.\n")

# Verify: are _imagenet norms consistently different?
for arch in ["resnet18", "resnet50", "alexnet", "googlenet"]:
    indices = archs[arch]
    for idx in indices:
        src = model_ids[idx].replace(arch + "_", "")
        n = np.linalg.norm(M[idx])
        is_img = " <-- IMAGENET" if src == "imagenet" else ""
        print(f"  {model_ids[idx]:<35} norm={n:.4f}{is_img}")

print(f"\n{SEP}")
print("ANALYSIS COMPLETE")
print(SEP)
