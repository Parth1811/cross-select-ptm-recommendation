#!/usr/bin/env python3
"""Compare Spider vs Cross-Select on the same 2 samples."""

import sys, json, random
from pathlib import Path
import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
np.set_printoptions(precision=4, suppress=True, linewidth=120)

sys.path.insert(0, "src")
from cross_select.data.tokens import (
    load_model_tokens, dataset_prototype, list_shards,
    build_accuracy_matrix, load_ground_truth,
)
from cross_select.models.cross_select import CrossSelect
from cross_select.models.model_spider import ModelSpider

SEP = "=" * 80

# ─── Load shared data ───
gt = load_ground_truth("constants/ground_truth_rankings.json")
datasets = sorted(gt.keys())
test_ds = ["cifar_10", "caltech_101"]

model_tokens_dict = load_model_tokens("artifacts/extracted/parc_model_embeddings")
model_ids = sorted(model_tokens_dict.keys())
model_matrix = np.stack([model_tokens_dict[m] for m in model_ids], axis=0).astype(np.float32)
num_models = len(model_ids)

rng = random.Random(42)
acc_matrix = build_accuracy_matrix(gt, model_ids, datasets, missing_value="random_rank", rng=rng)

ds_protos = {}
for ds in test_ds:
    for split in ["validation", "test", "train"]:
        try:
            shards = list_shards("artifacts/extracted/datasets", ds, split)
            ds_protos[ds] = dataset_prototype(shards)
            break
        except FileNotFoundError:
            continue

# ─── Load both models ───
print(SEP)
print("LOADING MODELS")
print(SEP)

# Cross-Select (4 layers based on checkpoint)
cs_ckpt = torch.load(
    "artifacts/experiments/2026-04-24_12-47-31/cs-listmle/last.pt",
    map_location="cpu", weights_only=False,
)
cs_model = CrossSelect(model_token_dim=512, dataset_token_dim=512, hidden_dim=512,
                        num_heads=8, num_layers=4, dropout=0.0)
cs_model.load_state_dict(cs_ckpt["model_state"], strict=True)
cs_model.eval()
cs_params = sum(p.numel() for p in cs_model.parameters())
print(f"CrossSelect: {cs_params:,} params, 4 cross-attn layers")

# Spider
sp_ckpt = torch.load(
    "artifacts/experiments/2026-04-24_12-47-31/spider-listmle/last.pt",
    map_location="cpu", weights_only=False,
)
# Infer num_layers from checkpoint
n_encoder_layers = max(
    int(k.split(".")[2]) for k in sp_ckpt["model_state"]
    if k.startswith("encoder.layers.")
) + 1
sp_model = ModelSpider(num_models=num_models, model_token_dim=512, dataset_token_dim=512,
                        hidden_dim=512, num_heads=8, num_layers=n_encoder_layers, dropout=0.0)
sp_model.load_state_dict(sp_ckpt["model_state"], strict=True)
sp_model.eval()
sp_params = sum(p.numel() for p in sp_model.parameters())
print(f"ModelSpider: {sp_params:,} params, {n_encoder_layers} self-attn layers")
print(f"  Learnable model embeddings: {sp_model.model_embeddings.shape}")
print(f"  Embedding norms: min={sp_model.model_embeddings.data.norm(dim=1).min():.4f}, "
      f"max={sp_model.model_embeddings.data.norm(dim=1).max():.4f}")

# ─── Compare on each dataset ───
for ds in test_ds:
    print(f"\n{SEP}")
    print(f"DATASET: {ds}")
    print(SEP)

    proto = ds_protos[ds]
    di = datasets.index(ds)
    true_accs = acc_matrix[:, di]
    true_rank = np.argsort(-true_accs)

    dataset_tok = torch.from_numpy(proto).unsqueeze(0)  # (1, C, 512)
    model_tok = torch.from_numpy(model_matrix).unsqueeze(0)  # (1, M, 512)
    model_idx = torch.arange(num_models).unsqueeze(0)  # (1, M)

    with torch.no_grad():
        cs_scores = cs_model(model_tok, dataset_tok).squeeze(0).numpy()
        sp_scores = sp_model(model_idx=model_idx, dataset_token=dataset_tok).squeeze(0).numpy()

    cs_rank = np.argsort(-cs_scores)
    sp_rank = np.argsort(-sp_scores)

    # Side-by-side ranking
    print(f"\n  {'Rank':<5} {'TRUE PTM':<35} {'Acc':>6}  |  {'Spider PTM':<35} {'Score':>8}  |  {'CrossSel PTM':<35} {'Score':>8}")
    print(f"  {'-'*5} {'-'*35} {'-'*6}  |  {'-'*35} {'-'*8}  |  {'-'*35} {'-'*8}")
    for i in range(min(15, num_models)):
        ti = true_rank[i]
        si = sp_rank[i]
        ci = cs_rank[i]
        sp_match = " <<" if si == ti else ""
        cs_match = " <<" if ci == ti else ""
        print(f"  {i+1:<5} {model_ids[ti]:<35} {true_accs[ti]:>6.2f}  |  "
              f"{model_ids[si]:<35} {sp_scores[si]:>8.4f}{sp_match}  |  "
              f"{model_ids[ci]:<35} {cs_scores[ci]:>8.4f}{cs_match}")

    # Metrics
    cs_tau, _ = kendalltau(cs_scores, true_accs)
    sp_tau, _ = kendalltau(sp_scores, true_accs)
    cs_rho, _ = spearmanr(cs_scores, true_accs)
    sp_rho, _ = spearmanr(sp_scores, true_accs)

    cs_top3 = len(set(cs_rank[:3].tolist()) & set(true_rank[:3].tolist())) / 3
    sp_top3 = len(set(sp_rank[:3].tolist()) & set(true_rank[:3].tolist())) / 3

    # Score distribution analysis
    print(f"\n  SCORE DISTRIBUTIONS:")
    print(f"    Spider:      min={sp_scores.min():.4f}, max={sp_scores.max():.4f}, "
          f"std={sp_scores.std():.4f}, range={sp_scores.max()-sp_scores.min():.4f}")
    print(f"    CrossSelect: min={cs_scores.min():.4f}, max={cs_scores.max():.4f}, "
          f"std={cs_scores.std():.4f}, range={cs_scores.max()-cs_scores.min():.4f}")

    # Unique scores (tied predictions?)
    cs_unique = len(np.unique(np.round(cs_scores, 4)))
    sp_unique = len(np.unique(np.round(sp_scores, 4)))
    print(f"    Spider unique scores:      {sp_unique}/{num_models}")
    print(f"    CrossSelect unique scores: {cs_unique}/{num_models}")

    print(f"\n  METRICS COMPARISON:")
    print(f"    {'Metric':<20} {'Spider':>10} {'CrossSelect':>12} {'Delta':>10}")
    print(f"    {'-'*20} {'-'*10} {'-'*12} {'-'*10}")
    print(f"    {'Kendall tau':<20} {sp_tau:>10.4f} {cs_tau:>12.4f} {cs_tau-sp_tau:>+10.4f}")
    print(f"    {'Spearman rho':<20} {sp_rho:>10.4f} {cs_rho:>12.4f} {cs_rho-sp_rho:>+10.4f}")
    print(f"    {'Precision@3':<20} {sp_top3:>10.4f} {cs_top3:>12.4f} {cs_top3-sp_top3:>+10.4f}")

    # Within-architecture analysis
    print(f"\n  WITHIN-ARCHITECTURE DISCRIMINATION:")
    for arch in ["resnet18", "resnet50", "alexnet", "googlenet"]:
        arch_idx = [i for i, m in enumerate(model_ids) if m.startswith(arch + "_")]
        if not arch_idx:
            continue
        cs_arch = cs_scores[arch_idx]
        sp_arch = sp_scores[arch_idx]
        true_arch = true_accs[arch_idx]
        arch_names = [model_ids[i].replace(arch + "_", "") for i in arch_idx]

        cs_arch_tau, _ = kendalltau(cs_arch, true_arch)
        sp_arch_tau, _ = kendalltau(sp_arch, true_arch)

        print(f"\n    {arch} ({len(arch_idx)} variants):")
        print(f"      Score spread — Spider: {sp_arch.std():.4f}  CS: {cs_arch.std():.4f}")
        print(f"      Within-arch tau — Spider: {sp_arch_tau:.4f}  CS: {cs_arch_tau:.4f}")
        # Show per-variant scores
        order = np.argsort(-true_arch)
        for j in order:
            print(f"        {arch_names[j]:<25} true={true_arch[j]:.2f}  "
                  f"spider={sp_arch[j]:.4f}  cs={cs_arch[j]:.4f}")

# ─── Spider learned embeddings analysis ───
print(f"\n{SEP}")
print("SPIDER LEARNED EMBEDDINGS vs PARC TOKENS")
print(SEP)
learned = sp_model.model_embeddings.data.numpy()  # (32, 512)
print(f"Learned shape: {learned.shape}, PARC shape: {model_matrix.shape}")

# Cosine similarity within architectures
from numpy.linalg import norm
for arch in ["resnet18", "resnet50", "alexnet", "googlenet"]:
    arch_idx = [i for i, m in enumerate(model_ids) if m.startswith(arch + "_")]
    if len(arch_idx) < 2:
        continue

    # PARC tokens: cosine sim between variants
    parc_vecs = model_matrix[arch_idx]
    parc_norms = parc_vecs / (norm(parc_vecs, axis=1, keepdims=True) + 1e-8)
    parc_cos = parc_norms @ parc_norms.T

    # Learned tokens: cosine sim between variants
    learn_vecs = learned[arch_idx]
    learn_norms = learn_vecs / (norm(learn_vecs, axis=1, keepdims=True) + 1e-8)
    learn_cos = learn_norms @ learn_norms.T

    # Off-diagonal stats
    n = len(arch_idx)
    mask = ~np.eye(n, dtype=bool)
    parc_off = parc_cos[mask]
    learn_off = learn_cos[mask]

    print(f"\n  {arch} ({n} variants):")
    print(f"    PARC cosine sim:    mean={parc_off.mean():.4f}, min={parc_off.min():.4f}, max={parc_off.max():.4f}")
    print(f"    Learned cosine sim: mean={learn_off.mean():.4f}, min={learn_off.min():.4f}, max={learn_off.max():.4f}")
    print(f"    -> Spider learned {('MORE' if learn_off.std() > parc_off.std() else 'LESS')} diverse embeddings")

print(f"\n{SEP}")
print("COMPARISON COMPLETE")
print(SEP)
