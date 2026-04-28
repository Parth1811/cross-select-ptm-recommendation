#!/usr/bin/env python3
"""End-to-end debug trace: 2 samples through Cross-Select + Spider."""

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

SEP = "=" * 80

# ─── 1. Ground truth ───
print(SEP)
print("STEP 1: GROUND TRUTH RANKINGS")
print(SEP)
gt = load_ground_truth("constants/ground_truth_rankings.json")
datasets = sorted(gt.keys())
test_ds = ["cifar_10", "caltech_101"]
for ds in test_ds:
    ranked = sorted(gt[ds].items(), key=lambda x: -x[1])
    print(f"\n  {ds} TRUE ranking (top 5 of {len(ranked)}):")
    for i, (m, a) in enumerate(ranked[:5]):
        tag = "<-- BEST" if i == 0 else ""
        print(f"    #{i+1}: {m:40s} acc={a:.2f}  {tag}")

# ─── 2. Model tokens ───
print(f"\n{SEP}")
print("STEP 2: MODEL TOKENS (PTM embeddings, dim=512)")
print(SEP)
model_tokens = load_model_tokens("artifacts/extracted/parc_model_embeddings")
model_ids = sorted(model_tokens.keys())
model_matrix = np.stack([model_tokens[m] for m in model_ids], axis=0).astype(np.float32)
print(f"Shape: {model_matrix.shape}  ({len(model_ids)} models x 512 dims)")
norms = np.linalg.norm(model_matrix, axis=1)
print(f"L2 norms: min={norms.min():.2f}, max={norms.max():.2f}, mean={norms.mean():.2f}")
# Show which models are in GT vs extra
gt_models = set()
for ds_gt in gt.values():
    gt_models.update(ds_gt.keys())
for mid in model_ids:
    in_gt = "GT" if mid in gt_models else "EXTRA"
    print(f"  {mid:55s} norm={np.linalg.norm(model_tokens[mid]):.2f}  [{in_gt}]")

# ─── 3. Dataset tokens ───
print(f"\n{SEP}")
print("STEP 3: DATASET TOKENS (prototype from eval shards)")
print(SEP)
ds_protos = {}
for ds in test_ds:
    for split in ["validation", "test", "train"]:
        try:
            shards = list_shards("artifacts/extracted/datasets", ds, split)
            proto = dataset_prototype(shards)
            ds_protos[ds] = proto
            print(f"  {ds} ({split}): {len(shards)} shards -> shape {proto.shape}")
            print(f"    Stats: mean={proto.mean():.6f}, std={proto.std():.6f}")
            print(f"    Per-class norms: {np.linalg.norm(proto, axis=1)}")
            break
        except FileNotFoundError:
            continue

# ─── 4. Accuracy matrix ───
print(f"\n{SEP}")
print("STEP 4: ACCURACY MATRIX (models x datasets)")
print(SEP)
rng = random.Random(42)
acc_matrix = build_accuracy_matrix(gt, model_ids, datasets, missing_value="random_rank", rng=rng)
print(f"Shape: {acc_matrix.shape}")
for ds in test_ds:
    di = datasets.index(ds)
    col = acc_matrix[:, di]
    top5 = np.argsort(-col)[:5]
    print(f"\n  {ds} (col {di}) top 5:")
    for rank, mi in enumerate(top5):
        in_gt = model_ids[mi] in gt[ds]
        label = "GT" if in_gt else "imputed"
        print(f"    #{rank+1}: {model_ids[mi]:45s} acc={col[mi]:.2f}  [{label}]")

# ─── 5. Cross-Select forward pass ───
print(f"\n{SEP}")
print("STEP 5: CROSS-SELECT MODEL FORWARD PASS")
print(SEP)
ckpt = torch.load(
    "artifacts/experiments/2026-04-24_12-47-31/cs-listmle/last.pt",
    map_location="cpu", weights_only=False,
)
sd = ckpt["model_state"]
total_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
print(f"Checkpoint params: {total_params:,}")

from cross_select.models.cross_select import CrossSelect
model = CrossSelect(model_token_dim=512, dataset_token_dim=512, hidden_dim=512, num_heads=8, num_layers=4, dropout=0.0)
try:
    model.load_state_dict(sd, strict=True)
    print("Model loaded (strict=True)")
except RuntimeError as e:
    missing = set(model.state_dict().keys()) - set(sd.keys())
    extra = set(sd.keys()) - set(model.state_dict().keys())
    if missing:
        print(f"Missing keys: {missing}")
    if extra:
        print(f"Extra keys: {extra}")
    model.load_state_dict(sd, strict=False)
    print("Model loaded (strict=False)")

model.eval()

for ds in test_ds:
    proto = ds_protos[ds]
    dataset_tokens = torch.from_numpy(proto).unsqueeze(0)   # (1, C, 512)
    model_embs = torch.from_numpy(model_matrix).unsqueeze(0) # (1, 32, 512)

    print(f"\n  --- {ds} ---")
    print(f"  Inputs: dataset_tokens={list(dataset_tokens.shape)}, model_embs={list(model_embs.shape)}")

    with torch.no_grad():
        scores = model(model_embs, dataset_tokens)  # (1, 32) — model queries, dataset KV

    sc = scores.squeeze(0).numpy()
    print(f"  Raw scores: min={sc.min():.4f}, max={sc.max():.4f}, mean={sc.mean():.4f}, std={sc.std():.4f}")

    pred_rank = np.argsort(-sc)
    di = datasets.index(ds)
    true_accs = acc_matrix[:, di]
    true_rank = np.argsort(-true_accs)

    print(f"\n  {'Rank':<5} {'Predicted PTM':<50} {'Score':>8}  |  {'True PTM':<50} {'Acc':>8}")
    print(f"  {'-'*5} {'-'*50} {'-'*8}  |  {'-'*50} {'-'*8}")
    for i in range(min(10, len(model_ids))):
        pi, ti = pred_rank[i], true_rank[i]
        match = " <<" if pi == ti else ""
        print(f"  {i+1:<5} {model_ids[pi]:<50} {sc[pi]:>8.4f}  |  {model_ids[ti]:<50} {true_accs[ti]:>8.2f}{match}")

    # Metrics
    tau, _ = kendalltau(sc, true_accs)
    rho, _ = spearmanr(sc, true_accs)
    pred_top3 = set(pred_rank[:3].tolist())
    true_top3 = set(true_rank[:3].tolist())
    p3 = len(pred_top3 & true_top3) / 3

    print(f"\n  Kendall tau:   {tau:.4f}")
    print(f"  Spearman rho:  {rho:.4f}")
    print(f"  Precision@3:   {p3:.4f}")

# ─── 6. Loss breakdown ───
print(f"\n{SEP}")
print("STEP 6: LOSS COMPUTATION (per sample)")
print(SEP)
from cross_select.losses.ranking import ListMLELoss, ListMLEMSELoss, PairwiseBCELoss, listmle_loss, mse_loss, listnet_loss

listmle_fn = ListMLELoss(pred_temperature=1.0)
listmle_mse_fn = ListMLEMSELoss(ranking_weight=1.0, mse_weight=1.0)
pairwise_fn = PairwiseBCELoss(margin=0.0)

for ds in test_ds:
    proto = ds_protos[ds]
    dt = torch.from_numpy(proto).unsqueeze(0)
    me = torch.from_numpy(model_matrix).unsqueeze(0)
    with torch.no_grad():
        scores = model(me, dt)  # model queries, dataset KV
    di = datasets.index(ds)
    targets = torch.from_numpy(acc_matrix[:, di]).unsqueeze(0)

    ml, ml_d = listmle_fn(scores, targets)
    mm, mm_d = listmle_mse_fn(scores, targets)
    pb, pb_d = pairwise_fn(scores, targets)

    print(f"\n  {ds}:")
    print(f"    Scores range:  [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    print(f"    Targets range: [{targets.min().item():.2f}, {targets.max().item():.2f}]")
    print(f"    ListMLE loss:      {ml.item():.6f}")
    print(f"    ListMLE+MSE loss:  {mm.item():.6f}  (rank={mm_d['rank'].item():.6f}, mse={mm_d['mse'].item():.6f})")
    print(f"    PairwiseBCE loss:  {pb.item():.6f}")

# ─── 7. Spider comparison ───
print(f"\n{SEP}")
print("STEP 7: CROSS-SELECT vs SPIDER (checkpoint metrics)")
print(SEP)
cs_m = ckpt["metrics"]
sp_ckpt = torch.load(
    "artifacts/experiments/2026-04-24_12-47-31/spider-listmle/last.pt",
    map_location="cpu", weights_only=False,
)
sp_m = sp_ckpt["metrics"]

print(f"\n  {'Metric':<25} {'Spider':>10} {'CrossSelect':>12} {'Delta':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10}")
for k in ["weighted_kendall_tau", "ndcg", "precision@3", "relAcc@3", "pearson", "mrr"]:
    s, c = sp_m[k], cs_m[k]
    arrow = ">>>" if c > s else "<<<" if c < s else "==="
    print(f"  {k:<25} {s:>10.4f} {c:>12.4f} {c-s:>+10.4f}  {arrow}")

print(f"\n  Per-dataset breakdown:")
for ds_name in sorted(cs_m["_per_dataset"].keys()):
    cs_d = cs_m["_per_dataset"][ds_name]
    sp_d = sp_m["_per_dataset"][ds_name]
    print(f"\n    {ds_name}:")
    for k in ["weighted_kendall_tau", "ndcg", "precision@3"]:
        s, c = sp_d[k], cs_d[k]
        print(f"      {k:<25} Spider={s:.4f}  CS={c:.4f}  delta={c-s:+.4f}")

print(f"\n{SEP}")
print("DEBUG TRACE COMPLETE")
print(SEP)
