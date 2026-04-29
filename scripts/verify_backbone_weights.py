#!/usr/bin/env python3
"""Strategy D: Verify whether PARC fine-tuned models have identical backbone weights.

Three verification modes:
  1. LOCAL: Load torchvision pretrained weights, simulate head-only fine-tuning,
     prove backbone stays identical (no checkpoints needed).
  2. PROBES: Load PARC probe .pkl files from Gautschi, compare probe weights
     across source datasets for each architecture.
  3. CHECKPOINTS: Load saved .pth model checkpoints if available.

Usage:
    # Local verification (no remote files needed):
    python scripts/verify_backbone_weights.py

    # With PARC probes from Gautschi:
    python scripts/verify_backbone_weights.py --probes-dir parc/cache/probes/fixed_budget_500

    # With saved checkpoints (if they exist):
    python scripts/verify_backbone_weights.py --checkpoints-dir parc/models
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as tv
from torchvision.models import (
    AlexNet_Weights, GoogLeNet_Weights, ResNet18_Weights, ResNet50_Weights,
)

ARCHS = ["alexnet", "googlenet", "resnet18", "resnet50"]
SOURCES = ["caltech101", "cifar10", "cub200", "imagenet", "nabird", "oxford_pets", "stanford_dogs", "voc2007"]
NUM_CLASSES = {"caltech101": 101, "cifar10": 10, "cub200": 200, "imagenet": 1000,
               "nabird": 555, "oxford_pets": 37, "stanford_dogs": 120, "voc2007": 21}
WEIGHTS = {"alexnet": AlexNet_Weights.IMAGENET1K_V1, "resnet18": ResNet18_Weights.IMAGENET1K_V1,
           "resnet50": ResNet50_Weights.IMAGENET1K_V2, "googlenet": GoogLeNet_Weights.IMAGENET1K_V1}
HEAD_PREFIXES = {"resnet18": ["fc."], "resnet50": ["fc."], "alexnet": ["classifier."], "googlenet": ["fc."]}


def get_backbone_params(sd: dict[str, torch.Tensor], arch: str) -> torch.Tensor:
    prefixes = HEAD_PREFIXES[arch]
    parts = [v.flatten() for k, v in sorted(sd.items()) if not any(k.startswith(p) for p in prefixes)]
    return torch.cat(parts).float()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


def compare_pair(v1: torch.Tensor, v2: torch.Tensor) -> tuple[float, float, bool]:
    return cosine_sim(v1, v2), (v1 - v2).norm().item(), torch.equal(v1, v2)


def print_table(rows: list[tuple], arch: str):
    print(f"\n  {'Pair':<50} {'Cosine':>8} {'L2 dist':>12} {'Identical?':>10}")
    print(f"  {'-'*50} {'-'*8} {'-'*12} {'-'*10}")
    identical = 0
    for s1, s2, cos, l2, ident in rows:
        tag = "YES ⚠️" if ident else "no"
        identical += ident
        print(f"  {arch}_{s1} vs {s2:<25} {cos:>8.6f} {l2:>12.6f} {tag:>10}")
    return identical, len(rows)


# ── Mode 1: Local torchvision verification ──────────────────────────────────

def verify_local():
    """Prove head-only fine-tuning leaves backbone identical using torchvision weights."""
    print("\n" + "=" * 74)
    print("  MODE 1: LOCAL VERIFICATION — Simulated Head-Only Fine-Tuning")
    print("  (Proves backbone identity without needing remote checkpoints)")
    print("=" * 74)

    total_identical, total_pairs = 0, 0
    for arch in ARCHS:
        print(f"\n{'─'*74}")
        print(f"  {arch.upper()}")
        print(f"{'─'*74}")

        # Load pretrained model
        base_model = getattr(tv, arch)(weights=WEIGHTS[arch])
        base_sd = base_model.state_dict()
        base_backbone = get_backbone_params(base_sd, arch)
        print(f"  Backbone params: {base_backbone.numel():,}")

        # Simulate head-only fine-tuning for different source datasets
        simulated: dict[str, torch.Tensor] = {"imagenet": base_backbone}
        for src in SOURCES:
            if src == "imagenet":
                continue
            model = getattr(tv, arch)(weights=WEIGHTS[arch])
            # Freeze backbone (exactly what PARC does)
            for name, p in model.named_parameters():
                if not any(name.startswith(pfx) for pfx in HEAD_PREFIXES[arch]):
                    p.requires_grad = False
            # Replace head with random init for target num_classes
            nc = NUM_CLASSES[src]
            if arch in ("resnet18", "resnet50"):
                model.fc = nn.Linear(model.fc.in_features, nc)
            elif arch == "alexnet":
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, nc)
            elif arch == "googlenet":
                model.fc = nn.Linear(model.fc.in_features, nc)
            # Simulate 1 step of training (only head updates)
            opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
            dummy_x = torch.randn(2, 3, 224, 224)
            out = model(dummy_x)
            if isinstance(out, tuple):  # googlenet returns tuple
                out = out[0]
            loss = out.sum()
            loss.backward()
            opt.step()
            simulated[src] = get_backbone_params(model.state_dict(), arch)

        # Pairwise comparison: imagenet vs others, and non-imagenet vs each other
        rows = []
        sources = sorted(simulated.keys())
        for i, s1 in enumerate(sources):
            for s2 in sources[i + 1:]:
                cos, l2, ident = compare_pair(simulated[s1], simulated[s2])
                rows.append((s1, s2, cos, l2, ident))

        ident, total = print_table(rows, arch)
        total_identical += ident
        total_pairs += total

        # Verdict per arch
        non_img_pairs = [(s1, s2, c, l, i) for s1, s2, c, l, i in rows if s1 != "imagenet" and s2 != "imagenet"]
        img_pairs = [(s1, s2, c, l, i) for s1, s2, c, l, i in rows if s1 == "imagenet" or s2 == "imagenet"]
        non_img_identical = sum(1 for *_, i in non_img_pairs if i)
        img_identical = sum(1 for *_, i in img_pairs if i)
        print(f"\n  Non-imagenet pairs: {non_img_identical}/{len(non_img_pairs)} identical")
        print(f"  Imagenet vs others: {img_identical}/{len(img_pairs)} identical")

    print(f"\n{'='*74}")
    print(f"  OVERALL: {total_identical}/{total_pairs} pairs have identical backbones")
    if total_identical > 0:
        print(f"  → Head-only fine-tuning CANNOT change backbone weights (by definition).")
        print(f"  → PARC probes trained on identical features → identical embeddings.")
        print(f"  → Strategy C (fix extraction) CANNOT help for collapsed pairs.")
        print(f"  → Must use Strategy A (learnable residuals) or Strategy B (concat features).")
    print(f"{'='*74}")


# ── Mode 2: PARC probe comparison ───────────────────────────────────────────

def verify_probes(probes_dir: Path):
    """Compare PARC probe weights across source datasets."""
    print(f"\n{'='*74}")
    print(f"  MODE 2: PARC PROBE COMPARISON — {probes_dir}")
    print(f"{'='*74}")

    for arch in ARCHS:
        print(f"\n{'─'*74}")
        print(f"  {arch.upper()} — Probe weights (target=caltech101, fold=0)")
        print(f"{'─'*74}")

        probes: dict[str, torch.Tensor] = {}
        for src in SOURCES:
            pkl = probes_dir / f"{arch}_{src}_caltech101_0.pkl"
            if not pkl.exists():
                print(f"  [SKIP] {pkl.name}")
                continue
            with open(pkl, "rb") as f:
                obj = pickle.load(f)
            # Extract probe weights (sklearn LinearRegression or similar)
            if hasattr(obj, "coef_"):
                w = torch.tensor(obj.coef_, dtype=torch.float32).flatten()
            elif isinstance(obj, dict) and "coef_" in obj:
                w = torch.tensor(obj["coef_"], dtype=torch.float32).flatten()
            else:
                print(f"  [SKIP] {pkl.name}: unknown format {type(obj)}")
                continue
            probes[src] = w

        if len(probes) < 2:
            print("  Not enough probes loaded.")
            continue

        print(f"  Probe dim: {probes[next(iter(probes))].numel()}")
        rows = []
        sources = sorted(probes.keys())
        for i, s1 in enumerate(sources):
            for s2 in sources[i + 1:]:
                cos, l2, ident = compare_pair(probes[s1], probes[s2])
                rows.append((s1, s2, cos, l2, ident))
        print_table(rows, arch)


# ── Mode 3: Checkpoint comparison (original script logic) ───────────────────

def verify_checkpoints(ckpt_dir: Path):
    """Compare backbone weights from saved .pth checkpoints."""
    print(f"\n{'='*74}")
    print(f"  MODE 3: CHECKPOINT COMPARISON — {ckpt_dir}")
    print(f"{'='*74}")

    for arch in ARCHS:
        print(f"\n{'─'*74}")
        print(f"  {arch.upper()}")
        print(f"{'─'*74}")

        models: dict[str, torch.Tensor] = {}
        for src in SOURCES:
            if src == "imagenet":
                model = getattr(tv, arch)(weights=WEIGHTS[arch])
                models[src] = get_backbone_params(model.state_dict(), arch)
                continue
            pth = ckpt_dir / arch / f"{arch}_{src}.pth"
            if not pth.exists():
                continue
            sd = torch.load(pth, map_location="cpu", weights_only=True)
            if all(k.startswith("module.") for k in sd):
                sd = {k[7:]: v for k, v in sd.items()}
            models[src] = get_backbone_params(sd, arch)

        if len(models) < 2:
            print("  Not enough checkpoints found.")
            continue

        print(f"  Backbone params: {models[next(iter(models))].numel():,}")
        rows = []
        sources = sorted(models.keys())
        for i, s1 in enumerate(sources):
            for s2 in sources[i + 1:]:
                cos, l2, ident = compare_pair(models[s1], models[s2])
                rows.append((s1, s2, cos, l2, ident))
        print_table(rows, arch)


def main():
    parser = argparse.ArgumentParser(description="Strategy D: Verify backbone weight identity")
    parser.add_argument("--probes-dir", type=Path, help="PARC probes directory (pkl files)")
    parser.add_argument("--checkpoints-dir", type=Path, help="Model checkpoints directory (pth files)")
    parser.add_argument("--arch", help="Single architecture to check")
    args = parser.parse_args()

    global ARCHS
    if args.arch:
        ARCHS = [args.arch]

    # Always run local verification
    verify_local()

    if args.probes_dir and args.probes_dir.exists():
        verify_probes(args.probes_dir)

    if args.checkpoints_dir and args.checkpoints_dir.exists():
        verify_checkpoints(args.checkpoints_dir)


if __name__ == "__main__":
    main()
