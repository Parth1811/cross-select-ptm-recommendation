"""Extract 8192-dim parameter vectors for all 32 PARC models using FAISS K-means.

Replicates the old repo's BaseExtractor pipeline:
1. Load model state_dict → list of column vectors (reversed order)
2. Per-layer FAISS K-means clustering to compress each layer
3. Concatenate centroids → pad/truncate to 8192

Usage:
    python scripts/extract_parc_raw_vectors.py --parc-models-dir parc/models
    python scripts/extract_parc_raw_vectors.py --parc-models-dir parc/models --no-kmeans
    python scripts/extract_parc_raw_vectors.py --skip-existing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import (
    AlexNet_Weights,
    GoogLeNet_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
)

ARCHITECTURES = ["resnet50", "resnet18", "googlenet", "alexnet"]
SOURCE_DATASETS = [
    "nabird", "oxford_pets", "cub200", "caltech101",
    "stanford_dogs", "voc2007", "cifar10", "imagenet",
]
NUM_CLASSES = {
    "nabird": 555, "oxford_pets": 37, "cub200": 200,
    "caltech101": 101, "stanford_dogs": 120, "voc2007": 21,
    "cifar10": 10, "imagenet": 1000,
}
IMAGENET_WEIGHTS = {
    "alexnet": AlexNet_Weights.IMAGENET1K_V1,
    "resnet18": ResNet18_Weights.IMAGENET1K_V1,
    "resnet50": ResNet50_Weights.IMAGENET1K_V2,
    "googlenet": GoogLeNet_Weights.IMAGENET1K_V1,
}
OUTPUT_SIZE = 8192


def load_model(arch: str, source: str, parc_models_dir: Path) -> nn.Module:
    if source == "imagenet":
        return getattr(tv_models, arch)(weights=IMAGENET_WEIGHTS[arch]).eval()
    num_classes = NUM_CLASSES[source]
    kwargs = {"aux_logits": False, "init_weights": False} if arch == "googlenet" else {}
    model = getattr(tv_models, arch)(weights=None, num_classes=num_classes, **kwargs)
    pth_path = parc_models_dir / arch / f"{arch}_{source}.pth"
    if not pth_path.exists():
        raise FileNotFoundError(f"Missing: {pth_path}")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    if all(k.startswith("module.") for k in state_dict):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.eval()


def kmeans_cluster(values: np.ndarray, n_clusters: int) -> np.ndarray:
    """FAISS K-means on 1-D scalar values, returns sorted centroids."""
    data = values.astype(np.float32).reshape(-1, 1)
    kmeans = faiss.Kmeans(d=1, k=n_clusters, gpu=False)
    kmeans.train(data)
    return np.sort(kmeans.centroids.ravel())[::-1][:n_clusters]


def extract_with_kmeans(model: nn.Module) -> np.ndarray:
    """Replicate old repo: reverse layers, K-means per layer, concat to 8192."""
    params = [t.detach().cpu().numpy().reshape(-1, 1) for t in model.state_dict().values()]
    params.reverse()

    cluster_count = OUTPUT_SIZE // len(params)
    min_points = cluster_count * 39  # FAISS requirement

    columns = []
    for col in params:
        flat = col.flatten()
        if len(flat) >= min_points:
            columns.append(kmeans_cluster(flat, cluster_count))
        else:
            columns.append(flat)

    concatenated = np.concatenate(columns).astype(np.float32)
    if concatenated.size >= OUTPUT_SIZE:
        return concatenated[:OUTPUT_SIZE]
    return np.pad(concatenated, (0, OUTPUT_SIZE - concatenated.size))


def extract_flat(model: nn.Module) -> np.ndarray:
    """Simple flatten + truncate/pad (no K-means)."""
    parts = [t.detach().cpu().numpy().flatten() for t in model.state_dict().values()]
    flat = np.concatenate(parts).astype(np.float32)
    if flat.size >= OUTPUT_SIZE:
        return flat[:OUTPUT_SIZE]
    return np.pad(flat, (0, OUTPUT_SIZE - flat.size))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parc-models-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/extracted/parc_models"))
    parser.add_argument("--no-kmeans", action="store_true", help="Use flat truncation instead of K-means")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ok = skipped = failed = 0
    for arch in ARCHITECTURES:
        for source in SOURCE_DATASETS:
            name = f"{arch}_{source}"
            out_path = args.output_dir / f"{name}.npz"
            if args.skip_existing and out_path.exists():
                skipped += 1
                continue
            try:
                model = load_model(arch, source, args.parc_models_dir)
                vec = extract_flat(model) if args.no_kmeans else extract_with_kmeans(model)
                np.savez_compressed(out_path, parameters=vec.reshape(-1, 1))
                print(f"[OK] {name}  shape={vec.shape}")
                ok += 1
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                failed += 1

    print(f"\nDone: {ok} extracted, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
