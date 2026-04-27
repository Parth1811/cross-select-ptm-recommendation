"""Extract CLIP image embeddings for PARC target datasets.

Produces class-balanced shards at:
    artifacts/extracted/datasets/<slug>/<split>/<split>_clip_NNNNN.npz

Shard format:
    features:       (batches_per_shard, num_classes, 512) float32
    class_ids:      (batches_per_shard, num_classes)      int64
    class_names:    (batches_per_shard, num_classes)      object
    actual_batches: (1,) int32
    target_batches: (1,) int32

Each batch row = one sample per class encoded by CLIP ViT-B/32.

Usage:
    python scripts/extract_clip_dataset_tokens.py --data-dir parc/data
    python scripts/extract_clip_dataset_tokens.py --datasets cifar10 cub200
    python scripts/extract_clip_dataset_tokens.py --skip-existing
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPModel, CLIPProcessor

# PARC target datasets
PARC_TARGETS = {
    "cifar10": "cifar_10",
    "oxford_pets": "oxford_pets",
    "cub200": "cub200",
    "caltech101": "caltech_101",
    "stanford_dogs": "stanford_dogs",
    "nabird": "nabird",
    "voc2007": "voc2007",
}
BATCHES_PER_SHARD = 16


class ClipEncoder:
    """Minimal CLIP ViT-B/32 image encoder."""

    def __init__(self, device: str = "cuda", precision: str = "fp16"):
        model_name = "openai/clip-vit-base-patch32"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.precision = precision
        if precision == "fp16":
            self.model.half()

        proc = CLIPProcessor.from_pretrained(model_name).image_processor
        size = proc.size
        shorter = int(size.get("shortest_edge", 224)) if isinstance(size, dict) else int(size)
        crop = 224
        self.transform = transforms.Compose([
            transforms.Resize(shorter, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=proc.image_mean, std=proc.image_std),
        ])

    @torch.inference_mode()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = pixel_values.to(self.device)
        if self.precision == "fp16":
            x = x.half()
        features = self.model.get_image_features(pixel_values=x)
        return F.normalize(features, dim=-1).float().cpu()


class ClassBalancedSampler(Sampler):
    """Yields batches of indices with one sample per class."""

    def __init__(self, labels: list[int], seed: int = 42):
        from collections import defaultdict
        self.class_indices: dict[int, list[int]] = defaultdict(list)
        for i, l in enumerate(labels):
            self.class_indices[l].append(i)
        self.classes = sorted(self.class_indices.keys())
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        # Shuffle within each class
        pools = {c: list(self._rng.permutation(idxs)) for c, idxs in self.class_indices.items()}
        # Yield batches until any class is exhausted
        max_batches = min(len(v) for v in pools.values())
        for i in range(max_batches):
            yield [pools[c][i] for c in self.classes]

    def __len__(self):
        return min(len(v) for v in self.class_indices.values())


def load_parc_dataset(name: str, data_dir: Path, train: bool, transform):
    """Load a PARC dataset using the parc/ data directory structure."""
    # Import PARC's dataset constructors
    parc_dir = data_dir.parent  # assumes data_dir = parc/data, parc_dir = parc/
    sys.path.insert(0, str(parc_dir))
    old_cwd = os.getcwd()
    os.chdir(parc_dir)
    try:
        from datasets import construct_dataset, get_dataset_path, dataset_objs
        dataset_cls = dataset_objs[name]
        ds = dataset_cls(get_dataset_path(name), train, transform=transform)
        return ds
    finally:
        os.chdir(old_cwd)
        sys.path.remove(str(parc_dir))


def get_labels(dataset) -> list[int]:
    """Extract labels from a PARC dataset."""
    for attr in ("labels", "targets", "_labels"):
        if hasattr(dataset, attr):
            return [int(x) for x in getattr(dataset, attr)]
    # Fallback: iterate
    labels = []
    for i in range(len(dataset)):
        target = dataset[i][1]
        if hasattr(target, "__len__") and not isinstance(target, (int, str)):
            arr = np.asarray(target).flatten()
            nz = np.flatnonzero(arr)
            labels.append(int(nz[0]) if len(nz) > 0 else 0)
        else:
            labels.append(int(target))
    return labels


def save_shard(path: Path, features, class_ids, class_names, actual, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features, class_ids=class_ids,
             class_names=class_names,
             actual_batches=np.array([actual], dtype=np.int32),
             target_batches=np.array([target], dtype=np.int32))


def process_dataset(
    name: str, slug: str, split: str, is_train: bool,
    clip: ClipEncoder, data_dir: Path, output_root: Path, skip_existing: bool,
) -> None:
    split_dir = output_root / slug / split
    if skip_existing and split_dir.exists() and any(split_dir.glob("*.npz")):
        print(f"  [SKIP] {slug}/{split}")
        return

    dataset = load_parc_dataset(name, data_dir, is_train, clip.transform)
    labels = get_labels(dataset)
    num_classes = len(set(labels))
    print(f"  {slug}/{split}: {len(dataset)} samples, {num_classes} classes")

    sampler = ClassBalancedSampler(labels)
    # Manual iteration since we use a batch sampler
    buffer = []
    shard_idx = 0

    for batch_indices in sampler:
        images = torch.stack([dataset[i][0] for i in batch_indices])
        batch_labels = np.array([labels[i] for i in batch_indices], dtype=np.int64)
        features = clip.encode(images).numpy()
        class_names = np.array([str(l) for l in batch_labels], dtype=object)

        buffer.append({"features": features, "class_ids": batch_labels, "class_names": class_names})

        if len(buffer) >= BATCHES_PER_SHARD:
            feats = np.stack([b["features"] for b in buffer[:BATCHES_PER_SHARD]])
            cids = np.stack([b["class_ids"] for b in buffer[:BATCHES_PER_SHARD]])
            cnames = np.stack([b["class_names"] for b in buffer[:BATCHES_PER_SHARD]])
            path = split_dir / f"{split}_clip_{shard_idx:05d}.npz"
            save_shard(path, feats, cids, cnames, BATCHES_PER_SHARD, BATCHES_PER_SHARD)
            print(f"    {path.name}  shape={feats.shape}")
            buffer = buffer[BATCHES_PER_SHARD:]
            shard_idx += 1

    # Flush remaining
    if buffer:
        feats = np.stack([b["features"] for b in buffer])
        cids = np.stack([b["class_ids"] for b in buffer])
        cnames = np.stack([b["class_names"] for b in buffer])
        path = split_dir / f"{split}_clip_{shard_idx:05d}.npz"
        save_shard(path, feats, cids, cnames, len(buffer), BATCHES_PER_SHARD)
        print(f"    {path.name}  shape={feats.shape}")
        shard_idx += 1

    print(f"  Done: {slug}/{split}  shards={shard_idx}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to parc/data/ directory with raw datasets")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/extracted/datasets"))
    parser.add_argument("--datasets", nargs="+", default=None,
                        help=f"Subset to extract. Default: all. Choices: {list(PARC_TARGETS)}")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    clip = ClipEncoder(device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = args.datasets or list(PARC_TARGETS.keys())
    for name in targets:
        if name not in PARC_TARGETS:
            print(f"[WARN] unknown: {name}")
            continue
        slug = PARC_TARGETS[name]
        print(f"\n=== {name} → {slug} ===")
        for split in args.splits:
            is_train = (split == "train")
            try:
                process_dataset(name, slug, split, is_train, clip,
                                args.data_dir, args.output_dir, args.skip_existing)
            except Exception as e:
                import traceback
                print(f"  [ERROR] {slug}/{split}: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
