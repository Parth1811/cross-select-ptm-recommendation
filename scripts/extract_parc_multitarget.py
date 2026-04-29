"""Extract multi-target PARC embeddings to fix the single-target collapse bug.

Instead of probing each model on only caltech101 (producing identical embeddings
for same-architecture variants), this script loads probes from ALL 7 target
datasets and aggregates them into richer, discriminative embeddings.

Probe path format: <arch>_<source>_<target>_<fold>.pkl
Each .pkl contains a sklearn LogisticRegression whose .coef_ is the probe weight
matrix (num_target_classes, feature_dim). We flatten coef_ per target and either
concatenate or mean-pool across targets.

Usage:
    python scripts/extract_parc_multitarget.py --probes-dir parc/cache/probes/fixed_budget_500
    python scripts/extract_parc_multitarget.py --probes-dir ... --aggregation concat
    python scripts/extract_parc_multitarget.py --probes-dir ... --aggregation mean --dim 512
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

ARCHITECTURES = ["resnet50", "resnet18", "googlenet", "alexnet"]
SOURCE_DATASETS = [
    "nabird", "oxford_pets", "cub200", "caltech101",
    "stanford_dogs", "voc2007", "cifar10", "imagenet",
]
TARGET_DATASETS = [
    "caltech101", "cifar10", "cub200", "nabird",
    "oxford_pets", "stanford_dogs", "voc2007",
]
FOLD = 0


def load_probe_weights(pkl_path: Path) -> np.ndarray:
    """Load probe .pkl and return flattened coef_ vector."""
    with open(pkl_path, "rb") as f:
        probe = pickle.load(f)
    # sklearn LogisticRegression stores coef_ as (n_classes, n_features)
    coef = probe.coef_ if hasattr(probe, "coef_") else probe["coef_"]
    return coef.astype(np.float32).flatten()


def extract_multitarget(
    arch: str, source: str, probes_dir: Path, dim: int,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Extract probe weights across all targets for one model.

    Returns (per_target_vectors, None) — caller aggregates.
    """
    vectors = {}
    for target in TARGET_DATASETS:
        pkl = probes_dir / f"{arch}_{source}_{target}_{FOLD}.pkl"
        if not pkl.exists():
            continue
        raw = load_probe_weights(pkl)
        # Truncate/pad to uniform dim per target
        if raw.size >= dim:
            vectors[target] = raw[:dim]
        else:
            vectors[target] = np.pad(raw, (0, dim - raw.size))
    return vectors


def aggregate_concat(vectors: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate across targets (ordered). Output dim = n_targets * per_target_dim."""
    parts = [vectors[t] for t in TARGET_DATASETS if t in vectors]
    return np.concatenate(parts)


def aggregate_mean(vectors: dict[str, np.ndarray]) -> np.ndarray:
    """Mean-pool across targets. Output dim = per_target_dim."""
    parts = [vectors[t] for t in TARGET_DATASETS if t in vectors]
    return np.mean(parts, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes-dir", type=Path, required=True,
                        help="Path to parc/cache/probes/fixed_budget_500/")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/extracted/parc_multitarget"))
    parser.add_argument("--aggregation", choices=["concat", "mean"], default="mean",
                        help="How to aggregate across targets (default: mean)")
    parser.add_argument("--dim", type=int, default=512,
                        help="Per-target truncation dim (default: 512)")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    agg_fn = aggregate_concat if args.aggregation == "concat" else aggregate_mean
    ok = skipped = failed = 0

    for arch in ARCHITECTURES:
        for source in SOURCE_DATASETS:
            name = f"{arch}_{source}"
            out_path = args.output_dir / f"{name}_embedding.npz"
            if args.skip_existing and out_path.exists():
                skipped += 1
                continue

            try:
                vectors = extract_multitarget(arch, source, args.probes_dir, args.dim)
                if not vectors:
                    print(f"[SKIP] {name}: no probes found")
                    skipped += 1
                    continue

                embedding = agg_fn(vectors)
                np.savez_compressed(out_path, embedding=embedding,
                                    targets=list(vectors.keys()),
                                    aggregation=args.aggregation)
                print(f"[OK] {name}  targets={len(vectors)}  shape={embedding.shape}")
                ok += 1
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                failed += 1

    print(f"\nDone: {ok} extracted, {skipped} skipped, {failed} failed")
    print(f"Output: {args.output_dir}")
    if ok > 0:
        sample = next(args.output_dir.glob("*_embedding.npz"))
        data = np.load(sample, allow_pickle=True)
        print(f"Embedding dim: {data['embedding'].shape}")


if __name__ == "__main__":
    main()
