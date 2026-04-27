"""Extract 512-dim model embeddings for the 32 PARC controlled-bank models.

Pipeline: load model weights → flatten → truncate/pad to 8192 → autoencoder → 512-dim.

Requires:
    - PARC .pth files at <parc_models_dir>/<arch>/<arch>_<source>.pth
    - Trained ModelAutoEncoder checkpoint

Usage:
    python scripts/extract_model_embeddings.py --parc-models-dir parc/models
    python scripts/extract_model_embeddings.py --no-autoencoder  # raw 8192-dim
    python scripts/extract_model_embeddings.py --skip-existing
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    """Load a torchvision model with appropriate weights."""
    if source == "imagenet":
        model = getattr(tv_models, arch)(weights=IMAGENET_WEIGHTS[arch])
    else:
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


def extract_param_vector(model: nn.Module) -> np.ndarray:
    """Flatten all parameters, truncate/pad to OUTPUT_SIZE."""
    parts = [t.detach().cpu().numpy().flatten() for t in model.state_dict().values()]
    flat = np.concatenate(parts).astype(np.float32)
    if flat.size >= OUTPUT_SIZE:
        return flat[:OUTPUT_SIZE]
    return np.pad(flat, (0, OUTPUT_SIZE - flat.size))


def build_autoencoder(checkpoint_path: Path) -> nn.Module:
    """Build and load the autoencoder (8192 → 1024 → 512)."""
    encoder = nn.Sequential(
        nn.Linear(8192, 1024), nn.GELU(), nn.Dropout(0.05),
        nn.Linear(1024, 512),
    )
    decoder = nn.Sequential(
        nn.Linear(512, 1024), nn.GELU(), nn.Dropout(0.05),
        nn.Linear(1024, 8192),
    )
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # Map old keys to encoder/decoder
    encoder_keys = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
    decoder_keys = {k.replace("decoder.", ""): v for k, v in state_dict.items() if k.startswith("decoder.")}
    encoder.load_state_dict(encoder_keys)
    decoder.load_state_dict(decoder_keys)
    encoder.eval()
    return encoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parc-models-dir", type=Path, required=True,
                        help="Path to parc/models/ with .pth files")
    parser.add_argument("--autoencoder-checkpoint", type=Path, default=None,
                        help="Path to ModelAutoEncoder .pt checkpoint")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/extracted/parc_model_embeddings"))
    parser.add_argument("--no-autoencoder", action="store_true",
                        help="Save raw 8192-dim vectors instead of compressed 512-dim")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    autoencoder = None
    if not args.no_autoencoder:
        if args.autoencoder_checkpoint is None:
            # Try default path
            default = Path("artifacts/models/model_autoencoder")
            candidates = sorted(default.glob("ModelAutoEncoder_best_*.pt"))
            if candidates:
                args.autoencoder_checkpoint = candidates[-1]
            else:
                print("[WARN] No autoencoder checkpoint found, saving raw 8192-dim vectors")
                args.no_autoencoder = True
        if not args.no_autoencoder:
            autoencoder = build_autoencoder(args.autoencoder_checkpoint)
            print(f"[AE] Loaded: {args.autoencoder_checkpoint.name}")

    ok = skipped = failed = 0
    for arch in ARCHITECTURES:
        for source in SOURCE_DATASETS:
            name = f"{arch}_{source}"
            out_path = args.output_dir / f"{name}_embedding.npz"
            if args.skip_existing and out_path.exists():
                skipped += 1
                continue

            try:
                model = load_model(arch, source, args.parc_models_dir)
                raw = extract_param_vector(model)

                if autoencoder is not None:
                    with torch.no_grad():
                        tensor = torch.tensor(raw).unsqueeze(0)
                        embedding = autoencoder(tensor).squeeze(0).numpy()
                else:
                    embedding = raw

                np.savez_compressed(out_path, embedding=embedding)
                print(f"[OK] {name}  shape={embedding.shape}")
                ok += 1
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                failed += 1

    print(f"\nDone: {ok} extracted, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
