"""Compute FID and optional CLIP similarity for rendered sketch folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

from utils.clip_manager import get_clip_manager


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def image_paths(directory: Path) -> list[Path]:
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No images found in {directory}.")
    return paths


def uint8_batch(paths: list[Path]) -> torch.Tensor:
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((299, 299))
        images.append(torch.from_numpy(np.asarray(image)).permute(2, 0, 1))
    return torch.stack(images)


@torch.no_grad()
def compute_fid(
    real_paths: list[Path],
    generated_paths: list[Path],
    batch_size: int,
    device: str,
) -> float:
    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    for start in range(0, len(real_paths), batch_size):
        metric.update(
            uint8_batch(real_paths[start : start + batch_size]).to(device),
            real=True,
        )
    for start in range(0, len(generated_paths), batch_size):
        metric.update(
            uint8_batch(generated_paths[start : start + batch_size]).to(device),
            real=False,
        )
    return float(metric.compute().cpu())


@torch.no_grad()
def compute_clip_similarity(
    generated_paths: list[Path],
    prompt: str,
    batch_size: int,
    device: str,
) -> float:
    manager = get_clip_manager(device=device)
    text_feature = manager.encode_text([prompt])
    scores = []
    for start in range(0, len(generated_paths), batch_size):
        paths = generated_paths[start : start + batch_size]
        images = torch.stack(
            [
                manager.preprocess(Image.open(path).convert("RGB"))
                for path in paths
            ]
        )
        image_features = manager.encode_image(images)
        scores.append((image_features @ text_feature.T).flatten().cpu())
    return float(torch.cat(scores).mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    real = image_paths(Path(args.real_dir))
    generated = image_paths(Path(args.generated_dir))
    results = {
        "fid": compute_fid(real, generated, args.batch_size, args.device),
        "num_real": len(real),
        "num_generated": len(generated),
    }
    if args.prompt:
        results["clip_similarity"] = compute_clip_similarity(
            generated,
            args.prompt,
            args.batch_size,
            args.device,
        )
    print(json.dumps(results, indent=2))
    if args.output:
        Path(args.output).write_text(
            json.dumps(results, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
