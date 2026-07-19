"""Build the caches used by SketchFlow training."""

from __future__ import annotations

import argparse

from data_process.quickdraw_categories import build_qdraw_cat_cache
from data_process.quickdraw_sketch import build_quickdraw_cache
from data_process.clip_tokens import build_clip_cache
from data_process.text_perturbation import build_qdraw_text_perturbation_cache


def main(
    data_path: str = "data/quickdraw",
    cache_dir: str = "cache/quickdraw",
    n_points: int = 256,
    splits: tuple[str, ...] = ("train", "valid"),
    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    clip_batch_size: int = 512,
    clip_num_workers: int = 16,
    clip_precision: str = "16-mixed",
    gpu_id: int = 0,
    save_clip_tokens: bool = False,
    build_text_stats: bool = False,
) -> None:
    for split in splits:
        build_quickdraw_cache(
            input_path=data_path,
            n_points=n_points,
            split=split,
            cache_dir=cache_dir,
        )

    build_qdraw_cat_cache(
        quickdraw_data_dir=data_path,
        cache_dir=cache_dir,
        qdraw_n_points=n_points,
        model_name=clip_model_name,
        pretrained=clip_pretrained,
    )

    for split in splits:
        build_clip_cache(
            input_dir=data_path,
            split_name=split,
            cache_dir=cache_dir,
            batch_size=clip_batch_size,
            num_workers=clip_num_workers,
            precision=clip_precision,
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            mode="quickdraw",
            n_points=n_points,
            save_embeddings=True,
            save_tokens=save_clip_tokens,
            gpu_id=gpu_id,
        )

    if build_text_stats:
        for split in splits:
            build_qdraw_text_perturbation_cache(
                input_dir=data_path,
                cache_dir=cache_dir,
                split_name=split,
                n_points=n_points,
                model_name=clip_model_name,
                pretrained=clip_pretrained,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/quickdraw")
    parser.add_argument("--cache-dir", default="cache/quickdraw")
    parser.add_argument("--n-points", type=int, default=256)
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--clip-model-name", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--clip-batch-size", type=int, default=512)
    parser.add_argument("--clip-num-workers", type=int, default=16)
    parser.add_argument("--clip-precision", default="16-mixed")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--save-clip-tokens", action="store_true")
    parser.add_argument("--build-text-stats", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        data_path=args.data_path,
        cache_dir=args.cache_dir,
        n_points=args.n_points,
        splits=tuple(args.splits),
        clip_model_name=args.clip_model_name,
        clip_pretrained=args.clip_pretrained,
        clip_batch_size=args.clip_batch_size,
        clip_num_workers=args.clip_num_workers,
        clip_precision=args.clip_precision,
        gpu_id=args.gpu_id,
        save_clip_tokens=args.save_clip_tokens,
        build_text_stats=args.build_text_stats,
    )
