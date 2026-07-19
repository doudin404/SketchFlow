"""Train SketchFlow from a prepared QuickDraw cache."""

from __future__ import annotations

import argparse

from script.train_diffusion import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SketchFlow on QuickDraw sketches and CLIP embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", default="data/quickdraw")
    parser.add_argument("--cache-path", default="cache/quickdraw")
    parser.add_argument("--sample-dir", default="val_samples/sketchflow")
    parser.add_argument("--log-name", default="sketchflow")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--devices", default="0", help="Comma-separated GPU IDs or 'auto'.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-points", type=int, default=256)
    parser.add_argument("--extra-scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-every-n-epochs", type=int, default=1)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--max-epochs", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-every-n-steps", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--idx-mode", default="binary")
    parser.add_argument("--in-proj-type", default="linear")
    parser.add_argument("--clip-model-name", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--cond-dim", type=int, default=512)
    parser.add_argument("--sigma-txt", type=float, default=0.025)
    parser.add_argument("--sigma-perturb-std", type=float, default=0.25)
    parser.add_argument(
        "--conditioner-type",
        default="clip_flow",
        choices=["clip_flow", "gaussian_flow"],
    )
    parser.add_argument(
        "--load-weights-only",
        action="store_true",
        help="Load model weights without restoring optimizer or trainer state.",
    )
    return parser.parse_args()


def parse_devices(value: str) -> list[int] | str:
    value = value.strip()
    if value.lower() == "auto":
        return "auto"
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    train(
        accelerator=args.accelerator,
        batch_size=args.batch_size,
        cache_path=args.cache_path,
        checkpoint_dir=args.checkpoint_dir,
        ckpt_path=args.ckpt_path,
        clip_model_name=args.clip_model_name,
        clip_pretrained=args.clip_pretrained,
        cond_dim=args.cond_dim,
        conditioner_type=args.conditioner_type,
        data_path=args.data_path,
        devices=parse_devices(args.devices),
        extra_scale=args.extra_scale,
        idx_mode=args.idx_mode,
        in_proj_type=args.in_proj_type,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        load_weights_only=args.load_weights_only,
        log_name=args.log_name,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        n_points=args.n_points,
        num_workers=args.num_workers,
        precision=args.precision,
        sample_dir=args.sample_dir,
        sample_every_n_epochs=args.sample_every_n_epochs,
        save_every_n_steps=args.save_every_n_steps,
        seed=args.seed,
        sigma_perturb_std=args.sigma_perturb_std,
        sigma_txt=args.sigma_txt,
    )


if __name__ == "__main__":
    main()
