"""Launch the SketchFlow Gradio demo."""

from __future__ import annotations

import argparse

from ui.app import create_app
from ui.inference import InferenceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the SketchFlow text-to-vector-sketch demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt-path",
        default="checkpoints/sketchflow_v1.ckpt",
    )
    parser.add_argument("--n-point", type=int, default=256)
    parser.add_argument("--extra-scale", type=float, default=0.1)
    parser.add_argument(
        "--conditioner-type",
        default="clip_flow",
        choices=["clip_flow", "gaussian_flow"],
    )
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--text-std-cache", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-dir", default="ui_results")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = InferenceEngine(
        ckpt_path=args.ckpt_path,
        n_point=args.n_point,
        extra_scale=args.extra_scale,
        conditioner_type=args.conditioner_type,
        txt_perturb_cache_path=args.text_std_cache,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=args.device,
    )
    app = create_app(engine=engine, default_save_dir=args.save_dir)
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
