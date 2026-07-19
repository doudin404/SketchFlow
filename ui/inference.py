"""Model loading, CLIP conditioning, generation, rendering, and export."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image

from model.sketchflow_model import SketchFlowModel
from utils.clip_manager import get_clip_manager
from utils.visualize import StrokeRenderer


def _slugify(value: str | None, max_length: int = 40) -> str:
    value = str(value or "unconditional").lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[-\s]+", "-", value).strip("-_")
    return (value or "unnamed")[:max_length]


def _to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu()
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = array.permute(1, 2, 0)
    if array.ndim == 2:
        array = array.unsqueeze(-1)
    array = array.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    if array.shape[-1] == 1:
        return Image.fromarray(array[..., 0], mode="L")
    return Image.fromarray(array[..., :3], mode="RGB")


class InferenceEngine:
    def __init__(
        self,
        ckpt_path: str,
        n_point: int = 256,
        extra_scale: float = 0.1,
        device: Optional[str] = None,
        conditioner_type: str = "clip_flow",
        txt_perturb_cache_path: Optional[str] = None,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
    ) -> None:
        torch.set_float32_matmul_precision("medium")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_point = int(n_point)
        self.clip_manager = get_clip_manager(
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            device=self.device,
        )
        self.model = SketchFlowModel(
            conditioner_type=conditioner_type,
            extra_scale=extra_scale,
            idx_mode="binary",
            in_proj_type="linear",
            loss_mode="mse_denoised",
            sample_dir="val_samples",
            sample_every_n_epochs=1,
            sigma_txt=0.025,
            sigma_perturb_std=0.25,
        )
        self._load_checkpoint(ckpt_path)
        self.model.to(self.device).eval()
        self.renderer = StrokeRenderer(idx_mode="binary")
        self.avg_txt_std = self._load_text_std(txt_perturb_cache_path)

    def _load_checkpoint(self, ckpt_path: str) -> None:
        path = Path(ckpt_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}. "
                "Download the release checkpoint or pass --ckpt-path."
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)

    def _load_text_std(self, cache_path: Optional[str]) -> Optional[torch.Tensor]:
        if not cache_path:
            return None
        path = Path(cache_path)
        if not path.is_file():
            raise FileNotFoundError(f"Text perturbation cache not found: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        if "avg_txt_std" in data:
            values = np.asarray(data["avg_txt_std"], dtype=np.float32)
        elif data.get("txt_stds_per_class"):
            values = np.mean(
                [
                    np.asarray(value, dtype=np.float32)
                    for value in data["txt_stds_per_class"].values()
                ],
                axis=0,
            )
        else:
            raise KeyError(
                f"{path} contains neither avg_txt_std nor txt_stds_per_class."
            )
        return torch.as_tensor(values, device=self.device)

    def _encode_condition(
        self,
        text: Optional[str],
        image: Optional[Image.Image],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if text and text.strip():
            return self.clip_manager.encode_text([text.strip()] * batch_size)
        if image is not None:
            image_tensor = self.clip_manager.preprocess(image).unsqueeze(0)
            return self.clip_manager.encode_image(
                image_tensor.repeat(batch_size, 1, 1, 1)
            )
        return None

    @torch.no_grad()
    def generate(
        self,
        text: Optional[str] = None,
        image: Optional[Image.Image] = None,
        B: int = 1,
        n_intermediate: int = 0,
        seed: Optional[int] = None,
        varscale: float = 1.0,
        second_text: Optional[str] = None,
        second_image: Optional[Image.Image] = None,
        flow_alpha: float = 1.0,
        denoising_steps: int = 60,
        sigma_txt_override: Optional[float] = None,
    ) -> tuple[list[Image.Image], torch.Tensor]:
        batch_size = int(B)
        cond_vec = self._encode_condition(text, image, batch_size)
        second_vec = self._encode_condition(
            second_text,
            second_image,
            batch_size,
        )
        if cond_vec is not None and second_vec is not None:
            alpha = torch.linspace(
                0,
                1,
                batch_size,
                device=self.device,
            ).unsqueeze(1)
            cond_vec = (1 - alpha) * cond_vec + alpha * second_vec
        elif second_vec is not None:
            cond_vec = second_vec
        if cond_vec is None:
            cond_vec = self.clip_manager.encode_text([""] * batch_size)

        cond_vec_std = None
        if self.avg_txt_std is not None:
            cond_vec_std = self.avg_txt_std.unsqueeze(0).repeat(batch_size, 1)

        generated = self.model.sample(
            B=batch_size,
            return_n_intermediate=n_intermediate,
            cond_vec=cond_vec,
            cond_vec_std=cond_vec_std,
            seed=seed,
            n_point=self.n_point,
            variance_scale=float(varscale),
            flow_alpha=float(flow_alpha),
            num_steps=int(denoising_steps),
            sigma_txt_override=sigma_txt_override,
        )
        strokes = torch.cat(generated, dim=0)
        images = []
        for sample in strokes:
            rendered = self.renderer.render(
                sample[:, :2],
                sample[:, 2:],
                return_tensor=True,
                extra_scale=self.model.extra_scale,
            )
            images.append(_to_pil(rendered))

        strokes = strokes.detach().cpu()
        strokes[..., 2] /= self.model.extra_scale
        return images, strokes

    def save_strokes(
        self,
        raw_strokes: Optional[torch.Tensor],
        save_dir: str,
        prompt: Optional[str] = None,
        b: int = 0,
        seed: Optional[int] = None,
        varscale: float = 1.0,
        flow_alpha: float = 1.0,
        selected_indices: Optional[Sequence[int]] = None,
    ) -> str:
        if raw_strokes is None:
            return "Generate sketches before exporting."

        array = torch.as_tensor(raw_strokes)[..., :3].detach().cpu().numpy()
        if selected_indices:
            indices = sorted(set(int(index) for index in selected_indices))
            if min(indices) < 0 or max(indices) >= array.shape[0]:
                return f"Selected index is outside 0..{array.shape[0] - 1}."
            array = array[indices]
        else:
            indices = list(range(array.shape[0]))

        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        seed_value = "random" if seed is None else str(seed)
        selection = "-".join(str(index) for index in indices[:8])
        if len(indices) > 8:
            selection = f"{len(indices)}-samples"
        name = (
            f"{_slugify(prompt, 30)}_seed-{seed_value}_"
            f"var-{varscale:.2f}_flow-{flow_alpha:.2f}_sel-{selection}.npy"
        )
        output_path = output_dir / name
        if output_path.exists():
            output_path = output_dir / f"{output_path.stem}_{int(time.time())}.npy"
        np.save(output_path, array)
        return f"Saved {output_path}"
