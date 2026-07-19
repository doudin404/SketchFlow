"""Lightning module for SketchFlow training and inference."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from nets.clip_latent_flow import ClipLatentFlowConditioner
from nets.diffusion_unet import DiffusionTransformer
from nets.encoder import TextCondEncoder
from nets.gaussian_latent_flow import GaussianLatentFlowConditioner
from nets.pipelines import SketchPipeline
from utils.visualize import StrokeRenderer


class SketchFlowModel(pl.LightningModule):
    def __init__(
        self,
        data_dim: int = 3,
        model_dim: int = 128,
        depth: int = 12,
        num_heads: int = 8,
        cond_dim: int = 512,
        max_steps: int = 1000,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        loss_mode: str = "mse",
        sample_every_n_epochs: int = 100,
        sample_dir: str = "val_samples",
        sample_num: int = 16,
        sample_num_steps: Optional[int] = None,
        idx_mode: str = "weighted",
        extra_scale: float = 1.0,
        use_text_cond: bool = False,
        text_model_name: Optional[str] = None,
        text_max_len: int = 10,
        freeze_text: bool = True,
        precomputed_text_hidden_dim: Optional[int] = 512,
        use_point_encoder: bool = False,
        point_in_dim: Optional[int] = None,
        point_hidden: int = 128,
        point_out_dim: int = 64,
        in_proj_type: str = "linear",
        sample_intermediate_steps: Optional[int] = 1,
        unet_dim_mults: Tuple[float, ...] = (1, 1.5, 2, 4),
        unet_depths: Tuple[int, ...] = (3, 3, 3, 3),
        unet_heads: Tuple[int, ...] = (4, 4, 8, 8),
        cond_tok_dim: Optional[int] = None,
        cond_vec_in_dim: Optional[int] = None,
        use_latent_conditioner: bool = True,
        use_learnable_delta: bool = True,
        interp_alpha_beta: Tuple[float, float] = (3, 1),
        conditioner_type: str = "clip_flow",
        flow_width: int = 1024,
        flow_depth: int = 10,
        t_eps: float = 0.02,
        sigma_txt: float = 0.025,
        sigma_perturb_std: float = 0.25,
        sigma_img: float = 0.001,
        flow_l2: float = 1e-6,
        use_dynamic_noise: bool = False,
        dynamic_noise_scale: float = 2.7,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = DiffusionTransformer(
            data_dim=data_dim,
            model_dim=model_dim,
            depth=depth,
            num_heads=num_heads,
            cond_in_dim=cond_tok_dim,
            cond_dim=cond_dim,
            max_steps=max_steps,
            use_cross_attn=use_text_cond,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_pos_emb=idx_mode != "weighted",
            in_proj_type=in_proj_type,
            dim_mults=unet_dim_mults,
            depths=unet_depths,
            heads=unet_heads,
        )
        self.scheduler = DDPMScheduler(
            num_train_timesteps=max_steps,
            prediction_type="epsilon",
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
        )
        self.pipeline = SketchPipeline(unet=self.model, scheduler=self.scheduler)

        self.cond_vec_proj = (
            nn.Linear(cond_vec_in_dim, cond_dim)
            if cond_vec_in_dim is not None
            else None
        )
        self.text_encoder = None
        self.cond_proj = None
        if use_text_cond:
            if text_model_name:
                self.text_encoder = TextCondEncoder(
                    model_name=text_model_name,
                    max_length=text_max_len,
                    trainable=not freeze_text,
                )
                self.cond_proj = nn.Linear(
                    self.text_encoder.encoder.config.hidden_size,
                    cond_tok_dim or model_dim,
                )
            elif precomputed_text_hidden_dim is not None:
                self.cond_proj = nn.Linear(
                    precomputed_text_hidden_dim,
                    cond_tok_dim or model_dim,
                )

        if use_latent_conditioner:
            conditioner_args = {
                "dim": cond_vec_in_dim or cond_dim,
                "width": flow_width,
                "depth": flow_depth,
                "t_eps": t_eps,
                "sigma_txt": sigma_txt,
                "sigma_img": sigma_img,
                "flow_l2": flow_l2,
                "sigma_perturb_std": sigma_perturb_std,
            }
            kind = conditioner_type.lower().replace("-", "_")
            if kind in {"clip_flow", "clip_latent_flow", "gmm_flow", "flow"}:
                self.latent_conditioner = ClipLatentFlowConditioner(
                    **conditioner_args
                )
            elif kind in {"gaussian_flow", "gaussian", "normal"}:
                self.latent_conditioner = GaussianLatentFlowConditioner(
                    **conditioner_args
                )
            else:
                raise ValueError(
                    f"Unknown conditioner_type '{conditioner_type}'. "
                    "Use 'clip_flow' or 'gaussian_flow'."
                )
        else:
            self.latent_conditioner = None

        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.loss_mode = loss_mode
        self.sample_every_n_epochs = int(sample_every_n_epochs)
        self.sample_dir = Path(sample_dir)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.sample_num = int(sample_num)
        self.sample_num_steps = sample_num_steps
        self.prediction_type = self.scheduler.config.prediction_type
        self.idx_mode = idx_mode
        self.extra_scale = extra_scale
        self.sample_intermediate_steps = sample_intermediate_steps
        self.use_text_cond = use_text_cond
        self.use_dynamic_noise = use_dynamic_noise
        self.dynamic_noise_scale = dynamic_noise_scale
        self._val_ref_shape = None
        self._val_ref_vec = None
        self._val_ref_text_cond = None

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        *args,
        **kwargs,
    ):
        """Load current checkpoints and migrate the original conditioner prefix."""
        migrated_state = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith("random_interp."):
                key = key.replace("random_interp.", "latent_conditioner.", 1)
            migrated_state[key] = value
        return super().load_state_dict(
            migrated_state,
            strict=strict,
            *args,
            **kwargs,
        )

    def _alpha_bar(self, timesteps: torch.LongTensor) -> torch.Tensor:
        return self.scheduler.alphas_cumprod.to(timesteps.device)[timesteps].view(
            -1, 1, 1
        )

    def _unpack_batch(self, batch):
        points = batch["points"]
        extra = batch["extra"] * self.extra_scale
        x0 = torch.cat([points, extra], dim=-1)
        return (
            x0,
            batch.get("cond_vec"),
            batch.get("cond_vec_image"),
            batch.get("cond_vec_std"),
        )

    def _cond_from_batch(self, batch):
        x0, text_features, image_features, feature_std = self._unpack_batch(batch)
        cond_loss = 0.0
        if (
            self.latent_conditioner is not None
            and image_features is not None
            and text_features is not None
        ):
            if self.use_dynamic_noise and feature_std is not None:
                cond_vec, cond_loss = self.latent_conditioner(
                    image_features,
                    text_features,
                    cond_vec_std=feature_std * self.dynamic_noise_scale,
                )
            else:
                cond_vec, cond_loss = self.latent_conditioner(
                    image_features,
                    text_features,
                )
        else:
            cond_vec = text_features

        if cond_vec is not None and self.cond_vec_proj is not None:
            cond_vec = self.cond_vec_proj(cond_vec)
        return x0, cond_vec, cond_loss

    def _compute_loss(self, batch):
        x0, cond_vec, cond_loss = self._cond_from_batch(batch)
        batch_size = x0.size(0)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=x0.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(x0)
        noisy = self.scheduler.add_noise(x0, noise, timesteps)
        prediction = self.model.predict(noisy, timesteps, cond_vec=cond_vec)

        if self.loss_mode == "mse":
            diffusion_loss = F.mse_loss(prediction, noise)
        elif self.loss_mode == "mse_denoised":
            alpha_bar = self._alpha_bar(timesteps)
            x0_prediction = (
                noisy - (1.0 - alpha_bar).sqrt() * prediction
            ) / (alpha_bar.sqrt() + 1e-8)
            weights = (
                alpha_bar / (1.0 - alpha_bar + 1e-8)
            ).flatten(start_dim=1).mean(dim=1)
            per_sample = F.mse_loss(
                x0_prediction,
                x0,
                reduction="none",
            ).mean(dim=(1, 2))
            diffusion_loss = (weights * per_sample).mean()
        else:
            raise ValueError(f"Unknown loss_mode: {self.loss_mode}")
        return diffusion_loss + cond_loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["points"].size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch["points"].size(0),
        )
        if batch_idx == 0:
            x0, cond_vec, _ = self._cond_from_batch(batch)
            self._val_ref_shape = (x0.shape[1], x0.shape[2])
            self._val_ref_vec = cond_vec
            self._val_ref_text_cond = batch.get("text_cond")
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.sample_every_n_epochs <= 0:
            return
        if (self.current_epoch + 1) % self.sample_every_n_epochs:
            return
        if self._val_ref_shape is None or not self.trainer.is_global_zero:
            return

        length, channels = self._val_ref_shape
        cond_vec = self._val_ref_vec
        batch_size = min(
            self.sample_num,
            cond_vec.shape[0] if cond_vec is not None else self.sample_num,
        )
        if cond_vec is not None:
            cond_vec = cond_vec[:batch_size].to(self.device)
        intermediate_count = int(self.sample_intermediate_steps or 1)

        self.model.eval()
        samples = self.pipeline(
            shape=(batch_size, length, channels),
            num_inference_steps=(
                self.sample_num_steps
                or self.scheduler.config.num_train_timesteps
            ),
            return_n_intermediate=max(intermediate_count, 1),
            cond_vec=cond_vec,
            mode="diffusion",
            show_progress_bar=False,
        )

        renderer = StrokeRenderer(idx_mode=self.idx_mode)
        renderer.cfg.min_alpha = 0
        for step_index, sample_batch in enumerate(samples, start=1):
            values = sample_batch.clone()
            if self.extra_scale != 1.0:
                values[..., 2:] /= self.extra_scale
            for sample_index in range(batch_size):
                label = ""
                if self._val_ref_text_cond is not None:
                    label = str(self._val_ref_text_cond[sample_index])
                safe_label = "".join(
                    char if char.isalnum() or char in "-_" else "_"
                    for char in label
                )
                output_path = self.sample_dir / (
                    f"epoch{self.current_epoch + 1:04d}_"
                    f"sample{sample_index + 1:02d}_"
                    f"step{step_index:02d}_{safe_label}.png"
                )
                renderer.render(
                    values[sample_index, :, :2].detach().cpu().numpy(),
                    values[sample_index, :, 2:].detach().cpu().numpy(),
                    out_path=str(output_path),
                    return_tensor=True,
                    invert_y=True,
                )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / 5000.0),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @torch.no_grad()
    def sample(
        self,
        B: int,
        mode: str = "diffusion",
        return_n_intermediate: Optional[int] = 1,
        cond_vec: Optional[torch.Tensor] = None,
        cond_vec_std: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        n_point: int = 256,
        num_steps: Optional[int] = None,
        variance_scale: float = 0.0,
        flow_alpha: float = 1.0,
        sigma_txt_override: Optional[float] = None,
    ):
        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )
        steps = int(
            num_steps
            or self.sample_num_steps
            or self.scheduler.config.num_train_timesteps
        )

        cond_embed = None
        if cond_vec is not None:
            cond_embed = cond_vec.to(self.device)
            if self.latent_conditioner is not None:
                shift_kwargs = {
                    "variance_scale": variance_scale,
                    "flow_alpha": flow_alpha,
                    "steps": steps,
                    "sigma_txt_override": sigma_txt_override,
                }
                if self.use_dynamic_noise and cond_vec_std is not None:
                    shift_kwargs["cond_vec_std"] = (
                        cond_vec_std.to(self.device) * self.dynamic_noise_scale
                    )
                cond_embed = self.latent_conditioner.shift(
                    cond_embed,
                    **shift_kwargs,
                )
            if self.cond_vec_proj is not None:
                cond_embed = self.cond_vec_proj(cond_embed)

        return self.pipeline(
            shape=(B, int(n_point), int(self.model.data_dim)),
            mode=mode,
            num_inference_steps=steps,
            generator=generator,
            cond_vec=cond_embed,
            return_n_intermediate=max(int(return_n_intermediate or 1), 1),
        )
