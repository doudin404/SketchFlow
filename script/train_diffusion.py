"""PyTorch Lightning training entry point for SketchFlow."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from dataset.SketchesDataset import SketchesDataModule
from model.sketchflow_model import SketchFlowModel


def train(
    *,
    accelerator: str = "auto",
    batch_size: int = 32,
    cache_path: str = "cache/quickdraw",
    checkpoint_dir: str | None = None,
    ckpt_path: str | None = None,
    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    cond_dim: int = 512,
    conditioner_type: str = "clip_flow",
    data_path: str = "data/quickdraw",
    devices: list[int] | str = "auto",
    extra_scale: float = 0.1,
    idx_mode: str = "binary",
    in_proj_type: str = "linear",
    limit_train_batches: float = 1.0,
    limit_val_batches: float = 1.0,
    load_weights_only: bool = False,
    log_name: str = "sketchflow",
    loss_mode: str = "mse_denoised",
    max_epochs: int = -1,
    max_steps: int = -1,
    n_points: int = 256,
    num_workers: int = 4,
    precision: str = "32-true",
    sample_dir: str = "val_samples/sketchflow",
    sample_every_n_epochs: int = 1,
    save_every_n_steps: int = 0,
    seed: int = 42,
    sigma_perturb_std: float = 0.25,
    sigma_txt: float = 0.025,
) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(seed, workers=True)

    data_module = SketchesDataModule(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        n_points=n_points,
        cache_path=cache_path,
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
    )
    model = SketchFlowModel(
        cond_dim=cond_dim,
        conditioner_type=conditioner_type,
        extra_scale=extra_scale,
        idx_mode=idx_mode,
        in_proj_type=in_proj_type,
        loss_mode=loss_mode,
        sample_dir=sample_dir,
        sample_every_n_epochs=sample_every_n_epochs,
        sigma_perturb_std=sigma_perturb_std,
        sigma_txt=sigma_txt,
    )

    resume_path = ckpt_path
    if load_weights_only:
        if not ckpt_path:
            raise ValueError("--load-weights-only requires --ckpt-path.")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
        resume_path = None

    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else None
    if save_every_n_steps > 0:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_root,
            every_n_train_steps=save_every_n_steps,
            filename="step-{step}",
            monitor=None,
            save_last=True,
            save_on_train_epoch_end=False,
            save_top_k=-1,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_root,
            every_n_epochs=1,
            filename="best-model",
            monitor="val/loss",
            mode="min",
            save_last=True,
            save_top_k=1,
        )

    trainer = pl.Trainer(
        accelerator=accelerator,
        callbacks=[checkpoint_callback],
        devices=devices,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=10,
        logger=TensorBoardLogger("tb_logs", name=log_name),
        max_epochs=max_epochs,
        max_steps=max_steps,
        precision=precision,
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_path)


if __name__ == "__main__":
    train()
