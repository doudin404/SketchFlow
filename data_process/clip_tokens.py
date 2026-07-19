from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_process.quickdraw_sketch import build_quickdraw_cache
from utils.cache import cache_filename
from utils.clip_manager import get_clip_manager
from utils.visualize import StrokeRenderer


class QuickDrawClipDataset(Dataset[dict[str, torch.Tensor]]):
    """Render cached vector trajectories and apply CLIP preprocessing."""

    def __init__(
        self,
        trajectories: np.ndarray,
        transform,
        start_index: int = 0,
    ) -> None:
        self.trajectories = trajectories
        self.transform = transform
        self.start_index = start_index
        self.renderer = StrokeRenderer()

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        trajectory = self.trajectories[index]
        image = self.renderer.render(
            trajectory[:, :2],
            trajectory[:, 2],
            output="pil",
        )
        return {
            "index": torch.tensor(self.start_index + index, dtype=torch.long),
            "image": self.transform(image),
        }


def clip_cache_path(
    input_dir: str | Path,
    cache_dir: str | Path | None,
    mode: str,
    model_name: str,
    split_name: str,
    pretrained: str,
) -> tuple[Path, Path, Path]:
    """Return token, embedding, and metadata cache paths."""
    input_path = Path(input_dir)
    cache_root = Path(cache_dir) if cache_dir is not None else input_path
    cache_name = cache_filename(
        f"clip_{mode}",
        input_path.name,
        model_name,
        split_name,
        pretrained,
        prefix_count=4,
    )
    return (
        cache_root / f"{cache_name}.clip_tok.npy",
        cache_root / f"{cache_name}.clip_emb.npy",
        cache_root / f"{cache_name}.meta.npz",
    )


@torch.no_grad()
def _encode_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    save_tokens: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    embeddings = model.encode_image(images)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    tokens = None
    if save_tokens:
        intermediate = model.visual.forward_intermediates(images)
        feature_map = intermediate["image_intermediates"][-1]
        tokens = feature_map.flatten(2).transpose(1, 2)
    return embeddings, tokens


def _resume_index(path: Path) -> int:
    values = np.load(path, mmap_mode="r")
    row_sums = np.abs(values).sum(axis=tuple(range(1, values.ndim)))
    nonzero = np.flatnonzero(row_sums > 0)
    return int(nonzero[-1] + 1) if len(nonzero) else 0


def build_clip_cache(
    input_dir: str | Path,
    split_name: str = "all",
    cache_dir: str | Path | None = None,
    batch_size: int = 128,
    num_workers: int = 8,
    precision: str = "16-mixed",
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    mode: str = "quickdraw",
    n_points: int = 256,
    start_idx: int = 0,
    save_embeddings: bool = True,
    save_tokens: bool = False,
    gpu_id: int = 0,
    **_: object,
) -> tuple[Path | None, Path | None, Path]:
    """Build normalized CLIP caches for rendered QuickDraw trajectories.

    Set ``start_idx=-1`` to resume after the last nonzero embedding row.
    """
    if mode != "quickdraw":
        raise ValueError("The public cache builder supports mode='quickdraw'")
    if not save_embeddings and not save_tokens:
        raise ValueError("Enable at least one of save_embeddings or save_tokens")

    input_path = Path(input_dir)
    cache_root = Path(cache_dir) if cache_dir is not None else input_path
    cache_root.mkdir(parents=True, exist_ok=True)
    token_path, embedding_path, metadata_path = clip_cache_path(
        input_path,
        cache_root,
        mode,
        model_name,
        split_name,
        pretrained,
    )

    requested_paths = [
        path
        for enabled, path in (
            (save_embeddings, embedding_path),
            (save_tokens, token_path),
        )
        if enabled
    ]
    if start_idx == 0 and requested_paths and all(path.exists() for path in requested_paths):
        print(f"CLIP cache already exists for split '{split_name}'")
        return (
            token_path if save_tokens else None,
            embedding_path if save_embeddings else None,
            metadata_path,
        )
    if start_idx == -1:
        existing = embedding_path if save_embeddings else token_path
        if not existing.exists():
            start_idx = 0
        else:
            start_idx = _resume_index(existing)
            print(f"Resuming CLIP cache at sample {start_idx}")

    trajectory_path, _ = build_quickdraw_cache(
        input_path,
        n_points=n_points,
        split=split_name,
        cache_dir=cache_root,
    )
    trajectories = np.load(trajectory_path, mmap_mode="r")
    total = len(trajectories)
    if start_idx >= total:
        print(f"CLIP cache is complete ({total} samples)")
        return (
            token_path if save_tokens else None,
            embedding_path if save_embeddings else None,
            metadata_path,
        )

    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    manager = get_clip_manager(model_name, pretrained, device)
    dataset = QuickDrawClipDataset(
        trajectories[start_idx:],
        manager.preprocess,
        start_index=start_idx,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )

    fresh_write = start_idx == 0
    process_id = os.getpid()
    working_embedding_path = (
        embedding_path.with_suffix(f"{embedding_path.suffix}.{process_id}.tmp")
        if fresh_write and save_embeddings
        else embedding_path
    )
    working_token_path = (
        token_path.with_suffix(f"{token_path.suffix}.{process_id}.tmp")
        if fresh_write and save_tokens
        else token_path
    )
    working_metadata_path = (
        metadata_path.parent / f"{metadata_path.stem}.{process_id}.tmp.npz"
        if fresh_write
        else metadata_path
    )

    embedding_mmap: np.memmap | None = None
    token_mmap: np.memmap | None = None
    embedding_dim = 0
    token_shape: tuple[int, int] | None = None

    try:
        for batch in tqdm(loader, desc=f"CLIP {split_name}", unit="batch"):
            indices = batch["index"].numpy()
            images = batch["image"].to(device, non_blocking=True)
            embeddings, tokens = _encode_batch(
                manager.model,
                images,
                save_tokens=save_tokens,
            )
            embeddings_np = embeddings.cpu().numpy().astype(np.float16)
            embedding_dim = embeddings_np.shape[1]

            if save_embeddings and embedding_mmap is None:
                if fresh_write:
                    embedding_mmap = np.lib.format.open_memmap(
                        working_embedding_path,
                        mode="w+",
                        dtype=np.float16,
                        shape=(total, embedding_dim),
                    )
                else:
                    embedding_mmap = np.load(
                        working_embedding_path,
                        mmap_mode="r+",
                    )
            if embedding_mmap is not None:
                embedding_mmap[indices] = embeddings_np

            if save_tokens and tokens is not None:
                tokens_np = tokens.cpu().numpy().astype(np.float16)
                token_shape = (tokens_np.shape[1], tokens_np.shape[2])
                if token_mmap is None:
                    if fresh_write:
                        token_mmap = np.lib.format.open_memmap(
                            working_token_path,
                            mode="w+",
                            dtype=np.float16,
                            shape=(total, *token_shape),
                        )
                    else:
                        token_mmap = np.load(
                            working_token_path,
                            mmap_mode="r+",
                        )
                token_mmap[indices] = tokens_np

        if embedding_mmap is not None:
            embedding_mmap.flush()
        if token_mmap is not None:
            token_mmap.flush()
        del embedding_mmap
        del token_mmap

        np.savez_compressed(
            working_metadata_path,
            input_dir=str(input_path),
            model_name=model_name,
            pretrained=pretrained,
            split_name=split_name,
            n_items=total,
            embedding_dim=embedding_dim,
            token_shape=token_shape,
            precision=precision,
        )

        if fresh_write:
            if save_embeddings:
                working_embedding_path.rename(embedding_path)
            if save_tokens:
                working_token_path.rename(token_path)
            working_metadata_path.rename(metadata_path)
    finally:
        if fresh_write:
            if save_embeddings:
                working_embedding_path.unlink(missing_ok=True)
            if save_tokens:
                working_token_path.unlink(missing_ok=True)
            working_metadata_path.unlink(missing_ok=True)

    return (
        token_path if save_tokens else None,
        embedding_path if save_embeddings else None,
        metadata_path,
    )
