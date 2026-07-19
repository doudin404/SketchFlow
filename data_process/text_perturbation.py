from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_process.clip_tokens import clip_cache_path
from data_process.quickdraw_categories import qdraw_cat_cache_path
from data_process.quickdraw_sketch import quickdraw_cache_path
from utils.cache import cache_filename


def qdraw_txt_perturb_cache_path(
    input_dir: str | Path,
    cache_dir: str | Path | None,
    model_name: str,
    split_name: str,
    pretrained: str,
) -> Path:
    """Return the adaptive text-prior statistics cache path."""
    input_path = Path(input_dir)
    cache_root = Path(cache_dir) if cache_dir is not None else input_path
    name = cache_filename(
        "qdraw_txt_perturb",
        input_path.name,
        split_name,
        model_name,
        pretrained,
        prefix_count=5,
    )
    return cache_root / f"{name}.pt"


def _class_statistics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    block_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate per-class means and standard deviations in bounded memory."""
    class_to_index = {name: index for index, name in enumerate(class_names)}
    dimension = embeddings.shape[1]
    counts = np.zeros(len(class_names), dtype=np.int64)
    sums = np.zeros((len(class_names), dimension), dtype=np.float64)
    squared_sums = np.zeros_like(sums)

    for start in tqdm(
        range(0, len(labels), block_size),
        desc="Class statistics",
        unit="block",
    ):
        stop = min(start + block_size, len(labels))
        label_block = np.asarray(labels[start:stop]).astype(str)
        embedding_block = np.asarray(
            embeddings[start:stop],
            dtype=np.float32,
        )
        for class_name in np.unique(label_block):
            class_index = class_to_index.get(str(class_name))
            if class_index is None:
                continue
            selected = embedding_block[label_block == class_name]
            counts[class_index] += len(selected)
            sums[class_index] += selected.sum(axis=0, dtype=np.float64)
            squared_sums[class_index] += np.square(
                selected,
                dtype=np.float64,
            ).sum(axis=0)

    valid = counts > 0
    means = np.zeros_like(sums, dtype=np.float32)
    standard_deviations = np.zeros_like(sums, dtype=np.float32)
    means[valid] = (sums[valid] / counts[valid, None]).astype(np.float32)
    variances = np.zeros_like(sums)
    variances[valid] = (
        squared_sums[valid] / counts[valid, None]
        - np.square(means[valid], dtype=np.float64)
    )
    standard_deviations[valid] = np.sqrt(
        np.maximum(variances[valid], 0.0)
    ).astype(np.float32)
    return counts, means, standard_deviations


def build_qdraw_text_perturbation_cache(
    input_dir: str | Path,
    cache_dir: str | Path,
    split_name: str = "train",
    n_points: int = 256,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    force_rebuild: bool = False,
) -> Path:
    """Estimate text-side GMM scales from rendered-sketch CLIP statistics."""
    input_path = Path(input_dir)
    cache_root = Path(cache_dir)
    output_path = qdraw_txt_perturb_cache_path(
        input_path,
        cache_root,
        model_name,
        split_name,
        pretrained,
    )
    if output_path.exists() and not force_rebuild:
        print(f"Text-prior statistics already exist: {output_path}")
        return output_path

    _, image_embedding_path, _ = clip_cache_path(
        input_path,
        cache_root,
        "quickdraw",
        model_name,
        split_name,
        pretrained,
    )
    _, label_path = quickdraw_cache_path(
        input_path,
        n_points,
        split_name,
        cache_root,
    )
    text_embedding_path, text_index_path = qdraw_cat_cache_path(
        cache_root,
        model_name,
    )
    required = (
        image_embedding_path,
        label_path,
        text_embedding_path,
        text_index_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prerequisite caches:\n" + "\n".join(missing)
        )

    image_embeddings = np.load(image_embedding_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    text_embeddings = np.load(text_embedding_path, mmap_mode="r")
    text_index = torch.load(text_index_path, weights_only=False)
    class_names = sorted(text_index, key=text_index.get)

    counts, image_means, image_stds = _class_statistics(
        image_embeddings,
        labels,
        class_names,
    )
    valid_indices = np.flatnonzero(counts > 0)
    if len(valid_indices) < 2:
        raise RuntimeError("At least two populated categories are required")

    text_targets = np.asarray(
        text_embeddings[valid_indices],
        dtype=np.float32,
    )
    image_targets = image_means[valid_indices]
    centered_image = image_targets - image_targets.mean(axis=0)
    centered_text = text_targets - text_targets.mean(axis=0)
    denominator = float(np.sum(centered_image * centered_image))
    scale = (
        float(np.sum(centered_image * centered_text) / denominator)
        if denominator > 1e-8
        else 1.0
    )
    offset = text_targets.mean(axis=0) - scale * image_targets.mean(axis=0)

    text_stds_per_class = {
        class_names[index]: image_stds[index] * abs(scale)
        for index in valid_indices
    }
    average_text_std = np.mean(
        image_stds[valid_indices],
        axis=0,
    ) * abs(scale)
    image_class_means = {
        class_names[index]: image_means[index]
        for index in valid_indices
    }

    payload = {
        "txt_stds_per_class": text_stds_per_class,
        "avg_txt_std": average_text_std.astype(np.float32),
        "k": scale,
        "b": offset.astype(np.float32),
        "img_class_means": image_class_means,
        "class_counts": {
            class_names[index]: int(counts[index])
            for index in valid_indices
        },
        "clip_model": model_name,
        "clip_pretrained": pretrained,
        "split": split_name,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f".{os.getpid()}.tmp.pt")
    try:
        torch.save(payload, temporary_path)
        temporary_path.rename(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Text-prior statistics saved: {output_path}")
    return output_path
