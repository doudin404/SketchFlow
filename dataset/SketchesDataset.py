from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from data_process.clip_tokens import clip_cache_path
from data_process.quickdraw_categories import qdraw_cat_cache_path
from data_process.quickdraw_sketch import quickdraw_cache_path
from data_process.text_perturbation import qdraw_txt_perturb_cache_path


def _load_average_text_std(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "avg_txt_std" in payload:
        return np.asarray(payload["avg_txt_std"], dtype=np.float32)
    per_class = payload.get("txt_stds_per_class", {})
    if per_class:
        return np.mean(list(per_class.values()), axis=0).astype(np.float32)
    return None


class SketchesDataset(Dataset[dict[str, Any]]):
    """Read the precomputed trajectory and CLIP caches for one split."""

    def __init__(
        self,
        data_path: str | Path,
        cache_dir: str | Path,
        split: str,
        n_points: int,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
    ) -> None:
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir)
        self.split = split

        points_path, labels_path = quickdraw_cache_path(
            self.data_path,
            n_points,
            split,
            self.cache_dir,
        )
        _, image_embedding_path, _ = clip_cache_path(
            input_dir=self.data_path,
            cache_dir=self.cache_dir,
            mode="quickdraw",
            model_name=clip_model_name,
            split_name=split,
            pretrained=clip_pretrained,
        )
        text_embedding_path, text_index_path = qdraw_cat_cache_path(
            self.cache_dir,
            clip_model_name,
        )
        text_std_path = qdraw_txt_perturb_cache_path(
            self.data_path,
            self.cache_dir,
            clip_model_name,
            split,
            clip_pretrained,
        )

        required = {
            "trajectory": points_path,
            "label": labels_path,
            "image embedding": image_embedding_path,
            "text embedding": text_embedding_path,
            "text index": text_index_path,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing SketchFlow caches. Run `python -m script.data_prepare` first:\n"
                + "\n".join(missing)
            )

        self.points = np.load(points_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.image_embeddings = np.load(image_embedding_path, mmap_mode="r")
        self.text_embeddings = np.load(text_embedding_path, mmap_mode="r")
        self.text_index = torch.load(text_index_path, weights_only=False)
        self.average_text_std = _load_average_text_std(text_std_path)

        if not (
            len(self.points)
            == len(self.labels)
            == len(self.image_embeddings)
        ):
            raise ValueError("Cache lengths differ; rebuild all caches together")

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int) -> dict[str, Any]:
        trajectory = self.points[index].astype(np.float32)
        label = str(self.labels[index])
        text_index = self.text_index[label]
        sample: dict[str, Any] = {
            "points": torch.from_numpy(trajectory[:, :2]),
            "extra": torch.from_numpy(trajectory[:, 2:]),
            "text_cond": label,
            "cond_vec": torch.from_numpy(
                self.text_embeddings[text_index].astype(np.float32)
            ),
            "cond_vec_image": torch.from_numpy(
                self.image_embeddings[index].astype(np.float32)
            ),
        }
        if self.average_text_std is not None:
            sample["cond_vec_std"] = torch.from_numpy(self.average_text_std)
        return sample


class SketchesDataModule(pl.LightningDataModule):
    """Lightning data module for cached QuickDraw trajectories."""

    def __init__(
        self,
        data_path: str,
        cache_path: str,
        n_points: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.cache_path = cache_path
        self.n_points = n_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.train_dataset: SketchesDataset | None = None
        self.val_dataset: SketchesDataset | None = None
        self.test_dataset: SketchesDataset | None = None

    def _dataset(self, split: str) -> SketchesDataset:
        return SketchesDataset(
            data_path=self.data_path,
            cache_dir=self.cache_path,
            split=split,
            n_points=self.n_points,
            clip_model_name=self.clip_model_name,
            clip_pretrained=self.clip_pretrained,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._dataset("train")
            self.val_dataset = self._dataset("valid")
        if stage in (None, "test"):
            self.test_dataset = self._dataset("test")

    def _loader(
        self,
        dataset: Dataset[Any],
        *,
        shuffle: bool,
    ) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.train_dataset is None:
            raise RuntimeError("Data module has not been set up")
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        if self.val_dataset is None:
            raise RuntimeError("Data module has not been set up")
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        if self.test_dataset is None:
            raise RuntimeError("Data module has not been set up")
        return self._loader(self.test_dataset, shuffle=False)
