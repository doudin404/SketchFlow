from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


def _is_npz(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".npz"


def _iter_npz_files(root: Path) -> Iterator[Path]:
    yield from root.rglob("*.npz")


class QuickDrawSampleIterator:
    """Stream samples from Sketch-RNN QuickDraw NPZ files.

    A directory input may contain one category per subdirectory. The parent
    directory name is used as the category label. A single NPZ input may
    instead provide ``<split>_desc`` arrays for labels.
    """

    def __init__(self, input_path: str | Path, split: str = "train") -> None:
        self.input_path = Path(input_path)
        self.split = split
        self._single_file = _is_npz(self.input_path)

    def stats(self) -> tuple[int, int]:
        """Return the sample count and maximum label length."""
        total = 0
        max_label_len = 1

        if self._single_file:
            with np.load(
                self.input_path,
                allow_pickle=True,
                encoding="latin1",
            ) as data:
                if self.split not in data.files:
                    raise KeyError(
                        f"Split '{self.split}' is absent from {self.input_path}"
                    )
                total = len(data[self.split])
                description_key = f"{self.split}_desc"
                if description_key in data.files:
                    max_label_len = max(
                        max_label_len,
                        *(len(str(value)) for value in data[description_key]),
                    )
            return total, max_label_len

        for npz_path in sorted(_iter_npz_files(self.input_path)):
            label = npz_path.parent.name
            max_label_len = max(max_label_len, len(label))
            with np.load(
                npz_path,
                allow_pickle=True,
                encoding="latin1",
            ) as data:
                if self.split in data.files:
                    total += len(data[self.split])

        return total, max_label_len

    def __iter__(self) -> Iterator[tuple[int, Any, str]]:
        index = 0

        if self._single_file:
            with np.load(
                self.input_path,
                allow_pickle=True,
                encoding="latin1",
            ) as data:
                if self.split not in data.files:
                    raise KeyError(
                        f"Split '{self.split}' is absent from {self.input_path}"
                    )
                values = data[self.split]
                description_key = f"{self.split}_desc"
                descriptions = (
                    data[description_key]
                    if description_key in data.files
                    else None
                )
                for local_index, strokes in enumerate(values):
                    label = (
                        str(descriptions[local_index])
                        if descriptions is not None
                        else ""
                    )
                    yield index, strokes, label
                    index += 1
            return

        for npz_path in sorted(_iter_npz_files(self.input_path)):
            label = npz_path.parent.name
            with np.load(
                npz_path,
                allow_pickle=True,
                encoding="latin1",
            ) as data:
                if self.split not in data.files:
                    continue
                for strokes in data[self.split]:
                    yield index, strokes, label
                    index += 1


def list_image_like_files(
    root: str | Path,
    exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".webp", ".bmp"),
    vector_exts: Sequence[str] = (".svg", ".pdf"),
    recursive: bool = True,
) -> list[Path]:
    """List raster and vector image files in deterministic order."""
    root_path = Path(root)
    allowed = {
        *(extension.lower() for extension in exts),
        *(extension.lower() for extension in vector_exts),
    }
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in root_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in allowed
    )
