from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from utils.cache import cache_filename
from utils.load import QuickDrawSampleIterator
from utils.resample import resample_polylines


def npz_sample_to_polylines(strokes: np.ndarray) -> list[np.ndarray]:
    """Convert a Sketch-RNN delta sequence into absolute polylines."""
    if strokes.ndim != 2 or strokes.shape[1] not in (3, 5):
        raise ValueError("Expected a Sketch-RNN array with shape (N, 3) or (N, 5)")

    xy = np.cumsum(strokes[:, :2], axis=0)
    if strokes.shape[1] == 3:
        pen_up = np.isin(strokes[:, 2].astype(int), (1, 2))
    else:
        pen_up = (strokes[:, 3] == 1) | (strokes[:, 4] == 1)

    polylines: list[np.ndarray] = []
    start = 0
    for index in range(len(xy) - 1):
        if pen_up[index]:
            segment = xy[start : index + 1]
            if len(segment) >= 2:
                polylines.append(segment)
            start = index + 1

    final_segment = xy[start:]
    if len(final_segment) >= 2:
        polylines.append(final_segment)
    return polylines


def normalize_to_unit(polylines: list[np.ndarray]) -> list[np.ndarray]:
    """Center polylines and uniformly scale them into [-1, 1]."""
    points = np.concatenate(polylines, axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    scale = 2.0 / max(float(np.max(maximum - minimum)), 1e-6)
    return [(polyline - center) * scale for polyline in polylines]


def _process_one_sample_from_strokes(
    strokes_object: Any,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize and resample one sketch, returning points and pen states."""
    try:
        polylines = npz_sample_to_polylines(np.asarray(strokes_object))
        if not polylines:
            raise ValueError("Sketch has no valid polyline")
        resampled = resample_polylines(
            normalize_to_unit(polylines),
            n_points,
            keep_empty=False,
        )
        points = np.concatenate(resampled, axis=0).astype(np.float32, copy=False)
        if len(points) > n_points:
            points = points[:n_points]
        elif len(points) < n_points:
            points = np.pad(points, ((0, n_points - len(points)), (0, 0)))

        pen_state = np.zeros(n_points, dtype=np.float32)
        offset = 0
        for polyline in resampled:
            length = min(len(polyline), n_points - offset)
            if length > 1:
                pen_state[offset + 1 : offset + length] = 1.0
            offset += length
            if offset >= n_points:
                break
        return points, pen_state
    except Exception:
        return (
            np.zeros((n_points, 2), dtype=np.float32),
            np.zeros(n_points, dtype=np.float32),
        )


def quickdraw_cache_path(
    input_path: str | Path,
    n_points: int,
    split: str,
    cache_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return the point and label cache paths for a dataset split."""
    input_path = Path(input_path)
    cache_root = Path(cache_dir) if cache_dir is not None else input_path.parent
    name = cache_filename(input_path.name, n_points, split)
    return cache_root / f"{name}.npy", cache_root / f"{name}.labels.npy"


def build_quickdraw_cache(
    input_path: str | Path,
    n_points: int,
    split: str,
    cache_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Build a memory-mapped fixed-length trajectory cache."""
    data_path, labels_path = quickdraw_cache_path(
        input_path,
        n_points,
        split,
        cache_dir,
    )
    if data_path.exists() and labels_path.exists():
        print(f"QuickDraw cache already exists: {data_path}")
        return data_path, labels_path

    data_path.parent.mkdir(parents=True, exist_ok=True)
    process_id = os.getpid()
    temporary_data = data_path.with_suffix(f"{data_path.suffix}.{process_id}.tmp")
    temporary_labels = labels_path.with_suffix(
        f"{labels_path.suffix}.{process_id}.tmp"
    )

    try:
        iterator = QuickDrawSampleIterator(input_path, split=split)
        sample_count, max_label_length = iterator.stats()
        if sample_count <= 0:
            raise RuntimeError(
                f"No samples found at {input_path} for split '{split}'"
            )

        data_mmap = np.lib.format.open_memmap(
            temporary_data,
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, n_points, 3),
        )
        labels_mmap = np.lib.format.open_memmap(
            temporary_labels,
            mode="w+",
            dtype=f"<U{max(1, max_label_length)}",
            shape=(sample_count,),
        )

        worker_count = min(max(1, os.cpu_count() or 1), 48)
        in_flight_limit = worker_count * 4
        futures: dict[Any, tuple[int, str]] = {}
        source = iter(iterator)

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            def fill_queue() -> None:
                while len(futures) < in_flight_limit:
                    try:
                        index, strokes, label = next(source)
                    except StopIteration:
                        break
                    future = executor.submit(
                        _process_one_sample_from_strokes,
                        strokes,
                        n_points,
                    )
                    futures[future] = (index, label)

            fill_queue()
            with tqdm(
                total=sample_count,
                desc=f"Resampling {split}",
                unit="sketch",
            ) as progress:
                while futures:
                    for future in as_completed(list(futures)):
                        index, label = futures.pop(future)
                        points, pen_state = future.result()
                        data_mmap[index, :, :2] = points
                        data_mmap[index, :, 2] = pen_state
                        labels_mmap[index] = label
                        progress.update(1)
                        fill_queue()

        del data_mmap
        del labels_mmap
        temporary_data.rename(data_path)
        temporary_labels.rename(labels_path)
        print(f"QuickDraw cache saved under {data_path.parent}")
    finally:
        temporary_data.unlink(missing_ok=True)
        temporary_labels.unlink(missing_ok=True)

    return data_path, labels_path
