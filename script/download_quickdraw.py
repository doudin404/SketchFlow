"""Download the official Sketch-RNN QuickDraw NPZ files."""

from __future__ import annotations

import argparse
import urllib.parse
import urllib.request
from pathlib import Path


CATEGORIES_URL = (
    "https://raw.githubusercontent.com/googlecreativelab/"
    "quickdraw-dataset/master/categories.txt"
)
DATASET_ROOT = "https://storage.googleapis.com/quickdraw_dataset/sketchrnn"


def read_categories(list_file: str | None = None) -> list[str]:
    if list_file:
        text = Path(list_file).read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(CATEGORIES_URL) as response:
            text = response.read().decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def download_category(
    category: str,
    output_dir: Path,
    full: bool,
    overwrite: bool,
) -> Path:
    suffix = ".full.npz" if full else ".npz"
    filename = f"{category}{suffix}"
    target_dir = output_dir / category
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if target.exists() and not overwrite:
        print(f"skip {target}")
        return target

    encoded = urllib.parse.quote(filename)
    url = f"{DATASET_ROOT}/{encoded}"
    temporary = target.with_suffix(target.suffix + ".part")
    print(f"download {category}: {url}")
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="data/quickdraw")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Category names. Omit to download all 345 categories.",
    )
    parser.add_argument(
        "--list-file",
        default=None,
        help="Optional newline-delimited category list.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download the full per-category archives used for paper training.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = args.categories or read_categories(args.list_file)
    output_dir = Path(args.output_dir)
    for category in categories:
        download_category(
            category,
            output_dir,
            full=args.full,
            overwrite=args.overwrite,
        )
    print(f"Downloaded {len(categories)} categories to {output_dir}.")


if __name__ == "__main__":
    main()
