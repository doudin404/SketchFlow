"""Download the official SketchFlow v1 release checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


DEFAULT_URL = (
    "https://github.com/doudin404/SketchFlow/releases/download/"
    "v1.0.0/sketchflow_v1.ckpt"
)
DEFAULT_SHA256 = "899f5a32e72acb349ab70cfbe2cac068faa4b05bc54d47ccbd97624087279dbf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default="checkpoints/sketchflow_v1.ckpt")
    parser.add_argument("--sha256", default=DEFAULT_SHA256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        if not args.sha256 or sha256(output) == args.sha256.lower():
            print(f"Checkpoint already exists: {output}")
            return
        raise RuntimeError(
            f"Existing checkpoint failed SHA-256 verification: {output}"
        )

    temporary = output.with_suffix(output.suffix + ".part")
    try:
        print(f"Downloading {args.url}")
        urllib.request.urlretrieve(args.url, temporary)
        if args.sha256:
            actual = sha256(temporary)
            if actual != args.sha256.lower():
                raise RuntimeError(
                    f"SHA-256 mismatch: expected {args.sha256}, got {actual}"
                )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
