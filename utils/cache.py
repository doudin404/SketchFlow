from __future__ import annotations

import hashlib
import re


def _sanitize_filename_part(value: str) -> str:
    """Replace characters that are unsafe in cross-platform filenames."""
    return re.sub(r'[\\/:*?"<>|]', "_", value)


def cache_filename(
    *values: object,
    prefix_count: int = 3,
    separator: str = "_",
) -> str:
    """Build a readable cache filename with a content-derived suffix."""
    string_values = [str(value) for value in values]
    digest_source = separator.join(string_values).encode("utf-8")
    digest = hashlib.md5(digest_source).hexdigest()
    prefix = [
        _sanitize_filename_part(value)
        for value in string_values[:prefix_count]
    ]
    return separator.join([*prefix, digest]) if prefix else digest
