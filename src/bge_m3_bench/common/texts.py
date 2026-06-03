"""Shared input-text loading for the client-side tools.

A single source of truth for the small built-in sample and the one-input-per-line
file loader used by both ``bge-m3-bench`` and ``bge-m3-embed``.

Accepts either a local file path or an ``https://`` URL to a plain-text file
(one input per line, blank lines dropped).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Embeddings turn text into dense vectors.",
    "BGE-M3 supports dense, sparse, and multi-vector retrieval.",
    "gRPC is a high-performance RPC framework.",
    "Tokenization splits text into model input ids.",
]


def _fetch_text(path: str) -> list[str]:
    """Read one-per-line text from an HTTPS URL."""
    try:
        with urllib.request.urlopen(path, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise ValueError(f"failed to fetch texts from {path}: {exc}") from exc
    lines = [line.strip() for line in raw.splitlines()]
    return [line for line in lines if line]


def load_texts(path: str | None) -> list[str]:
    """Load inputs from a file (one per line, blanks dropped) or the sample.

    * ``None`` / empty → built-in ``DEFAULT_TEXTS``.
    * ``https://...`` → fetched via HTTPS.
    * Everything else → local file path.
    """
    if not path:
        return list(DEFAULT_TEXTS)
    if path.startswith("http://") or path.startswith("https://"):
        return _fetch_text(path)
    lines = [line.strip() for line in Path(path).read_text().splitlines()]
    return [line for line in lines if line]
