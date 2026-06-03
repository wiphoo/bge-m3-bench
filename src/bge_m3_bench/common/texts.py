"""Shared input-text loading for the client-side tools.

A single source of truth for the small built-in sample and the one-input-per-line
file loader used by both ``bge-m3-bench`` and ``bge-m3-embed``.

Accepts either a local file path or an ``http(s)://`` URL to a plain-text file
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


_UA = "bge-m3-bench/1.0 (+https://github.com/wiphoo/bge-m3-bench)"


def _split_lines(raw: str) -> list[str]:
    """Split into stripped, non-blank lines (one input per line)."""
    return [stripped for line in raw.splitlines() if (stripped := line.strip())]


def _fetch_text(path: str) -> list[str]:
    """Read one-per-line text from an HTTP(S) URL."""
    req = urllib.request.Request(path, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.URLError, UnicodeDecodeError) as exc:
        raise ValueError(f"failed to fetch texts from {path}: {exc}") from exc
    return _split_lines(raw)


def load_texts(path: str | None) -> list[str]:
    """Load inputs from a file (one per line, blanks dropped) or the sample.

    * ``None`` / empty → built-in ``DEFAULT_TEXTS``.
    * ``http(s)://...`` → fetched over HTTP(S).
    * Everything else → local file path.

    Raises ``ValueError`` (with a clear message) if the URL fetch fails or the
    local file cannot be read.
    """
    if not path:
        return list(DEFAULT_TEXTS)
    if path.startswith(("http://", "https://")):
        return _fetch_text(path)
    try:
        raw = Path(path).read_text()
    except OSError as exc:
        raise ValueError(f"failed to read texts from {path}: {exc}") from exc
    return _split_lines(raw)
