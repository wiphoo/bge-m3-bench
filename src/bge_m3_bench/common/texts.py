"""Shared input-text loading for the client-side tools.

A single source of truth for the small built-in sample and the one-input-per-line
file loader used by both ``bge-m3-bench`` and ``bge-m3-embed``.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Embeddings turn text into dense vectors.",
    "BGE-M3 supports dense, sparse, and multi-vector retrieval.",
    "gRPC is a high-performance RPC framework.",
    "Tokenization splits text into model input ids.",
]


def load_texts(path: str | None) -> list[str]:
    """Load inputs from a file (one per line, blanks dropped) or the sample."""
    if not path:
        return list(DEFAULT_TEXTS)
    lines = [line.strip() for line in Path(path).read_text().splitlines()]
    return [line for line in lines if line]
