"""Percentile helpers for the benchmark summary.

The benchmark client is the only place latency percentiles are computed; the
server never aggregates.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def percentiles_ms(prefix: str, values_ms: Iterable[float]) -> dict[str, float]:
    """Return ``{prefix}_p50_ms / p95_ms / p99_ms`` for the given samples."""
    arr = np.asarray(list(values_ms), dtype=np.float64)
    if arr.size == 0:
        return {f"{prefix}_p50_ms": 0.0, f"{prefix}_p95_ms": 0.0, f"{prefix}_p99_ms": 0.0}
    return {
        f"{prefix}_p50_ms": round(float(np.percentile(arr, 50)), 4),
        f"{prefix}_p95_ms": round(float(np.percentile(arr, 95)), 4),
        f"{prefix}_p99_ms": round(float(np.percentile(arr, 99)), 4),
    }


def token_percentiles(token_counts: Iterable[int]) -> dict[str, float | int]:
    """p50/p95/max token count per input."""
    arr = np.asarray(list(token_counts), dtype=np.float64)
    if arr.size == 0:
        return {"p50_tokens_per_input": 0, "p95_tokens_per_input": 0, "max_tokens_per_input": 0}
    return {
        # Rounded floats: truncating to int understates the distribution when a
        # percentile falls between observed counts (e.g. p95 of 8.95 -> 8).
        "p50_tokens_per_input": round(float(np.percentile(arr, 50)), 2),
        "p95_tokens_per_input": round(float(np.percentile(arr, 95)), 2),
        "max_tokens_per_input": int(arr.max()),
    }
