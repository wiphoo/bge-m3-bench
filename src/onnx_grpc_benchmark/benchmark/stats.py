"""Latency statistics for benchmark runs (Milestone 1).

Percentiles use linear interpolation between closest ranks, matching
``numpy.percentile`` defaults, so p50/p95/p99 are well defined for any sample
size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    count: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_latency_stats(latencies_ms: list[float] | np.ndarray) -> LatencyStats:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot compute stats over an empty sample")
    mean = float(arr.mean())
    total_s = float(arr.sum()) / 1000.0
    throughput = arr.size / total_s if total_s > 0 else 0.0
    return LatencyStats(
        count=int(arr.size),
        mean_ms=mean,
        std_ms=float(arr.std()),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        p50_ms=float(np.percentile(arr, 50)),
        p90_ms=float(np.percentile(arr, 90)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        throughput_rps=throughput,
    )
