from __future__ import annotations

import numpy as np
import pytest

from onnx_grpc_benchmark.benchmark.stats import compute_latency_stats


def test_percentiles_monotonic():
    stats = compute_latency_stats(list(range(1, 101)))
    assert stats.count == 100
    assert stats.min_ms == 1.0
    assert stats.max_ms == 100.0
    assert stats.p50_ms <= stats.p90_ms <= stats.p95_ms <= stats.p99_ms


def test_throughput():
    # 10 requests each 100ms => 1s total => 10 rps.
    stats = compute_latency_stats([100.0] * 10)
    assert stats.throughput_rps == pytest.approx(10.0, rel=1e-6)


def test_empty_raises():
    with pytest.raises(ValueError):
        compute_latency_stats(np.array([]))
