"""Benchmark runners (Milestones 1 & 3).

``run_local`` benchmarks an ONNX model in-process. ``run_grpc`` benchmarks the
same model through the gRPC service. Both produce an identical
:class:`BenchmarkResult` so the cost of the gRPC/serialization layer can be
compared directly. Every result carries full machine metadata and a validation
report; a failed validation aborts the run.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..common.logging import get_logger
from ..metadata import collect_metadata
from ..server.runtime import OnnxModel
from .dataset import Dataset
from .stats import LatencyStats, compute_latency_stats
from .validation import ValidationReport, validate_grpc, validate_model

logger = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    model_name: str = "model"
    provider: str = "cpu"
    transport: str = "local"  # "local" or "grpc"
    warmup: int = 5
    iterations: int = 100
    dataset_name: str = "synthetic"
    dataset_version: str = "1.0.0"
    dataset_fingerprint: str | None = None


@dataclass
class BenchmarkResult:
    config: dict[str, Any]
    stats: dict[str, float | int]
    validation: dict[str, Any]
    metadata: dict[str, Any]
    started_at: float
    finished_at: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iter_samples(dataset: Dataset, total: int) -> Iterator[dict[str, np.ndarray]]:
    n = len(dataset)
    for i in range(total):
        yield dataset.samples[i % n]


def run_local(
    model: OnnxModel,
    dataset: Dataset,
    config: BenchmarkConfig,
    *,
    validate: bool = True,
) -> BenchmarkResult:
    """Benchmark a model in-process (no gRPC)."""
    report = ValidationReport()
    if validate:
        report = validate_model(model, dataset.samples[0])
        report.raise_if_failed()

    # Warmup (not measured).
    for sample in _iter_samples(dataset, config.warmup):
        model.run(sample)

    latencies_ms: list[float] = []
    started = time.time()
    for sample in _iter_samples(dataset, config.iterations):
        t0 = time.perf_counter_ns()
        model.run(sample)
        latencies_ms.append((time.perf_counter_ns() - t0) / 1e6)
    finished = time.time()

    result = _assemble(config, latencies_ms, report, started, finished)
    # Record the execution provider the session actually used. ``config.provider``
    # is the *requested* logical name, which silently falls back to CPU when the
    # hardware/EP is absent; this is the ground truth a verifier needs.
    result.extra["active_provider"] = model.active_provider
    return result


def run_grpc(
    client: Any,
    dataset: Dataset,
    config: BenchmarkConfig,
    *,
    model_name: str = "",
    reference: OnnxModel | None = None,
    validate: bool = True,
) -> BenchmarkResult:
    """Benchmark through the gRPC service. ``client`` is an InferenceClient.

    When ``validate`` is set, the served outputs are checked (and compared
    against ``reference`` if provided) before timing; a failed check aborts the
    run, matching the local benchmark contract. As with ``run_local``,
    ``result.metadata`` is the environment of the process that produced the
    result (the benchmark client); the server's own environment (the host that
    actually ran inference) is captured separately under
    ``extra['server_metadata']`` to keep remote/VPS/Kubernetes results honest.
    """
    report = ValidationReport()
    if validate:
        report = validate_grpc(
            client, dataset.samples[0], model_name=model_name, reference=reference
        )
        report.raise_if_failed()

    # Warmup.
    for sample in _iter_samples(dataset, config.warmup):
        client.predict(sample, model=model_name)

    latencies_ms: list[float] = []
    server_us: list[int] = []
    started = time.time()
    for sample in _iter_samples(dataset, config.iterations):
        t0 = time.perf_counter_ns()
        _, inf_us = client.predict(sample, model=model_name)
        latencies_ms.append((time.perf_counter_ns() - t0) / 1e6)
        server_us.append(inf_us)
    finished = time.time()

    result = _assemble(config, latencies_ms, report, started, finished)
    # Record server-side inference time so transport overhead is visible. The
    # overhead is mean client latency minus mean server time; it can be slightly
    # negative under timing noise.
    server_ms = np.asarray(server_us, dtype=np.float64) / 1000.0
    result.extra["server_inference"] = compute_latency_stats(server_ms).to_dict()
    result.extra["transport_overhead_ms"] = result.stats["mean_ms"] - float(server_ms.mean())
    # ``result.metadata`` is the client's; the EP that actually ran inference and
    # the server's environment live on the server. Record both so provider
    # provenance reflects the inference host, not the caller.
    server_info = _fetch_server_info(client, model_name)
    result.extra["active_provider"] = server_info["active_provider"]
    result.extra["server_metadata"] = server_info["metadata"]
    return result


def _fetch_server_info(client: Any, model_name: str) -> dict[str, Any]:
    """Best-effort fetch of the server's execution provider and environment.

    Returns ``{"active_provider": str | None, "metadata": dict | None}`` from one
    ModelMetadata call. An unreachable RPC yields both ``None``; a server that
    does not publish its environment (older builds) yields ``None`` metadata.
    """
    info: dict[str, Any] = {"active_provider": None, "metadata": None}
    try:
        response = client.model_metadata(model_name)
    except Exception:  # pragma: no cover - server may not implement it
        return info
    info["active_provider"] = response.provider or None
    raw = dict(response.metadata).get("environment")
    if raw:
        with contextlib.suppress(TypeError, ValueError):  # defensive: tolerate bad JSON
            info["metadata"] = json.loads(raw)
    return info


def _assemble(
    config: BenchmarkConfig,
    latencies_ms: list[float],
    report: ValidationReport,
    started: float,
    finished: float,
) -> BenchmarkResult:
    stats: LatencyStats = compute_latency_stats(latencies_ms)
    logger.info(
        "benchmark complete",
        extra={
            "fields": {
                "transport": config.transport,
                "provider": config.provider,
                "iterations": config.iterations,
                "p50_ms": round(stats.p50_ms, 3),
                "p95_ms": round(stats.p95_ms, 3),
                "p99_ms": round(stats.p99_ms, 3),
            }
        },
    )
    return BenchmarkResult(
        config=asdict(config),
        stats=stats.to_dict(),
        validation=report.to_dict(),
        metadata=collect_metadata(extra={"benchmark": config.model_name}),
        started_at=started,
        finished_at=finished,
    )
