from __future__ import annotations

from onnx_grpc_benchmark.benchmark.runner import BenchmarkConfig, run_local


def test_run_local_produces_result(model, dataset):
    config = BenchmarkConfig(
        model_name=model.name,
        provider="cpu",
        transport="local",
        warmup=2,
        iterations=10,
    )
    result = run_local(model, dataset, config)
    assert result.stats["count"] == 10
    assert result.validation["passed"] is True
    assert result.metadata["schema_version"]
    assert result.stats["p99_ms"] >= result.stats["p50_ms"]


def test_metadata_attached(model, dataset):
    config = BenchmarkConfig(iterations=5, warmup=1)
    result = run_local(model, dataset, config)
    meta = result.metadata
    assert "cpu" in meta
    assert "runtime" in meta
    assert meta["runtime"]["available_providers"]
