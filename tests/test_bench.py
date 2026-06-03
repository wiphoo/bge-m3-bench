from __future__ import annotations

import numpy as np

from bge_m3_bench.bench.metrics import (
    RequestSample,
    RunContext,
    build_summary,
    error_row,
    request_row,
)
from bge_m3_bench.bench.stats import percentiles_ms, token_percentiles
from bge_m3_bench.bench.validation import validate_embeddings


def test_percentiles_keys():
    p = percentiles_ms("inference", [1.0, 2.0, 3.0, 4.0])
    assert set(p) == {"inference_p50_ms", "inference_p95_ms", "inference_p99_ms"}


def test_token_percentiles():
    t = token_percentiles([1, 2, 3, 10])
    assert t["max_tokens_per_input"] == 10


def test_validate_finite_and_norm():
    emb = np.ones((4, 8), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    report, passed, reasons = validate_embeddings(emb, normalize=True)
    assert passed and not reasons
    assert report["embedding_dim"] == 8
    assert abs(report["embedding_norm_mean"] - 1.0) < 1e-5
    assert report["nan_count"] == 0 and report["zero_vector_count"] == 0
    assert report["passed"] is True and report["reasons"] == []


def test_validate_reference_match_and_mismatch():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((5, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    report, passed, _ = validate_embeddings(emb, normalize=True, reference=emb)
    assert passed
    assert report["cosine_similarity_mean_vs_reference"] > 0.999
    assert report["max_abs_diff_vs_reference"] < 1e-5

    _, passed_bad, reasons = validate_embeddings(emb, normalize=True, reference=-emb)
    assert not passed_bad and any("cosine" in r for r in reasons)


def test_validate_reference_shape_mismatch_does_not_crash():
    emb = np.ones((8, 16), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    reference = np.ones((5, 32), dtype=np.float32)
    report, passed, reasons = validate_embeddings(emb, normalize=True, reference=reference)
    assert not passed and any("shape" in r for r in reasons)
    assert report["cosine_similarity_mean_vs_reference"] is None
    assert report["max_abs_diff_vs_reference"] is None
    assert report["passed"] is False and report["reasons"] == reasons


def test_validate_norm_checks_each_row_not_just_mean():
    # Norms 0.5 and 1.5 average to 1.0 but neither row is normalized.
    emb = np.zeros((2, 4), dtype=np.float32)
    emb[0, 0] = 0.5
    emb[1, 0] = 1.5
    _, passed, reasons = validate_embeddings(emb, normalize=True)
    assert not passed and any("norm" in r for r in reasons)


def test_validate_non_finite_report_is_json_safe():
    import json

    emb = np.full((2, 4), np.nan, dtype=np.float32)
    emb[1, 0] = np.inf
    report, passed, reasons = validate_embeddings(emb, normalize=True)
    assert not passed and any("non-finite" in r for r in reasons)
    # Non-finite aggregates are emitted as None, not bare NaN/Infinity.
    assert report["embedding_norm_mean"] is None
    assert report["embedding_norm_std"] is None
    # Strict JSON encoding must succeed (no NaN/Infinity tokens).
    json.dumps(report, allow_nan=False)


def test_validate_non_finite_reference_fails():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((4, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    bad_ref = emb.copy()
    bad_ref[0, 0] = np.nan  # broken reference, finite served embeddings
    report, passed, reasons = validate_embeddings(emb, normalize=True, reference=bad_ref)
    assert not passed and any("non-finite reference" in r for r in reasons)
    assert report["cosine_similarity_mean_vs_reference"] is None


def _sample(tokens, tok_us, inf_us, post_us, client_us, rss):
    return RequestSample(
        num_inputs=len(tokens),
        token_counts=tokens,
        tokenize_us=tok_us,
        inference_us=inf_us,
        postprocess_us=post_us,
        client_e2e_us=client_us,
        request_bytes=100,
        response_bytes=400,
    )


def test_request_row_shape():
    s = _sample([3, 4], 10, 20, 5, 80, 100.0)
    row = request_row(7, s)
    assert row["type"] == "request" and row["i"] == 7
    assert row["ok"] is True
    assert row["server_e2e_us"] == 35 and row["total_tokens"] == 7
    assert row["token_counts"] == [3, 4]


def test_error_row_shape():
    row = error_row(3, "UNAVAILABLE", "connection refused")
    assert row["type"] == "request" and row["i"] == 3
    assert row["ok"] is False
    assert row["error_code"] == "UNAVAILABLE"
    assert row["error"] == "connection refused"


def test_build_summary_sections():
    samples = [
        _sample([3, 4], 10, 100, 5, 200, 0),
        _sample([5, 6], 12, 120, 6, 240, 0),
        _sample([2, 2], 8, 90, 4, 180, 0),
    ]
    resources = [
        {"t_unix": 1.0, "rss_mb": 100.0, "cpu_percent": 50.0},
        {"t_unix": 1.1, "rss_mb": 150.0, "cpu_percent": 90.0},
    ]
    spec = {
        "model": {"name": "m", "embedding_dim": 8, "inputs": [], "outputs": []},
        "config": {"provider": "cpu"},
        "runtime": {"runtime": "onnxruntime"},
        "machine": {"hostname": "h"},
    }
    ctx = RunContext(
        benchmark_id="bid",
        duration_sec=1.0,
        warmup_sec=0.0,
        batch_size=2,
        concurrency=1,
        model_name="",
        model_revision="rev",
        precision="fp32",
        quantization="none",
    )
    summary = build_summary(
        samples=samples, resource_samples=resources, spec=spec, ctx=ctx, validation=None
    )
    assert summary["type"] == "summary"
    assert summary["input"]["num_inputs"] == 6
    assert summary["input"]["total_tokens"] == 22
    assert summary["grpc_metrics"]["total_requests"] == 3
    assert summary["grpc_metrics"]["client_concurrency"] == 1
    assert "client_e2e_p50_ms" in summary["grpc_metrics"]
    assert "grpc_overhead_p50_ms" in summary["grpc_metrics"]
    assert "inference_p99_ms" in summary["raw_onnx_metrics"]
    assert summary["resource_metrics"]["memory_rss_peak_mb"] == 150.0
    assert summary["resource_metrics"]["cpu_percent_peak"] == 90.0
    assert summary["model"]["embedding_dim"] == 8
    assert summary["model"]["model_name"] == "m"  # falls back to spec name
    # Analysis section is present; with a bare machine block the normalized
    # efficiency ratios and the memory verdict are None (unknown, not guessed).
    analysis = summary["analysis"]
    assert analysis["efficiency"]["inputs_per_sec_per_physical_core"] is None
    assert analysis["memory"]["sufficient"] is None
    assert isinstance(analysis["notes"], list)


def test_build_analysis_full():
    import json

    samples = [
        _sample([3, 4], 10, 100, 5, 200, 0),
        _sample([5, 6], 12, 120, 6, 240, 0),
        _sample([2, 2], 8, 90, 4, 180, 0),
    ]
    resources = [
        {"t_unix": 1.0, "rss_mb": 100.0, "cpu_percent": 50.0},
        {"t_unix": 1.1, "rss_mb": 150.0, "cpu_percent": 90.0},
    ]
    spec = {
        "model": {"name": "m", "embedding_dim": 8, "inputs": [], "outputs": []},
        "config": {"provider": "cpu"},
        "runtime": {},
        "machine": {
            "cpu_physical_cores": 8,
            "cpu_logical_cores": 16,
            "cpu_freq_max_mhz": 4000.0,
            "ram_total_mb": 32000.0,
            "cpu_isa_extensions": ["avx2", "avx512f"],
        },
    }
    ctx = RunContext(
        benchmark_id="bid",
        duration_sec=1.0,
        warmup_sec=0.0,
        batch_size=2,
        concurrency=1,
        model_name="",
        model_revision="rev",
        precision="fp32",
        quantization="none",
    )
    summary = build_summary(
        samples=samples, resource_samples=resources, spec=spec, ctx=ctx, validation=None
    )
    a = summary["analysis"]
    # inputs_per_sec = 6 / 1.0 ; tokens_per_sec = 22 / 1.0
    eff = a["efficiency"]
    assert eff["inputs_per_sec_per_physical_core"] == 0.75  # 6/8
    assert eff["inputs_per_sec_per_logical_core"] == 0.375  # 6/16
    assert eff["inputs_per_sec_per_ghz"] == 1.5  # 6 / 4.0 GHz
    assert eff["inputs_per_sec_per_physical_core_ghz"] == 0.1875  # 6 / (8*4)
    assert eff["tokens_per_sec_per_physical_core"] == 2.75  # 22/8
    mem = a["memory"]
    assert mem["headroom_mb"] == 31850.0  # 32000 - 150
    assert mem["sufficient"] is True
    assert abs(a["cpu_utilization"]["core_utilization_pct"] - 4.375) < 0.02  # 70/(16*100)*100
    assert a["notes"] and any("AVX-512" in n for n in a["notes"])
    # Whole summary must encode as strict JSON (no NaN/Infinity).
    json.dumps(summary, allow_nan=False)


def _ctx(**kw):
    base = {
        "benchmark_id": "bid",
        "duration_sec": 1.0,
        "warmup_sec": 0.0,
        "batch_size": 2,
        "concurrency": 1,
        "model_name": "",
        "model_revision": "rev",
        "precision": "fp32",
        "quantization": "none",
    }
    return RunContext(**{**base, **kw})


def test_build_analysis_container_uses_effective_limits():
    # 2-vCPU / 2 GB container on a big host: saturation and memory must be judged
    # against the cgroup limits, not the host counts.
    samples = [_sample([3, 4], 10, 100, 5, 200, 0)]
    resources = [{"t_unix": 1.0, "rss_mb": 1900.0, "cpu_percent": 190.0}]
    spec = {
        "model": {},
        "config": {},
        "runtime": {},
        "machine": {
            "containerized": True,
            "cpu_logical_cores": 64,
            "cpu_effective_cores": 2.0,
            "ram_total_mb": 64000.0,
            "ram_limit_mb": 2048.0,
        },
    }
    a = build_summary(
        samples=samples,
        resource_samples=resources,
        spec=spec,
        ctx=_ctx(concurrency=2),
        validation=None,
    )["analysis"]
    mem = a["memory"]
    assert mem["budget_mb"] == 2048.0 and mem["budget_source"] == "cgroup_limit"
    assert mem["headroom_mb"] == 148.0  # 2048 - 1900
    assert mem["sufficient"] is False  # 92.8% > 90%
    cpu = a["cpu_utilization"]
    assert cpu["effective_cores"] == 2.0
    assert cpu["core_utilization_pct"] == 95.0  # 190 / (2*100) * 100
    # Fully saturating its 2 vCPUs -> no misleading "raise --concurrency" note.
    assert not any("under-utilized" in n for n in a["notes"])
    assert any("cgroup limit" in n for n in a["notes"])


def test_build_analysis_container_without_limit_is_unknown():
    samples = [_sample([3, 4], 10, 100, 5, 200, 0)]
    resources = [{"t_unix": 1.0, "rss_mb": 1000.0, "cpu_percent": 50.0}]
    spec = {
        "model": {},
        "config": {},
        "runtime": {},
        "machine": {
            "containerized": True,
            "cpu_logical_cores": 64,
            "cpu_effective_cores": 4.0,
            "ram_total_mb": 64000.0,
            # no ram_limit_mb -> host RAM must not be trusted for the verdict
        },
    }
    a = build_summary(
        samples=samples, resource_samples=resources, spec=spec, ctx=_ctx(), validation=None
    )["analysis"]
    mem = a["memory"]
    assert mem["budget_mb"] is None and mem["budget_source"] is None
    assert mem["sufficient"] is None and mem["utilization_pct"] is None
    assert any("verdict unknown" in n for n in a["notes"])


def test_build_summary_failure_tracking():
    samples = [
        _sample([3, 4], 10, 100, 5, 200, 0),
        _sample([5, 6], 12, 120, 6, 240, 0),
        _sample([2, 2], 8, 90, 4, 180, 0),
    ]
    spec = {"model": {}, "config": {}, "runtime": {}, "machine": {}}
    ctx = RunContext(
        benchmark_id="bid",
        duration_sec=2.0,
        warmup_sec=0.0,
        batch_size=2,
        concurrency=4,
        model_name="",
        model_revision="rev",
        precision="fp32",
        quantization="none",
    )
    summary = build_summary(
        samples=samples,
        resource_samples=[],
        spec=spec,
        ctx=ctx,
        validation=None,
        failed_requests=1,
    )
    g = summary["grpc_metrics"]
    assert g["client_concurrency"] == 4
    assert g["successful_requests"] == 3
    assert g["failed_requests"] == 1
    assert g["total_requests"] == 4
    assert g["error_rate"] == 0.25
    # Throughput and size averages are over successful requests only.
    assert g["requests_per_sec"] == 1.5
    assert g["request_size_bytes_avg"] == 100
