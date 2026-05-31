from __future__ import annotations

import numpy as np

from bge_m3_bench.bench.metrics import RequestSample, RunContext, build_summary, request_row
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
    assert row["server_e2e_us"] == 35 and row["total_tokens"] == 7


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
    assert "client_e2e_p50_ms" in summary["grpc_metrics"]
    assert "grpc_overhead_p50_ms" in summary["grpc_metrics"]
    assert "inference_p99_ms" in summary["raw_onnx_metrics"]
    assert summary["resource_metrics"]["memory_rss_peak_mb"] == 150.0
    assert summary["resource_metrics"]["cpu_percent_peak"] == 90.0
    assert summary["model"]["embedding_dim"] == 8
    assert summary["model"]["model_name"] == "m"  # falls back to spec name
