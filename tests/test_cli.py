from __future__ import annotations

import json

import grpc
from click.testing import CliRunner

from bge_m3_bench.bench.cli import _provider_label, cli
from bge_m3_bench.client import EmbeddingClient


class _FakeRpcError(grpc.RpcError):
    """Minimal RpcError mimicking a per-request gRPC failure."""

    def code(self) -> grpc.StatusCode:
        return grpc.StatusCode.DEADLINE_EXCEEDED

    def details(self) -> str:
        return "Deadline Exceeded"


def test_provider_label_uses_active_provider():
    # Falls back to CPU at runtime but was requested as cuda -> label is cpu.
    spec = {
        "config": {"provider": "cuda"},
        "runtime": {"execution_provider": "CPUExecutionProvider"},
    }
    assert _provider_label(spec) == "cpu"
    # Active CUDA -> cuda.
    assert _provider_label({"runtime": {"execution_provider": "CUDAExecutionProvider"}}) == "cuda"
    # No runtime info -> fall back to requested logical provider.
    assert _provider_label({"config": {"provider": "cuda"}}) == "cuda"


def test_cli_writes_jsonl_summary(running_server, tmp_path):
    out = tmp_path / "run.jsonl"
    result = CliRunner().invoke(
        cli,
        [
            "--address",
            running_server,
            "--warmup-sec",
            "0.2",
            "--duration-sec",
            "0.8",
            "--batch-size",
            "4",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) >= 2
    assert rows[0]["type"] == "request"
    summary = rows[-1]
    assert summary["type"] == "summary"

    grpc_metrics = summary["grpc_metrics"]
    assert grpc_metrics["total_requests"] == len(rows) - 1
    assert grpc_metrics["client_concurrency"] == 1
    assert "client_e2e_p50_ms" in grpc_metrics
    assert "grpc_overhead_p99_ms" in grpc_metrics
    assert summary["resource_metrics"]["memory_rss_peak_mb"] > 0
    assert summary["model"]["embedding_dim"] == 8
    assert summary["validation"]["embedding_dim"] == 8
    assert summary["validation"]["nan_count"] == 0
    assert summary["validation"]["passed"] is True
    assert summary["validation"]["reasons"] == []
    assert summary["input"]["num_inputs"] == grpc_metrics["total_requests"] * 4


def test_cli_concurrency_runs(running_server, tmp_path):
    out = tmp_path / "run.jsonl"
    result = CliRunner().invoke(
        cli,
        [
            "--address",
            running_server,
            "--warmup-sec",
            "0.2",
            "--duration-sec",
            "0.8",
            "--batch-size",
            "4",
            "--concurrency",
            "4",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    summary = rows[-1]
    assert summary["type"] == "summary"

    request_rows = [r for r in rows if r["type"] == "request"]
    grpc_metrics = summary["grpc_metrics"]
    assert grpc_metrics["client_concurrency"] == 4
    # One record per measured request (success or failure): row count == total.
    assert len(request_rows) == grpc_metrics["total_requests"]
    assert all(r["ok"] is True for r in request_rows)
    assert grpc_metrics["successful_requests"] == len(request_rows)
    assert grpc_metrics["total_requests"] == grpc_metrics["successful_requests"]
    assert grpc_metrics["failed_requests"] == 0
    assert grpc_metrics["error_rate"] == 0.0
    assert grpc_metrics["error_codes"] == {}
    # A healthy run reports run_ok and exits zero.
    assert '"run_ok": true' in result.output
    assert summary["benchmark"]["benchmark_id"].endswith("-c4")
    assert summary["input"]["num_inputs"] == grpc_metrics["total_requests"] * 4


def test_cli_all_requests_fail_signals_loudly(running_server, tmp_path, monkeypatch):
    # Every measured Embed raises DEADLINE_EXCEEDED (control RPCs — spec, resource
    # samples — still work). Skip validation so it doesn't hit the same failure first.
    def _boom(self, texts):
        raise _FakeRpcError()

    monkeypatch.setattr(EmbeddingClient, "embed", _boom)

    out = tmp_path / "run.jsonl"
    result = CliRunner().invoke(
        cli,
        [
            "--address",
            running_server,
            "--no-validate",
            "--warmup-sec",
            "0",
            "--duration-sec",
            "0.3",
            "--concurrency",
            "2",
            "--out",
            str(out),
        ],
    )
    # No successful request -> non-zero exit + loud guidance on stderr.
    assert result.exit_code == 1, result.output
    assert '"run_ok": false' in result.output
    assert "DEADLINE_EXCEEDED" in result.output
    assert "Raise --timeout" in result.output

    summary = [json.loads(line) for line in out.read_text().splitlines()][-1]
    g = summary["grpc_metrics"]
    assert g["successful_requests"] == 0
    assert g["error_rate"] == 1.0
    assert g["failed_requests"] == g["error_codes"]["DEADLINE_EXCEEDED"]
    assert g["error_codes"] == {"DEADLINE_EXCEEDED": g["failed_requests"]}


def test_cli_rejects_zero_concurrency(running_server, tmp_path):
    result = CliRunner().invoke(
        cli,
        ["--address", running_server, "--concurrency", "0", "--out", str(tmp_path / "x.jsonl")],
    )
    assert result.exit_code != 0
    assert "concurrency" in result.output.lower()


def test_cli_rejects_incomplete_reference_pair(tmp_path):
    ref = tmp_path / "ref_model.onnx"
    ref.write_bytes(b"")  # only needs to exist for click's exists=True check
    result = CliRunner().invoke(
        cli,
        ["--ref-model", str(ref), "--out", str(tmp_path / "x.jsonl")],
    )
    assert result.exit_code != 0
    assert "together" in result.output.lower()
