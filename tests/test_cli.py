from __future__ import annotations

import json

from click.testing import CliRunner

from bge_m3_bench.bench.cli import cli


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
    assert summary["input"]["num_inputs"] == grpc_metrics["total_requests"] * 4


def test_cli_rejects_concurrency(running_server, tmp_path):
    result = CliRunner().invoke(
        cli,
        ["--address", running_server, "--concurrency", "4", "--out", str(tmp_path / "x.jsonl")],
    )
    assert result.exit_code != 0
    assert "concurrency" in result.output.lower()
