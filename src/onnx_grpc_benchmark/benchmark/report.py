"""Persist benchmark results as JSON and CSV (Milestone 3)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .runner import BenchmarkResult

# Flat columns surfaced into CSV (nested data stays in the JSON sidecar).
_CSV_COLUMNS = [
    "model",
    "transport",
    "provider",
    "iterations",
    "mean_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "throughput_rps",
    "validation_passed",
    "hostname",
    "active_provider",
]


def _flatten(result: BenchmarkResult) -> dict:
    cfg = result.config
    stats = result.stats
    meta = result.metadata
    providers = meta.get("runtime", {}).get("available_providers", [])
    return {
        "model": cfg.get("model_name"),
        "transport": cfg.get("transport"),
        "provider": cfg.get("provider"),
        "iterations": cfg.get("iterations"),
        "mean_ms": round(stats["mean_ms"], 4),
        "p50_ms": round(stats["p50_ms"], 4),
        "p90_ms": round(stats["p90_ms"], 4),
        "p95_ms": round(stats["p95_ms"], 4),
        "p99_ms": round(stats["p99_ms"], 4),
        "throughput_rps": round(stats["throughput_rps"], 2),
        "validation_passed": result.validation.get("passed"),
        "hostname": meta.get("hostname"),
        "active_provider": providers[0] if providers else None,
    }


def save_json(result: BenchmarkResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
    return path


def save_csv(results: list[BenchmarkResult], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(_flatten(result))
    return path
