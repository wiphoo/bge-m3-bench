"""Chart generation for benchmark reports (Milestone 4).

Reads one or more benchmark result JSON files and renders comparison charts.
Matplotlib is an optional (``viz``) dependency, imported lazily so the core
package stays slim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_results(paths: list[str | Path]) -> list[dict[str, Any]]:
    results = []
    for p in paths:
        results.append(json.loads(Path(p).read_text()))
    return results


def _label(result: dict[str, Any]) -> str:
    cfg = result.get("config", {})
    return f"{cfg.get('model_name', '?')}/{cfg.get('transport', '?')}/{cfg.get('provider', '?')}"


def latency_bar_chart(results: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Grouped bar chart of p50/p95/p99 latency per result."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [_label(r) for r in results]
    metrics = ["p50_ms", "p95_ms", "p99_ms"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.6), 4.5))
    for i, metric in enumerate(metrics):
        values = [r["stats"][metric] for r in results]
        ax.bar(x + (i - 1) * width, values, width, label=metric.replace("_ms", ""))
    ax.set_ylabel("latency (ms)")
    ax.set_title("Latency percentiles by configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def throughput_chart(results: list[dict[str, Any]], out_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [_label(r) for r in results]
    values = [r["stats"]["throughput_rps"] for r in results]
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.6), 4.5))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_ylabel("throughput (req/s)")
    ax.set_title("Throughput by configuration")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
