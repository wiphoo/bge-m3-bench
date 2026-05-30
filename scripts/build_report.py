#!/usr/bin/env python3
"""Build a reproducible benchmark report (Milestone 4).

Generates a Jupyter notebook from result JSON files, executes it, and exports
PNG charts and a standalone HTML report.

Usage:
    python scripts/build_report.py results/*.json --out-dir results/report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_notebook(result_paths: list[str], out_dir: Path) -> Path:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

    paths_repr = repr([str(Path(p).resolve()) for p in result_paths])
    out_repr = repr(str(out_dir.resolve()))

    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# ONNX Runtime gRPC Benchmark Report\n\n"
            "Auto-generated and reproducible: regenerate with "
            "`python scripts/build_report.py`."
        ),
        new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from onnx_grpc_benchmark.benchmark import viz\n"
            f"RESULT_PATHS = {paths_repr}\n"
            f"OUT_DIR = Path({out_repr})\n"
            "OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
            "results = viz.load_results(RESULT_PATHS)\n"
            "print(f'loaded {len(results)} result(s)')"
        ),
        new_markdown_cell("## Summary table"),
        new_code_cell(
            "rows = []\n"
            "for r in results:\n"
            "    cfg, st = r['config'], r['stats']\n"
            "    rows.append({\n"
            "        'model': cfg.get('model_name'),\n"
            "        'transport': cfg.get('transport'),\n"
            "        'provider': cfg.get('provider'),\n"
            "        'p50_ms': round(st['p50_ms'], 3),\n"
            "        'p95_ms': round(st['p95_ms'], 3),\n"
            "        'p99_ms': round(st['p99_ms'], 3),\n"
            "        'rps': round(st['throughput_rps'], 1),\n"
            "        'host': r['metadata'].get('hostname'),\n"
            "    })\n"
            "df = pd.DataFrame(rows)\n"
            "df"
        ),
        new_markdown_cell("## Latency percentiles"),
        new_code_cell(
            "p = viz.latency_bar_chart(results, OUT_DIR / 'latency.png')\n"
            "from IPython.display import Image\n"
            "Image(str(p))"
        ),
        new_markdown_cell("## Throughput"),
        new_code_cell(
            "p = viz.throughput_chart(results, OUT_DIR / 'throughput.png')\nImage(str(p))"
        ),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    nb_path = out_dir / "report.ipynb"
    nbformat.write(nb, nb_path)
    return nb_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build benchmark report")
    parser.add_argument("results", nargs="+", help="Benchmark result JSON files")
    parser.add_argument("--out-dir", default="results/report")
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    nb_path = build_notebook(args.results, out_dir)
    print(f"wrote {nb_path}")

    if args.no_execute:
        return 0

    try:
        import nbformat
        from nbconvert import HTMLExporter
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError:
        print("install the 'viz' extra to execute/export the report", file=sys.stderr)
        return 0

    nb = nbformat.read(nb_path, as_version=4)
    ExecutePreprocessor(timeout=120, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": str(Path.cwd())}}
    )
    nbformat.write(nb, nb_path)

    html, _ = HTMLExporter().from_notebook_node(nb)
    html_path = out_dir / "report.html"
    html_path.write_text(html)
    print(f"wrote {html_path}")
    print(f"charts in {out_dir} (latency.png, throughput.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
