#!/usr/bin/env python3
"""CoreML memory-leak repro / diagnostic harness.

A single-threaded inference loop that loads the model + tokenizer and calls
``OnnxModel.run`` repeatedly while reporting RSS — to isolate whether the memory
growth / ``Context leak detected, msgtracer returned -1`` seen under
``--provider coreml`` originates in the CoreML execution provider itself or in
the gRPC server's threading.

No gRPC, no thread pool: if RSS still climbs here, the server's concurrency is
not the cause. Compare runs to localize the leak:

- ``--provider cpu`` baseline (RSS should stay flat).
- ``--provider coreml`` with dynamic shapes vs a fixed ``--pad-length`` (if it
  grows only with dynamic shapes, it's per-shape CoreML model caching).
- ``--vary-length`` to deliberately change the padded sequence length per
  iteration (stresses the shape-cache hypothesis).
- ``--tracemalloc`` to tell Python-side growth (our code) from native growth
  (CoreML/Metal/ORT).
- ``--ort-verbose`` to surface CoreML EP graph partitioning / per-shape compiles.

The process prints its PID and can ``--hold`` at the end so you can attach the
macOS command-line tools while it is live, e.g.::

    MallocStackLogging=1 uv run python scripts/coreml_memcheck.py \
        --model models/bge-m3/fp32/model.onnx \
        --tokenizer models/bge-m3/fp32/tokenizer.json \
        --provider coreml --iters 4000 --hold 180

    leaks <pid> | head -60          # leaked blocks + allocation stacks
    heap <pid>                       # live objects by class (watch MTL*/CoreML*)
    vmmap <pid> ; footprint <pid>    # which memory regions grow
    malloc_history <pid> <address>   # stack for an address from leaks/vmmap

If those tools fail with "process <pid> is not debuggable" (the uv Python is a
hardened-runtime build without get-task-allow, and sudo does NOT bypass this),
either rely on this script's in-process ``malloc_mb`` / ``non_malloc_mb`` /
``--tracemalloc`` breakdown (no entitlement needed), or re-sign the interpreter
to unlock leaks/malloc_history/heap::

    printf '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC ' \
      '"-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n' \
      '<plist version="1.0"><dict><key>com.apple.security.get-task-allow</key>' \
      '<true/></dict></plist>\n' > /tmp/gta.entitlements
    codesign -s - -f --entitlements /tmp/gta.entitlements \
      "$(uv run python -c 'import sys; print(sys.executable)')"

Interpreting the in-process numbers:
  RSS grows, malloc_mb flat, non_malloc_mb grows  -> native Metal/CoreML (GPU)
  RSS grows and malloc_mb grows together          -> C/C++ heap leak (use stacks)
  --tracemalloc top grows                          -> leak is in our Python code
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-threaded CoreML inference memory repro.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to the ONNX model.")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json.")
    parser.add_argument("--provider", default="coreml", help="cpu|cuda|openvino|coreml")
    parser.add_argument(
        "--provider-option",
        dest="provider_option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Provider-specific option (repeatable).",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--pad-length",
        type=int,
        default=0,
        help="Fixed pad length (0 = dynamic per-batch). A fixed value pins one static shape.",
    )
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--texts", default=None, help="Input file, one text per line (else synthetic)."
    )
    parser.add_argument(
        "--synthetic", type=int, default=256, help="How many synthetic texts to generate."
    )
    parser.add_argument(
        "--vary-length",
        action="store_true",
        help="Shift the batch window each iteration so the padded sequence length varies.",
    )
    parser.add_argument("--rss-every", type=int, default=50, help="Log RSS every N iterations.")
    parser.add_argument(
        "--gc-every", type=int, default=0, help="Call gc.collect() every N iterations (0 = off)."
    )
    parser.add_argument(
        "--tracemalloc", action="store_true", help="Report top Python allocations at the end."
    )
    parser.add_argument(
        "--ort-verbose", action="store_true", help="Set ONNX Runtime logger to VERBOSE."
    )
    parser.add_argument(
        "--vmmap",
        action="store_true",
        help="Log `vmmap -summary` + `footprint` (region-level) at start and end (macOS).",
    )
    parser.add_argument(
        "--hold", type=int, default=0, help="Seconds to keep the process alive at the end."
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1e6


def _malloc_in_use_mb() -> float | None:
    """Total live bytes across all malloc zones (macOS), else None.

    Works from inside the process — no entitlement / debuggability needed — so it
    survives the "process is not debuggable" restriction that blocks ``leaks``.
    Compared against RSS it splits C-heap growth from non-malloc growth (Metal /
    IOAccelerator GPU buffers, mmap'd CoreML models).
    """
    if sys.platform != "darwin":
        return None
    import ctypes

    class _Stats(ctypes.Structure):
        _fields_ = [
            ("blocks_in_use", ctypes.c_uint),
            ("size_in_use", ctypes.c_size_t),
            ("max_size_in_use", ctypes.c_size_t),
            ("size_allocated", ctypes.c_size_t),
        ]

    try:
        libc = ctypes.CDLL(None)
        stats = _Stats()
        # zone=NULL aggregates statistics across every malloc zone.
        libc.malloc_zone_statistics(None, ctypes.byref(stats))
        return stats.size_in_use / 1e6
    except Exception:
        return None


def _vmmap_snapshot(pid: int) -> str:
    """Best-effort `vmmap -summary` + `footprint` for region-level growth.

    These read the task's region map, which is often still permitted when
    memory-content scanning (leaks) is not; on denial we return the error text.
    """
    import subprocess

    out = []
    for cmd in (["vmmap", "-summary", str(pid)], ["footprint", str(pid)]):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            out.append(f"$ {' '.join(cmd)}\n{res.stdout or res.stderr}")
        except Exception as exc:  # tool missing / denied
            out.append(f"$ {' '.join(cmd)}\n<failed: {exc}>")
    return "\n".join(out)


def _load_texts(args: argparse.Namespace) -> list[str]:
    if args.texts:
        lines = [ln.strip() for ln in Path(args.texts).read_text().splitlines() if ln.strip()]
        if not lines:
            raise SystemExit(f"no usable lines in {args.texts}")
        return lines
    # Synthetic texts of varying word counts so --vary-length produces a range of
    # sequence lengths (4..64 words) for the dynamic-shape stress test.
    base = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    pool = base * 12
    count = max(args.synthetic, args.batch_size)
    return [" ".join(pool[: ((i % 16) + 1) * 4]) for i in range(count)]


def _build_feed(tokenizer: object, input_names: set[str], texts: list[str]):
    import numpy as np

    enc = tokenizer.encode_batch(texts)  # type: ignore[attr-defined]
    feed: dict[str, object] = {}
    if "input_ids" in input_names:
        feed["input_ids"] = enc.input_ids
    if "attention_mask" in input_names:
        feed["attention_mask"] = enc.attention_mask
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.zeros_like(enc.input_ids)
    return feed, tuple(enc.input_ids.shape)


def run_loop(args: argparse.Namespace) -> int:
    from bge_m3_bench.common.config import parse_provider_option_items
    from bge_m3_bench.common.logging import configure_logging, get_logger
    from bge_m3_bench.server.runtime import OnnxModel
    from bge_m3_bench.server.tokenizer import BgeTokenizer

    configure_logging(args.log_level)
    log = get_logger("coreml_memcheck")

    if args.ort_verbose:
        import onnxruntime as ort

        ort.set_default_logger_severity(0)  # 0 = VERBOSE

    provider_options = parse_provider_option_items(args.provider_option or [])
    model = OnnxModel(args.model, provider=args.provider, provider_options=provider_options)
    tokenizer = BgeTokenizer.from_file(
        args.tokenizer, max_length=args.max_length, pad_length=args.pad_length or None
    )
    input_names = set(model.input_names())
    texts = _load_texts(args)
    n = len(texts)
    bs = args.batch_size
    pid = os.getpid()

    log.info(
        "memcheck start",
        extra={
            "fields": {
                "pid": pid,
                "provider_requested": args.provider,
                "provider_active": model.active_provider,
                "coreml_serialized": model.coreml_serialized,
                "coreml_autorelease_pool": model.coreml_autorelease_pool,
                "pad_length": args.pad_length,
                "vary_length": args.vary_length,
                "iters": args.iters,
                "batch_size": bs,
            }
        },
    )
    print(f"[pid {pid}] attach native tools while running, e.g.:", file=sys.stderr)
    print(f"  leaks {pid} | head -60 ; vmmap {pid} ; heap {pid}", file=sys.stderr)
    print(f"  (run under MallocStackLogging=1 for: malloc_history {pid} <addr>)", file=sys.stderr)
    print("  if 'not debuggable': re-sign python with get-task-allow (see --help)", file=sys.stderr)

    if args.tracemalloc:
        import tracemalloc

        tracemalloc.start(25)
    if args.vmmap:
        log.info("vmmap_start", extra={"fields": {"snapshot": _vmmap_snapshot(pid)}})

    def _mb_fields(rss: float, malloc: float | None) -> dict[str, object]:
        # non_malloc isolates GPU/Metal/mmap growth (CoreML) from the C heap.
        fields: dict[str, object] = {"rss_mb": round(rss, 1)}
        if malloc is not None:
            fields["malloc_mb"] = round(malloc, 1)
            fields["non_malloc_mb"] = round(rss - malloc, 1)
        return fields

    rss0 = _rss_mb()
    malloc0 = _malloc_in_use_mb()
    rss_peak = rss0
    for i in range(1, args.iters + 1):
        if args.vary_length:
            batch = [texts[(i + j) % n] for j in range(bs)]
        else:
            batch = [texts[j % n] for j in range(bs)]
        feed, shape = _build_feed(tokenizer, input_names, batch)
        model.run(feed)  # type: ignore[arg-type]
        rss = _rss_mb()
        rss_peak = max(rss_peak, rss)
        if i == 1 or i % args.rss_every == 0:
            fields = {"i": i, "shape": list(shape), "delta_mb": round(rss - rss0, 1)}
            fields.update(_mb_fields(rss, _malloc_in_use_mb()))
            log.info("iter", extra={"fields": fields})
        if args.gc_every and i % args.gc_every == 0:
            gc.collect()

    rss_end = _rss_mb()
    malloc_end = _malloc_in_use_mb()
    done: dict[str, object] = {
        "iters": args.iters,
        "rss_start_mb": round(rss0, 1),
        "rss_end_mb": round(rss_end, 1),
        "rss_peak_mb": round(rss_peak, 1),
        "growth_mb": round(rss_end - rss0, 1),
    }
    if malloc0 is not None and malloc_end is not None:
        done["malloc_start_mb"] = round(malloc0, 1)
        done["malloc_end_mb"] = round(malloc_end, 1)
        done["malloc_growth_mb"] = round(malloc_end - malloc0, 1)
        # The growth not accounted for by the C heap — CoreML/Metal native memory.
        done["non_malloc_growth_mb"] = round((rss_end - rss0) - (malloc_end - malloc0), 1)
    log.info("memcheck done", extra={"fields": done})
    if args.vmmap:
        log.info("vmmap_end", extra={"fields": {"snapshot": _vmmap_snapshot(pid)}})

    if args.tracemalloc:
        import tracemalloc

        gc.collect()
        snapshot = tracemalloc.take_snapshot()
        for stat in snapshot.statistics("lineno")[:10]:
            log.info("tracemalloc_top", extra={"fields": {"stat": str(stat)}})

    if args.hold:
        log.info(
            "holding for native inspection",
            extra={"fields": {"pid": pid, "hold_sec": args.hold}},
        )
        time.sleep(args.hold)
    return 0


def main() -> None:
    raise SystemExit(run_loop(build_parser().parse_args()))


if __name__ == "__main__":
    main()
