# Debugging guide

Troubleshooting for `bge-m3-bench` — execution providers, the CoreML memory
issue, OpenVINO install, and threading. For the metrics schema see
[`metrics.md`](metrics.md).

First step for almost anything: turn on debug logs.

```bash
BGE_M3_LOG_LEVEL=DEBUG bge-m3-server --model ... --tokenizer ...
```

Logs are single-line JSON on stderr. The server records the requested vs.
resolved provider and mitigation flags in `GetSpec` / the JSONL summary's
`config` block (`provider`, `execution_provider`, `coreml_serialized`,
`coreml_autorelease_pool`, `pad_length`).

---

## CoreML: memory growth / "Context leak detected, msgtracer returned -1"

### What's happening

`Context leak detected, msgtracer returned -1` is a macOS os_log/Metal
diagnostic, not an onnxruntime error. On its own it is often benign noise, but it
commonly accompanies **real RSS growth** under the CoreML provider, from three
compounding sources:

1. **Concurrency.** CoreML/Metal is not reliably safe when driven concurrently
   from multiple non-Cocoa threads. Under `--concurrency > 1` several gRPC worker
   threads hit the single CoreML session at once → native instability + N× peak
   memory → the process can be OOM-killed.
2. **Dynamic input shapes.** Variable padded sequence length makes the CoreML EP
   compile and cache a new model per shape → unbounded growth.
3. **Autorelease accumulation.** CoreML inference creates autoreleased
   Objective-C objects on worker threads that have no autorelease pool.

### Mitigations already in the server

- **Inference is serialized** for CoreML (one inference at a time — it's a single
  ANE/GPU resource anyway). `config.coreml_serialized: true` confirms it.
- **Autorelease pool** wraps each CoreML inference via `pyobjc-core` (installed by
  default on macOS / `make sync-coreml`). `config.coreml_autorelease_pool: true`
  confirms it engaged.
- **`--pad-length N`** (or `BGE_M3_PAD_LENGTH`) pins one static input shape (every
  batch padded/truncated to `N`, `≤ --max-length`) so the EP compiles once. Start
  here — it's the highest-leverage memory fix.

```bash
OS_ACTIVITY_MODE=disable \
  bge-m3-server --provider coreml --pad-length 512 --model ... --tokenizer ...
```

`OS_ACTIVITY_MODE=disable` silences the residual os_log line. Other knobs: try
`--provider-option ModelFormat=MLProgram` (newer CoreML backend); lower
`BGE_M3_MAX_WORKERS` or the client `--concurrency`; upgrade `onnxruntime`.

### Isolating the root cause with `coreml_memcheck.py`

`scripts/coreml_memcheck.py` runs a **single-threaded** inference loop (no gRPC,
no thread pool) and prints memory per iteration, so you can attribute growth.

```bash
make coreml-memcheck ARGS="--model models/bge-m3/fp32/model.onnx \
  --tokenizer models/bge-m3/fp32/tokenizer.json --provider coreml --iters 4000 --vmmap"
# or: uv run python scripts/coreml_memcheck.py --help
```

The tooling's own tests live under `tests/debug/` (marked `debug`, excluded from
`make test`); run them with `make test-debug`.

Useful comparisons:

| Run | Tells you |
|-----|-----------|
| `--provider cpu` baseline | Should be flat. Confirms the harness/model aren't the leak. |
| `--provider coreml` single-thread | If it still grows, the leak is the EP — **not** our gRPC threading. |
| `--vary-length` vs `--pad-length 512` | Grows only with `--vary-length` ⇒ per-shape CoreML model caching. |
| `--ort-verbose` | Shows the CoreML EP partitioning / recompiling per shape. |

### Reading the in-process numbers (works despite "not debuggable")

The script reports, from inside the process (no permissions needed):

- `rss_mb` — total resident memory.
- `malloc_mb` — live bytes across all malloc zones (the C/C++ heap).
- `non_malloc_mb` = `rss_mb − malloc_mb` — everything else: Metal/GPU
  (IOAccelerator) buffers and mmap'd CoreML models.
- `--tracemalloc` — top Python allocations at the end.

| Observation | Root cause → next move |
|-------------|------------------------|
| RSS grows, `malloc_mb` flat, `non_malloc_mb` grows | **Native CoreML/Metal GPU memory** → onnxruntime CoreML EP / upstream; not our Python. Try `--pad-length`, `ModelFormat=MLProgram`, onnxruntime upgrade. |
| RSS grows **and** `malloc_mb` grows together | C/C++ heap leak → capture allocation stacks (below). |
| `--tracemalloc` top grows | Leak in our Python code. |
| Grows only via the server, flat single-threaded | Concurrency/threading — serialization should cover it; check `config.coreml_serialized`. |

### macOS native tools and the "not debuggable" restriction

`leaks`, `malloc_history`, and `heap` give allocation stacks but need to read the
target's memory. The uv-managed Python is a hardened-runtime build **without**
`com.apple.security.get-task-allow`, so they fail with:

> process <pid> is not debuggable. due to security restrictions, leaks can only
> show or save contents of readonly memory

`sudo` does **not** bypass this (AMFI denies the task port). Options:

- **No setup:** rely on the script's `malloc_mb` / `non_malloc_mb` /
  `--tracemalloc` split above, plus `--vmmap` (which logs `vmmap -summary` +
  `footprint` for region-level growth — usually permitted even when `leaks` is
  not). Watch for growing `IOAccelerator` / `Metal` regions.
- **Re-sign the interpreter** to unlock `leaks`/`malloc_history`/`heap` (local,
  opt-in):

  ```bash
  cat > /tmp/gta.entitlements <<'EOF'
  <?xml version="1.0" encoding="UTF-8"?>
  <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
  <plist version="1.0">
    <dict><key>com.apple.security.get-task-allow</key><true/></dict>
  </plist>
  EOF
  PY=$(uv run python -c 'import sys; print(sys.executable)')
  codesign -s - -f --entitlements /tmp/gta.entitlements "$PY"
  ```

  Then, with the process running under `MallocStackLogging=1 ... --hold 180`:

  ```bash
  leaks <pid> | head -60          # leaked blocks + stacks
  heap <pid>                       # live objects by class (watch MTL*/CoreML*)
  malloc_history <pid> <address>   # full alloc stack for an address
  ```

---

## Provider selection: requested vs. resolved

If a provider "doesn't seem to apply", check the resolved provider. When the
requested EP isn't available in the installed build, the server logs a warning
and **falls back to CPU** — `config.provider` (requested) then differs from
`runtime.execution_provider` (resolved).

```bash
# What does this onnxruntime build actually support?
uv run python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

`config.coreml_autorelease_pool` / `coreml_serialized` are only `true` when CoreML
is actually active — a quick way to confirm you're not silently on CPU.

---

## OpenVINO (Intel) install issues

`onnxruntime-openvino` **replaces** base `onnxruntime` (both own the `onnxruntime`
import; they can't coexist). Install with `make sync-openvino`, which keeps the
base wheel out via `uv sync --group openvino --no-install-package onnxruntime`.

- **Python 3.14:** there is no cp314 wheel; `make sync-openvino` fails fast with a
  clear message. Use Python 3.12 / 3.13 for OpenVINO.
- **Swap undone after running:** a plain `uv run` re-syncs the lockfile and pulls
  base `onnxruntime` back. Run the server with `uv run --no-sync bge-m3-server
  ...` (or activate `.venv` directly).

---

## Threading / throughput

- `--intra-op-threads` / `--inter-op-threads` (or `BGE_M3_INTRA_OP` /
  `BGE_M3_INTER_OP`); `0` leaves ORT's defaults. For AMD/Intel CPU, set
  intra-op to the physical core count as a starting point.
- The gRPC pool size is `BGE_M3_MAX_WORKERS` (default 8); the client drives N
  in-flight requests with `--concurrency`. Note CoreML inference is serialized
  regardless, so raising `--concurrency` won't parallelize it.
- `--provider-option KEY=VALUE` is a passthrough to the EP (e.g.
  `device_type=CPU` for OpenVINO, `MLComputeUnits=ALL` /
  `ModelCacheDirectory=/tmp/coreml-cache` for CoreML).
