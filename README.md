# bge-m3-bench

Benchmark **BGE-M3 text embeddings served over gRPC**. The measured pipeline is
**gRPC → tokenize → ONNX inference → embedding**. The server is a thin raw-data
provider; the benchmark client drives it and writes a **JSONL** artifact — one
raw record per request plus a final summary record with aggregated stats.

CPU-only today, with a `cuda` provider seam for later.

## Architecture

- **Embedding service** (`bge-m3-server`) — receives text, tokenizes (`tokenizers`
  + `tokenizer.json`), runs ONNX, pools (`none|cls|mean`) and optionally
  L2-normalizes, and returns **only raw data**: embeddings, per-request timings
  (tokenize / inference / postprocess), token counts, plus raw resource samples.
  It also serves embedded **grpc.health.v1** and **server reflection**, and a
  `GetSpec` RPC describing the model/runtime/machine.
- **Benchmark client** (`bge-m3-bench`) — owns **all** metric aggregation:
  percentiles, throughput, resource avg/peak, and the summary. The server never
  computes metrics.

## Install

```bash
make sync     # uv sync
```

## Quickstart (tiny demo model, no download)

```bash
make model    # writes models/tiny/fp32/{model.onnx,tokenizer.json}

uv run bge-m3-server \
  --model models/tiny/fp32/model.onnx \
  --tokenizer models/tiny/fp32/tokenizer.json \
  --pooling cls --port 50071 &

uv run bge-m3-bench \
  --address localhost:50071 \
  --warmup-sec 1 --duration-sec 5 --batch-size 8 \
  --ref-model models/tiny/fp32/model.onnx --ref-tokenizer models/tiny/fp32/tokenizer.json \
  --out results/run.jsonl
```

`make model` takes `MODEL` (`tiny` | `bge-m3`) and `PRECISION` (`fp32` | `fp16` |
`int8`), writing `models/<model>/<precision>/{model.onnx,tokenizer.json}` so
variants coexist. fp16/int8 are derived locally from the fp32 ONNX with ONNX
Runtime (int8 via dynamic quantization, fp16 via float16 conversion) — no external
pre-quantized downloads. fp16 on the CPU provider is up-cast to fp32 by ONNX
Runtime (the server logs a warning), so it is mainly for the future `cuda` path.

`results/run.jsonl` contains one `{"type":"request", ...}` row per request and a
final `{"type":"summary", ...}` record. See [docs/metrics.md](docs/metrics.md)
for every field.

## Dump embeddings

`bge-m3-bench` measures performance and does **not** persist the vectors. To get
the actual dense embeddings out of a running server, use `bge-m3-embed`:

```bash
uv run bge-m3-embed \
  --address localhost:50071 \
  --texts inputs.txt \
  --out results/embeddings.jsonl
```

It writes one JSON record per input — `{"i": <index>, "text": ..., "dim": <D>,
"embedding": [<D floats>]}` — calling the same `Embed` RPC the benchmark uses.
Inputs come from `--text` (an inline string, repeatable), else `--texts` (a file
with one input per line), else a small built-in sample — for example:

```bash
uv run bge-m3-embed --address localhost:50071 \
  --text "the quick brown fox" --text "a second sentence" \
  --out results/embeddings.jsonl
```

`--text` and `--texts` are mutually exclusive; `--batch-size` (default `16`)
chunks inputs across `Embed` calls. This is a raw vector dump, distinct from the
`bge-m3-bench` metrics JSONL.

## Real BGE-M3

Export the real model (and its `tokenizer.json`) at the precision you want, then
point the server at the artifacts. The heavy export deps (transformers/torch/optimum)
live in the optional `export` dependency-group, pulled in only for `MODEL=bge-m3`
(or up front with `make sync-export`) — the tiny path above stays lean:

```bash
make model MODEL=bge-m3 PRECISION=fp32   # also: fp16, int8

uv run bge-m3-server \
  --model models/bge-m3/fp32/model.onnx \
  --tokenizer models/bge-m3/fp32/tokenizer.json \
  --pooling cls --normalize

uv run bge-m3-bench \
  --address localhost:50051 \
  --texts inputs.txt --batch-size 16 --concurrency 8 \
  --warmup-sec 10 --duration-sec 60 \
  --model-name BAAI/bge-m3 --model-revision main --precision fp32 \
  --out results/bge_m3.jsonl
```

`--model-name`, `--model-revision`, and `--precision` are **report labels** —
they don't affect inference, only the JSONL summary (and `--precision` feeds the
auto `benchmark_id`). Set them to match the artifact you serve. Served ONNX dtypes
are recorded separately in the summary's `inputs`/`outputs`.

`--texts` is a file with one input per line (a small built-in sample is used if
omitted). Pass `--ref-model/--ref-tokenizer` to validate server embeddings
against a local reference (cosine similarity + max abs diff).

`--concurrency N` drives `N` requests in flight at once (default `1`), each on
its own gRPC channel — raise it to saturate the server's worker pool (set with
`BGE_M3_MAX_WORKERS`, default 8) and measure real throughput. `grpc_metrics`
then reports the true `client_concurrency`, `failed_requests`, and `error_rate`.

Tune ONNX Runtime threading with `--intra-op-threads` / `--inter-op-threads`
(or `BGE_M3_INTRA_OP` / `BGE_M3_INTER_OP`); `0` (the default) leaves ORT's own
defaults in place.

### Choosing an execution provider

`--provider` selects the ONNX Runtime execution provider for your hardware
(`cpu` by default). Pass provider-specific tuning via repeatable
`--provider-option KEY=VALUE` (or `BGE_M3_PROVIDER_OPTIONS="k=v,k2=v2"`). If the
requested provider isn't available in the installed build, the server logs a
warning and falls back to `cpu`, so the same command runs anywhere — the
requested vs. resolved provider are both recorded in the JSONL summary
(`config.provider` vs. `runtime.execution_provider`).

- **Intel x86 → `openvino`.** The `OpenVINOExecutionProvider` ships in the
  `onnxruntime-openvino` wheel, which **replaces** the base `onnxruntime` package
  (the two cannot coexist). Install it deliberately with `make sync-openvino`
  (Python 3.12/3.13 only — there is no cp314 wheel yet), then run e.g. `uv run
  --no-sync bge-m3-server --provider openvino --provider-option device_type=CPU
  ...`. Use `--no-sync` (or activate `.venv` directly) so an implicit `uv run`
  sync doesn't reinstall the base `onnxruntime` wheel and undo the swap.
- **Apple Silicon → `coreml`.** The `CoreMLExecutionProvider` is bundled in the
  standard macOS `onnxruntime` wheel — no extra install. Run `bge-m3-server
  --provider coreml ...` (optionally `--provider-option MLComputeUnits=ALL`).
  CoreML compiles the model on load and caches it; if the default location isn't
  writable (e.g. a read-only mount or container), point it at a writable path
  with `--provider-option ModelCacheDirectory=/tmp/coreml-cache`.
  - *Concurrency & memory:* CoreML/Metal is not reliably safe when driven
    concurrently from multiple gRPC worker threads — under `--concurrency > 1`
    the server could crash / be OOM-killed. CoreML inference is therefore
    **serialized** (one inference at a time; CoreML is a single ANE/GPU resource
    anyway), and each call is wrapped in an autorelease pool (via `pyobjc-core`,
    installed by `make sync` / `make sync-coreml`) to drain Objective-C
    temporaries. For stable memory, set a fixed `--pad-length` (e.g. `512`, ≤
    `--max-length`) so every batch is one **static** input shape — otherwise the
    CoreML EP compiles and caches a new model per sequence length and RSS climbs.
    Further knobs: try the newer backend with `--provider-option
    ModelFormat=MLProgram`; silence the residual `Context leak detected,
    msgtracer returned -1` os_log line with `OS_ACTIVITY_MODE=disable
    bge-m3-server ...`; and if still OOM-killed, lower `BGE_M3_MAX_WORKERS` or the
    client `--concurrency`, or upgrade `onnxruntime`. The artifact records
    `config.coreml_serialized`, `config.coreml_autorelease_pool`, and
    `config.pad_length`; watch `resource_metrics.memory_rss_peak_mb`.
- **AMD x86 → `cpu`.** There is no pip-installable AMD execution provider; the
  default MLAS-backed `cpu` provider is already well-tuned. Get the most from it
  by setting `--intra-op-threads` to your physical core count (and experiment
  with `--inter-op-threads` for concurrent requests).

> **Security:** the server uses plaintext (insecure) gRPC and binds `0.0.0.0`
> by default, exposing the model on all interfaces with no auth. Run it only on
> a trusted network, or bind loopback with `--host 127.0.0.1` (or
> `BGE_M3_HOST=127.0.0.1`).

## Docker

```bash
docker build -t bge-m3-bench .
docker run --rm -p 50051:50051 -v "$PWD/models:/models:ro" bge-m3-bench \
  --model /models/model.onnx --tokenizer /models/tokenizer.json
```

## Commands

```bash
make sync       # install deps
make sync-export # install optional model-export deps (transformers/torch/optimum)
make sync-openvino # swap to the Intel OpenVINO ORT build (replaces onnxruntime)
make sync-coreml # ensure the CoreML autorelease-pool dep (pyobjc-core, macOS)
make model      # build a model: MODEL=tiny|bge-m3 PRECISION=fp32|fp16|int8
make proto      # regenerate protobuf/gRPC stubs
make lint       # ruff check
make typecheck  # mypy
make test       # pytest
```

## Configuration

Server flags mirror `BGE_M3_*` env vars: `BGE_M3_HOST`, `BGE_M3_PORT`,
`BGE_M3_PROVIDER` (`cpu`/`cuda`/`openvino`/`coreml`), `BGE_M3_PROVIDER_OPTIONS`
(`k=v,k2=v2`), `BGE_M3_POOLING`, `BGE_M3_NORMALIZE`, `BGE_M3_MAX_LENGTH`,
`BGE_M3_PAD_LENGTH`, `BGE_M3_INTRA_OP`, `BGE_M3_INTER_OP`, `BGE_M3_MODEL`,
`BGE_M3_TOKENIZER`.
