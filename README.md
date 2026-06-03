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

`--texts` accepts either a local file path or an `https://` URL to a
one-sentence-per-line text file (a small built-in sample is used if omitted).
Pass `--ref-model/--ref-tokenizer` to validate server embeddings against a
local reference (cosine similarity + max abs diff).

`--concurrency N` drives `N` requests in flight at once (default `1`), each on
its own gRPC channel — raise it to saturate the server's worker pool (`--max-workers`
or `BGE_M3_MAX_WORKERS`, default 8) and measure real throughput. `grpc_metrics`
then reports the true `client_concurrency`, `failed_requests`, `error_rate`, and an
`error_codes` breakdown. `--timeout` (default `120`s) bounds each request; the CLI
prints `run_ok` and exits non-zero if **no** request succeeds.

Tune ONNX Runtime threading with `--intra-op-threads` / `--inter-op-threads`
(or `BGE_M3_INTRA_OP` / `BGE_M3_INTER_OP`). `--intra-op-threads` defaults to `-1`
(**auto**): the server caps each ONNX session at `max(1, usable_cores // max_workers)`
threads so `max_workers` concurrent requests don't oversubscribe the CPU. `usable_cores`
is the cores the process can actually use — host physical cores capped by any cgroup CPU
quota / affinity — so it stays correct under Docker/Kubernetes limits or `taskset`. Use `0`
for ORT's own default (all cores — best for `--concurrency 1`), or a positive explicit count.

### Troubleshooting: zero results / all `DEADLINE_EXCEEDED`

If a run reports `successful_requests: 0` and every metric is `0`/`null`, check
`grpc_metrics.error_codes`. All `DEADLINE_EXCEEDED` means requests didn't finish within
`--timeout`: concurrent heavy inferences (e.g. real `BAAI/bge-m3` fp32, batch 16, 512-token
inputs) overran the deadline. Each request needs roughly `single_request_latency × (concurrency / usable_cores)` wall time. Fixes: raise `--timeout`, lower `--concurrency`,
or keep the auto intra-op default so threads ≈ cores. Example: on an 8-core CPU,
`--concurrency 8` with the real fp32 model is far heavier than `--concurrency 2`; start low
and scale up while watching `error_rate`.

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
make model      # build a model: MODEL=tiny|bge-m3 PRECISION=fp32|fp16|int8
make proto      # regenerate protobuf/gRPC stubs
make lint       # ruff check
make typecheck  # mypy
make test       # pytest
```

## Reference datasets

Pre-generated benchmark input files are available for reproducible runs:

| Dataset | Description | Files |
|---|---|---|
| **restaurant/v1** | Thai/English restaurant texts in fixed-ish length buckets (t32, t64, t128, t256, t512, mixed, smoke). Generated from 268 seed restaurant rows. | See [`data/restaurant/manifest.json`](data/restaurant/manifest.json) |

Hosted on Cloudflare R2, each file can be passed directly to `--texts`:

```bash
uv run bge-m3-bench --address localhost:50051 \
  --texts https://public-assets.wiphoo.dev/datasets/restaurants/v1/restaurant_t128_1000.txt \
  --batch-size 16 --concurrency 4 --warmup-sec 10 --duration-sec 60
```

The generation script is hosted alongside the data files (`generate_bge_m3_bench_inputs.py`)
and the manifest lives in [`data/restaurant/manifest.json`](data/restaurant/manifest.json).

## Configuration

Server flags mirror `BGE_M3_*` env vars: `BGE_M3_HOST`, `BGE_M3_PORT`,
`BGE_M3_PROVIDER` (`cpu`/`cuda`), `BGE_M3_POOLING`, `BGE_M3_NORMALIZE`,
`BGE_M3_MAX_LENGTH`, `BGE_M3_INTRA_OP`, `BGE_M3_INTER_OP`, `BGE_M3_MODEL`,
`BGE_M3_TOKENIZER`.
