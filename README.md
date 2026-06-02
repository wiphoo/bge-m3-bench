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
make model    # writes models/tiny_embed.onnx + models/tiny_tokenizer.json

uv run bge-m3-server \
  --model models/tiny_embed.onnx \
  --tokenizer models/tiny_tokenizer.json \
  --pooling cls --port 50071 &

uv run bge-m3-bench \
  --address localhost:50071 \
  --warmup-sec 1 --duration-sec 5 --batch-size 8 \
  --ref-model models/tiny_embed.onnx --ref-tokenizer models/tiny_tokenizer.json \
  --out results/run.jsonl
```

`results/run.jsonl` contains one `{"type":"request", ...}` row per request and a
final `{"type":"summary", ...}` record. See [docs/metrics.md](docs/metrics.md)
for every field.

## Real BGE-M3

Export the model to ONNX and grab its `tokenizer.json`, then point the server at
them:

```bash
uv run bge-m3-server \
  --model path/to/bge-m3/model.onnx \
  --tokenizer path/to/bge-m3/tokenizer.json \
  --pooling cls --normalize

uv run bge-m3-bench \
  --address localhost:50051 \
  --texts inputs.txt --batch-size 16 --concurrency 8 \
  --warmup-sec 10 --duration-sec 60 \
  --model-name BAAI/bge-m3 --model-revision main --precision fp32 \
  --out results/bge_m3.jsonl
```

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
make model      # build the tiny demo model + tokenizer
make proto      # regenerate protobuf/gRPC stubs
make lint       # ruff check
make typecheck  # mypy
make test       # pytest
```

## Configuration

Server flags mirror `BGE_M3_*` env vars: `BGE_M3_HOST`, `BGE_M3_PORT`,
`BGE_M3_PROVIDER` (`cpu`/`cuda`), `BGE_M3_POOLING`, `BGE_M3_NORMALIZE`,
`BGE_M3_MAX_LENGTH`, `BGE_M3_INTRA_OP`, `BGE_M3_INTER_OP`, `BGE_M3_MODEL`,
`BGE_M3_TOKENIZER`.
