# Repository Agent Instructions

## Project Overview

`bge-m3-bench` is a focused benchmark for **BGE-M3 text embeddings served over
gRPC**. The pipeline is **gRPC → tokenize → ONNX inference → embedding**. The
server is a thin, raw-data provider; the benchmark client drives it and computes
all metrics, writing a **JSONL** artifact (one raw record per request + a final
summary record).

The Python distribution and import package are both `bge_m3_bench`.

## Architecture (important)

- **Separation of concerns:** the embedding *service* returns only raw data
  (raw per-request timings/counts/embeddings + raw resource samples). **All**
  metric aggregation — percentiles, throughput, resource avg/peak, summary —
  lives in the *bench* layer. The server has no metrics/summary code.
- **CPU only for now**, with a `cuda` provider seam for later.

## Tech Stack

- Python `>=3.12`; package manager `uv` (`uv.lock`, run via `uv run`).
- Build backend `hatchling`; wheel packages `src/bge_m3_bench`.
- Core deps: `onnxruntime`, `numpy`, `grpcio`, `grpcio-health-checking`,
  `grpcio-reflection`, `protobuf`, `psutil`, `click`, `tokenizers`.
- Tooling: ruff (lint+format, line 100, rules E,F,I,UP,B,C4,SIM,RUF), mypy
  (`disallow_untyped_defs`), pytest. Generated protobuf stubs are excluded from
  ruff/mypy by config — do not work around that.

## Repository Structure

- `src/bge_m3_bench/`
  - `common/` — `config.py`, `logging.py`, `serialize.py` (float32 matrix ↔ bytes).
  - `server/` — `runtime.py` (`OnnxModel`), `providers.py`, `tokenizer.py`,
    `embedder.py`, `resources.py` (raw RSS/CPU sampler), `spec.py`,
    `service.py`, `grpc_server.py`, `__main__.py`.
  - `client.py` — gRPC client.
  - `bench/` — `stats.py`, `validation.py`, `metrics.py`, `cli.py`.
  - `generated/` — generated protobuf/gRPC stubs. **Do not hand-edit.**
- `proto/embedding.proto` — RPC contract; regenerate stubs after changes.
- `scripts/` — `gen_proto.py`, `make_test_embedding_model.py`.
- Console entry points: `bge-m3-server`, `bge-m3-bench`.

## Development Workflow

1. `make sync` to set up the venv.
2. If you change `proto/embedding.proto`, regenerate with `make proto` (never
   edit `src/bge_m3_bench/generated/` by hand).
3. Add/update tests under `tests/`.
4. Run `make lint`, `make typecheck`, `make test` before declaring completion.

### Common commands

```bash
make sync       # uv sync (install deps)
make model      # build the tiny test embedding model + tokenizer
make proto      # regenerate protobuf/gRPC stubs
make lint       # ruff check
make format     # ruff format + ruff check --fix
make typecheck  # mypy src
make test       # pytest
```

## Rules

- Keep the service raw-data-only; never compute metrics server-side.
- All functions typed; line length 100; ruff rule sets as configured.
- Keep the gRPC/proto contract and CLI flags backward compatible unless the task
  says otherwise.
- Update `README.md` / `docs/metrics.md` when commands, the gRPC/CLI API, or the
  metrics schema change.
- Do not commit secrets, model weights, or large artifacts.

## Agent File Policy

This file is canonical. Tool-specific files symlink to it: `CLAUDE.md`,
`GEMINI.md`, `codex.md`, `opencode.md` → `AGENTS.md`. Do not duplicate content.
