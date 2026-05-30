# Milestone Status

Status of each milestone from the project plan. ✅ implemented & testable in CI;
🟡 implemented, full verification requires specific hardware not present in CI.

## Milestone 0 — Foundation ✅
- Repo structure (`proto/ src/{server,benchmark,metadata,common} models/ results/ notebooks/ docs/ tests/`), `Dockerfile`, `Makefile`, `pyproject.toml`, lockfile.
- Tooling: uv, ruff, mypy, pytest, pre-commit.
- CI: lint, type check, unit tests, proto generation validation (`.github/workflows/ci.yml`).
- Generated protobuf checked in under `src/onnx_grpc_benchmark/generated/`.

## Milestone 1 — ONNX Runtime Baseline ✅
- `OnnxModel` loads and runs a CPU/FP32 model; `onnx-bench local` produces
  p50/p95/p99 latency and saves JSON. See `tests/test_runner.py`.

## Milestone 1.1 — Benchmark Dataset ✅
- Deterministic, versioned, fingerprinted datasets (`benchmark/dataset.py`),
  save/load with reproducibility verification. See `tests/test_dataset.py`.

## Milestone 1.2 — Machine Metadata ✅
- `metadata/collector.py` exports a full snapshot; JSON schema documented
  (`docs/metadata-schema.md`); attached to every result. See `tests/test_metadata.py`.

## Milestone 1.3 — Validation Framework ✅
- Finite-output + reproducibility checks; failure aborts the run
  (`benchmark/validation.py`). See `tests/test_runtime_and_validation.py`.

## Milestone 2 — gRPC Service MVP ✅
- Server starts, model loaded once, client inference works, Health endpoint.
  See `tests/test_grpc_integration.py`.

## Milestone 2.1 — Tensor Serialization ✅
- Raw-bytes tensor transport with roundtrip/no-corruption tests
  (`server/serialization.py`, `tests/test_serialization.py`); transport overhead
  measured by `run_grpc`.

## Milestone 3 — Benchmark Framework ✅
- `onnx-bench` CLI; JSON and CSV output (`benchmark/cli.py`, `report.py`).

## Milestone 4 — Visualization ✅
- Reproducible notebook generated and executed by `scripts/build_report.py`;
  auto charts; PNG + HTML export (`benchmark/viz.py`).

## Milestone 5 — Docker Standardization ✅
- `Dockerfile` (CPU), deterministic build from lockfile, versioned image tags.

## Milestone 6 — OpenVINO Provider 🟡
- `Dockerfile.openvino` + `--provider openvino`; provider abstraction wired.
  Comparison report via the standard CLI on Intel hardware.

## Milestone 7 — Apple Silicon Support 🟡
- Native arm64 (slim base is multi-arch) + CoreML provider supported via the
  provider abstraction (`--provider coreml`). Report via the standard CLI on
  Apple hardware.

## Milestone 8 — GPU Support 🟡
- `Dockerfile.cuda` + `--provider cuda|tensorrt`; GPU metadata captured via
  `nvidia-smi` in `metadata/collector.py`. Reports via the standard CLI on a GPU host.

## Milestone 9 — Kubernetes Execution 🟡
- `deploy/k8s/` Deployment + Job + `run-benchmark.sh` (one command); results
  exported; pod/node metadata injected via downward API and recorded in metadata.

## Milestone 10 — Multi-Model Registry ✅
- `ModelRegistry` loads many models; dynamic selection per request; model
  metadata endpoint (`ModelMetadata`/`ListModels`). See `tests/test_runtime_and_validation.py`.

## Notes on hardware-gated milestones (6–9)

The provider abstraction (`common/providers.py`) means a single code path and a
single CLI cover every provider; only the container base image and host
hardware change. On a machine without the requested accelerator the run falls
back to CPU and records that fact, so the platform is runnable everywhere while
remaining accurate where the hardware exists.
