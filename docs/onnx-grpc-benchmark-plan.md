# ONNX Runtime gRPC Service Benchmark Platform

## Project Plan, Milestones, and Definition of Done (DoD)

## Goals

Build a production-ready gRPC service wrapping ONNX Runtime that:

- Serves ONNX models via gRPC
- Supports benchmarking across CPU/GPU architectures
- Captures comprehensive machine metadata
- Produces reproducible benchmark results
- Runs identically on local machines, Docker, and Kubernetes
- Supports future execution providers:
  - CPU
  - OpenVINO
  - CUDA
  - CoreML
  - TensorRT

---

# Milestone 0: Foundation

## Objective

Establish repository structure and developer experience.

## Deliverables

### Repository

```text
onnx-grpc-benchmark/
├── proto/
├── src/
│   ├── server/
│   ├── benchmark/
│   ├── metadata/
│   └── common/
├── models/
├── results/
├── notebooks/
├── docs/
├── tests/
├── Dockerfile
├── Makefile
├── pyproject.toml
└── uv.lock
```

### Tooling

- uv
- ruff
- mypy
- pytest
- pre-commit

### CI

- lint
- unit tests
- proto generation validation

## DoD

- [ ] `uv sync` works on clean machine
- [ ] `make lint`
- [ ] `make test`
- [ ] `make proto`
- [ ] GitHub Actions passing
- [ ] Generated protobuf checked

---

# Milestone 1: ONNX Runtime Baseline

## Objective

Run a single ONNX model locally.

## Scope

- CPU only
- FP32 only
- Single model

## DoD

- [ ] Load ONNX model
- [ ] Execute inference
- [ ] Output valid results
- [ ] p50 latency
- [ ] p95 latency
- [ ] p99 latency
- [ ] Results saved as JSON

---

# Milestone 1.1: Benchmark Dataset

## Objective

Create repeatable benchmark inputs.

## DoD

- [ ] Dataset versioned
- [ ] Dataset documented
- [ ] Reproducible benchmark runs

---

# Milestone 1.2: Machine Metadata Collection

## Objective

Capture benchmark environment completely.

## DoD

- [ ] Metadata exported
- [ ] JSON schema documented
- [ ] Metadata attached to every benchmark

---

# Milestone 1.3: Validation Framework

## Objective

Ensure benchmark correctness.

## DoD

- [ ] Validation report generated
- [ ] Failed validation causes benchmark failure
- [ ] Output reproducibility verified

---

# Milestone 2: gRPC Service MVP

## Objective

Wrap ONNX Runtime with gRPC.

## DoD

- [ ] gRPC server starts
- [ ] Model loaded once
- [ ] Client inference works
- [ ] Health endpoint works

---

# Milestone 2.1: Tensor Serialization

## Objective

Efficient tensor transport.

## DoD

- [ ] Roundtrip tests pass
- [ ] No tensor corruption
- [ ] Benchmark serialization overhead

---

# Milestone 3: Benchmark Framework

## Objective

Benchmark through gRPC.

## DoD

- [ ] CLI benchmark tool
- [ ] JSON output
- [ ] CSV output

---

# Milestone 4: Visualization

## Objective

Generate benchmark reports.

## DoD

- [ ] Notebook reproducible
- [ ] Charts generated automatically
- [ ] Export PNG and HTML

---

# Milestone 5: Docker Standardization

## Objective

Run benchmark identically everywhere.

## DoD

- [ ] Docker build works
- [ ] Results identical to local
- [ ] Image version tagged

---

# Milestone 6: OpenVINO Provider

## Objective

Add Intel optimization.

## DoD

- [ ] OpenVINO image built
- [ ] Benchmark comparison report

---

# Milestone 7: Apple Silicon Support

## Objective

Benchmark Apple machines.

## DoD

- [ ] Native arm64 support
- [ ] Apple benchmark report

---

# Milestone 8: GPU Support

## Objective

Benchmark NVIDIA GPUs.

## DoD

- [ ] GPU metadata captured
- [ ] CUDA benchmark report
- [ ] TensorRT benchmark report

---

# Milestone 9: Kubernetes Execution

## Objective

Run benchmarks on any cluster.

## DoD

- [ ] One-command benchmark execution
- [ ] Results exported
- [ ] Pod metadata attached

---

# Milestone 10: Multi-Model Registry

## Objective

Support many models.

## DoD

- [ ] Multiple models loaded
- [ ] Dynamic model selection
- [ ] Model metadata endpoint

---

# Production Release Criteria

## Functional

- [ ] gRPC inference works
- [ ] Multi-provider support
- [ ] Benchmark automation

## Reliability

- [ ] Health checks
- [ ] Graceful shutdown
- [ ] Structured logging

## Benchmark Quality

- [ ] Metadata attached
- [ ] Validation enforced
- [ ] Reproducible results

## Portability

- [ ] Local
- [ ] Docker
- [ ] Kubernetes
- [ ] x86
- [ ] ARM64
- [ ] Apple Silicon
