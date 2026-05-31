# bge-m3-bench

Benchmark ONNX Runtime inference locally, through gRPC, in Docker, on VPS
machines, or in Kubernetes. Results include latency stats, validation status,
and environment metadata.

## Quick Start

```bash
make sync
make model

uv run --extra cpu onnx-bench local \
  --model models/tiny_mlp.onnx \
  --provider cpu \
  --iterations 100 \
  --out results/local_cpu.json
```

ONNX Runtime ships as three wheels that all provide the same `onnxruntime`
import package, so exactly one must be installed. Select it with a mutually
exclusive extra: `cpu` (default for local dev and CI), `gpu`
(`onnxruntime-gpu`, CUDA/TensorRT), or `openvino` (`onnxruntime-openvino`).
`make sync` installs the `cpu` extra; the CUDA and OpenVINO Docker images select
`gpu`/`openvino` instead.

## E2E Runbooks

- [Local and gRPC](docs/e2e-local-grpc.md): local baseline, local server, gRPC benchmark, reusable datasets.
- [Docker](docs/e2e-docker.md): build the image, run the server in Docker, benchmark from the host.
- [VPS over gRPC](docs/e2e-vps-grpc.md): run the server on DigitalOcean, Hetzner, Linode, Vultr, AWS, GCP, Azure, or any Ubuntu VPS.
- [Kubernetes](docs/e2e-kubernetes.md): deploy the server, run the benchmark Job, copy results.

## Model And API Guides

- [Export FP16 and INT8 models](docs/model-export-fp16-int8.md)
- [Call the gRPC API with grpcurl](docs/grpcurl.md)
- [Architecture](docs/architecture.md)
- [Metadata schema](docs/metadata-schema.md)

## Common Commands

```bash
make sync       # install dependencies
make model      # build models/tiny_mlp.onnx
make proto      # regenerate protobuf stubs
make lint       # ruff lint
make typecheck  # mypy
make test       # pytest
make serve      # start local gRPC server with the demo model
```
