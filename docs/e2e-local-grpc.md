# E2E: Local And gRPC Benchmarking

Run everything on one machine: create a demo model, benchmark it in-process,
then benchmark the same model through the gRPC server.

## Setup

```bash
make sync
make model
```

This writes `models/tiny_mlp.onnx`.

## Local Baseline

```bash
uv run --extra cpu onnx-bench local \
  --model models/tiny_mlp.onnx \
  --provider cpu \
  --warmup 5 \
  --iterations 100 \
  --out results/local_cpu.json
```

The command writes `results/local_cpu.json` and `results/local_cpu.csv`.

## gRPC Server

Start the server in one terminal:

```bash
uv run --extra cpu onnx-server \
  --model models/tiny_mlp.onnx \
  --provider cpu
```

If port `50051` is already in use, start the server on another port:

```bash
uv run --extra cpu onnx-server \
  --model models/tiny_mlp.onnx \
  --provider cpu \
  --port 50052
```

Run the gRPC benchmark from another terminal:

```bash
uv run --extra cpu onnx-bench grpc \
  --address localhost:50051 \
  --ref-model models/tiny_mlp.onnx \
  --warmup 5 \
  --iterations 100 \
  --out results/grpc_cpu.json
```

`--ref-model` is a local copy of the model used to generate correctly shaped
synthetic inputs. Inference is performed by the running gRPC server.

## Reusable Dataset

Create deterministic inputs once:

```bash
uv run --extra cpu onnx-bench gen-dataset \
  --model models/tiny_mlp.onnx \
  --out results/tiny_dataset.npz \
  --num-samples 64 \
  --seed 1234
```

Use that dataset for local runs:

```bash
uv run --extra cpu onnx-bench local \
  --model models/tiny_mlp.onnx \
  --dataset results/tiny_dataset.npz \
  --out results/local_dataset.json
```
