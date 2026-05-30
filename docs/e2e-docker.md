# E2E: Docker gRPC Benchmarking

Build the container, run the gRPC server in Docker, and benchmark it from the
host.

## Build

```bash
docker build -t onnx-grpc-benchmark:local .
```

## Run The Server

```bash
docker run --rm -p 50051:50051 onnx-grpc-benchmark:local
```

If host port `50051` is already in use, map another host port:

```bash
docker run --rm -p 50052:50051 onnx-grpc-benchmark:local
```

## Benchmark From The Host

```bash
uv run onnx-bench grpc \
  --address localhost:50051 \
  --ref-model models/tiny_mlp.onnx \
  --out results/docker_grpc.json
```

Use `--address localhost:50052` if you mapped the alternate host port above.

## Use A Custom Model

Mount your model at `/models` and pass the model path to `onnx-server`:

```bash
docker run --rm -p 50051:50051 \
  -v "$PWD/models:/models:ro" \
  onnx-grpc-benchmark:local \
  --model /models/tiny_mlp.onnx \
  --provider cpu
```
