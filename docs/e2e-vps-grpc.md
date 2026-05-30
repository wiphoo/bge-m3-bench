# E2E: VPS gRPC Benchmarking

Use this flow for DigitalOcean Droplets, Hetzner Cloud servers, Linode, Vultr,
AWS EC2, GCP Compute Engine, Azure VMs, or any Ubuntu VPS. The VPS runs only the
gRPC server; your local machine sends benchmark traffic to it and writes the
result files.

## Start The Server

Create a VPS with Ubuntu, SSH in, and install Docker:

```bash
ssh root@$VPS_HOST
apt-get update
apt-get install -y ca-certificates curl docker.io
systemctl enable --now docker
```

Start the server with the bundled demo model:

```bash
docker run -d \
  --name onnx-grpc-benchmark \
  --restart unless-stopped \
  -p 50051:50051 \
  ghcr.io/wiphoo/bge-m3-bench:latest
```

## Serve A Custom ONNX Model

Copy the model to the VPS and mount it into the container:

```bash
scp models/bge-m3-onnx/model.int8.onnx root@$VPS_HOST:/opt/model.onnx

ssh root@$VPS_HOST
docker rm -f onnx-grpc-benchmark || true
docker run -d \
  --name onnx-grpc-benchmark \
  --restart unless-stopped \
  -p 50051:50051 \
  -v /opt/model.onnx:/models/model.onnx:ro \
  ghcr.io/wiphoo/bge-m3-bench:latest \
  --model /models/model.onnx=bge-m3-int8 \
  --provider cpu
```

## Check The Remote Service

Install `grpcurl` on your local machine before running these checks.

```bash
grpcurl \
  -plaintext \
  -import-path proto \
  -proto inference.proto \
  $VPS_HOST:50051 \
  onnx_grpc_benchmark.v1.InferenceService/ListModels
```

## Benchmark Over gRPC

Run this from your local machine:

```bash
uv run onnx-bench grpc \
  --address $VPS_HOST:50051 \
  --model bge-m3-int8 \
  --ref-model models/bge-m3-onnx/model.onnx \
  --warmup 10 \
  --iterations 500 \
  --out results/vps_bge_m3_int8_grpc.json
```

For the bundled demo model, omit `--model`:

```bash
uv run onnx-bench grpc \
  --address $VPS_HOST:50051 \
  --ref-model models/tiny_mlp.onnx \
  --warmup 10 \
  --iterations 500 \
  --out results/vps_tiny_grpc.json
```

## Use An SSH Tunnel

Allow TCP `50051` only from your benchmark runner's public IP, or skip the public
port and tunnel it:

```bash
ssh -N -L 50051:127.0.0.1:50051 root@$VPS_HOST
```

With the tunnel open:

```bash
uv run onnx-bench grpc \
  --address localhost:50051 \
  --ref-model models/tiny_mlp.onnx \
  --out results/vps_tunnel_grpc.json
```

Direct public gRPC benchmarking includes internet round-trip time. To measure
only server-side inference, run `onnx-bench grpc` on the VPS against
`localhost:50051`.
