# E2E: Kubernetes Benchmarking

Deploy the gRPC server, run a benchmark Job, and copy the result files back.

The manifests use the published image `ghcr.io/wiphoo/bge-m3-bench:latest` and
the bundled demo model.

## Run

```bash
deploy/k8s/run-benchmark.sh
```

Optional namespace and output directory:

```bash
NAMESPACE=bench OUT_DIR="$PWD/results" deploy/k8s/run-benchmark.sh
```

The script applies the Deployment, waits for rollout, runs the benchmark Job,
and copies `k8s_run.json` plus `k8s_run.csv` into the output directory.

## What It Runs

- `deploy/k8s/deployment.yaml` starts the gRPC server.
- `deploy/k8s/benchmark-job.yaml` runs `onnx-bench grpc`.
- `deploy/k8s/run-benchmark.sh` orchestrates deploy, wait, benchmark, and copy.
