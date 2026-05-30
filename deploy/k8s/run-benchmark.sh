#!/usr/bin/env bash
# One-command Kubernetes benchmark execution (Milestone 9).
#
# Deploys the server, runs the benchmark Job, waits for completion, and copies
# results out of the Job pod into ./results/.
set -euo pipefail

NAMESPACE="${NAMESPACE:-default}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$HERE/../../results}"

echo ">> Deploying server"
kubectl -n "$NAMESPACE" apply -f "$HERE/deployment.yaml"
kubectl -n "$NAMESPACE" rollout status deploy/onnx-grpc-benchmark --timeout=120s

echo ">> Running benchmark job"
kubectl -n "$NAMESPACE" delete job onnx-grpc-benchmark-run --ignore-not-found
kubectl -n "$NAMESPACE" apply -f "$HERE/benchmark-job.yaml"
kubectl -n "$NAMESPACE" wait --for=condition=complete job/onnx-grpc-benchmark-run --timeout=300s

POD="$(kubectl -n "$NAMESPACE" get pods -l app=onnx-grpc-benchmark-run -o jsonpath='{.items[0].metadata.name}')"
mkdir -p "$OUT_DIR"
echo ">> Copying results from $POD"
kubectl -n "$NAMESPACE" cp "$POD:/results/k8s_run.json" "$OUT_DIR/k8s_run.json"
kubectl -n "$NAMESPACE" cp "$POD:/results/k8s_run.csv" "$OUT_DIR/k8s_run.csv" || true
echo ">> Results written to $OUT_DIR"
