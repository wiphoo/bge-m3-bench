# Architecture

## Components

```
                    ┌─────────────────────────────┐
                    │        InferenceService     │  proto/inference.proto
                    │  Predict / Health /         │
                    │  ModelMetadata / ListModels │
                    └──────────────┬──────────────┘
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        │                  server/                             │
        │  grpc_server  ──►  service  ──►  ModelRegistry       │
        │                     │              │                 │
        │              serialization     OnnxModel (runtime)   │
        └─────────────────────┬───────────────┬───────────────┘
                              │               │
                       common/providers   onnxruntime.InferenceSession
                       (cpu/openvino/cuda/coreml/tensorrt)
```

`OnnxModel` (`server/runtime.py`) is the single execution path. Both the
in-process baseline benchmark and the gRPC service call it, so latency
differences between `local` and `grpc` transports isolate the gRPC +
serialization overhead, not model differences.

## Execution providers

`common/providers.py` maps a logical provider name to the concrete ONNX Runtime
provider and resolves it against the locally available runtime. If the requested
provider is unavailable (e.g. CUDA on a CPU-only box) it transparently falls
back to CPU while still recording the *requested* provider in metadata, so the
same command runs everywhere.

## Benchmark pipeline

1. **Dataset** (`benchmark/dataset.py`) — deterministic, versioned, fingerprinted
   inputs shaped to the model.
2. **Validation** (`benchmark/validation.py`) — finite outputs + reproducibility;
   failure aborts the run.
3. **Run** (`benchmark/runner.py`) — warmup, timed iterations, latency capture.
4. **Stats** (`benchmark/stats.py`) — p50/p90/p95/p99, throughput.
5. **Metadata** (`metadata/collector.py`) — full environment snapshot attached
   to every result.
6. **Report** (`benchmark/report.py`, `viz.py`, `scripts/build_report.py`) —
   JSON + CSV, charts, HTML notebook.

## Wire format

Tensors travel as raw little-endian, C-contiguous bytes plus an explicit dtype
and shape (`server/serialization.py`), avoiding protobuf repeated-field overhead
and keeping (de)serialization a single buffer copy.
