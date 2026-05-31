# Benchmark Metadata Schema

Every benchmark result embeds a complete environment snapshot so runs are
reproducible and comparable across machines (Milestone 1.2). The authoritative
schema is produced by `onnx_grpc_benchmark.metadata.metadata_schema()`:

```bash
uv run --extra cpu onnx-bench metadata --schema
```

`schema_version` is **1.0.0**.

## Fields

| Field | Type | Notes |
| --- | --- | --- |
| `schema_version` | string | Metadata document version |
| `collected_at` | string (date-time) | UTC ISO-8601 timestamp |
| `hostname` | string | Machine hostname |
| `os` | object | `system`, `release`, `version`, `platform` |
| `cpu` | object | `arch`, `model_name`, logical/physical cores, `max_freq_mhz` |
| `memory` | object | `total_bytes`, `available_bytes` |
| `gpu` | object | `present`; `devices[]` with name/memory/driver/compute capability (Milestone 8) |
| `runtime` | object | Python, ONNX Runtime, numpy versions; `available_providers` |
| `container` | object | `in_docker`, `in_kubernetes`, `kubernetes` pod/node metadata (Milestone 9) |
| `git_revision` | string \| null | HEAD commit of the working tree |
| `argv` | string[] | Process arguments |
| `extra` | object | Caller-supplied context |

## Reproducibility

* **Datasets** are generated from a fixed seed and content-hashed
  (`Dataset.fingerprint()`); the fingerprint is recorded in each result's
  config and verified on load.
* **Validation** asserts outputs are finite and deterministic across repeated
  runs; a failure aborts the benchmark (Milestone 1.3).
* **Metadata** captures the exact runtime/providers so results are attributable
  to a specific environment.
