# Metrics (JSONL)

`bge-m3-bench` writes one `.jsonl` file: **one record per measured request**
(raw), then **one final summary record**. The server emits only raw data; every
aggregate below is computed by the benchmark client.

## Request records (`type: "request"`)

One per measured request — raw, unprocessed. Times are microseconds; sizes are
serialized protobuf bytes.

| field | meaning |
|---|---|
| `i` | request index (0-based) |
| `num_inputs` | texts in this batch |
| `total_tokens` | tokens across the batch |
| `token_counts` | per-input token counts (raw; summary token percentiles derive from these) |
| `tokenize_us` / `inference_us` / `postprocess_us` | server-side phase times (`tokenize` includes assembling the ONNX feed) |
| `server_e2e_us` | `tokenize + inference + postprocess` |
| `client_e2e_us` | client wall round-trip |
| `request_bytes` / `response_bytes` | wire sizes |

## Summary record (`type: "summary"`)

### benchmark
`benchmark_id` (auto `bge-m3-grpc-<provider>-<precision>-bs<N>-c1` unless
`--benchmark-id`), `timestamp` (UTC), `benchmark_type` (`grpc_service`),
`duration_sec` (measured window), `warmup_sec`.

### model
`model_name` (`--model-name`, else server's), `model_revision`
(`--model-revision`), `embedding_dim`, `precision` (`--precision`),
`quantization` (`--quantization`), `inputs`/`outputs` (ONNX I/O specs from
`GetSpec`).

### runtime / machine
From the server's `GetSpec`. `runtime`: `runtime`, `runtime_version`,
`execution_provider`, available providers. `machine`: `hostname`, `os`,
`architecture`, `cpu_model`, `cpu_physical_cores`, `cpu_logical_cores`,
`ram_total_mb`, `containerized`.

### input
`num_inputs`, `total_tokens`, `avg_tokens_per_input`,
`p50_tokens_per_input`, `p95_tokens_per_input` (rounded floats, not truncated),
`max_tokens_per_input` (from per-input `token_counts`).

### raw_onnx_metrics (server-side)
`tokenize_total_time_ms`, `inference_total_time_ms`, `postprocess_total_time_ms`,
`e2e_total_time_ms`; `{tokenize,inference,e2e}_{tokens,inputs}_per_sec`;
`inference_p50_ms`, `inference_p95_ms`, `inference_p99_ms`.

### grpc_metrics
`total_requests`, `successful_requests`, `failed_requests`, `error_rate`,
`client_concurrency` (1 in MVP), `client_batch_size`, `requests_per_sec`,
`inputs_per_sec`, `tokens_per_sec`, `client_e2e_p50/p95/p99_ms`,
`grpc_overhead_p50/p95/p99_ms` (= `client_e2e − server_e2e`, may be slightly
negative under timing noise), `request_size_bytes_avg`,
`response_size_bytes_avg`.

### resource_metrics
`cpu_percent_avg`, `cpu_percent_peak` (process-wide, can exceed 100% across
cores), `memory_rss_peak_mb` — bench-reduced from the server's raw resource
samples. The server retains samples in a bounded ring buffer (default 100k) and
the client resets it at the start of each measured window. `gpu_utilization_avg`
/ `gpu_memory_peak_mb` are `null` (CPU MVP).

### validation
`embedding_dim`, `embedding_dtype`, `embedding_norm_mean`, `embedding_norm_std`,
`nan_count`, `inf_count`, `zero_vector_count`, `passed` (bool — all checks
passed), `reasons` (list of failure strings; empty when `passed`). With
`--ref-model/--ref-tokenizer`: `cosine_similarity_mean_vs_reference`,
`max_abs_diff_vs_reference` (else `null`). The normalize check is **per row**
(worst row deviation, not the batch mean), and a reference whose shape differs
from the served embeddings is reported as a failed validation reason rather than
crashing the run. `--fail-on-invalid` aborts the run when a check fails;
otherwise the failure is recorded in `passed`/`reasons` and the run completes.
Non-finite norm/reference scalars are emitted as `null` (the failure is still
flagged via `nan_count`/`inf_count`) so the JSONL stays valid JSON.
`--ref-model` and `--ref-tokenizer` must be supplied together.

## Deferred (not in the MVP)

Omitted for now, to be added later: client concurrency > 1; `server_e2e`,
`tokenize`, and `e2e` latency percentiles; `queue_wait_*`; request-decode /
response-encode split; `embeddings_per_sec` (≡ `inputs_per_sec`); GPU
utilization/memory.
