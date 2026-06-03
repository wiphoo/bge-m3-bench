# Metrics (JSONL)

`bge-m3-bench` writes one `.jsonl` file: **one record per measured request**
(raw), then **one final summary record**. The server emits only raw data; every
aggregate below is computed by the benchmark client.

## Request records (`type: "request"`)

One per measured request — raw, unprocessed. Times are microseconds; sizes are
serialized protobuf bytes. There is exactly one record per measured request
(success or failure), so the number of `type: "request"` rows equals
`grpc_metrics.total_requests`.

**Successful request** (`ok: true`):

| field | meaning |
|---|---|
| `i` | request index (0-based, monotonic across all workers, ordered by completion) |
| `ok` | `true` |
| `num_inputs` | texts in this batch |
| `total_tokens` | tokens across the batch |
| `token_counts` | per-input token counts (raw; summary token percentiles derive from these) |
| `tokenize_us` / `inference_us` / `postprocess_us` | server-side phase times (`tokenize` includes assembling the ONNX feed) |
| `server_e2e_us` | `tokenize + inference + postprocess` |
| `client_e2e_us` | client wall round-trip |
| `request_bytes` / `response_bytes` | wire sizes |

**Failed request** (`ok: false`) — emitted when an `Embed` RPC raises
`grpc.RpcError` during the measured window (carries no timings):

| field | meaning |
|---|---|
| `i` | request index (same monotonic sequence as successes) |
| `ok` | `false` |
| `error_code` | gRPC status code name (e.g. `UNAVAILABLE`, `DEADLINE_EXCEEDED`) |
| `error` | gRPC status detail string |

## Summary record (`type: "summary"`)

### benchmark
`benchmark_id` (auto `bge-m3-grpc-<provider>-<precision>-bs<N>-c<concurrency>`
unless `--benchmark-id`), `timestamp` (UTC), `benchmark_type` (`grpc_service`),
`duration_sec` (measured window), `warmup_sec`.

### model
`model_name` (`--model-name`, else server's), `model_revision`
(`--model-revision`), `embedding_dim`, `precision` (`--precision`),
`quantization` (`--quantization`), `inputs`/`outputs` (ONNX I/O specs from
`GetSpec`). `precision`/`quantization` are run **labels** for the report — the
actual artifact precision is chosen at build time with
`make model PRECISION=fp32|fp16|int8`; set these flags to match the model you
serve. The served ONNX dtypes are always reflected in `inputs`/`outputs`.

### runtime / machine
From the server's `GetSpec`. `runtime`: `runtime`, `runtime_version`,
`execution_provider`, available providers. `machine`: `hostname`, `os`,
`architecture`, `cpu_model`, `cpu_physical_cores`, `cpu_logical_cores`,
`cpu_freq_max_mhz`, `cpu_freq_min_mhz`, `cpu_freq_current_mhz`,
`cpu_isa_extensions`, `ram_total_mb`, `containerized`. The frequency fields and
`cpu_isa_extensions` (a filtered list of throughput-relevant SIMD/ISA flags —
`avx2`, `avx512f`, `avx512_vnni`, `amx_*`, ARM `neon`/`sve`/`i8mm`, …) describe
what most explains throughput differences between CPUs; any field is `null`/`[]`
when the host doesn't expose it (e.g. frequency in many VMs/containers, flags on
non-Linux).

### input
`num_inputs`, `total_tokens`, `avg_tokens_per_input`,
`p50_tokens_per_input`, `p95_tokens_per_input` (rounded floats, not truncated),
`max_tokens_per_input` (from per-input `token_counts`).

### raw_onnx_metrics (server-side)
`tokenize_total_time_ms`, `inference_total_time_ms`, `postprocess_total_time_ms`,
`e2e_total_time_ms`; `{tokenize,inference,e2e}_{tokens,inputs}_per_sec`;
`inference_p50_ms`, `inference_p95_ms`, `inference_p99_ms`.

### grpc_metrics
`total_requests` (= successful + failed), `successful_requests`,
`failed_requests` (gRPC errors caught during the measured window),
`error_rate` (`failed / total`), `client_concurrency` (the `--concurrency`
value: number of concurrent in-flight requests, each on its own gRPC channel),
`client_batch_size`, `requests_per_sec`, `inputs_per_sec`, `tokens_per_sec`,
`client_e2e_p50/p95/p99_ms`, `grpc_overhead_p50/p95/p99_ms` (= `client_e2e −
server_e2e`, may be slightly negative under timing noise),
`request_size_bytes_avg`, `response_size_bytes_avg`. Throughput rates and size
averages are computed over **successful** requests; percentiles likewise cover
only successful requests.

### resource_metrics
`cpu_percent_avg`, `cpu_percent_peak` (process-wide, can exceed 100% across
cores), `memory_rss_peak_mb` — bench-reduced from the server's raw resource
samples. The server retains samples in a bounded ring buffer (default 100k) and
the client resets it at the start of each measured window. `gpu_utilization_avg`
/ `gpu_memory_peak_mb` are `null` (CPU MVP).

### analysis
Derived, normalized read of the run for comparing CPUs and judging headroom
(computed entirely client-side from `grpc_metrics`, `resource_metrics`, and
`machine`). Any value whose inputs are unknown is `null` — never guessed.

- `efficiency`: `inputs_per_sec` / `tokens_per_sec` (echoed), plus throughput
  normalized per core and per clock: `inputs_per_sec_per_physical_core`,
  `inputs_per_sec_per_logical_core`, `inputs_per_sec_per_ghz` (embeddings per
  clock, vs `cpu_freq_max_mhz`), `inputs_per_sec_per_physical_core_ghz` (the most
  apples-to-apples cross-CPU number), `tokens_per_sec_per_physical_core`.
- `memory`: `ram_total_mb`, `rss_peak_mb`, `headroom_mb`, `utilization_pct`, and
  `sufficient` (bool: peak RSS below 90% of RAM; `null` when RAM/peak unknown).
- `cpu_utilization`: `cpu_percent_avg`, `logical_cores`, `concurrency`, and
  `core_utilization_pct` (share of total core capacity used — low values flag an
  under-saturated run; raise `--concurrency`).
- `notes`: short human/LLM-readable strings interpreting the above (memory
  verdict, CPU saturation, available SIMD/ISA, and an efficiency one-liner).

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

Omitted for now, to be added later: `server_e2e`, `tokenize`, and `e2e` latency
percentiles; `queue_wait_*`; request-decode / response-encode split;
`embeddings_per_sec` (≡ `inputs_per_sec`); GPU utilization/memory.
