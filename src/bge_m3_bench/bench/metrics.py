"""Aggregate raw per-request samples + raw resource samples into the summary.

This is the ONLY place metrics are derived. The gRPC service provides raw data;
everything here (sums, throughput, percentiles, resource reduction) is computed
client-side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .stats import percentiles_ms, token_percentiles


@dataclass
class RequestSample:
    """Raw measurements for one Embed request."""

    num_inputs: int
    token_counts: list[int]
    tokenize_us: int
    inference_us: int
    postprocess_us: int
    client_e2e_us: int
    request_bytes: int
    response_bytes: int

    @property
    def total_tokens(self) -> int:
        return int(sum(self.token_counts))

    @property
    def server_e2e_us(self) -> int:
        return self.tokenize_us + self.inference_us + self.postprocess_us


@dataclass
class RunContext:
    benchmark_id: str
    duration_sec: float
    warmup_sec: float
    batch_size: int
    concurrency: int
    model_name: str
    model_revision: str
    precision: str
    quantization: str
    extra: dict[str, Any] = field(default_factory=dict)


def request_row(i: int, s: RequestSample) -> dict[str, Any]:
    """The raw JSONL record streamed per successful request."""
    return {
        "type": "request",
        "i": i,
        "ok": True,
        "num_inputs": s.num_inputs,
        "total_tokens": s.total_tokens,
        "token_counts": s.token_counts,
        "tokenize_us": s.tokenize_us,
        "inference_us": s.inference_us,
        "postprocess_us": s.postprocess_us,
        "server_e2e_us": s.server_e2e_us,
        "client_e2e_us": s.client_e2e_us,
        "request_bytes": s.request_bytes,
        "response_bytes": s.response_bytes,
    }


def error_row(i: int, code: str, detail: str) -> dict[str, Any]:
    """Raw JSONL record streamed for a failed request.

    Keeps the one-record-per-measured-request contract so the number of
    ``type: "request"`` rows equals ``grpc_metrics.total_requests`` and failures
    are reconcilable from the raw artifact. Carries no timings (none exist).
    """
    return {
        "type": "request",
        "i": i,
        "ok": False,
        "error_code": code,
        "error": detail,
    }


def _div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


# A run whose peak RSS exceeds this share of total RAM is flagged as memory-tight.
MEMORY_SUFFICIENT_MAX_PCT = 90.0


def _ratio(num: float | None, den: float | None) -> float | None:
    """``num/den`` rounded, or ``None`` when either side is missing/zero."""
    return round(num / den, 4) if num and den else None


def build_analysis(
    *,
    grpc: dict[str, Any],
    resources: dict[str, Any],
    machine: dict[str, Any],
    concurrency: int,
) -> dict[str, Any]:
    """Derive normalized efficiency + memory/CPU judgements for cross-CPU comparison.

    Consumes only already-computed values (throughput, reduced resources, machine
    facts). Unknown inputs yield ``None`` rather than a guess, and every value
    stays strict-JSON safe.
    """
    ips = grpc.get("inputs_per_sec")
    tps = grpc.get("tokens_per_sec")
    physical = machine.get("cpu_physical_cores")
    logical = machine.get("cpu_logical_cores")
    ghz = _ratio(machine.get("cpu_freq_max_mhz"), 1000.0)
    isa = machine.get("cpu_isa_extensions") or []

    ram = machine.get("ram_total_mb")
    rss_peak = resources.get("memory_rss_peak_mb")
    headroom = round(ram - rss_peak, 2) if ram and rss_peak is not None else None
    mem_util = _ratio(rss_peak, ram)
    mem_util_pct = round(mem_util * 100, 2) if mem_util is not None else None
    sufficient = mem_util_pct < MEMORY_SUFFICIENT_MAX_PCT if mem_util_pct is not None else None

    cpu_avg = resources.get("cpu_percent_avg")
    core_util_pct = _ratio(cpu_avg, (logical or 0) * 100)
    core_util_pct = round(core_util_pct * 100, 2) if core_util_pct is not None else None

    efficiency = {
        "inputs_per_sec": ips,
        "tokens_per_sec": tps,
        "inputs_per_sec_per_physical_core": _ratio(ips, physical),
        "inputs_per_sec_per_logical_core": _ratio(ips, logical),
        "inputs_per_sec_per_ghz": _ratio(ips, ghz),
        "inputs_per_sec_per_physical_core_ghz": _ratio(
            ips, (physical * ghz) if physical and ghz else None
        ),
        "tokens_per_sec_per_physical_core": _ratio(tps, physical),
    }
    memory = {
        "ram_total_mb": ram,
        "rss_peak_mb": rss_peak,
        "headroom_mb": headroom,
        "utilization_pct": mem_util_pct,
        "sufficient": sufficient,
    }
    cpu_utilization = {
        "cpu_percent_avg": cpu_avg,
        "logical_cores": logical,
        "concurrency": concurrency,
        "core_utilization_pct": core_util_pct,
    }

    notes: list[str] = []
    if mem_util_pct is not None:
        if sufficient:
            notes.append(
                f"Memory sufficient: peak {rss_peak} MB / {ram} MB "
                f"({mem_util_pct}%), headroom {headroom} MB."
            )
        else:
            notes.append(
                f"Memory pressure: peak {rss_peak} MB / {ram} MB ({mem_util_pct}%) "
                f"— at or above {MEMORY_SUFFICIENT_MAX_PCT}% of RAM."
            )
    if core_util_pct is not None and core_util_pct < 50.0:
        notes.append(
            f"CPU under-utilized: avg {cpu_avg}% of {logical} logical cores "
            f"(~{core_util_pct}% capacity) at concurrency={concurrency} "
            "— raise --concurrency to saturate."
        )
    if isa:
        has_avx512 = any(x.startswith("avx512") for x in isa)
        has_vnni = any("vnni" in x for x in isa)
        has_amx = any(x.startswith("amx") for x in isa)
        if has_avx512:
            label = "AVX-512" + (" + VNNI" if has_vnni else "") + (" + AMX" if has_amx else "")
            notes.append(f"{label} available.")
        elif "avx2" in isa:
            notes.append("AVX2 available, no AVX-512.")
    else:
        notes.append("No SIMD/ISA info available.")
    eff_core = efficiency["inputs_per_sec_per_physical_core"]
    eff_core_ghz = efficiency["inputs_per_sec_per_physical_core_ghz"]
    if ips and eff_core is not None:
        msg = f"{ips} emb/s = {eff_core} emb/s/physical-core"
        if eff_core_ghz is not None:
            msg += f", {eff_core_ghz} emb/s/core-GHz"
        notes.append(msg + ".")

    return {
        "efficiency": efficiency,
        "memory": memory,
        "cpu_utilization": cpu_utilization,
        "notes": notes,
    }


def _reduce_resources(samples: list[dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "cpu_percent_avg": None,
        "cpu_percent_peak": None,
        "memory_rss_peak_mb": None,
        "gpu_utilization_avg": None,
        "gpu_memory_peak_mb": None,
    }
    if samples:
        cpu = [s["cpu_percent"] for s in samples]
        rss = [s["rss_mb"] for s in samples]
        out["cpu_percent_avg"] = round(float(np.mean(cpu)), 2)
        out["cpu_percent_peak"] = round(float(np.max(cpu)), 2)
        out["memory_rss_peak_mb"] = round(float(np.max(rss)), 2)
    return out


def build_summary(
    *,
    samples: list[RequestSample],
    resource_samples: list[dict[str, float]],
    spec: dict[str, Any],
    ctx: RunContext,
    validation: dict[str, Any] | None,
    failed_requests: int = 0,
) -> dict[str, Any]:
    successful_requests = len(samples)
    total_requests = successful_requests + failed_requests
    num_inputs = sum(s.num_inputs for s in samples)
    total_tokens = sum(s.total_tokens for s in samples)
    all_token_counts = [c for s in samples for c in s.token_counts]

    tok_us = sum(s.tokenize_us for s in samples)
    inf_us = sum(s.inference_us for s in samples)
    post_us = sum(s.postprocess_us for s in samples)
    server_e2e_us = tok_us + inf_us + post_us
    duration = ctx.duration_sec

    spec_model = spec.get("model", {})
    emb_dim = spec_model.get("embedding_dim")
    if emb_dim is None and validation is not None:
        emb_dim = validation.get("embedding_dim")

    raw_onnx = {
        "tokenize_total_time_ms": round(tok_us / 1000, 3),
        "inference_total_time_ms": round(inf_us / 1000, 3),
        "postprocess_total_time_ms": round(post_us / 1000, 3),
        "e2e_total_time_ms": round(server_e2e_us / 1000, 3),
        "tokenize_tokens_per_sec": round(_div(total_tokens, tok_us / 1e6), 2),
        "tokenize_inputs_per_sec": round(_div(num_inputs, tok_us / 1e6), 2),
        "inference_tokens_per_sec": round(_div(total_tokens, inf_us / 1e6), 2),
        "inference_inputs_per_sec": round(_div(num_inputs, inf_us / 1e6), 2),
        "e2e_tokens_per_sec": round(_div(total_tokens, server_e2e_us / 1e6), 2),
        "e2e_inputs_per_sec": round(_div(num_inputs, server_e2e_us / 1e6), 2),
        **percentiles_ms("inference", [s.inference_us / 1000 for s in samples]),
    }

    grpc = {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "error_rate": round(_div(failed_requests, total_requests), 4),
        "client_concurrency": ctx.concurrency,
        "client_batch_size": ctx.batch_size,
        "requests_per_sec": round(_div(successful_requests, duration), 2),
        "inputs_per_sec": round(_div(num_inputs, duration), 2),
        "tokens_per_sec": round(_div(total_tokens, duration), 2),
        **percentiles_ms("client_e2e", [s.client_e2e_us / 1000 for s in samples]),
        **percentiles_ms(
            "grpc_overhead", [(s.client_e2e_us - s.server_e2e_us) / 1000 for s in samples]
        ),
        "request_size_bytes_avg": round(
            _div(sum(s.request_bytes for s in samples), successful_requests)
        ),
        "response_size_bytes_avg": round(
            _div(sum(s.response_bytes for s in samples), successful_requests)
        ),
    }

    machine = spec.get("machine", {})
    resources = _reduce_resources(resource_samples)
    analysis = build_analysis(
        grpc=grpc, resources=resources, machine=machine, concurrency=ctx.concurrency
    )

    return {
        "type": "summary",
        "benchmark": {
            "benchmark_id": ctx.benchmark_id,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "benchmark_type": "grpc_service",
            "duration_sec": round(duration, 3),
            "warmup_sec": ctx.warmup_sec,
        },
        "model": {
            "model_name": ctx.model_name or spec_model.get("name"),
            "model_revision": ctx.model_revision,
            "embedding_dim": emb_dim,
            "precision": ctx.precision,
            "quantization": ctx.quantization,
            "inputs": spec_model.get("inputs"),
            "outputs": spec_model.get("outputs"),
        },
        "runtime": spec.get("runtime", {}),
        "machine": machine,
        "input": {
            "num_inputs": num_inputs,
            "total_tokens": total_tokens,
            "avg_tokens_per_input": round(_div(total_tokens, num_inputs), 2),
            **token_percentiles(all_token_counts),
        },
        "raw_onnx_metrics": raw_onnx,
        "grpc_metrics": grpc,
        "resource_metrics": resources,
        "analysis": analysis,
        "validation": validation,
    }
