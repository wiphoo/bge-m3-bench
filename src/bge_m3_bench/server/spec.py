"""Build the service spec / metadata returned by ``GetSpec``.

These are raw descriptive facts about the running service (model I/O, config,
tokenizer, runtime, machine) so the benchmark client can fill the summary's
model/runtime/machine sections without guessing. No metrics here.
"""

from __future__ import annotations

import os
import platform
import socket
from typing import Any

from .. import __version__
from ..common.config import ServerConfig
from .runtime import OnnxModel, TensorSpec


def _tensor_spec(spec: TensorSpec) -> dict[str, Any]:
    return {"name": spec.name, "dtype": spec.dtype, "shape": list(spec.shape)}


def embedding_dim(model: OnnxModel) -> int | None:
    """Best-effort embedding dimension = last static dim of the first output."""
    outputs = model.output_specs()
    if not outputs:
        return None
    last = outputs[0].shape[-1] if outputs[0].shape else -1
    return int(last) if last and last > 0 else None


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


# Throughput-relevant instruction sets, surfaced so runs on different CPUs can be
# compared (AVX-512 / VNNI / AMX dominate ONNX CPU inference, esp. int8).
_ISA_WHITELIST = frozenset(
    {
        # x86
        "avx",
        "avx2",
        "avx512f",
        "avx512bw",
        "avx512vl",
        "avx512dq",
        "avx512_vnni",
        "avx_vnni",
        "amx_tile",
        "amx_int8",
        "amx_bf16",
        "f16c",
        "fma",
        "sse4_1",
        "sse4_2",
        # ARM
        "neon",
        "asimd",
        "asimddp",
        "sve",
        "sve2",
        "i8mm",
        "bf16",
    }
)


def _filter_isa(flags: set[str]) -> list[str]:
    """Keep only throughput-relevant ISA extensions, sorted."""
    return sorted(_ISA_WHITELIST & flags)


def _cpu_isa_extensions() -> list[str]:
    """Best-effort relevant ISA extensions from ``/proc/cpuinfo`` (Linux)."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                key = line.split(":", 1)[0].strip().lower()
                if key in ("flags", "features"):
                    return _filter_isa(set(line.split(":", 1)[1].split()))
    except OSError:
        pass
    return []


def _read_first_line(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.readline().strip()
    except OSError:
        return None


def _parse_cpu_max(content: str) -> float | None:
    """cgroup-v2 ``cpu.max`` (``"<quota> <period>"`` or ``"max ..."``) -> vCPUs."""
    parts = content.split()
    if not parts or parts[0] == "max":
        return None
    try:
        quota = float(parts[0])
        period = float(parts[1]) if len(parts) > 1 else 100000.0
    except ValueError:
        return None
    return quota / period if quota > 0 and period > 0 else None


def _cgroup_cpu_quota() -> float | None:
    """Effective CPU limit (fractional vCPUs) from a cgroup quota, or ``None``."""
    v2 = _read_first_line("/sys/fs/cgroup/cpu.max")
    if v2 is not None:
        return _parse_cpu_max(v2)
    quota = _read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        if quota and period:
            q, p = int(quota), int(period)
            if q > 0 and p > 0:
                return q / p
    except ValueError:
        pass
    return None


def _effective_cpu_cores() -> float | None:
    """CPUs actually usable by this process: min(cgroup quota, CPU affinity).

    Unlike host logical-core counts this reflects cpuset/quota/taskset limits, so
    saturation analysis stays correct in containers and under affinity pinning.
    """
    candidates: list[float] = []
    quota = _cgroup_cpu_quota()
    if quota:
        candidates.append(quota)
    try:
        candidates.append(float(len(os.sched_getaffinity(0))))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        if os.cpu_count():
            candidates.append(float(os.cpu_count() or 0))
    return round(min(candidates), 2) if candidates else None


# cgroup-v1 "unlimited" sentinels sit near INT64_MAX; treat anything that large
# (or >= host RAM) as "no real limit".
_NO_MEM_LIMIT = 1 << 62


def _cgroup_mem_limit_mb(host_total_bytes: int | None) -> float | None:
    """Container memory limit in MB from the cgroup, or ``None`` when unlimited."""
    raw = _read_first_line("/sys/fs/cgroup/memory.max")  # v2
    if raw is None:
        raw = _read_first_line("/sys/fs/cgroup/memory/memory.limit_in_bytes")  # v1
    if not raw or raw == "max":
        return None
    try:
        val = int(raw)
    except ValueError:
        return None
    if val <= 0 or val >= _NO_MEM_LIMIT:
        return None
    if host_total_bytes and val >= host_total_bytes:
        return None  # a limit >= host RAM is not an effective constraint
    return round(val / 1e6, 1)


def _machine() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "cpu_physical_cores": None,
        "cpu_effective_cores": _effective_cpu_cores(),
        "cpu_freq_max_mhz": None,
        "cpu_freq_min_mhz": None,
        "cpu_freq_current_mhz": None,
        "cpu_isa_extensions": _cpu_isa_extensions(),
        "ram_total_mb": None,
        "ram_limit_mb": None,
        "containerized": os.path.exists("/.dockerenv")
        or os.getenv("KUBERNETES_SERVICE_HOST") is not None,
    }
    try:
        import psutil

        info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        total_bytes = psutil.virtual_memory().total
        info["ram_total_mb"] = round(total_bytes / 1e6, 1)
        info["ram_limit_mb"] = _cgroup_mem_limit_mb(total_bytes)
        freq = psutil.cpu_freq()
        if freq is not None:
            info["cpu_freq_max_mhz"] = round(freq.max, 1) or None
            info["cpu_freq_min_mhz"] = round(freq.min, 1) or None
            info["cpu_freq_current_mhz"] = round(freq.current, 1) or None
    except Exception:  # pragma: no cover - psutil optional at runtime
        pass
    return info


def _runtime(model: OnnxModel) -> dict[str, Any]:
    info: dict[str, Any] = {"runtime": "onnxruntime"}
    try:
        import onnxruntime as ort

        info["runtime_version"] = ort.__version__
        info["available_providers"] = list(ort.get_available_providers())
    except Exception:  # pragma: no cover
        info["runtime_version"] = None
        info["available_providers"] = []
    info["execution_provider"] = model.active_provider
    return info


def build_spec(config: ServerConfig, model: OnnxModel) -> dict[str, Any]:
    return {
        "service_version": __version__,
        "model": {
            "name": model.name,
            "embedding_dim": embedding_dim(model),
            "inputs": [_tensor_spec(s) for s in model.input_specs()],
            "outputs": [_tensor_spec(s) for s in model.output_specs()],
        },
        "config": {
            "pooling": config.pooling,
            "normalize": config.normalize,
            "max_length": config.max_length,
            "provider": config.provider,
            "execution_provider": model.active_provider,
            "intra_op_threads": config.intra_op_threads,
            "inter_op_threads": config.inter_op_threads,
        },
        "tokenizer": {"path": config.tokenizer_path},
        "runtime": _runtime(model),
        "machine": _machine(),
    }
