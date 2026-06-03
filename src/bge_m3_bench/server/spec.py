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


_CGROUP_ROOT = "/sys/fs/cgroup"


def _proc_self_cgroup() -> list[tuple[str, str]]:
    """Parse ``/proc/self/cgroup`` into ``(controllers, relpath)`` rows.

    The cgroup-v2 unified row has empty controllers (``"0::/path"``); v1 rows
    carry a comma-separated controller list (``"4:cpu,cpuacct:/path"``).
    """
    rows: list[tuple[str, str]] = []
    try:
        with open("/proc/self/cgroup") as fh:
            for line in fh:
                parts = line.strip().split(":", 2)
                if len(parts) == 3:
                    rows.append((parts[1], parts[2]))
    except OSError:
        pass
    return rows


def _dirs_leaf_to_root(rel: str | None, base: str) -> list[str]:
    """cgroup dirs from the process's own cgroup up to the mount root.

    Limits can live on the leaf cgroup or any ancestor slice, so callers read
    each level and keep the most restrictive. ``base`` is always included as the
    final fallback (covers an unreadable/absent ``/proc/self/cgroup``).
    """
    dirs: list[str] = []
    rel = (rel or "").strip("/")
    cur = os.path.join(base, rel) if rel else base
    while True:
        dirs.append(cur)
        if os.path.normpath(cur) == os.path.normpath(base) or len(cur) <= len(base):
            break
        cur = os.path.dirname(cur)
    if base not in dirs:
        dirs.append(base)
    return dirs


def _cgroup_cpu_quota() -> float | None:
    """Effective CPU limit (fractional vCPUs) from the process cgroup, or ``None``.

    Resolves this process's own cgroup from ``/proc/self/cgroup`` (not the root),
    walking leaf->ancestors and taking the most restrictive quota, so limits set
    on a parent slice or in a non-root container cgroup are honored.
    """
    rows = _proc_self_cgroup()
    quotas: list[float] = []

    # cgroup v2 (unified): the row with empty controllers.
    rel_v2 = next((rel for ctrls, rel in rows if ctrls == ""), "" if not rows else None)
    if rel_v2 is not None:
        for d in _dirs_leaf_to_root(rel_v2, _CGROUP_ROOT):
            content = _read_first_line(os.path.join(d, "cpu.max"))
            if content is not None:
                q = _parse_cpu_max(content)
                if q:
                    quotas.append(q)
    if quotas:
        return min(quotas)

    # cgroup v1: the row whose controllers include "cpu".
    rel_v1 = next(
        (rel for ctrls, rel in rows if "cpu" in ctrls.split(",")), "" if not rows else None
    )
    if rel_v1 is not None:
        for mount in (f"{_CGROUP_ROOT}/cpu", f"{_CGROUP_ROOT}/cpu,cpuacct"):
            for d in _dirs_leaf_to_root(rel_v1, mount):
                quota = _read_first_line(os.path.join(d, "cpu.cfs_quota_us"))
                period = _read_first_line(os.path.join(d, "cpu.cfs_period_us"))
                try:
                    if quota and period:
                        q, p = int(quota), int(period)
                        if q > 0 and p > 0:
                            quotas.append(q / p)
                except ValueError:
                    pass
    return min(quotas) if quotas else None


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
    """Container memory limit in MB from the process cgroup, or ``None`` if unlimited.

    Resolves this process's own cgroup (not the root) and keeps the most
    restrictive real limit found from leaf to ancestors.
    """
    rows = _proc_self_cgroup()
    raws: list[str] = []

    rel_v2 = next((rel for ctrls, rel in rows if ctrls == ""), "" if not rows else None)
    if rel_v2 is not None:
        for d in _dirs_leaf_to_root(rel_v2, _CGROUP_ROOT):
            v = _read_first_line(os.path.join(d, "memory.max"))
            if v is not None:
                raws.append(v)

    if not raws:
        rel_v1 = next(
            (rel for ctrls, rel in rows if "memory" in ctrls.split(",")),
            "" if not rows else None,
        )
        if rel_v1 is not None:
            for d in _dirs_leaf_to_root(rel_v1, f"{_CGROUP_ROOT}/memory"):
                v = _read_first_line(os.path.join(d, "memory.limit_in_bytes"))
                if v is not None:
                    raws.append(v)

    limits: list[int] = []
    for raw in raws:
        if not raw or raw == "max":
            continue
        try:
            val = int(raw)
        except ValueError:
            continue
        if val <= 0 or val >= _NO_MEM_LIMIT:
            continue
        if host_total_bytes and val >= host_total_bytes:
            continue  # a limit >= host RAM is not an effective constraint
        limits.append(val)
    return round(min(limits) / 1e6, 1) if limits else None


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
