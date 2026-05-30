"""Capture the benchmark environment completely (Milestone 1.2 / 8 / 9).

The collected metadata is attached to every benchmark result so runs are
reproducible and comparable across machines. Collection is best-effort: every
field degrades gracefully when a probe is unavailable (e.g. no GPU, running
outside Kubernetes) so the same code path works on local, Docker and cluster.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

# Bump when the metadata structure changes in a backwards-incompatible way.
METADATA_SCHEMA_VERSION = "1.0.0"


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "arch": platform.machine(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
    }
    try:
        import psutil

        info["physical_cores"] = psutil.cpu_count(logical=False)
        freq = psutil.cpu_freq()
        if freq is not None:
            info["max_freq_mhz"] = round(freq.max, 2)
    except Exception:  # pragma: no cover - psutil optional at runtime
        pass

    # Linux: pull a human-readable model name from /proc/cpuinfo.
    model_name = None
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    model_name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if model_name:
        info["model_name"] = model_name
    return info


def _memory_info() -> dict[str, Any]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {"total_bytes": vm.total, "available_bytes": vm.available}
    except Exception:  # pragma: no cover
        return {}


def _gpu_info() -> dict[str, Any]:
    """Best-effort NVIDIA GPU metadata via nvidia-smi (Milestone 8)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"present": False}
    query = "name,memory.total,driver_version,compute_cap"
    try:
        out = subprocess.run(
            [smi, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - depends on hardware
        return {"present": False}
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append(
                {
                    "name": parts[0],
                    "memory_total_mib": _to_number(parts[1]),
                    "driver_version": parts[2],
                    "compute_capability": parts[3],
                }
            )
    return {"present": bool(gpus), "devices": gpus}


def _to_number(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def _runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    try:
        import onnxruntime as ort

        info["onnxruntime_version"] = ort.__version__
        info["available_providers"] = list(ort.get_available_providers())
    except Exception:  # pragma: no cover
        info["onnxruntime_version"] = None
        info["available_providers"] = []
    try:
        import numpy as np

        info["numpy_version"] = np.__version__
    except Exception:  # pragma: no cover
        pass
    return info


def _container_info() -> dict[str, Any]:
    """Detect Docker / Kubernetes context (Milestone 5 / 9)."""
    in_docker = os.path.exists("/.dockerenv")
    k8s = {
        "namespace": os.getenv("POD_NAMESPACE"),
        "pod_name": os.getenv("POD_NAME") or os.getenv("HOSTNAME")
        if os.getenv("KUBERNETES_SERVICE_HOST")
        else None,
        "node_name": os.getenv("NODE_NAME"),
    }
    in_k8s = os.getenv("KUBERNETES_SERVICE_HOST") is not None
    return {
        "in_docker": in_docker,
        "in_kubernetes": in_k8s,
        "kubernetes": {k: v for k, v in k8s.items() if v} if in_k8s else {},
    }


def _git_revision() -> str | None:
    git = shutil.which("git")
    if not git:
        return None
    try:
        return subprocess.run(
            [git, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def collect_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect a complete, JSON-serialisable snapshot of the environment."""
    meta: dict[str, Any] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "cpu": _cpu_info(),
        "memory": _memory_info(),
        "gpu": _gpu_info(),
        "runtime": _runtime_info(),
        "container": _container_info(),
        "git_revision": _git_revision(),
        "argv": sys.argv,
    }
    if extra:
        meta["extra"] = extra
    return meta


def metadata_schema() -> dict[str, Any]:
    """Return a JSON-schema-style description of the metadata document."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "BenchmarkMetadata",
        "version": METADATA_SCHEMA_VERSION,
        "type": "object",
        "required": [
            "schema_version",
            "collected_at",
            "hostname",
            "os",
            "cpu",
            "runtime",
        ],
        "properties": {
            "schema_version": {"type": "string"},
            "collected_at": {"type": "string", "format": "date-time"},
            "hostname": {"type": "string"},
            "os": {"type": "object"},
            "cpu": {"type": "object"},
            "memory": {"type": "object"},
            "gpu": {"type": "object"},
            "runtime": {"type": "object"},
            "container": {"type": "object"},
            "git_revision": {"type": ["string", "null"]},
            "argv": {"type": "array", "items": {"type": "string"}},
            "extra": {"type": "object"},
        },
    }
