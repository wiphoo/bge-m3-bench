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


def _machine() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "cpu_physical_cores": None,
        "ram_total_mb": None,
        "containerized": os.path.exists("/.dockerenv")
        or os.getenv("KUBERNETES_SERVICE_HOST") is not None,
    }
    try:
        import psutil

        info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        info["ram_total_mb"] = round(psutil.virtual_memory().total / 1e6, 1)
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
            "provider_options": config.provider_options,
            "execution_provider": model.active_provider,
            "intra_op_threads": config.intra_op_threads,
            "inter_op_threads": config.inter_op_threads,
        },
        "tokenizer": {"path": config.tokenizer_path},
        "runtime": _runtime(model),
        "machine": _machine(),
    }
