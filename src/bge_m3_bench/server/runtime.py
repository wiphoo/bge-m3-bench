"""Thin, reusable wrapper around an ONNX Runtime inference session.

The single place where the embedding model is loaded and executed. Used by the
gRPC server and by the optional local reference embedder in the benchmark.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..common.logging import get_logger
from .providers import ResolvedProvider, resolve_provider

logger = get_logger(__name__)

# CoreML inference creates autoreleased Objective-C objects. The gRPC worker
# threads that call into ORT are not Cocoa run-loop threads and never exit, so
# without an enclosing autorelease pool those objects accumulate and RSS grows
# (macOS prints "Context leak detected, msgtracer returned -1"). We wrap each
# CoreML inference in a pool via pyobjc-core so they drain per request. The
# factory is resolved once and cached; absent pyobjc-core we degrade to a no-op
# (with a one-time warning) so the server still runs everywhere.
_POOL_FACTORY: Callable[[], AbstractContextManager[Any]] | None = None
_POOL_WARNED = False


def _resolve_pool_factory() -> Callable[[], AbstractContextManager[Any]]:
    """Resolve (once, cached) the autorelease pool factory.

    Returns ``objc.autorelease_pool`` when pyobjc-core is importable, else
    ``contextlib.nullcontext`` (a no-op) with a one-time warning.
    """
    global _POOL_FACTORY, _POOL_WARNED
    if _POOL_FACTORY is None:
        try:
            import objc  # pyobjc-core

            _POOL_FACTORY = objc.autorelease_pool
        except Exception:
            _POOL_FACTORY = contextlib.nullcontext
            if not _POOL_WARNED:
                logger.warning(
                    "CoreML active but pyobjc-core is missing; autorelease pool "
                    "disabled and RSS may grow — install with `make sync-coreml`"
                )
                _POOL_WARNED = True
    return _POOL_FACTORY


def _autorelease_pool() -> AbstractContextManager[Any]:
    """Return a per-call autorelease pool context manager for CoreML inference."""
    return _resolve_pool_factory()()


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str  # numpy dtype string, e.g. "float32" or "int64"
    shape: tuple[int, ...]  # -1 denotes a dynamic dimension


@dataclass(frozen=True)
class InferenceResult:
    outputs: dict[str, np.ndarray]
    inference_us: int


class OnnxModel:
    """A loaded ONNX model bound to a resolved execution provider."""

    def __init__(
        self,
        model_path: str | Path,
        provider: str = "cpu",
        provider_options: dict[str, str] | None = None,
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
        name: str | None = None,
    ) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self.name = name or self.model_path.stem
        self.resolved: ResolvedProvider = resolve_provider(provider, provider_options)

        sess_options = ort.SessionOptions()
        if intra_op_threads > 0:
            sess_options.intra_op_num_threads = intra_op_threads
        if inter_op_threads > 0:
            sess_options.inter_op_num_threads = inter_op_threads
            # ORT only uses the inter-op pool in parallel execution mode; the
            # default (ORT_SEQUENTIAL) would ignore inter_op_num_threads.
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=self.resolved.session_providers(),
        )
        # Wrap inference in an autorelease pool only when CoreML is the active
        # provider (computed once; other providers keep the zero-overhead path).
        self._coreml_active = self.active_provider == "CoreMLExecutionProvider"
        logger.info(
            "model loaded",
            extra={
                "fields": {
                    "model": self.name,
                    "provider_requested": self.resolved.requested,
                    "provider_active": self.session.get_providers()[0],
                    "fell_back": self.resolved.fell_back,
                }
            },
        )

    @property
    def active_provider(self) -> str:
        return self.session.get_providers()[0]

    @property
    def coreml_autorelease_pool(self) -> bool:
        """Whether CoreML inference is wrapped in an autorelease pool.

        True only when CoreML is active *and* pyobjc-core is importable (so the
        pool is a real drain rather than the no-op fallback).
        """
        if not self._coreml_active:
            return False
        return _resolve_pool_factory() is not contextlib.nullcontext

    def input_specs(self) -> list[TensorSpec]:
        return [self._spec(i) for i in self.session.get_inputs()]

    def output_specs(self) -> list[TensorSpec]:
        return [self._spec(o) for o in self.session.get_outputs()]

    def input_names(self) -> list[str]:
        return [i.name for i in self.session.get_inputs()]

    @staticmethod
    def _spec(node: Any) -> TensorSpec:
        shape = tuple(d if isinstance(d, int) else -1 for d in (node.shape or ()))
        dtype = _onnx_type_to_numpy(node.type)
        return TensorSpec(name=node.name, dtype=dtype, shape=shape)

    def run(
        self,
        inputs: dict[str, np.ndarray],
        output_names: list[str] | None = None,
    ) -> InferenceResult:
        names = output_names or [o.name for o in self.session.get_outputs()]
        pool = _autorelease_pool() if self._coreml_active else contextlib.nullcontext()
        start = time.perf_counter_ns()
        with pool:
            results = self.session.run(names, inputs)
        elapsed_us = (time.perf_counter_ns() - start) // 1000
        return InferenceResult(
            outputs=dict(zip(names, results, strict=True)),
            inference_us=int(elapsed_us),
        )


_ONNX_TYPE_TO_NUMPY = {
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(float16)": "float16",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(int16)": "int16",
    "tensor(int8)": "int8",
    "tensor(uint64)": "uint64",
    "tensor(uint32)": "uint32",
    "tensor(uint16)": "uint16",
    "tensor(uint8)": "uint8",
    "tensor(bool)": "bool",
}


def _onnx_type_to_numpy(onnx_type: str) -> str:
    """Map an ONNX value-info type string to a numpy dtype string.

    Raises ``ValueError`` for element types we do not model instead of silently
    defaulting to float32.
    """
    try:
        return _ONNX_TYPE_TO_NUMPY[onnx_type]
    except KeyError as exc:
        raise ValueError(f"unsupported ONNX tensor dtype: {onnx_type!r}") from exc
