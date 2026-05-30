"""Thin, reusable wrapper around an ONNX Runtime inference session.

This is the single place where a model is loaded and executed. It is used by
both the local baseline benchmark (Milestone 1) and the gRPC service
(Milestone 2), guaranteeing identical execution semantics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..common.logging import get_logger
from ..common.providers import ResolvedProvider, resolve_provider

logger = get_logger(__name__)


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
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=self.resolved.session_providers(),
        )
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

    def input_specs(self) -> list[TensorSpec]:
        return [self._spec(i) for i in self.session.get_inputs()]

    def output_specs(self) -> list[TensorSpec]:
        return [self._spec(o) for o in self.session.get_outputs()]

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
        start = time.perf_counter_ns()
        results = self.session.run(names, inputs)
        elapsed_us = (time.perf_counter_ns() - start) // 1000
        return InferenceResult(
            outputs=dict(zip(names, results, strict=True)),
            inference_us=int(elapsed_us),
        )


def _onnx_type_to_numpy(onnx_type: str) -> str:
    """Map an ONNX value-info type string to a numpy dtype string."""
    mapping = {
        "tensor(float)": "float32",
        "tensor(double)": "float64",
        "tensor(float16)": "float16",
        "tensor(int64)": "int64",
        "tensor(int32)": "int32",
        "tensor(uint8)": "uint8",
        "tensor(bool)": "bool",
    }
    return mapping.get(onnx_type, "float32")
