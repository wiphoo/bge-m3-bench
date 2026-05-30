"""Multi-model registry (Milestone 10).

Loads one or more ONNX models and exposes them by id. Supports a default model
so single-model deployments need not specify a name on every request.
"""

from __future__ import annotations

from pathlib import Path

from ..common.logging import get_logger
from .runtime import OnnxModel

logger = get_logger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, OnnxModel] = {}
        self._default: str | None = None

    def add(self, model: OnnxModel, *, default: bool = False) -> None:
        if model.name in self._models:
            raise ValueError(f"model already registered: {model.name}")
        self._models[model.name] = model
        if default or self._default is None:
            self._default = model.name

    def load(
        self,
        model_path: str | Path,
        *,
        provider: str = "cpu",
        provider_options: dict[str, str] | None = None,
        name: str | None = None,
        default: bool = False,
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
    ) -> OnnxModel:
        model = OnnxModel(
            model_path,
            provider=provider,
            provider_options=provider_options,
            name=name,
            intra_op_threads=intra_op_threads,
            inter_op_threads=inter_op_threads,
        )
        self.add(model, default=default)
        return model

    def get(self, name: str | None = None) -> OnnxModel:
        key = name or self._default
        if key is None:
            raise KeyError("no models registered")
        if key not in self._models:
            raise KeyError(f"unknown model: {key}")
        return self._models[key]

    @property
    def default_model(self) -> str | None:
        return self._default

    def names(self) -> list[str]:
        return list(self._models)

    def __len__(self) -> int:
        return len(self._models)
