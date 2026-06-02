"""Execution provider abstraction.

Maps a logical provider name to the concrete ONNX Runtime provider and resolves
it against the locally available build, falling back to CPU (with a warning) so
the same command runs anywhere while still recording the requested provider.

CPU is the focus today; ``cuda`` is wired as a seam for later GPU support.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..common.logging import get_logger

logger = get_logger(__name__)

# Logical provider name -> ONNX Runtime provider identifier.
_PROVIDER_MAP: dict[str, str] = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",  # seam: works when onnxruntime-gpu is installed
    "openvino": "OpenVINOExecutionProvider",  # Intel x86; needs onnxruntime-openvino
    "coreml": "CoreMLExecutionProvider",  # Apple Silicon; bundled in the macOS wheel
}


@dataclass(frozen=True)
class ResolvedProvider:
    """A provider resolved against the locally available ONNX Runtime build."""

    requested: str
    ort_provider: str
    options: dict[str, str] = field(default_factory=dict)
    available: bool = True
    fell_back: bool = False

    def session_providers(self) -> list:
        """Return the ``providers`` argument for ``onnxruntime.InferenceSession``."""
        if self.options:
            return [(self.ort_provider, self.options)]
        return [self.ort_provider]


def available_providers() -> list[str]:
    """Return the ONNX Runtime providers available in this environment."""
    try:
        import onnxruntime as ort
    except Exception:  # pragma: no cover - onnxruntime always present in deps
        return ["CPUExecutionProvider"]
    return list(ort.get_available_providers())


def resolve_provider(
    name: str,
    options: dict[str, str] | None = None,
) -> ResolvedProvider:
    """Resolve a logical provider name; fall back to CPU when unavailable."""
    key = name.strip().lower()
    if key not in _PROVIDER_MAP:
        raise ValueError(f"unknown provider {name!r}; supported: {sorted(_PROVIDER_MAP)}")
    ort_provider = _PROVIDER_MAP[key]
    options = dict(options or {})

    if ort_provider in available_providers():
        return ResolvedProvider(
            requested=key, ort_provider=ort_provider, options=options, available=True
        )

    logger.warning(
        "requested provider unavailable; falling back to CPU",
        extra={"fields": {"requested": key, "ort_provider": ort_provider}},
    )
    return ResolvedProvider(
        requested=key,
        ort_provider="CPUExecutionProvider",
        options={},
        available=False,
        fell_back=True,
    )
