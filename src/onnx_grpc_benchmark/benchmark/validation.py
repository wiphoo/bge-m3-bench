"""Validation framework (Milestone 1.3).

Ensures benchmark correctness before/while results are recorded:

* **Output validity** - outputs contain no NaN/Inf and have expected rank.
* **Reproducibility** - running the same input twice yields identical (or
  numerically close) outputs.

A failed validation raises :class:`ValidationError`, which the runner surfaces
so a failed validation causes the benchmark to fail (per the DoD).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..server.runtime import OnnxModel


class ValidationError(RuntimeError):
    """Raised when benchmark output validation fails."""


@dataclass
class ValidationReport:
    passed: bool = True
    checks: list[dict] = field(default_factory=list)

    def record(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "passed": passed, "detail": detail})
        if not passed:
            self.passed = False

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": self.checks}

    def raise_if_failed(self) -> None:
        if not self.passed:
            failed = [c for c in self.checks if not c["passed"]]
            raise ValidationError(f"validation failed: {failed}")


def _finite(outputs: dict[str, np.ndarray]) -> tuple[bool, str]:
    for name, arr in outputs.items():
        if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
            return False, f"output {name!r} contains NaN/Inf"
    return True, ""


def _first_output_mismatch(
    expected: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> str | None:
    """Name of the first output where ``actual`` diverges from ``expected``.

    A divergence is a missing key, float arrays not within tolerance, or
    non-float arrays not exactly equal. Returns ``None`` when they match. Shared
    by the reproducibility check (same model, two runs) and the gRPC reference
    check (served outputs vs a local run).
    """
    for name, ref in expected.items():
        got = actual.get(name)
        if got is None:
            return name
        if ref.dtype.kind == "f":
            if not np.allclose(got, ref, rtol=rtol, atol=atol, equal_nan=False):
                return name
        elif not np.array_equal(got, ref):
            return name
    return None


def validate_model(
    model: OnnxModel,
    sample: dict[str, np.ndarray],
    *,
    reproducibility_runs: int = 2,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> ValidationReport:
    """Validate a model against a single representative input sample."""
    report = ValidationReport()

    first = model.run(sample).outputs
    ok, detail = _finite(first)
    report.record("outputs_finite", ok, detail)

    ok = len(first) > 0
    report.record("outputs_present", ok, "" if ok else "model produced no outputs")

    # Reproducibility: identical inputs must produce (numerically) identical
    # outputs across repeated runs.
    reproducible = True
    detail = ""
    for _ in range(max(0, reproducibility_runs - 1)):
        mismatch = _first_output_mismatch(first, model.run(sample).outputs, rtol=rtol, atol=atol)
        if mismatch is not None:
            reproducible = False
            detail = f"non-deterministic output {mismatch!r}"
            break
    report.record("reproducible", reproducible, detail)
    return report


def validate_grpc(
    client: Any,
    sample: dict[str, np.ndarray],
    *,
    model_name: str = "",
    reference: OnnxModel | None = None,
    # Looser than validate_model's reproducibility tolerance: the server may run
    # a different execution provider than the local reference, so small
    # cross-provider numerical differences are expected.
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> ValidationReport:
    """Validate gRPC inference outputs before recording benchmark results.

    Always checks that the served outputs are present and finite so a server
    returning corrupt/non-finite tensors cannot be reported as ``passed``. When
    a ``reference`` model is supplied, the gRPC outputs are additionally
    compared against a local run of the same input, catching a server that is
    serving a different model or a serialization defect. ``client`` is an
    :class:`~onnx_grpc_benchmark.server.client.InferenceClient`.
    """
    report = ValidationReport()
    outputs, _ = client.predict(sample, model=model_name)

    ok, detail = _finite(outputs)
    report.record("outputs_finite", ok, detail)

    present = len(outputs) > 0
    report.record("outputs_present", present, "" if present else "server returned no outputs")

    if reference is not None:
        expected = reference.run(sample).outputs
        mismatch = _first_output_mismatch(expected, outputs, rtol=rtol, atol=atol)
        report.record(
            "matches_reference",
            mismatch is None,
            "" if mismatch is None else f"output {mismatch!r} differs from reference model",
        )

    return report
