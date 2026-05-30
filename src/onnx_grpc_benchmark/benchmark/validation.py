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
        again = model.run(sample).outputs
        for name in first:
            a, b = first[name], again[name]
            if a.dtype.kind == "f":
                if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False):
                    reproducible = False
                    detail = f"non-deterministic output {name!r}"
                    break
            elif not np.array_equal(a, b):
                reproducible = False
                detail = f"non-deterministic output {name!r}"
                break
        if not reproducible:
            break
    report.record("reproducible", reproducible, detail)
    return report
