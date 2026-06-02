from __future__ import annotations

import pytest

from bge_m3_bench.server import providers
from bge_m3_bench.server.providers import resolve_provider


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        resolve_provider("tpu")


@pytest.mark.parametrize("name", ["openvino", "coreml"])
def test_unavailable_provider_falls_back_to_cpu(monkeypatch, name):
    # Pretend only the CPU provider is available in this environment.
    monkeypatch.setattr(providers, "available_providers", lambda: ["CPUExecutionProvider"])
    resolved = resolve_provider(name, {"device_type": "CPU"})
    assert resolved.requested == name
    assert resolved.ort_provider == "CPUExecutionProvider"
    assert resolved.fell_back is True
    assert resolved.available is False
    # Options are dropped on fallback so we never pass EP-specific keys to CPU.
    assert resolved.options == {}
    assert resolved.session_providers() == ["CPUExecutionProvider"]


def test_available_provider_keeps_options(monkeypatch):
    monkeypatch.setattr(
        providers,
        "available_providers",
        lambda: ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    )
    resolved = resolve_provider("openvino", {"device_type": "CPU"})
    assert resolved.ort_provider == "OpenVINOExecutionProvider"
    assert resolved.fell_back is False
    assert resolved.session_providers() == [("OpenVINOExecutionProvider", {"device_type": "CPU"})]
