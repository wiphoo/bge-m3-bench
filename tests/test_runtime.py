from __future__ import annotations

import contextlib
import sys
import types

import numpy as np
import pytest

from bge_m3_bench.server import runtime


@pytest.fixture(autouse=True)
def _reset_pool_cache():
    """The pool factory is module-cached; reset it around every test."""
    runtime._POOL_FACTORY = None
    runtime._POOL_WARNED = False
    yield
    runtime._POOL_FACTORY = None
    runtime._POOL_WARNED = False


class _FakeSession:
    """Minimal stand-in for an ORT InferenceSession used by OnnxModel.run."""

    class _Out:
        name = "y"

    def get_outputs(self):
        return [self._Out()]

    def run(self, names, inputs):
        return [np.zeros((1, 8), dtype=np.float32) for _ in names]


def test_pool_factory_falls_back_to_nullcontext_without_pyobjc(monkeypatch):
    # No 'objc' module on this platform -> nullcontext (no-op) + one-time warning.
    monkeypatch.setitem(sys.modules, "objc", None)  # makes `import objc` raise
    assert runtime._resolve_pool_factory() is contextlib.nullcontext
    # The no-op context manager still works as a context manager.
    with runtime._autorelease_pool():
        pass


def test_pool_factory_uses_objc_autorelease_pool_when_available(monkeypatch):
    fake = types.ModuleType("objc")

    @contextlib.contextmanager
    def autorelease_pool():
        yield "pool"

    fake.autorelease_pool = autorelease_pool
    monkeypatch.setitem(sys.modules, "objc", fake)
    assert runtime._resolve_pool_factory() is autorelease_pool


def test_cpu_model_does_not_wrap_inference(model, monkeypatch):
    assert model._coreml_active is False
    assert model.coreml_autorelease_pool is False

    entered: list[str] = []

    @contextlib.contextmanager
    def tracking():
        entered.append("enter")
        try:
            yield
        finally:
            entered.append("exit")

    monkeypatch.setattr(runtime, "_autorelease_pool", tracking)
    monkeypatch.setattr(model, "session", _FakeSession())
    result = model.run({"x": np.zeros((1, 1), dtype=np.int64)})

    assert entered == []  # CPU path never touches the pool
    assert result.outputs["y"].shape == (1, 8)


def test_coreml_model_wraps_inference_in_pool(model, monkeypatch):
    entered: list[str] = []

    @contextlib.contextmanager
    def tracking():
        entered.append("enter")
        try:
            yield
        finally:
            entered.append("exit")

    monkeypatch.setattr(runtime, "_autorelease_pool", tracking)
    monkeypatch.setattr(model, "session", _FakeSession())
    model._coreml_active = True  # simulate CoreML being the active provider

    result = model.run({"x": np.zeros((1, 1), dtype=np.int64)})

    assert entered == ["enter", "exit"]  # pool entered and exited around run
    assert result.outputs["y"].shape == (1, 8)


def test_cpu_model_has_no_inference_lock(model):
    assert model._infer_lock is None
    assert model.coreml_serialized is False


def test_coreml_inference_is_serialized(model, monkeypatch):
    # A CoreML-active model holds a lock and takes it around session.run.
    events: list[str] = []

    class _TrackingLock:
        def __enter__(self):
            events.append("acquire")
            return self

        def __exit__(self, *exc):
            events.append("release")
            return False

    monkeypatch.setattr(model, "_infer_lock", _TrackingLock())
    monkeypatch.setattr(model, "_coreml_active", True)
    monkeypatch.setattr(runtime, "_autorelease_pool", contextlib.nullcontext)
    monkeypatch.setattr(model, "session", _FakeSession())

    model.run({"x": np.zeros((1, 1), dtype=np.int64)})

    assert events == ["acquire", "release"]
    assert model.coreml_serialized is True


def test_coreml_autorelease_pool_property_reflects_pyobjc_availability(model, monkeypatch):
    model._coreml_active = True
    # pyobjc available -> property True (any factory that isn't nullcontext)
    fake = types.ModuleType("objc")
    fake.autorelease_pool = lambda: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "objc", fake)
    assert model.coreml_autorelease_pool is True

    # pyobjc missing -> property False
    runtime._POOL_FACTORY = None
    monkeypatch.setitem(sys.modules, "objc", None)
    assert model.coreml_autorelease_pool is False
