from __future__ import annotations

import numpy as np
import pytest

from onnx_grpc_benchmark.benchmark.validation import validate_model
from onnx_grpc_benchmark.server.registry import ModelRegistry
from onnx_grpc_benchmark.server.runtime import _onnx_type_to_numpy


def test_model_runs_and_outputs_valid(model, dataset):
    result = model.run(dataset.samples[0])
    assert "output" in result.outputs
    assert result.inference_us >= 0
    assert np.all(np.isfinite(result.outputs["output"]))


def test_input_output_specs(model):
    ins = model.input_specs()
    outs = model.output_specs()
    assert ins[0].name == "input"
    assert ins[0].dtype == "float32"
    assert outs[0].name == "output"


def test_validation_passes(model, dataset):
    report = validate_model(model, dataset.samples[0])
    assert report.passed
    names = {c["check"] for c in report.checks}
    assert {"outputs_finite", "outputs_present", "reproducible"} <= names


def test_onnx_type_mapping_supported():
    assert _onnx_type_to_numpy("tensor(float)") == "float32"
    assert _onnx_type_to_numpy("tensor(int8)") == "int8"
    assert _onnx_type_to_numpy("tensor(uint16)") == "uint16"


def test_onnx_type_mapping_unsupported_raises():
    # Unknown element types must fail loudly instead of defaulting to float32,
    # which would silently feed a model the wrong input dtype.
    with pytest.raises(ValueError, match="unsupported ONNX tensor dtype"):
        _onnx_type_to_numpy("tensor(string)")


def test_registry_default_and_lookup(tiny_model_path):
    reg = ModelRegistry()
    reg.load(tiny_model_path, name="m1", default=True)
    reg.load(tiny_model_path, name="m2")
    assert reg.default_model == "m1"
    assert set(reg.names()) == {"m1", "m2"}
    assert reg.get().name == "m1"
    assert reg.get("m2").name == "m2"
