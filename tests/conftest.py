"""Shared pytest fixtures: a tiny on-disk ONNX model and a dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable without an editable install.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Make the test model builder importable.
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory) -> Path:
    import onnx
    from make_test_model import build_model

    path = tmp_path_factory.mktemp("models") / "tiny_mlp.onnx"
    onnx.save(build_model(seed=0), path)
    return path


@pytest.fixture()
def model(tiny_model_path):
    from onnx_grpc_benchmark.server.runtime import OnnxModel

    return OnnxModel(tiny_model_path, provider="cpu")


@pytest.fixture()
def dataset(model):
    from onnx_grpc_benchmark.benchmark import dataset as ds

    return ds.generate_for_model(model, num_samples=8, seed=7)
