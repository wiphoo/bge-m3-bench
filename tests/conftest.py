"""Shared pytest fixtures and import-path setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ and scripts/ are importable without an editable install.
ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    _path = str(ROOT / sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory) -> Path:
    import onnx
    from make_test_embedding_model import build_model

    path = tmp_path_factory.mktemp("model") / "tiny_embed.onnx"
    onnx.save(build_model(seed=0), path)
    return path


@pytest.fixture(scope="session")
def tiny_tokenizer_path(tmp_path_factory) -> Path:
    from make_test_embedding_model import build_tokenizer

    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    build_tokenizer().save(str(path))
    return path


@pytest.fixture()
def model(tiny_model_path):
    from bge_m3_bench.server.runtime import OnnxModel

    return OnnxModel(tiny_model_path, provider="cpu")


@pytest.fixture()
def tokenizer(tiny_tokenizer_path):
    from bge_m3_bench.server.tokenizer import BgeTokenizer

    return BgeTokenizer.from_file(tiny_tokenizer_path, max_length=32)


@pytest.fixture()
def embedder(model, tokenizer):
    from bge_m3_bench.server.embedder import Embedder

    return Embedder(model, tokenizer, pooling="cls", normalize=True)


@pytest.fixture()
def texts() -> list[str]:
    return ["hello world", "foo bar baz", "the quick brown fox"]


@pytest.fixture()
def running_server(model, tokenizer):
    """A real EmbeddingService on a loopback port (cls pooling, normalized)."""
    from bge_m3_bench.common.config import ServerConfig
    from bge_m3_bench.server.embedder import Embedder
    from bge_m3_bench.server.grpc_server import build_server
    from bge_m3_bench.server.resources import ResourceSampler
    from bge_m3_bench.server.spec import build_spec

    config = ServerConfig(port=0, pooling="cls", normalize=True, tokenizer_path="tok.json")
    embedder = Embedder(model, tokenizer, pooling="cls", normalize=True)
    sampler = ResourceSampler(interval_sec=0.01)
    sampler.start()
    server = build_server(embedder, build_spec(config, model), sampler, config)
    port = server.add_insecure_port("localhost:0")
    server.start()
    yield f"localhost:{port}"
    server.stop(grace=0).wait()
    sampler.stop()
