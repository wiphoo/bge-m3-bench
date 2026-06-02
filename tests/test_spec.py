from __future__ import annotations

from bge_m3_bench.common.config import ServerConfig
from bge_m3_bench.server.spec import build_spec, embedding_dim


def test_embedding_dim(model):
    assert embedding_dim(model) == 8


def test_build_spec_sections(model):
    config = ServerConfig(pooling="cls", normalize=True, tokenizer_path="/x/tok.json")
    spec = build_spec(config, model)
    assert set(spec) >= {"service_version", "model", "config", "tokenizer", "runtime", "machine"}
    assert spec["model"]["embedding_dim"] == 8
    assert {s["name"] for s in spec["model"]["inputs"]} == {"input_ids", "attention_mask"}
    assert spec["config"]["pooling"] == "cls"
    assert spec["runtime"]["execution_provider"] == "CPUExecutionProvider"
    assert spec["machine"]["cpu_logical_cores"]
    assert spec["tokenizer"]["path"] == "/x/tok.json"
