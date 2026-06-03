from __future__ import annotations

from bge_m3_bench.common.config import ServerConfig
from bge_m3_bench.server.spec import _filter_isa, _parse_cpu_max, build_spec, embedding_dim


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


def test_machine_extra_fields(model):
    config = ServerConfig(pooling="cls", normalize=True, tokenizer_path="/x/tok.json")
    machine = build_spec(config, model)["machine"]
    assert isinstance(machine["cpu_isa_extensions"], list)
    for key in ("cpu_freq_max_mhz", "cpu_freq_min_mhz", "cpu_freq_current_mhz"):
        assert machine[key] is None or isinstance(machine[key], float)
    # Container/affinity-aware capacity fields (best-effort; may be None).
    assert machine["cpu_effective_cores"] is None or isinstance(
        machine["cpu_effective_cores"], float
    )
    assert machine["ram_limit_mb"] is None or isinstance(machine["ram_limit_mb"], float)


def test_filter_isa():
    flags = {"avx2", "avx512f", "avx512_vnni", "sse4_2", "mmx", "fpu"}
    assert _filter_isa(flags) == ["avx2", "avx512_vnni", "avx512f", "sse4_2"]
    assert _filter_isa(set()) == []


def test_parse_cpu_max():
    assert _parse_cpu_max("50000 100000") == 0.5  # half a vCPU
    assert _parse_cpu_max("200000 100000") == 2.0  # 2 vCPUs
    assert _parse_cpu_max("100000") == 1.0  # default 100ms period
    assert _parse_cpu_max("max 100000") is None  # unlimited
    assert _parse_cpu_max("") is None
