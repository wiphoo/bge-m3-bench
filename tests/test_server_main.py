from __future__ import annotations

import pytest

from bge_m3_bench.common.config import ServerConfig, parse_provider_options
from bge_m3_bench.server.__main__ import build_parser, config_from_args


def test_parse_provider_options_skips_blanks_and_trims():
    assert parse_provider_options(" a = b , , c=d ") == {"a": "b", "c": "d"}
    assert parse_provider_options("") == {}


def test_parse_provider_options_rejects_missing_equals():
    with pytest.raises(ValueError):
        parse_provider_options("device_type")


def test_parse_provider_options_rejects_empty_key():
    with pytest.raises(ValueError):
        parse_provider_options("=b")


def test_server_config_is_hashable():
    # provider_options is a dict on a frozen dataclass; it must stay hashable
    # (excluded from eq/hash) so the config can be used as a set/dict key.
    assert hash(ServerConfig()) == hash(ServerConfig())
    assert hash(ServerConfig(provider_options={"a": "b"})) is not None


def _config(argv: list[str]):
    return config_from_args(build_parser().parse_args(argv))


def test_thread_flags_wire_into_config():
    cfg = _config(
        [
            "--model",
            "m.onnx",
            "--tokenizer",
            "t.json",
            "--intra-op-threads",
            "4",
            "--inter-op-threads",
            "2",
        ]
    )
    assert cfg.intra_op_threads == 4
    assert cfg.inter_op_threads == 2


def test_thread_flags_default_to_zero_when_omitted():
    cfg = _config(["--model", "m.onnx", "--tokenizer", "t.json"])
    assert cfg.intra_op_threads == 0
    assert cfg.inter_op_threads == 0


def test_explicit_zero_is_honored_not_treated_as_falsy(monkeypatch):
    # Env sets a non-zero default; an explicit --intra-op-threads 0 must win
    # (regression: `or` would wrongly fall back to the env value for 0).
    monkeypatch.setenv("BGE_M3_INTRA_OP", "8")
    cfg = _config(["--model", "m.onnx", "--tokenizer", "t.json", "--intra-op-threads", "0"])
    assert cfg.intra_op_threads == 0

    # And when the flag is omitted, the env default flows through.
    cfg_env = _config(["--model", "m.onnx", "--tokenizer", "t.json"])
    assert cfg_env.intra_op_threads == 8


def test_provider_options_flag_parses_repeated_pairs():
    cfg = _config(
        [
            "--model",
            "m.onnx",
            "--tokenizer",
            "t.json",
            "--provider",
            "openvino",
            "--provider-option",
            "device_type=CPU",
            "--provider-option",
            "num_of_threads=4",
        ]
    )
    assert cfg.provider == "openvino"
    assert cfg.provider_options == {"device_type": "CPU", "num_of_threads": "4"}


def test_provider_options_default_empty_when_omitted():
    cfg = _config(["--model", "m.onnx", "--tokenizer", "t.json"])
    assert cfg.provider_options == {}


def test_provider_option_value_may_contain_commas():
    # Repeated CLI flags are parsed one pair each (no comma splitting), so a
    # value containing commas (e.g. a path or URL) survives intact.
    cfg = _config(
        [
            "--model",
            "m.onnx",
            "--tokenizer",
            "t.json",
            "--provider-option",
            "cache_dir=/tmp/a,b",
        ]
    )
    assert cfg.provider_options == {"cache_dir": "/tmp/a,b"}


def test_provider_options_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("BGE_M3_PROVIDER_OPTIONS", "device_type=GPU")
    # Flag wins over env.
    cfg = _config(
        ["--model", "m.onnx", "--tokenizer", "t.json", "--provider-option", "device_type=CPU"]
    )
    assert cfg.provider_options == {"device_type": "CPU"}
    # Without the flag, the env value flows through.
    cfg_env = _config(["--model", "m.onnx", "--tokenizer", "t.json"])
    assert cfg_env.provider_options == {"device_type": "GPU"}
