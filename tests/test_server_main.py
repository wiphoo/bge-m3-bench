from __future__ import annotations

from bge_m3_bench.server.__main__ import build_parser, config_from_args


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
    cfg = _config(
        ["--model", "m.onnx", "--tokenizer", "t.json", "--intra-op-threads", "0"]
    )
    assert cfg.intra_op_threads == 0

    # And when the flag is omitted, the env default flows through.
    cfg_env = _config(["--model", "m.onnx", "--tokenizer", "t.json"])
    assert cfg_env.intra_op_threads == 8
