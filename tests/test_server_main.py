from __future__ import annotations

from bge_m3_bench.server.__main__ import build_parser, config_from_args
from bge_m3_bench.server.grpc_server import resolve_intra_op_threads


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


def test_intra_op_defaults_to_auto_sentinel_when_omitted():
    cfg = _config(["--model", "m.onnx", "--tokenizer", "t.json"])
    assert cfg.intra_op_threads == -1  # -1 = auto (resolved at build time)
    assert cfg.inter_op_threads == 0


def test_max_workers_flag_wires_into_config():
    cfg = _config(["--model", "m.onnx", "--tokenizer", "t.json", "--max-workers", "4"])
    assert cfg.max_workers == 4
    cfg_default = _config(["--model", "m.onnx", "--tokenizer", "t.json"])
    assert cfg_default.max_workers == 8


def test_resolve_intra_op_auto_bounds_oversubscription():
    # auto (-1): max(1, physical // max_workers)
    assert resolve_intra_op_threads(-1, max_workers=8, physical_cores=8) == 1
    assert resolve_intra_op_threads(-1, max_workers=4, physical_cores=16) == 4
    # never below 1, even when workers exceed cores
    assert resolve_intra_op_threads(-1, max_workers=16, physical_cores=8) == 1
    assert resolve_intra_op_threads(-1, max_workers=0, physical_cores=8) == 8


def test_resolve_intra_op_passes_through_explicit_values():
    # 0 = ORT default (all cores); >0 = explicit. Both bypass auto.
    assert resolve_intra_op_threads(0, max_workers=8, physical_cores=8) == 0
    assert resolve_intra_op_threads(2, max_workers=8, physical_cores=8) == 2


def test_explicit_zero_is_honored_not_treated_as_falsy(monkeypatch):
    # Env sets a non-zero default; an explicit --intra-op-threads 0 must win
    # (regression: `or` would wrongly fall back to the env value for 0).
    monkeypatch.setenv("BGE_M3_INTRA_OP", "8")
    cfg = _config(["--model", "m.onnx", "--tokenizer", "t.json", "--intra-op-threads", "0"])
    assert cfg.intra_op_threads == 0

    # And when the flag is omitted, the env default flows through.
    cfg_env = _config(["--model", "m.onnx", "--tokenizer", "t.json"])
    assert cfg_env.intra_op_threads == 8
