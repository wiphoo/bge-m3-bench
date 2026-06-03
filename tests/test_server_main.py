from __future__ import annotations

from bge_m3_bench.server.__main__ import build_parser, config_from_args
from bge_m3_bench.server.grpc_server import _resolve_usable_cores, resolve_intra_op_threads


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
    # auto (-1): max(1, usable_cores // max_workers)
    assert resolve_intra_op_threads(-1, max_workers=8, usable_cores=8) == 1
    assert resolve_intra_op_threads(-1, max_workers=4, usable_cores=16) == 4
    # never below 1, even when workers exceed cores
    assert resolve_intra_op_threads(-1, max_workers=16, usable_cores=8) == 1
    assert resolve_intra_op_threads(-1, max_workers=0, usable_cores=8) == 8


def test_resolve_intra_op_passes_through_explicit_values():
    # 0 = ORT default (all cores); >0 = explicit. Both bypass auto.
    assert resolve_intra_op_threads(0, max_workers=8, usable_cores=8) == 0
    assert resolve_intra_op_threads(2, max_workers=8, usable_cores=8) == 2


def test_resolve_usable_cores_caps_by_effective_limit():
    # Unconstrained host (effective unknown) -> physical count.
    assert _resolve_usable_cores(8, None) == 8
    # Container limited below host physical -> the effective (quota) count, so
    # auto threading respects the cgroup quota instead of the host's 64 cores.
    assert _resolve_usable_cores(64, 8) == 8
    # Effective above physical (e.g. logical/affinity on an HT host) -> capped at physical.
    assert _resolve_usable_cores(8, 16) == 8
    # Fractional / sub-1 quotas floor but never drop below 1.
    assert _resolve_usable_cores(64, 1.5) == 1
    assert _resolve_usable_cores(64, 0.5) == 1
    assert _resolve_usable_cores(8, 0) == 8  # 0/None treated as "unknown"


def test_explicit_zero_is_honored_not_treated_as_falsy(monkeypatch):
    # Env sets a non-zero default; an explicit --intra-op-threads 0 must win
    # (regression: `or` would wrongly fall back to the env value for 0).
    monkeypatch.setenv("BGE_M3_INTRA_OP", "8")
    cfg = _config(["--model", "m.onnx", "--tokenizer", "t.json", "--intra-op-threads", "0"])
    assert cfg.intra_op_threads == 0

    # And when the flag is omitted, the env default flows through.
    cfg_env = _config(["--model", "m.onnx", "--tokenizer", "t.json"])
    assert cfg_env.intra_op_threads == 8
