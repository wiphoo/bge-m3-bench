"""Tests for the diagnostic tooling (scripts/coreml_memcheck.py).

Kept out of the core suite — marked ``debug`` and excluded from ``make test`` —
so the benchmark/server tests stay focused. Run with ``make test-debug``.
"""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.debug


def test_malloc_in_use_mb_platform_aware():
    from coreml_memcheck import _malloc_in_use_mb

    result = _malloc_in_use_mb()
    if sys.platform == "darwin":
        assert result is None or result >= 0
    else:
        assert result is None  # macOS-only introspection


def test_memcheck_smoke_cpu(tiny_model_path, tiny_tokenizer_path):
    # The diagnostic harness runs end-to-end on CPU with the tiny fixtures: this
    # guards the reuse wiring (parser, tokenizer/model load, feed building, loop)
    # on every platform. CoreML-specific behavior is exercised manually on macOS.
    from coreml_memcheck import build_parser, run_loop

    args = build_parser().parse_args(
        [
            "--model",
            str(tiny_model_path),
            "--tokenizer",
            str(tiny_tokenizer_path),
            "--provider",
            "cpu",
            "--synthetic",
            "8",
            "--iters",
            "3",
            "--batch-size",
            "2",
            "--rss-every",
            "1",
            "--vary-length",
        ]
    )
    assert run_loop(args) == 0


def test_memcheck_smoke_fixed_pad_length(tiny_model_path, tiny_tokenizer_path):
    from coreml_memcheck import build_parser, run_loop

    args = build_parser().parse_args(
        [
            "--model",
            str(tiny_model_path),
            "--tokenizer",
            str(tiny_tokenizer_path),
            "--provider",
            "cpu",
            "--synthetic",
            "8",
            "--iters",
            "2",
            "--batch-size",
            "2",
            "--pad-length",
            "16",
        ]
    )
    assert run_loop(args) == 0
