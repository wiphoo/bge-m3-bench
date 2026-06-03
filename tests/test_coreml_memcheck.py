from __future__ import annotations


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
