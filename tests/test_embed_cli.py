"""Tests for the ``bge-m3-embed`` vector-dump CLI."""

from __future__ import annotations

import json
import math

from click.testing import CliRunner

from bge_m3_bench.common.texts import DEFAULT_TEXTS
from bge_m3_bench.embed_cli import cli


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_embed_cli_writes_vectors(running_server, tmp_path):
    texts = ["hello world", "foo bar baz", "the quick brown fox"]
    texts_file = tmp_path / "inputs.txt"
    texts_file.write_text("\n".join(texts) + "\n")
    out = tmp_path / "emb.jsonl"

    result = CliRunner().invoke(
        cli,
        ["--address", running_server, "--texts", str(texts_file), "--out", str(out)],
    )
    assert result.exit_code == 0, result.output

    rows = _read_jsonl(out)
    assert [r["text"] for r in rows] == texts
    assert [r["i"] for r in rows] == [0, 1, 2]
    for r in rows:
        assert r["dim"] == 8
        assert len(r["embedding"]) == 8
        assert all(math.isfinite(x) for x in r["embedding"])


def test_embed_cli_default_texts(running_server, tmp_path):
    out = tmp_path / "emb.jsonl"
    result = CliRunner().invoke(cli, ["--address", running_server, "--out", str(out)])
    assert result.exit_code == 0, result.output

    rows = _read_jsonl(out)
    assert [r["text"] for r in rows] == DEFAULT_TEXTS


def test_embed_cli_inline_text(running_server, tmp_path):
    out = tmp_path / "emb.jsonl"
    result = CliRunner().invoke(
        cli,
        [
            "--address",
            running_server,
            "--text",
            "hello world",
            "--text",
            "a second sentence",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    rows = _read_jsonl(out)
    assert [r["text"] for r in rows] == ["hello world", "a second sentence"]
    assert all(len(r["embedding"]) == 8 for r in rows)


def test_embed_cli_text_and_texts_mutually_exclusive(running_server, tmp_path):
    texts_file = tmp_path / "inputs.txt"
    texts_file.write_text("from file\n")
    result = CliRunner().invoke(
        cli,
        [
            "--address",
            running_server,
            "--text",
            "inline",
            "--texts",
            str(texts_file),
            "--out",
            str(tmp_path / "emb.jsonl"),
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_embed_cli_batches_preserve_order(running_server, tmp_path):
    texts = ["a", "b", "c", "d", "e"]
    texts_file = tmp_path / "inputs.txt"
    texts_file.write_text("\n".join(texts) + "\n")
    out = tmp_path / "emb.jsonl"

    result = CliRunner().invoke(
        cli,
        [
            "--address",
            running_server,
            "--texts",
            str(texts_file),
            "--batch-size",
            "2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    rows = _read_jsonl(out)
    assert [r["i"] for r in rows] == [0, 1, 2, 3, 4]
    assert [r["text"] for r in rows] == texts
