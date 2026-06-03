"""``bge-m3-embed``: call the gRPC service and dump dense vectors to JSONL.

A thin client tool, separate from the ``bge-m3-bench`` performance benchmark: it
writes the actual embeddings (one JSON record per input text), not metrics. The
gRPC ``Embed`` endpoint and the float32 wire transport are reused unchanged.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import click

from .client import EmbeddingClient
from .common.logging import configure_logging
from .common.texts import load_texts


def _chunks(pool: list[str], batch_size: int) -> list[list[str]]:
    return [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]


@click.command()
@click.option("--address", default="localhost:50051", show_default=True)
@click.option("--texts", "texts_path", type=str, default=None,
              help="Path to a one-sentence-per-line text file, or an http(s):// URL to one.")
@click.option(
    "--text",
    "inline_texts",
    multiple=True,
    help="Inline input text (repeatable). Mutually exclusive with --texts.",
)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--out", default="results/embeddings.jsonl", show_default=True)
@click.option("--log-level", default="WARNING", show_default=True)
def cli(
    address: str,
    texts_path: str | None,
    inline_texts: tuple[str, ...],
    batch_size: int,
    out: str,
    log_level: str,
) -> None:
    """Dump dense BGE-M3 embeddings for the given texts as JSONL.

    Inputs come from ``--text`` (inline, repeatable), else ``--texts`` (a file
    with one input per line, or an ``http(s)://`` URL to such a file), else a
    small built-in sample.
    """
    configure_logging(log_level)
    if batch_size < 1:
        raise click.ClickException("--batch-size must be >= 1")
    if inline_texts and texts_path:
        raise click.ClickException("--text and --texts are mutually exclusive")
    try:
        pool = [t for t in inline_texts if t.strip()] if inline_texts else load_texts(texts_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not pool:
        raise click.ClickException("no input texts")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dim = 0
    counter = itertools.count()
    with EmbeddingClient(address) as client, out_path.open("w") as fh:
        client.wait_ready()
        for batch in _chunks(pool, batch_size):
            res = client.embed(batch)
            dim = res.embedding_dim
            for j, text in enumerate(batch):
                record = {
                    "i": next(counter),
                    "text": text,
                    "dim": res.embedding_dim,
                    "embedding": res.embeddings[j].tolist(),
                }
                fh.write(json.dumps(record) + "\n")

    click.echo(json.dumps({"out": str(out_path), "count": len(pool), "dim": dim}, indent=2))


if __name__ == "__main__":
    cli()
