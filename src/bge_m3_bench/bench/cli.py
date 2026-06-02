"""``bge-m3-bench``: drive the gRPC embedding service and write JSONL metrics.

Single-stream, duration-based. Streams one raw record per request and a final
summary record. All aggregation lives in :mod:`bge_m3_bench.bench.metrics`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np

from ..client import EmbeddingClient, EmbedResult
from ..common.logging import configure_logging
from .metrics import RequestSample, RunContext, build_summary, request_row
from .validation import validate_embeddings

DEFAULT_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Embeddings turn text into dense vectors.",
    "BGE-M3 supports dense, sparse, and multi-vector retrieval.",
    "gRPC is a high-performance RPC framework.",
    "Tokenization splits text into model input ids.",
]


def _load_texts(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_TEXTS)
    lines = [line.strip() for line in Path(path).read_text().splitlines()]
    return [line for line in lines if line]


def _batch(pool: list[str], i: int, batch_size: int) -> list[str]:
    n = len(pool)
    return [pool[(i * batch_size + j) % n] for j in range(batch_size)]


def _provider_label(spec: dict) -> str:
    """Short label for the *active* execution provider (for the benchmark id).

    Uses the resolved provider the server actually runs on (``runtime.
    execution_provider``, e.g. ``CPUExecutionProvider`` -> ``cpu``) so a
    ``cuda`` request that fell back to CPU is not mislabeled. Falls back to the
    requested logical provider when the active one is unavailable.
    """
    active = str(spec.get("runtime", {}).get("execution_provider") or "")
    label = active.removesuffix("ExecutionProvider").lower()
    return label or str(spec.get("config", {}).get("provider", "cpu"))


def _to_sample(res: EmbedResult) -> RequestSample:
    return RequestSample(
        num_inputs=res.num_inputs,
        token_counts=res.token_counts,
        tokenize_us=res.tokenize_us,
        inference_us=res.inference_us,
        postprocess_us=res.postprocess_us,
        client_e2e_us=res.client_e2e_us,
        request_bytes=res.request_bytes,
        response_bytes=res.response_bytes,
    )


def _reference_embeddings(
    ref_model: str, ref_tokenizer: str, texts: list[str], spec: dict
) -> np.ndarray:
    from ..server.embedder import Embedder
    from ..server.runtime import OnnxModel
    from ..server.tokenizer import BgeTokenizer

    cfg = spec.get("config", {})
    model = OnnxModel(ref_model, provider="cpu")
    tokenizer = BgeTokenizer.from_file(ref_tokenizer, max_length=int(cfg.get("max_length", 512)))
    embedder = Embedder(
        model,
        tokenizer,
        pooling=str(cfg.get("pooling", "cls")),
        normalize=bool(cfg.get("normalize", True)),
    )
    return embedder.embed(texts).embeddings


@click.command()
@click.option("--address", default="localhost:50051", show_default=True)
@click.option("--duration-sec", type=float, default=30.0, show_default=True)
@click.option("--warmup-sec", type=float, default=5.0, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--concurrency", type=int, default=1, show_default=True, help="MVP supports 1.")
@click.option("--texts", "texts_path", type=click.Path(exists=True), default=None)
@click.option("--benchmark-id", default=None)
@click.option("--model-name", default="")
@click.option("--model-revision", default="unknown")
@click.option("--precision", default="fp32", show_default=True)
@click.option("--quantization", default="none", show_default=True)
@click.option("--ref-model", type=click.Path(exists=True), default=None)
@click.option("--ref-tokenizer", type=click.Path(exists=True), default=None)
@click.option("--validate/--no-validate", default=True, show_default=True)
@click.option("--fail-on-invalid", is_flag=True, default=False)
@click.option("--out", default="results/run.jsonl", show_default=True)
@click.option("--log-level", default="WARNING", show_default=True)
def cli(
    address: str,
    duration_sec: float,
    warmup_sec: float,
    batch_size: int,
    concurrency: int,
    texts_path: str | None,
    benchmark_id: str | None,
    model_name: str,
    model_revision: str,
    precision: str,
    quantization: str,
    ref_model: str | None,
    ref_tokenizer: str | None,
    validate: bool,
    fail_on_invalid: bool,
    out: str,
    log_level: str,
) -> None:
    """Benchmark a running BGE-M3 embedding gRPC server."""
    configure_logging(log_level)
    if concurrency != 1:
        raise click.ClickException("MVP supports --concurrency 1 only")
    if bool(ref_model) != bool(ref_tokenizer):
        raise click.ClickException("--ref-model and --ref-tokenizer must be provided together")
    pool = _load_texts(texts_path)
    if not pool:
        raise click.ClickException("no input texts")

    with EmbeddingClient(address) as client:
        client.wait_ready()
        spec = client.get_spec()

        # Warmup (not measured).
        warm_start = time.perf_counter()
        i = 0
        while time.perf_counter() - warm_start < warmup_sec:
            client.embed(_batch(pool, i, batch_size))
            i += 1

        # Validation on a representative batch.
        validation = None
        if validate:
            sample_batch = _batch(pool, 0, batch_size)
            vres = client.embed(sample_batch)
            reference = None
            if ref_model and ref_tokenizer:
                reference = _reference_embeddings(ref_model, ref_tokenizer, sample_batch, spec)
            validation, passed, reasons = validate_embeddings(
                vres.embeddings,
                normalize=bool(spec.get("config", {}).get("normalize", True)),
                reference=reference,
            )
            if not passed and fail_on_invalid:
                raise click.ClickException(f"embedding validation failed: {reasons}")

        # Fresh resource window for the measured phase.
        client.resource_samples(reset=True)

        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        samples: list[RequestSample] = []
        start = time.perf_counter()
        i = 0
        with out_path.open("w") as fh:
            while time.perf_counter() - start < duration_sec:
                res = client.embed(_batch(pool, i, batch_size))
                sample = _to_sample(res)
                samples.append(sample)
                fh.write(json.dumps(request_row(i, sample)) + "\n")
                i += 1
            duration = time.perf_counter() - start
            resource_samples = client.resource_samples()

            provider = _provider_label(spec)
            ctx = RunContext(
                benchmark_id=benchmark_id
                or f"bge-m3-grpc-{provider}-{precision}-bs{batch_size}-c1",
                duration_sec=duration,
                warmup_sec=warmup_sec,
                batch_size=batch_size,
                model_name=model_name,
                model_revision=model_revision,
                precision=precision,
                quantization=quantization,
            )
            summary = build_summary(
                samples=samples,
                resource_samples=resource_samples,
                spec=spec,
                ctx=ctx,
                validation=validation,
            )
            fh.write(json.dumps(summary) + "\n")

    click.echo(
        json.dumps(
            {
                "out": str(out_path),
                "requests": len(samples),
                "grpc_metrics": summary["grpc_metrics"],
                "validation": summary["validation"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    cli()
