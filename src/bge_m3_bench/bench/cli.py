"""``bge-m3-bench``: drive the gRPC embedding service and write JSONL metrics.

Single-stream, duration-based. Streams one raw record per request and a final
summary record. All aggregation lives in :mod:`bge_m3_bench.bench.metrics`.
"""

from __future__ import annotations

import itertools
import json
import sys
import threading
import time
from collections import Counter
from concurrent import futures
from contextlib import ExitStack
from pathlib import Path
from typing import Any, TextIO

import click
import grpc
import numpy as np

from ..client import EmbeddingClient, EmbedResult
from ..common.logging import configure_logging
from ..common.texts import load_texts as _load_texts
from .metrics import RequestSample, RunContext, build_summary, error_row, request_row
from .validation import validate_embeddings


def _batch(pool: list[str], i: int, batch_size: int) -> list[str]:
    n = len(pool)
    return [pool[(i * batch_size + j) % n] for j in range(batch_size)]


def _provider_label(spec: dict[str, Any]) -> str:
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


def _run_phase(
    clients: list[EmbeddingClient],
    pool: list[str],
    batch_size: int,
    duration_sec: float,
    *,
    fh: TextIO | None = None,
    counter: itertools.count[int] | None = None,
) -> tuple[list[RequestSample], Counter[str]]:
    """Drive ``len(clients)`` concurrent workers for ``duration_sec`` wall seconds.

    Each worker owns one client (one gRPC channel) and loops blocking ``embed``
    calls until the deadline. Returns the collected successful samples and a
    ``Counter`` of gRPC error codes for failed requests (its total is the failed
    count). When ``fh``/``counter`` are given, each successful request is streamed
    as a JSONL row under a lock (preserving the single-stream streaming behavior);
    ordering is by completion time.
    """
    stop_at = time.perf_counter() + duration_sec
    samples: list[RequestSample] = []
    error_codes: Counter[str] = Counter()
    lock = threading.Lock()
    stride = len(clients)

    def worker(wid: int, client: EmbeddingClient) -> None:
        i = wid  # per-worker offset so workers don't all send identical batches
        while time.perf_counter() < stop_at:
            try:
                res = client.embed(_batch(pool, i, batch_size))
            except grpc.RpcError as exc:
                code, detail = exc.code().name, exc.details()
                with lock:
                    error_codes[code] += 1
                    # Emit a per-request error record so the count of "request"
                    # rows equals total_requests (success + failed).
                    if fh is not None and counter is not None:
                        fh.write(json.dumps(error_row(next(counter), code, detail)) + "\n")
                i += stride
                continue
            except Exception as exc:
                # A non-gRPC failure (e.g. a malformed response in
                # bytes_to_matrix) would otherwise propagate out of ex.map and
                # abort the whole run, discarding every sample already collected.
                # Record it as a CLIENT_ERROR row instead so the run degrades.
                with lock:
                    error_codes["CLIENT_ERROR"] += 1
                    if fh is not None and counter is not None:
                        fh.write(
                            json.dumps(error_row(next(counter), "CLIENT_ERROR", str(exc))) + "\n"
                        )
                i += stride
                continue
            s = _to_sample(res)
            with lock:
                samples.append(s)
                if fh is not None and counter is not None:
                    fh.write(json.dumps(request_row(next(counter), s)) + "\n")
            i += stride

    with futures.ThreadPoolExecutor(max_workers=len(clients)) as ex:
        list(ex.map(lambda args: worker(*args), enumerate(clients)))
    return samples, error_codes


def _reference_embeddings(
    ref_model: str, ref_tokenizer: str, texts: list[str], spec: dict[str, Any]
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
@click.option(
    "--concurrency",
    type=int,
    default=1,
    show_default=True,
    help="Number of concurrent in-flight requests (worker threads, one gRPC channel each).",
)
@click.option(
    "--timeout",
    type=float,
    default=120.0,
    show_default=True,
    help="Per-request gRPC timeout (seconds). Raise for heavy models / high concurrency.",
)
@click.option(
    "--texts",
    "texts_path",
    type=str,
    default=None,
    help="Path to a one-input-per-line text file, or an http(s):// URL to one.",
)
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
    timeout: float,
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
    if concurrency < 1:
        raise click.ClickException("--concurrency must be >= 1")
    if bool(ref_model) != bool(ref_tokenizer):
        raise click.ClickException("--ref-model and --ref-tokenizer must be provided together")
    try:
        pool = _load_texts(texts_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    if not pool:
        raise click.ClickException("no input texts")

    with ExitStack() as stack:
        # One control client (spec, validation, resource window) + N worker
        # clients, each with its own gRPC channel (≈ N independent clients).
        control = stack.enter_context(EmbeddingClient(address, timeout=timeout))
        control.wait_ready()
        spec = control.get_spec()
        workers = [
            stack.enter_context(EmbeddingClient(address, timeout=timeout))
            for _ in range(concurrency)
        ]
        for w in workers:
            w.wait_ready()

        # Warmup (not measured), driven at full concurrency.
        _run_phase(workers, pool, batch_size, warmup_sec)

        # Validation on a representative batch (single-stream, via control client).
        validation = None
        if validate:
            sample_batch = _batch(pool, 0, batch_size)
            vres = control.embed(sample_batch)
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
        control.resource_samples(reset=True)

        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            counter = itertools.count()
            start = time.perf_counter()
            samples, error_codes = _run_phase(
                workers, pool, batch_size, duration_sec, fh=fh, counter=counter
            )
            duration = time.perf_counter() - start
            try:
                resource_samples = control.resource_samples()
            except grpc.RpcError:
                # Server may be unavailable (e.g. crashed mid-run, inflating
                # failed_requests). Still emit a summary with what we have.
                resource_samples = []

            provider = _provider_label(spec)
            ctx = RunContext(
                benchmark_id=benchmark_id
                or f"bge-m3-grpc-{provider}-{precision}-bs{batch_size}-c{concurrency}",
                duration_sec=duration,
                warmup_sec=warmup_sec,
                batch_size=batch_size,
                concurrency=concurrency,
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
                error_codes=dict(error_codes),
            )
            fh.write(json.dumps(summary) + "\n")

    grpc_metrics = summary["grpc_metrics"]
    run_ok = grpc_metrics["successful_requests"] > 0
    click.echo(
        json.dumps(
            {
                "out": str(out_path),
                "run_ok": run_ok,
                "requests": len(samples),
                "grpc_metrics": grpc_metrics,
                "validation": summary["validation"],
            },
            indent=2,
            default=str,
        )
    )
    # A run with zero successful requests yields all-zero metrics; say why loudly
    # (the per-request error rows live in the JSONL) and exit non-zero for CI.
    if not run_ok:
        breakdown = ", ".join(f"{code} x{n}" for code, n in grpc_metrics["error_codes"].items())
        click.echo(
            f"0/{grpc_metrics['total_requests']} requests succeeded"
            + (f" — {breakdown}." if breakdown else ".")
            + " Raise --timeout, lower --concurrency, or set the server's"
            " --intra-op-threads to reduce CPU oversubscription.",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    cli()
