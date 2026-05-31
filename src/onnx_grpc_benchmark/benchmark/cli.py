"""Command-line benchmark tool (Milestone 3): ``onnx-bench``."""

from __future__ import annotations

import json
from pathlib import Path

import click

from ..common.logging import configure_logging
from ..metadata import collect_metadata, metadata_schema
from ..server.runtime import OnnxModel
from . import dataset as ds
from .report import save_csv, save_json
from .runner import BenchmarkConfig, BenchmarkResult, run_grpc, run_local


@click.group()
@click.option("--log-level", default="INFO", show_default=True)
def cli(log_level: str) -> None:
    """ONNX Runtime gRPC benchmark toolkit."""
    configure_logging(log_level)


@cli.command("metadata")
@click.option("--schema", is_flag=True, help="Print the JSON schema instead.")
def metadata_cmd(schema: bool) -> None:
    """Print machine/environment metadata (or its JSON schema)."""
    payload = metadata_schema() if schema else collect_metadata()
    click.echo(json.dumps(payload, indent=2, default=str))


@cli.command("gen-dataset")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--name", default="synthetic", show_default=True)
@click.option("--num-samples", default=64, show_default=True)
@click.option("--batch-size", default=1, show_default=True)
@click.option("--seed", default=1234, show_default=True)
def gen_dataset_cmd(
    model_path: str,
    out: str,
    name: str,
    num_samples: int,
    batch_size: int,
    seed: int,
) -> None:
    """Generate a deterministic, versioned benchmark dataset for a model."""
    model = OnnxModel(model_path)
    dataset = ds.generate_for_model(
        model,
        name=name,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
    )
    path = dataset.save(out)
    click.echo(
        json.dumps(
            {
                "saved": str(path),
                "samples": len(dataset),
                "fingerprint": dataset.fingerprint(),
            },
            indent=2,
        )
    )


@cli.command("local")
@click.option("--model", "model_path", required=True, type=click.Path(exists=True))
@click.option("--dataset", "dataset_path", type=click.Path(exists=True), default=None)
@click.option("--provider", default="cpu", show_default=True)
@click.option("--warmup", default=5, show_default=True)
@click.option("--iterations", default=100, show_default=True)
@click.option("--num-samples", default=32, show_default=True, help="If no dataset given.")
@click.option("--seed", default=1234, show_default=True)
@click.option("--out", type=click.Path(), default=None, help="JSON result path.")
def local_cmd(
    model_path: str,
    dataset_path: str | None,
    provider: str,
    warmup: int,
    iterations: int,
    num_samples: int,
    seed: int,
    out: str | None,
) -> None:
    """Benchmark a model in-process (Milestone 1)."""
    model = OnnxModel(model_path, provider=provider)
    dataset = (
        ds.Dataset.load(dataset_path)
        if dataset_path
        else ds.generate_for_model(model, num_samples=num_samples, seed=seed)
    )
    config = BenchmarkConfig(
        model_name=model.name,
        provider=provider,
        transport="local",
        warmup=warmup,
        iterations=iterations,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_fingerprint=dataset.fingerprint(),
    )
    result = run_local(model, dataset, config)
    _emit(result, out)


@cli.command("grpc")
@click.option("--address", default="localhost:50051", show_default=True)
@click.option("--model", "model_name", default="", help="Model id (default model if empty).")
@click.option(
    "--ref-model",
    "ref_model_path",
    type=click.Path(exists=True),
    required=True,
    help="Local copy of the model used only to shape the dataset.",
)
@click.option("--provider", default="cpu", show_default=True)
@click.option("--warmup", default=5, show_default=True)
@click.option("--iterations", default=100, show_default=True)
@click.option("--num-samples", default=32, show_default=True)
@click.option("--seed", default=1234, show_default=True)
@click.option("--out", type=click.Path(), default=None)
def grpc_cmd(
    address: str,
    model_name: str,
    ref_model_path: str,
    provider: str,
    warmup: int,
    iterations: int,
    num_samples: int,
    seed: int,
    out: str | None,
) -> None:
    """Benchmark a model through a running gRPC server (Milestone 3)."""
    from ..server.client import InferenceClient

    ref_model = OnnxModel(ref_model_path)
    dataset = ds.generate_for_model(ref_model, num_samples=num_samples, seed=seed)
    config = BenchmarkConfig(
        model_name=model_name or ref_model.name,
        provider=provider,
        transport="grpc",
        warmup=warmup,
        iterations=iterations,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_fingerprint=dataset.fingerprint(),
    )
    with InferenceClient(address) as client:
        client.wait_ready()
        result = run_grpc(client, dataset, config, model_name=model_name, reference=ref_model)
    _emit(result, out)


def _emit(result: BenchmarkResult, out: str | None) -> None:
    if out:
        path = save_json(result, out)
        csv_path = Path(out).with_suffix(".csv")
        save_csv([result], csv_path)
        click.echo(json.dumps({"json": str(path), "csv": str(csv_path)}, indent=2))
    click.echo(json.dumps({"stats": result.stats, "validation": result.validation}, indent=2))


if __name__ == "__main__":
    cli()
