"""Construct and run the embedding gRPC server with health + reflection."""

from __future__ import annotations

import os
import signal
import threading
from concurrent import futures
from dataclasses import replace
from typing import Any

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from ..common.config import ServerConfig
from ..common.logging import get_logger
from ..generated import embedding_pb2 as pb
from ..generated import embedding_pb2_grpc as pb_grpc
from .embedder import Embedder
from .resources import ResourceSampler
from .service import EmbeddingServicer

logger = get_logger(__name__)

SERVICE_NAME = pb.DESCRIPTOR.services_by_name["EmbeddingService"].full_name


def build_server(
    embedder: Embedder,
    spec: dict[str, Any],
    sampler: ResourceSampler,
    config: ServerConfig,
) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.max_workers),
        options=[
            ("grpc.max_send_message_length", config.max_message_bytes),
            ("grpc.max_receive_message_length", config.max_message_bytes),
        ],
    )
    pb_grpc.add_EmbeddingServiceServicer_to_server(
        EmbeddingServicer(embedder, spec, sampler), server
    )

    # Standard grpc.health.v1: the single embedder is always serving once built.
    health_servicer = health.HealthServicer()
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set(SERVICE_NAME, health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Embedded server reflection for discovery (grpcurl-friendly).
    reflection.enable_server_reflection(
        (
            SERVICE_NAME,
            health_pb2.DESCRIPTOR.services_by_name["Health"].full_name,
            reflection.SERVICE_NAME,
        ),
        server,
    )

    server.add_insecure_port(config.address)
    return server


def _physical_cores() -> int:
    """Best-effort physical core count, falling back to logical/1."""
    try:
        import psutil

        cores = psutil.cpu_count(logical=False)
        if cores:
            return int(cores)
    except Exception:  # pragma: no cover - psutil import/probe failure
        pass
    return os.cpu_count() or 1


def _resolve_usable_cores(physical_cores: int, effective_cores: float | None) -> int:
    """Cores actually usable by this process for auto thread sizing.

    Host physical cores capped by the *effective* limit (cgroup CPU quota / CPU
    affinity) when known, so a container limited to N CPUs on a big host — or a
    ``taskset`` pin — sizes threads to N, not the host's physical count. Mirrors
    the ``min(physical, effective)`` normalization used in the bench layer.
    Falls back to the physical count when the effective limit is unknown.
    """
    if effective_cores and effective_cores > 0:
        return max(1, min(physical_cores, int(effective_cores)))
    return physical_cores


def resolve_intra_op_threads(intra_op_threads: int, max_workers: int, usable_cores: int) -> int:
    """Resolve the ``-1`` auto sentinel into a concrete intra-op thread count.

    Auto bounds CPU oversubscription: with ``max_workers`` requests potentially
    in flight, each ONNX session is capped at ``max(1, usable_cores //
    max_workers)`` threads so the total stays near the cores the process can
    actually use. ``0`` (ORT default = all cores) and explicit ``>0`` values
    pass through.
    """
    if intra_op_threads != -1:
        return intra_op_threads
    return max(1, usable_cores // max(1, max_workers))


def build_from_config(config: ServerConfig) -> tuple[Embedder, dict[str, Any], ResourceSampler]:
    """Load model + tokenizer + build the embedder, spec, and resource sampler."""
    from .runtime import OnnxModel
    from .spec import _effective_cpu_cores, build_spec
    from .tokenizer import BgeTokenizer

    if not config.model_path or not config.tokenizer_path:
        raise ValueError("both --model and --tokenizer are required")
    # Size auto threads off the cores the process can actually use (cgroup quota /
    # affinity), not the host physical count, so containers/taskset don't oversubscribe.
    usable_cores = _resolve_usable_cores(_physical_cores(), _effective_cpu_cores())
    intra_op = resolve_intra_op_threads(config.intra_op_threads, config.max_workers, usable_cores)
    if config.intra_op_threads == -1:
        logger.info(
            "intra-op threads auto-resolved",
            extra={
                "fields": {
                    "intra_op_threads": intra_op,
                    "max_workers": config.max_workers,
                    "usable_cores": usable_cores,
                }
            },
        )
    # Carry the resolved value forward so the spec reports the concrete thread
    # count (not the -1 sentinel) and the model runs with it.
    config = replace(config, intra_op_threads=intra_op)
    model = OnnxModel(
        config.model_path,
        provider=config.provider,
        intra_op_threads=config.intra_op_threads,
        inter_op_threads=config.inter_op_threads,
    )
    output_specs = model.output_specs()
    if output_specs and output_specs[0].dtype == "float16" and "CPU" in model.active_provider:
        logger.warning(
            "fp16 model on CPU provider — onnxruntime up-casts to fp32; expect no speedup",
            extra={"fields": {"model": model.name, "provider": model.active_provider}},
        )
    tokenizer = BgeTokenizer.from_file(config.tokenizer_path, max_length=config.max_length)
    embedder = Embedder(model, tokenizer, pooling=config.pooling, normalize=config.normalize)
    spec = build_spec(config, model)
    return embedder, spec, ResourceSampler()


def serve(config: ServerConfig) -> None:
    """Build everything from config, start the server, and block until signalled."""
    embedder, spec, sampler = build_from_config(config)
    sampler.start()
    server = build_server(embedder, spec, sampler, config)
    server.start()
    logger.info("server started", extra={"fields": {"address": config.address}})

    stop_event = threading.Event()

    def _handle(signum: int, _frame: object) -> None:
        logger.info("shutdown signal received", extra={"fields": {"signal": signum}})
        stop_event.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    stop_event.wait()
    server.stop(grace=10).wait()
    sampler.stop()
    logger.info("server stopped")
