"""Construct and run the embedding gRPC server with health + reflection."""

from __future__ import annotations

import signal
import threading
from concurrent import futures
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


def build_from_config(config: ServerConfig) -> tuple[Embedder, dict[str, Any], ResourceSampler]:
    """Load model + tokenizer + build the embedder, spec, and resource sampler."""
    from .runtime import OnnxModel
    from .spec import build_spec
    from .tokenizer import BgeTokenizer

    if not config.model_path or not config.tokenizer_path:
        raise ValueError("both --model and --tokenizer are required")
    model = OnnxModel(
        config.model_path,
        provider=config.provider,
        provider_options=config.provider_options,
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
