"""Construct and run the gRPC server with graceful shutdown."""

from __future__ import annotations

import signal
import threading
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from ..common.config import ServerConfig
from ..common.logging import get_logger
from ..generated import inference_pb2_grpc as pb_grpc
from .registry import ModelRegistry
from .service import InferenceServicer

logger = get_logger(__name__)

INFERENCE_SERVICE_NAME = "onnx_grpc_benchmark.v1.InferenceService"


def build_server(registry: ModelRegistry, config: ServerConfig) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.max_workers),
        options=[
            ("grpc.max_send_message_length", config.max_message_bytes),
            ("grpc.max_receive_message_length", config.max_message_bytes),
        ],
    )
    pb_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(registry), server)
    # Standard grpc.health.v1 status, set once from the (currently immutable)
    # registry. If dynamic model load/unload is ever added, call
    # health_servicer.set(...) on those mutations to keep this in sync.
    health_servicer = health.HealthServicer()
    serving_status = (
        health_pb2.HealthCheckResponse.SERVING
        if len(registry) > 0
        else health_pb2.HealthCheckResponse.NOT_SERVING
    )
    health_servicer.set("", serving_status)
    health_servicer.set(INFERENCE_SERVICE_NAME, serving_status)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    server.add_insecure_port(config.address)
    return server


def serve(registry: ModelRegistry, config: ServerConfig) -> None:
    """Start the server and block until SIGINT/SIGTERM, then drain gracefully."""
    server = build_server(registry, config)
    server.start()
    logger.info(
        "server started",
        extra={"fields": {"address": config.address, "models": registry.names()}},
    )

    stop_event = threading.Event()

    def _handle(signum: int, _frame: object) -> None:
        logger.info("shutdown signal received", extra={"fields": {"signal": signum}})
        stop_event.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    stop_event.wait()
    # Graceful shutdown: stop accepting new RPCs, allow in-flight to finish.
    server.stop(grace=10).wait()
    logger.info("server stopped")
