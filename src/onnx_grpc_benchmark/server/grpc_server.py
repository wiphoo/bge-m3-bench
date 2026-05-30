"""Construct and run the gRPC server with graceful shutdown."""

from __future__ import annotations

import signal
import threading
from concurrent import futures

import grpc

from ..common.config import ServerConfig
from ..common.logging import get_logger
from ..generated import inference_pb2_grpc as pb_grpc
from .registry import ModelRegistry
from .service import InferenceServicer

logger = get_logger(__name__)


def build_server(registry: ModelRegistry, config: ServerConfig) -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.max_workers),
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )
    pb_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(registry), server)
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
