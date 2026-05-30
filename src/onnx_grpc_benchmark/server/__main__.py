"""Server entrypoint: ``python -m onnx_grpc_benchmark.server`` / ``onnx-server``."""

from __future__ import annotations

import argparse

from ..common.config import ServerConfig
from ..common.logging import configure_logging, get_logger
from .grpc_server import serve
from .registry import ModelRegistry

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX Runtime gRPC server")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="PATH[=NAME]",
        help="ONNX model to load; repeatable. Optional =NAME sets the model id.",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--provider", default=None, help="cpu|openvino|cuda|coreml|tensorrt")
    args = parser.parse_args()

    base = ServerConfig.from_env()
    config = ServerConfig(
        host=args.host or base.host,
        port=args.port or base.port,
        max_workers=base.max_workers,
        provider=args.provider or base.provider,
        intra_op_threads=base.intra_op_threads,
        inter_op_threads=base.inter_op_threads,
        log_level=base.log_level,
    )
    configure_logging(config.log_level)

    if not args.model:
        parser.error("at least one --model is required")

    registry = ModelRegistry()
    for i, spec in enumerate(args.model):
        path, _, name = spec.partition("=")
        registry.load(
            path,
            provider=config.provider,
            name=name or None,
            default=(i == 0),
            intra_op_threads=config.intra_op_threads,
            inter_op_threads=config.inter_op_threads,
        )

    serve(registry, config)


if __name__ == "__main__":
    main()
