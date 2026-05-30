"""Runtime configuration loaded from environment variables / CLI defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 8
    provider: str = "cpu"
    intra_op_threads: int = 0  # 0 -> ONNX Runtime default
    inter_op_threads: int = 0
    log_level: str = "INFO"

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @classmethod
    def from_env(cls) -> ServerConfig:
        return cls(
            host=os.getenv("ONNX_GRPC_HOST", cls.host),
            port=int(os.getenv("ONNX_GRPC_PORT", str(cls.port))),
            max_workers=int(os.getenv("ONNX_GRPC_MAX_WORKERS", str(cls.max_workers))),
            provider=os.getenv("ONNX_GRPC_PROVIDER", cls.provider),
            intra_op_threads=int(os.getenv("ONNX_GRPC_INTRA_OP", "0")),
            inter_op_threads=int(os.getenv("ONNX_GRPC_INTER_OP", "0")),
            log_level=os.getenv("ONNX_GRPC_LOG_LEVEL", cls.log_level),
        )
