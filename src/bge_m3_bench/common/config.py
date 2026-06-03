"""Runtime configuration loaded from environment variables / CLI defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Valid pooling strategies for turning model output into a sentence embedding.
POOLING_MODES = ("none", "cls", "mean")


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 8
    provider: str = "cpu"
    # -1 -> auto: bound oversubscription as max(1, usable_cores // max_workers),
    # where usable_cores is the host physical count capped by any cgroup quota /
    # affinity (resolved in build_from_config). 0 -> ONNX Runtime default (all
    # cores). >0 -> explicit thread count.
    intra_op_threads: int = -1
    inter_op_threads: int = 0
    log_level: str = "INFO"
    max_message_mb: int = 256  # gRPC send/receive message size cap
    model_path: str = ""
    tokenizer_path: str = ""
    pooling: str = "cls"  # none|cls|mean
    normalize: bool = True
    max_length: int = 512  # tokenizer truncation length

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def max_message_bytes(self) -> int:
        return self.max_message_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> ServerConfig:
        return cls(
            host=os.getenv("BGE_M3_HOST", cls.host),
            port=int(os.getenv("BGE_M3_PORT", str(cls.port))),
            max_workers=int(os.getenv("BGE_M3_MAX_WORKERS", str(cls.max_workers))),
            provider=os.getenv("BGE_M3_PROVIDER", cls.provider),
            intra_op_threads=int(os.getenv("BGE_M3_INTRA_OP", str(cls.intra_op_threads))),
            inter_op_threads=int(os.getenv("BGE_M3_INTER_OP", "0")),
            log_level=os.getenv("BGE_M3_LOG_LEVEL", cls.log_level),
            max_message_mb=int(os.getenv("BGE_M3_MAX_MESSAGE_MB", str(cls.max_message_mb))),
            model_path=os.getenv("BGE_M3_MODEL", cls.model_path),
            tokenizer_path=os.getenv("BGE_M3_TOKENIZER", cls.tokenizer_path),
            pooling=os.getenv("BGE_M3_POOLING", cls.pooling),
            normalize=os.getenv("BGE_M3_NORMALIZE", "1") not in ("0", "false", "False"),
            max_length=int(os.getenv("BGE_M3_MAX_LENGTH", str(cls.max_length))),
        )
