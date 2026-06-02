"""Runtime configuration loaded from environment variables / CLI defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Valid pooling strategies for turning model output into a sentence embedding.
POOLING_MODES = ("none", "cls", "mean")


def parse_provider_options(raw: str) -> dict[str, str]:
    """Parse a ``KEY=VALUE,KEY=VALUE`` string into a provider-options dict.

    Blank entries are skipped; an entry without ``=`` raises ``ValueError`` so a
    typo surfaces instead of being silently ignored.
    """
    options: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid provider option {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        options[key.strip()] = value.strip()
    return options


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 8
    provider: str = "cpu"
    provider_options: dict[str, str] = field(default_factory=dict)
    intra_op_threads: int = 0  # 0 -> ONNX Runtime default
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
            provider_options=parse_provider_options(os.getenv("BGE_M3_PROVIDER_OPTIONS", "")),
            intra_op_threads=int(os.getenv("BGE_M3_INTRA_OP", "0")),
            inter_op_threads=int(os.getenv("BGE_M3_INTER_OP", "0")),
            log_level=os.getenv("BGE_M3_LOG_LEVEL", cls.log_level),
            max_message_mb=int(os.getenv("BGE_M3_MAX_MESSAGE_MB", str(cls.max_message_mb))),
            model_path=os.getenv("BGE_M3_MODEL", cls.model_path),
            tokenizer_path=os.getenv("BGE_M3_TOKENIZER", cls.tokenizer_path),
            pooling=os.getenv("BGE_M3_POOLING", cls.pooling),
            normalize=os.getenv("BGE_M3_NORMALIZE", "1") not in ("0", "false", "False"),
            max_length=int(os.getenv("BGE_M3_MAX_LENGTH", str(cls.max_length))),
        )
