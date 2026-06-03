"""Runtime configuration loaded from environment variables / CLI defaults."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field

# Valid pooling strategies for turning model output into a sentence embedding.
POOLING_MODES = ("none", "cls", "mean")


def _parse_option_pair(item: str) -> tuple[str, str] | None:
    """Parse a single ``KEY=VALUE`` provider option.

    Returns ``None`` for a blank entry. Raises ``ValueError`` on a missing ``=``
    or an empty key so typos surface instead of being silently accepted. The
    value is taken verbatim after the first ``=`` (it may itself contain ``=``).
    """
    item = item.strip()
    if not item:
        return None
    if "=" not in item:
        raise ValueError(f"invalid provider option {item!r}; expected KEY=VALUE")
    key, value = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"invalid provider option {item!r}; empty key")
    return key, value.strip()


def parse_provider_options(raw: str) -> dict[str, str]:
    """Parse a comma-separated ``KEY=VALUE,KEY=VALUE`` string (env var form)."""
    options: dict[str, str] = {}
    for item in raw.split(","):
        pair = _parse_option_pair(item)
        if pair is not None:
            options[pair[0]] = pair[1]
    return options


def parse_provider_option_items(items: Iterable[str]) -> dict[str, str]:
    """Parse repeated ``KEY=VALUE`` CLI flags into a provider-options dict.

    Each item is one pair; unlike :func:`parse_provider_options` it does **not**
    split on commas, so values may contain commas (e.g. a cache path or URL).
    """
    options: dict[str, str] = {}
    for item in items:
        pair = _parse_option_pair(item)
        if pair is not None:
            options[pair[0]] = pair[1]
    return options


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 8
    provider: str = "cpu"
    # Excluded from eq/hash so the frozen dataclass stays hashable despite the
    # mutable dict field (provider_options does not participate in equality).
    provider_options: dict[str, str] = field(default_factory=dict, compare=False)
    intra_op_threads: int = 0  # 0 -> ONNX Runtime default
    inter_op_threads: int = 0
    log_level: str = "INFO"
    max_message_mb: int = 256  # gRPC send/receive message size cap
    model_path: str = ""
    tokenizer_path: str = ""
    pooling: str = "cls"  # none|cls|mean
    normalize: bool = True
    max_length: int = 512  # tokenizer truncation length
    # 0 -> pad each batch to its longest sequence (dynamic). A fixed value pads
    # every batch to that length so the CoreML EP sees one static shape.
    pad_length: int = 0

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
            pad_length=int(os.getenv("BGE_M3_PAD_LENGTH", "0")),
        )
