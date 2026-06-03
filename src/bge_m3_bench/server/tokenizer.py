"""Tokenization for BGE-M3 via HF ``tokenizers`` (local ``tokenizer.json``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Candidate pad tokens, tried in order (XLM-R uses ``<pad>``). Intentionally no
# UNK fallback: padding with an UNK id is only safe while attention_mask hides
# those positions, so we fall back to id 0 instead.
_PAD_CANDIDATES = ("<pad>", "[PAD]")


@dataclass(frozen=True)
class TokenizeResult:
    input_ids: np.ndarray  # int64 [batch, seq]
    attention_mask: np.ndarray  # int64 [batch, seq]
    token_counts: np.ndarray  # int64 [batch] — real (non-pad) tokens per input


class BgeTokenizer:
    """Batch tokenizer producing padded int64 ``input_ids`` / ``attention_mask``."""

    def __init__(
        self, tokenizer: Any, max_length: int = 512, pad_length: int | None = None
    ) -> None:
        self._tok = tokenizer
        self.max_length = max_length
        self.pad_length = pad_length
        pad_id, pad_token = self._resolve_pad()
        # Default: pad to the longest sequence in each batch (dynamic shape). When
        # pad_length is set, pad every batch to that fixed length instead — this
        # gives the CoreML EP a single static input shape so it compiles once
        # rather than caching a new model per shape (bounds memory).
        if pad_length is not None:
            # Truncate to pad_length too (padding only adds tokens, never removes),
            # otherwise an input longer than pad_length would exceed it and the
            # shape would vary again — defeating the static-shape guarantee.
            self._tok.enable_truncation(min(max_length, pad_length))
            self._tok.enable_padding(pad_id=pad_id, pad_token=pad_token, length=pad_length)
        else:
            self._tok.enable_truncation(max_length)
            self._tok.enable_padding(pad_id=pad_id, pad_token=pad_token)

    @classmethod
    def from_file(
        cls, path: str | Path, max_length: int = 512, pad_length: int | None = None
    ) -> BgeTokenizer:
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)), max_length=max_length, pad_length=pad_length)

    def _resolve_pad(self) -> tuple[int, str]:
        for token in _PAD_CANDIDATES:
            tid = self._tok.token_to_id(token)
            if tid is not None:
                return tid, token
        return 0, "[PAD]"

    def encode_batch(self, texts: list[str]) -> TokenizeResult:
        encodings = self._tok.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_counts = attention_mask.sum(axis=1).astype(np.int64)
        return TokenizeResult(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_counts=token_counts,
        )
