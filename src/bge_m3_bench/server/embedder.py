"""Embedder: tokenize -> ONNX inference -> pool -> normalize.

Times each phase (tokenize / inference / postprocess) and returns raw timings so
the gRPC service can pass them to the benchmark client unaggregated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..common.config import POOLING_MODES
from .runtime import OnnxModel
from .tokenizer import BgeTokenizer


@dataclass(frozen=True)
class EmbedOutput:
    embeddings: np.ndarray  # float32 [num_inputs, dim]
    token_counts: np.ndarray  # int64 [num_inputs]
    tokenize_us: int
    inference_us: int
    postprocess_us: int


class Embedder:
    def __init__(
        self,
        model: OnnxModel,
        tokenizer: BgeTokenizer,
        pooling: str = "cls",
        normalize: bool = True,
    ) -> None:
        if pooling not in POOLING_MODES:
            raise ValueError(f"unknown pooling {pooling!r}; choose from {POOLING_MODES}")
        self.model = model
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.normalize = normalize
        self._input_names = set(model.input_names())

    def embed(self, texts: list[str]) -> EmbedOutput:
        # Tokenize timing covers building the model feed too (incl. the
        # token_type_ids zeros), so no per-request prep falls outside a phase.
        t0 = time.perf_counter_ns()
        enc = self.tokenizer.encode_batch(texts)
        feed: dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            feed["input_ids"] = enc.input_ids
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = enc.attention_mask
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(enc.input_ids)
        tokenize_us = (time.perf_counter_ns() - t0) // 1000

        result = self.model.run(feed)
        hidden = next(iter(result.outputs.values()))

        t1 = time.perf_counter_ns()
        embeddings = _pool(hidden, enc.attention_mask, self.pooling)
        if self.normalize:
            embeddings = _l2_normalize(embeddings)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        postprocess_us = (time.perf_counter_ns() - t1) // 1000

        return EmbedOutput(
            embeddings=embeddings,
            token_counts=enc.token_counts,
            tokenize_us=int(tokenize_us),
            inference_us=result.inference_us,
            postprocess_us=int(postprocess_us),
        )


def _pool(hidden: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        if hidden.ndim != 2:
            raise ValueError("pooling=none expects a 2-D model output [batch, dim]")
        return hidden
    if hidden.ndim != 3:
        raise ValueError(f"pooling={mode} expects a 3-D output [batch, seq, dim]")
    if mode == "cls":
        return hidden[:, 0, :]
    # mean: mask-weighted average over the sequence dimension.
    weights = mask.astype(np.float32)[:, :, None]
    summed = (hidden * weights).sum(axis=1)
    counts = np.clip(weights.sum(axis=1), 1e-9, None)
    return summed / counts


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)
