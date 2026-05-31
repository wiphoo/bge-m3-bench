"""gRPC servicer for the EmbeddingService.

Returns only raw per-request data and raw resource samples; never aggregates.
"""

from __future__ import annotations

import json
from typing import Any

import grpc

from ..common.logging import get_logger
from ..common.serialize import matrix_to_bytes
from ..generated import embedding_pb2 as pb
from ..generated import embedding_pb2_grpc as pb_grpc
from .embedder import Embedder
from .resources import ResourceSampler

logger = get_logger(__name__)


class EmbeddingServicer(pb_grpc.EmbeddingServiceServicer):
    def __init__(
        self,
        embedder: Embedder,
        spec: dict[str, Any],
        sampler: ResourceSampler,
    ) -> None:
        self._embedder = embedder
        self._spec_json = json.dumps(spec, default=str)
        self._sampler = sampler

    def Embed(self, request: pb.EmbedRequest, context: Any) -> pb.EmbedResponse:
        texts = list(request.texts)
        if not texts:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "texts must not be empty")
            raise RuntimeError("unreachable")  # pragma: no cover - abort raises
        try:
            out = self._embedder.embed(texts)
        except Exception as exc:  # server-side failure -> INTERNAL
            context.abort(grpc.StatusCode.INTERNAL, f"embed error: {exc}")
            raise  # pragma: no cover - abort raises

        embeddings = out.embeddings
        return pb.EmbedResponse(
            embeddings=matrix_to_bytes(embeddings),
            num_inputs=int(embeddings.shape[0]),
            embedding_dim=int(embeddings.shape[1]),
            token_counts=out.token_counts.tolist(),
            tokenize_us=out.tokenize_us,
            inference_us=out.inference_us,
            postprocess_us=out.postprocess_us,
        )

    def GetSpec(self, request: pb.SpecRequest, context: Any) -> pb.SpecResponse:
        return pb.SpecResponse(spec_json=self._spec_json)

    def ResourceSamples(
        self, request: pb.ResourceSamplesRequest, context: Any
    ) -> pb.ResourceSamplesResponse:
        if request.reset:
            self._sampler.reset()
            return pb.ResourceSamplesResponse(samples=[])
        samples = [
            pb.ResourceSample(t_unix=s.t_unix, rss_mb=s.rss_mb, cpu_percent=s.cpu_percent)
            for s in self._sampler.samples()
        ]
        return pb.ResourceSamplesResponse(samples=samples)
