"""Thin gRPC client for the EmbeddingService.

Returns raw per-request measurements (server timings, client wall time, wire
sizes). All aggregation happens in the benchmark layer.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import grpc
import numpy as np

from .common.serialize import bytes_to_matrix
from .generated import embedding_pb2 as pb
from .generated import embedding_pb2_grpc as pb_grpc


@dataclass(frozen=True)
class EmbedResult:
    embeddings: np.ndarray  # float32 [num_inputs, dim]
    num_inputs: int
    embedding_dim: int
    token_counts: list[int]
    tokenize_us: int
    inference_us: int
    postprocess_us: int
    client_e2e_us: int  # wall round-trip measured by the client
    request_bytes: int
    response_bytes: int


class EmbeddingClient:
    def __init__(
        self,
        address: str = "localhost:50051",
        timeout: float = 60.0,
        max_message_mb: int = 256,
    ) -> None:
        self.address = address
        self.timeout = timeout
        max_message_bytes = max_message_mb * 1024 * 1024
        self._channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_send_message_length", max_message_bytes),
                ("grpc.max_receive_message_length", max_message_bytes),
            ],
        )
        self._stub = pb_grpc.EmbeddingServiceStub(self._channel)

    def __enter__(self) -> EmbeddingClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._channel.close()

    def wait_ready(self, timeout: float = 10.0) -> None:
        grpc.channel_ready_future(self._channel).result(timeout=timeout)

    def embed(self, texts: list[str]) -> EmbedResult:
        request = pb.EmbedRequest(texts=texts)
        request_bytes = request.ByteSize()
        t0 = time.perf_counter_ns()
        response = self._stub.Embed(request, timeout=self.timeout)
        client_e2e_us = (time.perf_counter_ns() - t0) // 1000
        embeddings = bytes_to_matrix(
            response.embeddings, response.num_inputs, response.embedding_dim
        )
        return EmbedResult(
            embeddings=embeddings,
            num_inputs=response.num_inputs,
            embedding_dim=response.embedding_dim,
            token_counts=list(response.token_counts),
            tokenize_us=response.tokenize_us,
            inference_us=response.inference_us,
            postprocess_us=response.postprocess_us,
            client_e2e_us=int(client_e2e_us),
            request_bytes=request_bytes,
            response_bytes=response.ByteSize(),
        )

    def get_spec(self) -> dict[str, Any]:
        response = self._stub.GetSpec(pb.SpecRequest(), timeout=self.timeout)
        spec: dict[str, Any] = json.loads(response.spec_json)
        return spec

    def resource_samples(self, reset: bool = False) -> list[dict[str, float]]:
        response = self._stub.ResourceSamples(
            pb.ResourceSamplesRequest(reset=reset), timeout=self.timeout
        )
        return [
            {"t_unix": s.t_unix, "rss_mb": s.rss_mb, "cpu_percent": s.cpu_percent}
            for s in response.samples
        ]
