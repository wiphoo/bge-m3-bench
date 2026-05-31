"""End-to-end gRPC test: real server + client over a loopback port."""

from __future__ import annotations

import time

import grpc
import numpy as np
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc

from bge_m3_bench.client import EmbeddingClient
from bge_m3_bench.server.grpc_server import SERVICE_NAME


def test_embed_matches_local(running_server, embedder, texts):
    with EmbeddingClient(running_server) as client:
        client.wait_ready()
        result = client.embed(texts)

    assert result.embeddings.shape == (len(texts), 8)
    assert result.num_inputs == len(texts)
    assert result.embedding_dim == 8
    assert result.tokenize_us >= 0 and result.inference_us >= 0 and result.postprocess_us >= 0
    assert result.client_e2e_us >= 0
    assert result.request_bytes > 0 and result.response_bytes > 0
    # token_counts match the server-side tokenizer.
    local = embedder.embed(texts)
    assert result.token_counts == local.token_counts.tolist()
    # Raw-bytes transport is lossless; gRPC embeddings equal local embeddings.
    np.testing.assert_allclose(result.embeddings, local.embeddings, rtol=1e-6, atol=1e-6)


def test_get_spec(running_server):
    with EmbeddingClient(running_server) as client:
        client.wait_ready()
        spec = client.get_spec()
    assert spec["model"]["embedding_dim"] == 8
    assert {i["name"] for i in spec["model"]["inputs"]} == {"input_ids", "attention_mask"}
    assert spec["runtime"]["execution_provider"] == "CPUExecutionProvider"
    assert spec["machine"]["cpu_logical_cores"]


def test_resource_samples(running_server):
    with EmbeddingClient(running_server) as client:
        client.wait_ready()
        client.resource_samples(reset=True)
        time.sleep(0.1)
        samples = client.resource_samples()
    assert samples, "expected at least one raw resource sample"
    assert all(s["rss_mb"] > 0 for s in samples)


def test_empty_texts_rejected(running_server):
    with EmbeddingClient(running_server) as client:
        client.wait_ready()
        with pytest.raises(grpc.RpcError) as exc:
            client.embed([])
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_health_serving(running_server):
    with grpc.insecure_channel(running_server) as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        response = stub.Check(health_pb2.HealthCheckRequest(service=SERVICE_NAME), timeout=5)
    assert response.status == health_pb2.HealthCheckResponse.SERVING


def test_reflection_lists_service(running_server):
    with grpc.insecure_channel(running_server) as channel:
        stub = reflection_pb2_grpc.ServerReflectionStub(channel)
        req = reflection_pb2.ServerReflectionRequest(list_services="")
        names: list[str] = []
        for response in stub.ServerReflectionInfo(iter([req])):
            names = [s.name for s in response.list_services_response.service]
            break
    assert SERVICE_NAME in names
