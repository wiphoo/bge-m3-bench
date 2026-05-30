"""End-to-end gRPC test: real server + client over a loopback port.

Skipped automatically if the generated protobuf stubs are not present.
"""

from __future__ import annotations

import grpc
import numpy as np
import pytest
from grpc_health.v1 import health_pb2, health_pb2_grpc

pytest.importorskip("onnx_grpc_benchmark.generated.inference_pb2")

from onnx_grpc_benchmark.common.config import ServerConfig
from onnx_grpc_benchmark.server.grpc_server import INFERENCE_SERVICE_NAME, build_server
from onnx_grpc_benchmark.server.registry import ModelRegistry


@pytest.fixture()
def running_server(tiny_model_path):
    registry = ModelRegistry()
    registry.load(tiny_model_path, name="tiny", default=True)
    server = build_server(registry, ServerConfig(port=0))
    port = server.add_insecure_port("localhost:0")
    server.start()
    yield f"localhost:{port}"
    server.stop(grace=0).wait()


def test_grpc_predict_roundtrip(running_server, model, dataset):
    from onnx_grpc_benchmark.server.client import InferenceClient

    with InferenceClient(running_server) as client:
        client.wait_ready()
        assert client.health().version
        assert "tiny" in client.list_models().models
        meta = client.model_metadata("tiny")
        assert meta.inputs[0].name == "input"

        sample = dataset.samples[0]
        outputs, inf_us = client.predict(sample, model="tiny")
        assert inf_us >= 0
        # gRPC outputs must match local inference exactly (no corruption).
        local = model.run(sample).outputs
        for name, arr in local.items():
            np.testing.assert_allclose(outputs[name], arr, rtol=1e-5, atol=1e-6)


def test_standard_grpc_health_check(running_server):
    with grpc.insecure_channel(running_server) as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        response = stub.Check(
            health_pb2.HealthCheckRequest(service=INFERENCE_SERVICE_NAME),
            timeout=5,
        )

    assert response.status == health_pb2.HealthCheckResponse.SERVING


def test_standard_grpc_health_check_not_serving():
    """A server with no models loaded reports NOT_SERVING."""
    server = build_server(ModelRegistry(), ServerConfig(port=0))
    port = server.add_insecure_port("localhost:0")
    server.start()
    try:
        with grpc.insecure_channel(f"localhost:{port}") as channel:
            stub = health_pb2_grpc.HealthStub(channel)
            response = stub.Check(
                health_pb2.HealthCheckRequest(service=INFERENCE_SERVICE_NAME),
                timeout=5,
            )
        assert response.status == health_pb2.HealthCheckResponse.NOT_SERVING
    finally:
        server.stop(grace=0).wait()


def test_grpc_benchmark(running_server, model, dataset):
    from onnx_grpc_benchmark.benchmark.runner import BenchmarkConfig, run_grpc
    from onnx_grpc_benchmark.server.client import InferenceClient

    with InferenceClient(running_server) as client:
        client.wait_ready()
        config = BenchmarkConfig(transport="grpc", warmup=2, iterations=8)
        result = run_grpc(client, dataset, config, model_name="tiny")
        assert result.stats["count"] == 8
        assert "server_inference" in result.extra
        assert "transport_overhead_ms" in result.extra
