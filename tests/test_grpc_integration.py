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
        result = run_grpc(client, dataset, config, model_name="tiny", reference=model)
        assert result.stats["count"] == 8
        assert "server_inference" in result.extra
        assert "transport_overhead_ms" in result.extra
        # gRPC outputs are validated against the reference model before timing.
        assert result.validation["passed"] is True
        names = {c["check"] for c in result.validation["checks"]}
        assert {"outputs_finite", "outputs_present", "matches_reference"} <= names
        # The server's own environment (the inference host) is recorded
        # separately from ``result.metadata`` (the client's).
        server_meta = result.extra["server_metadata"]
        assert server_meta is not None
        assert server_meta["extra"]["role"] == "server"
        assert "runtime" in server_meta
        # Provider provenance reflects the server (inference host), not the client.
        assert result.extra["active_provider"] == "CPUExecutionProvider"


def test_grpc_validation_detects_wrong_reference(running_server, dataset):
    """A reference whose outputs disagree with the server fails validation."""
    import numpy as np

    from onnx_grpc_benchmark.benchmark.runner import BenchmarkConfig, run_grpc
    from onnx_grpc_benchmark.benchmark.validation import ValidationError
    from onnx_grpc_benchmark.server.client import InferenceClient

    class WrongModel:
        def run(self, sample):
            from onnx_grpc_benchmark.server.runtime import InferenceResult

            return InferenceResult(outputs={"output": np.zeros((1, 1), np.float32)}, inference_us=0)

    with InferenceClient(running_server) as client:
        client.wait_ready()
        config = BenchmarkConfig(transport="grpc", warmup=1, iterations=4)
        with pytest.raises(ValidationError):
            run_grpc(client, dataset, config, model_name="tiny", reference=WrongModel())
