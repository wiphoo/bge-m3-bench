"""Thin gRPC client for the InferenceService."""

from __future__ import annotations

from types import TracebackType

import grpc
import numpy as np

from ..generated import inference_pb2 as pb
from ..generated import inference_pb2_grpc as pb_grpc
from .serialization import ndarray_to_tensor, tensor_to_ndarray


class InferenceClient:
    def __init__(
        self,
        address: str = "localhost:50051",
        timeout: float = 30.0,
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
        self._stub = pb_grpc.InferenceServiceStub(self._channel)

    def __enter__(self) -> InferenceClient:
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

    def health(self) -> pb.HealthResponse:
        return self._stub.Health(pb.HealthRequest(), timeout=self.timeout)

    def list_models(self) -> pb.ListModelsResponse:
        return self._stub.ListModels(pb.ListModelsRequest(), timeout=self.timeout)

    def model_metadata(self, model: str = "") -> pb.ModelMetadataResponse:
        return self._stub.ModelMetadata(pb.ModelMetadataRequest(model=model), timeout=self.timeout)

    def predict(
        self,
        inputs: dict[str, np.ndarray],
        model: str = "",
        output_names: list[str] | None = None,
    ) -> tuple[dict[str, np.ndarray], int]:
        request = pb.PredictRequest(
            model=model,
            inputs=[ndarray_to_tensor(n, a, pb.Tensor) for n, a in inputs.items()],
            output_names=output_names or [],
        )
        response = self._stub.Predict(request, timeout=self.timeout)
        outputs = {t.name: tensor_to_ndarray(t) for t in response.outputs}
        return outputs, response.inference_us
