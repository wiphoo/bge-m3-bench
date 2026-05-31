"""gRPC servicer implementing InferenceService over a ModelRegistry."""

from __future__ import annotations

import json
from typing import Any

import grpc

from ..common.logging import get_logger
from ..generated import inference_pb2 as pb
from ..generated import inference_pb2_grpc as pb_grpc
from ..metadata import collect_metadata
from .registry import ModelRegistry
from .serialization import ndarray_to_tensor, tensor_to_ndarray

logger = get_logger(__name__)


class InferenceServicer(pb_grpc.InferenceServiceServicer):
    def __init__(self, registry: ModelRegistry, version: str = "0.1.0") -> None:
        self._registry = registry
        self._version = version
        # Snapshot the server's environment once so ModelMetadata can advertise
        # the machine that actually runs inference. Benchmark clients embed this
        # in their results so remote/VPS/k8s runs are not mislabelled with the
        # caller's hardware.
        self._environment_json = json.dumps(collect_metadata(extra={"role": "server"}), default=str)

    def Predict(self, request: pb.PredictRequest, context: Any) -> pb.PredictResponse:
        try:
            model = self._registry.get(request.model or None)
        except KeyError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
            raise  # pragma: no cover

        inputs = {t.name: tensor_to_ndarray(t) for t in request.inputs}
        output_names = list(request.output_names) or None
        try:
            result = model.run(inputs, output_names=output_names)
        except Exception as exc:
            # Server-side failure during inference -> INTERNAL, not a caller error.
            context.abort(grpc.StatusCode.INTERNAL, f"inference error: {exc}")
            raise  # pragma: no cover

        return pb.PredictResponse(
            outputs=[ndarray_to_tensor(n, a, pb.Tensor) for n, a in result.outputs.items()],
            inference_us=result.inference_us,
            model=model.name,
        )

    def Health(self, request: pb.HealthRequest, context: Any) -> pb.HealthResponse:
        serving = len(self._registry) > 0
        status = pb.HealthResponse.SERVING if serving else pb.HealthResponse.NOT_SERVING
        return pb.HealthResponse(status=status, version=self._version)

    def ModelMetadata(
        self, request: pb.ModelMetadataRequest, context: Any
    ) -> pb.ModelMetadataResponse:
        try:
            model = self._registry.get(request.model or None)
        except KeyError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
            raise  # pragma: no cover

        def info(spec: Any) -> pb.TensorInfo:
            from ..common.dtypes import numpy_to_proto

            try:
                dtype = numpy_to_proto(spec.dtype)
            except ValueError:
                dtype = pb.DATA_TYPE_UNSPECIFIED
            return pb.TensorInfo(name=spec.name, dtype=dtype, shape=list(spec.shape))

        return pb.ModelMetadataResponse(
            model=model.name,
            inputs=[info(s) for s in model.input_specs()],
            outputs=[info(s) for s in model.output_specs()],
            provider=model.active_provider,
            metadata={
                "requested_provider": model.resolved.requested,
                "environment": self._environment_json,
            },
        )

    def ListModels(self, request: pb.ListModelsRequest, context: Any) -> pb.ListModelsResponse:
        return pb.ListModelsResponse(
            models=self._registry.names(),
            default_model=self._registry.default_model or "",
        )
