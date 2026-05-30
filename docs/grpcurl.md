# grpcurl Examples

The server does not enable gRPC reflection, so point `grpcurl` at the checked-in
proto file.

Install `grpcurl` first:

```bash
# macOS
brew install grpcurl

# Go toolchain
go install github.com/fullstorydev/grpcurl/cmd/grpcurl@latest
```

## List Models

```bash
grpcurl \
  -plaintext \
  -import-path proto \
  -proto inference.proto \
  localhost:50051 \
  onnx_grpc_benchmark.v1.InferenceService/ListModels
```

## Inspect Model Metadata

```bash
grpcurl \
  -plaintext \
  -import-path proto \
  -proto inference.proto \
  -d '{"model":"bge-m3-int8"}' \
  localhost:50051 \
  onnx_grpc_benchmark.v1.InferenceService/ModelMetadata
```

Use this before calling `Predict`; input names and shapes vary by exported model.

## Call Predict For An Embedding Model

For an embedding model that accepts `input_ids` and `attention_mask` as `int64`
tensors, `raw_data` must be base64-encoded little-endian tensor bytes.

Build a request JSON file:

```bash
uv run python - <<'PY'
import base64
import json
from pathlib import Path

import numpy as np

input_ids = np.array([[101, 2023, 2003, 1037, 3231, 102]], dtype="<i8")
attention_mask = np.ones_like(input_ids, dtype="<i8")

request = {
    "model": "bge-m3-int8",
    "inputs": [
        {
            "name": "input_ids",
            "dtype": "DATA_TYPE_INT64",
            "shape": list(input_ids.shape),
            "raw_data": base64.b64encode(input_ids.tobytes()).decode("ascii"),
        },
        {
            "name": "attention_mask",
            "dtype": "DATA_TYPE_INT64",
            "shape": list(attention_mask.shape),
            "raw_data": base64.b64encode(attention_mask.tobytes()).decode("ascii"),
        },
    ],
}

Path("results/grpcurl_embedding_request.json").parent.mkdir(parents=True, exist_ok=True)
Path("results/grpcurl_embedding_request.json").write_text(json.dumps(request, indent=2))
PY
```

Call `Predict`:

```bash
grpcurl \
  -plaintext \
  -import-path proto \
  -proto inference.proto \
  -d @ \
  localhost:50051 \
  onnx_grpc_benchmark.v1.InferenceService/Predict \
  < results/grpcurl_embedding_request.json
```

If your model uses additional inputs, such as `token_type_ids`, mirror the names,
dtypes, and shapes from `ModelMetadata`.
