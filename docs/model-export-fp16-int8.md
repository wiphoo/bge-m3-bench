# Model Export: FP16 And INT8

Use this guide to export a Hugging Face embedding model to ONNX, then create
FP16 and INT8 variants for benchmarking.

## Export ONNX

```bash
uv run --with "optimum[exporters,onnxruntime]" optimum-cli export onnx \
  --model BAAI/bge-m3 \
  --task feature-extraction \
  models/bge-m3-onnx
```

## Export FP16

```bash
uv run --with onnxconverter-common python - <<'PY'
from pathlib import Path

import onnx
from onnxconverter_common import float16

src = Path("models/bge-m3-onnx/model.onnx")
dst = Path("models/bge-m3-onnx/model.fp16.onnx")

model = onnx.load(src)
model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
onnx.save(model_fp16, dst)
print(f"wrote {dst}")
PY
```

## Export INT8

Dynamic INT8 quantization is the easiest CPU baseline:

```bash
uv run python - <<'PY'
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

src = Path("models/bge-m3-onnx/model.onnx")
dst = Path("models/bge-m3-onnx/model.int8.onnx")

quantize_dynamic(
    model_input=src,
    model_output=dst,
    weight_type=QuantType.QInt8,
)
print(f"wrote {dst}")
PY
```

Use static INT8 quantization only when you have a representative calibration
dataset.

## Benchmark Exported Models

```bash
uv run --extra cpu onnx-bench local \
  --model models/bge-m3-onnx/model.fp16.onnx \
  --iterations 100 \
  --out results/bge_m3_fp16.json

uv run --extra cpu onnx-bench local \
  --model models/bge-m3-onnx/model.int8.onnx \
  --iterations 100 \
  --out results/bge_m3_int8.json
```

For gRPC:

```bash
uv run --extra cpu onnx-server \
  --model models/bge-m3-onnx/model.int8.onnx=bge-m3-int8 \
  --provider cpu

uv run --extra cpu onnx-bench grpc \
  --address localhost:50051 \
  --model bge-m3-int8 \
  --ref-model models/bge-m3-onnx/model.onnx \
  --iterations 100 \
  --out results/bge_m3_int8_grpc.json
```
