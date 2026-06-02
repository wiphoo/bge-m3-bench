#!/usr/bin/env python3
"""Build an embedding model + tokenizer at a chosen precision.

Unified entry point for ``make model``. Selects between:

- ``tiny``   — the tiny, deterministic synthetic model from
  ``make_test_embedding_model`` (no download; for tests/demos).
- ``bge-m3`` — the real ``BAAI/bge-m3`` model, exported to ONNX via Optimum
  (requires the optional ``export`` dependency-group: transformers/torch/optimum).

and between precisions ``fp32`` / ``fp16`` / ``int8``. fp16 and int8 are derived
locally from the fp32 ONNX with ONNX Runtime — no external pre-quantized downloads:

- ``int8`` → ``onnxruntime.quantization.quantize_dynamic`` (QInt8 weights).
- ``fp16`` → ``onnxruntime.transformers.float16.convert_float_to_float16``.
  (``quantize_dynamic`` only produces int8; fp16 is a separate graph conversion.)

Artifacts are written to ``<out-dir>/<model>/<precision>/{model.onnx,tokenizer.json}``
so variants coexist. Point the server at those paths via ``--model`` / ``--tokenizer``.

Note: for the ``tiny`` model ``int8`` is effectively a no-op — it is a single
``Gather`` with no MatMul/Conv for ``quantize_dynamic`` to quantize. The tiny model
exists only to exercise the pipeline.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

MODELS = ("tiny", "bge-m3")
PRECISIONS = ("fp32", "fp16", "int8")


def build_tiny_fp32(out_dir: Path, seed: int) -> None:
    """Write the synthetic tiny ``model.onnx`` + ``tokenizer.json`` (fp32)."""
    import onnx
    from make_test_embedding_model import build_model, build_tokenizer

    onnx.save(build_model(seed), str(out_dir / "model.onnx"))
    build_tokenizer().save(str(out_dir / "tokenizer.json"))


def export_bge_m3_fp32(out_dir: Path, model_id: str) -> None:
    """Export the real BGE-M3 model + tokenizer to ONNX (fp32) into ``out_dir``."""
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)

    # Optimum already writes ``model.onnx``; guard the name in case that changes.
    produced = out_dir / "model.onnx"
    if not produced.exists():
        onnxs = list(out_dir.glob("*.onnx"))
        if len(onnxs) != 1:
            raise RuntimeError(f"expected one exported .onnx in {out_dir}, found {onnxs}")
        onnxs[0].rename(produced)


def convert_to_fp16(onnx_path: Path) -> None:
    """Convert an fp32 ONNX model to float16 in place."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(str(onnx_path))
    onnx.save(convert_float_to_float16(model, keep_io_types=False), str(onnx_path))


def quantize_int8(onnx_path: Path) -> None:
    """Dynamically quantize an fp32 ONNX model to int8 weights in place."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    with tempfile.TemporaryDirectory() as tmp:
        tmp_out = Path(tmp) / "quant.onnx"
        quantize_dynamic(str(onnx_path), str(tmp_out), weight_type=QuantType.QInt8)
        shutil.copyfile(tmp_out, onnx_path)


def apply_precision(onnx_path: Path, precision: str) -> None:
    """Convert ``model.onnx`` in place to the requested precision (fp32 = no-op)."""
    if precision == "fp16":
        convert_to_fp16(onnx_path)
    elif precision == "int8":
        quantize_int8(onnx_path)


def embedding_dim(onnx_path: Path) -> int | str:
    """Best-effort embedding dim = last dim of the first output (for the log line)."""
    import onnx

    dims = onnx.load(str(onnx_path), load_external_data=False).graph.output[0]
    shape = dims.type.tensor_type.shape.dim
    if not shape:
        return "?"
    last = shape[-1]
    if last.HasField("dim_value") and last.dim_value > 0:
        return last.dim_value
    return last.dim_param or "?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, default="tiny")
    parser.add_argument("--precision", choices=PRECISIONS, default="fp32")
    parser.add_argument(
        "--out-dir",
        default="models",
        help="Root output dir; artifacts go in <out-dir>/<model>/<precision>/.",
    )
    parser.add_argument("--model-id", default="BAAI/bge-m3", help="HuggingFace id for bge-m3.")
    parser.add_argument("--seed", type=int, default=0, help="Tiny model RNG seed.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.model / args.precision
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "tiny":
        build_tiny_fp32(out_dir, args.seed)
    else:
        export_bge_m3_fp32(out_dir, args.model_id)

    onnx_path = out_dir / "model.onnx"
    apply_precision(onnx_path, args.precision)

    print(
        f"wrote {onnx_path} and {out_dir / 'tokenizer.json'} "
        f"(model={args.model} precision={args.precision} embedding_dim={embedding_dim(onnx_path)})"
    )


if __name__ == "__main__":
    main()
