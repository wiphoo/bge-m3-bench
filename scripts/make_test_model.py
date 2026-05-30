#!/usr/bin/env python3
"""Build a tiny, deterministic ONNX model for tests and demos.

The model computes ``y = relu(x @ W + b)`` for a fixed W/b, giving a small
FP32 CPU model with a single dynamic-batch float input and one output. It needs
no training data and is fully reproducible (fixed seed), so it works as the
"single model" baseline for Milestone 1 and as a fixture in the test suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

IN_FEATURES = 16
OUT_FEATURES = 8


def build_model(seed: int = 0) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((IN_FEATURES, OUT_FEATURES)).astype(np.float32)
    bias = rng.standard_normal((OUT_FEATURES,)).astype(np.float32)

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", IN_FEATURES])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", OUT_FEATURES])

    w_init = numpy_helper.from_array(weight, name="W")
    b_init = numpy_helper.from_array(bias, name="B")

    matmul = helper.make_node("MatMul", ["input", "W"], ["mm"])
    add = helper.make_node("Add", ["mm", "B"], ["pre"])
    relu = helper.make_node("Relu", ["pre"], ["output"])

    graph = helper.make_graph(
        [matmul, add, relu],
        "tiny_mlp",
        [x],
        [y],
        initializer=[w_init, b_init],
    )
    model = helper.make_model(
        graph,
        producer_name="bge-m3-bench",
        opset_imports=[helper.make_operatorsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/tiny_mlp.onnx")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(args.seed), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
