from __future__ import annotations

import numpy as np
import pytest

from onnx_grpc_benchmark.common import dtypes
from onnx_grpc_benchmark.server.serialization import ndarray_to_tensor, tensor_to_ndarray


class FakeTensor:
    """Stand-in for the proto Tensor message (duck-typed)."""

    def __init__(self, name="", dtype=0, shape=(), raw_data=b""):
        self.name = name
        self.dtype = dtype
        self.shape = list(shape)
        self.raw_data = raw_data


@pytest.mark.parametrize(
    "array",
    [
        np.random.default_rng(0).standard_normal((2, 3)).astype(np.float32),
        np.arange(12, dtype=np.int64).reshape(3, 4),
        np.array([True, False, True], dtype=np.bool_),
        np.random.default_rng(1).standard_normal((4,)).astype(np.float16),
        np.array(5.0, dtype=np.float64),  # scalar
    ],
)
def test_roundtrip_no_corruption(array):
    tensor = ndarray_to_tensor("t", array, FakeTensor)
    restored = tensor_to_ndarray(tensor)
    assert restored.shape == array.shape
    assert restored.dtype == array.dtype
    np.testing.assert_array_equal(restored, array)


def test_dtype_mapping_roundtrip():
    for code in (
        dtypes.DATA_TYPE_FLOAT32,
        dtypes.DATA_TYPE_INT64,
        dtypes.DATA_TYPE_BOOL,
        dtypes.DATA_TYPE_FLOAT16,
    ):
        np_dtype = dtypes.proto_to_numpy(code)
        assert dtypes.numpy_to_proto(np_dtype) == code


def test_bad_buffer_length_raises():
    t = FakeTensor(name="x", dtype=dtypes.DATA_TYPE_FLOAT32, shape=(4,), raw_data=b"\x00")
    with pytest.raises(ValueError):
        tensor_to_ndarray(t)


def test_big_endian_input_serialized_as_little_endian():
    # A big-endian array must serialize to the same logical values; the wire
    # format is always little-endian and decode returns native byte order.
    array = np.arange(6, dtype=">i4").reshape(2, 3)
    tensor = ndarray_to_tensor("be", array, FakeTensor)
    assert tensor.raw_data == array.astype("<i4").tobytes()
    restored = tensor_to_ndarray(tensor)
    np.testing.assert_array_equal(restored, array)
    assert restored.dtype.byteorder in ("=", "<", "|")


def test_empty_tensor_roundtrip():
    array = np.empty((0,), dtype=np.float32)
    tensor = ndarray_to_tensor("e", array, FakeTensor)
    restored = tensor_to_ndarray(tensor)
    assert restored.shape == (0,)
    assert restored.size == 0
