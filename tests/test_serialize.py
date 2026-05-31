from __future__ import annotations

import numpy as np
import pytest

from bge_m3_bench.common.serialize import bytes_to_matrix, matrix_to_bytes


def test_matrix_roundtrip():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    restored = bytes_to_matrix(matrix_to_bytes(arr), 3, 4)
    np.testing.assert_array_equal(restored, arr)
    assert restored.dtype == np.float32


def test_roundtrip_from_non_contiguous_non_float():
    # A float64, non-contiguous view must still serialize losslessly to f32.
    arr = np.asfortranarray(np.ones((2, 5), dtype=np.float64))
    restored = bytes_to_matrix(matrix_to_bytes(arr), 2, 5)
    np.testing.assert_allclose(restored, np.ones((2, 5)), rtol=0, atol=0)


def test_size_mismatch_raises():
    data = matrix_to_bytes(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="expects"):
        bytes_to_matrix(data, 2, 5)
