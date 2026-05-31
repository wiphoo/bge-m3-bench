"""Compact float32 matrix (de)serialization for gRPC transport.

Embeddings travel as raw little-endian, C-contiguous float32 bytes plus an
explicit row count and dimension, avoiding protobuf's per-element repeated-field
overhead. (De)serialization is a single memcpy. Shared by the server and client
so the wire format is defined in exactly one place.
"""

from __future__ import annotations

import numpy as np

_F32_LE = np.dtype("<f4")


def matrix_to_bytes(matrix: np.ndarray) -> bytes:
    """Serialize a 2-D float32 array to little-endian raw bytes."""
    arr = np.ascontiguousarray(matrix, dtype=_F32_LE)
    return arr.tobytes()


def bytes_to_matrix(data: bytes, num_rows: int, dim: int) -> np.ndarray:
    """Deserialize raw little-endian float32 bytes into a ``[num_rows, dim]`` array."""
    arr = np.frombuffer(data, dtype=_F32_LE)
    expected = num_rows * dim
    if arr.size != expected:
        raise ValueError(
            f"embedding buffer has {arr.size} floats, [{num_rows}, {dim}] expects {expected}"
        )
    # Return native byte order; copy so the result owns its buffer.
    return arr.reshape(num_rows, dim).astype(np.float32, copy=True)
