"""Zero-copy-ish tensor (de)serialization for gRPC transport (Milestone 2.1).

Tensors are carried as raw little-endian, C-contiguous bytes plus an explicit
dtype and shape, avoiding protobuf's per-element repeated-field overhead. This
keeps the wire format compact and makes (de)serialization a single memcpy.

The functions accept/return the protobuf ``Tensor`` message but are written to
also work in tests via duck typing against a simple stand-in object, so they do
not import the generated stubs at module import time.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import numpy as np

from ..common.dtypes import numpy_to_proto, proto_to_numpy


def ndarray_to_tensor(name: str, array: np.ndarray, tensor_cls: Callable[..., Any]) -> Any:
    """Serialize a numpy array into a proto ``Tensor`` message."""
    # Preserve the original shape: ascontiguousarray promotes 0-d scalars to
    # shape (1,), so capture the true shape before any normalisation.
    shape = list(array.shape)
    arr = np.ascontiguousarray(array)
    # Always emit little-endian so consumers on any architecture agree.
    if arr.dtype.byteorder == ">" or (arr.dtype.byteorder == "=" and sys.byteorder == "big"):
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    return tensor_cls(
        name=name,
        dtype=numpy_to_proto(arr.dtype),
        shape=shape,
        raw_data=arr.tobytes(),
    )


def tensor_to_ndarray(tensor: Any) -> np.ndarray:
    """Deserialize a proto ``Tensor`` message into a numpy array."""
    np_dtype = proto_to_numpy(tensor.dtype).newbyteorder("<")
    shape = tuple(tensor.shape)
    array = np.frombuffer(tensor.raw_data, dtype=np_dtype)
    expected = int(np.prod(shape)) if shape else 1
    if array.size != expected:
        raise ValueError(
            f"tensor {tensor.name!r}: buffer has {array.size} elements, "
            f"shape {shape} expects {expected}"
        )
    # Return native byte order for downstream compute; copy to own the buffer.
    return array.reshape(shape).astype(np_dtype.newbyteorder("="), copy=True)
