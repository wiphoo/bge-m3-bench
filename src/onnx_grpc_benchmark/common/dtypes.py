"""Mapping helpers between numpy dtypes and the proto ``DataType`` enum.

These helpers are import-safe even before the protobuf stubs are generated:
the proto enum values are mirrored here so unit tests can run on the integer
codes directly. ``proto_to_numpy``/``numpy_to_proto`` accept the plain ints.
"""

from __future__ import annotations

import numpy as np

# Mirror of proto ``DataType`` enum values (see proto/inference.proto).
DATA_TYPE_UNSPECIFIED = 0
DATA_TYPE_FLOAT32 = 1
DATA_TYPE_FLOAT64 = 2
DATA_TYPE_INT32 = 3
DATA_TYPE_INT64 = 4
DATA_TYPE_UINT8 = 5
DATA_TYPE_BOOL = 6
DATA_TYPE_FLOAT16 = 7

_PROTO_TO_NP: dict[int, np.dtype] = {
    DATA_TYPE_FLOAT32: np.dtype(np.float32),
    DATA_TYPE_FLOAT64: np.dtype(np.float64),
    DATA_TYPE_INT32: np.dtype(np.int32),
    DATA_TYPE_INT64: np.dtype(np.int64),
    DATA_TYPE_UINT8: np.dtype(np.uint8),
    DATA_TYPE_BOOL: np.dtype(np.bool_),
    DATA_TYPE_FLOAT16: np.dtype(np.float16),
}

_NP_TO_PROTO: dict[np.dtype, int] = {v: k for k, v in _PROTO_TO_NP.items()}


def proto_to_numpy(dtype: int) -> np.dtype:
    try:
        return _PROTO_TO_NP[int(dtype)]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unsupported proto DataType: {dtype}") from exc


def numpy_to_proto(dtype: np.dtype | type) -> int:
    np_dtype = np.dtype(dtype)
    try:
        return _NP_TO_PROTO[np_dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported numpy dtype: {np_dtype}") from exc
