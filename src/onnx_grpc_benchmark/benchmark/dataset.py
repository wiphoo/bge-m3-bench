"""Reproducible benchmark datasets (Milestone 1.1).

A dataset is a versioned, deterministic collection of input samples. Each
sample maps input tensor names to numpy arrays. Datasets are generated from a
fixed seed so the exact same inputs are produced on any machine, and they can
be saved to / loaded from ``.npz`` for versioning alongside the repo.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..server.runtime import OnnxModel, TensorSpec

# Bump when the generation algorithm changes (affects reproducibility hash).
DATASET_FORMAT_VERSION = "1.0.0"


def _npz_path(path: str | Path) -> Path:
    """Normalize ``path`` to a ``.npz`` filename.

    ``np.savez_compressed`` silently appends ``.npz`` when the filename lacks
    it, so a caller passing ``results/foo`` would get data at
    ``results/foo.npz`` while the returned path and ``.meta.json`` sidecar still
    referenced ``results/foo``. Normalizing up front keeps the returned path,
    the data file, and the metadata sidecar in agreement and loadable.
    """
    path = Path(path)
    if path.suffix != ".npz":
        return path.with_name(path.name + ".npz")
    return path


@dataclass(frozen=True)
class Dataset:
    name: str
    version: str
    samples: list[dict[str, np.ndarray]]
    seed: int

    def __len__(self) -> int:
        return len(self.samples)

    def fingerprint(self) -> str:
        """Stable content hash, used to assert reproducibility."""
        hasher = hashlib.sha256()
        hasher.update(f"{self.name}:{self.version}:{self.seed}".encode())
        for sample in self.samples:
            for key in sorted(sample):
                arr = np.ascontiguousarray(sample[key])
                hasher.update(key.encode())
                hasher.update(str(arr.dtype).encode())
                hasher.update(str(arr.shape).encode())
                hasher.update(arr.tobytes())
        return hasher.hexdigest()

    def save(self, path: str | Path) -> Path:
        path = _npz_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        flat: dict[str, np.ndarray] = {}
        for i, sample in enumerate(self.samples):
            for key, arr in sample.items():
                flat[f"{i}::{key}"] = arr
        # mypy mis-resolves the savez_compressed overload when fed **ndarrays;
        # the file-arg + keyword-arrays form is the documented usage.
        np.savez_compressed(str(path), **flat)  # type: ignore[arg-type]
        meta = {
            "name": self.name,
            "version": self.version,
            "seed": self.seed,
            "format_version": DATASET_FORMAT_VERSION,
            "num_samples": len(self.samples),
            "fingerprint": self.fingerprint(),
        }
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        path = _npz_path(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        with np.load(path) as data:
            buckets: dict[int, dict[str, np.ndarray]] = {}
            for flat_key in data.files:
                idx_str, key = flat_key.split("::", 1)
                buckets.setdefault(int(idx_str), {})[key] = data[flat_key]
        samples = [buckets[i] for i in sorted(buckets)]
        ds = cls(
            name=meta["name"],
            version=meta["version"],
            samples=samples,
            seed=meta["seed"],
        )
        expected = meta.get("fingerprint")
        if expected and expected != ds.fingerprint():
            raise ValueError(
                f"dataset fingerprint mismatch for {path}: "
                f"expected {expected}, got {ds.fingerprint()}"
            )
        return ds


def _concrete_shape(spec: TensorSpec, batch_size: int) -> tuple[int, ...]:
    # Replace dynamic dims (-1): first dim -> batch_size, others -> 1. A truly
    # scalar input (empty shape) stays scalar so the model's input contract is
    # preserved rather than promoted to a 1-D tensor.
    dims = []
    for i, d in enumerate(spec.shape):
        if d == -1:
            dims.append(batch_size if i == 0 else 1)
        else:
            dims.append(d)
    return tuple(dims)


def generate_for_model(
    model: OnnxModel,
    *,
    name: str = "synthetic",
    num_samples: int = 64,
    batch_size: int = 1,
    seed: int = 1234,
    version: str = DATASET_FORMAT_VERSION,
) -> Dataset:
    """Generate a deterministic synthetic dataset matching a model's inputs."""
    rng = np.random.default_rng(seed)
    specs = model.input_specs()
    samples: list[dict[str, np.ndarray]] = []
    for _ in range(num_samples):
        sample: dict[str, np.ndarray] = {}
        for spec in specs:
            shape = _concrete_shape(spec, batch_size)
            sample[spec.name] = _random_array(rng, spec.dtype, shape)
        samples.append(sample)
    return Dataset(name=name, version=version, samples=samples, seed=seed)


def _random_array(rng: np.random.Generator, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    np_dtype = np.dtype(dtype)
    if np_dtype.kind == "f":
        return rng.standard_normal(shape).astype(np_dtype)
    if np_dtype.kind in ("i", "u"):
        # Token-id-like inputs: small non-negative integers.
        return rng.integers(0, 100, size=shape, dtype=np_dtype)
    if np_dtype.kind == "b":
        return rng.integers(0, 2, size=shape).astype(np.bool_)
    return rng.standard_normal(shape).astype(np.float32)
