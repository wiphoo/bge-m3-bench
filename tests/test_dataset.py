from __future__ import annotations

from onnx_grpc_benchmark.benchmark import dataset as ds


def test_generation_is_deterministic(model):
    a = ds.generate_for_model(model, num_samples=5, seed=42)
    b = ds.generate_for_model(model, num_samples=5, seed=42)
    assert a.fingerprint() == b.fingerprint()


def test_different_seed_differs(model):
    a = ds.generate_for_model(model, num_samples=5, seed=1)
    b = ds.generate_for_model(model, num_samples=5, seed=2)
    assert a.fingerprint() != b.fingerprint()


def test_save_load_roundtrip(model, tmp_path):
    a = ds.generate_for_model(model, num_samples=4, seed=3)
    path = a.save(tmp_path / "ds.npz")
    loaded = ds.Dataset.load(path)
    assert loaded.fingerprint() == a.fingerprint()
    assert len(loaded) == len(a)


def test_save_normalizes_missing_npz_suffix(model, tmp_path):
    """A path without ``.npz`` is normalized so the returned path is loadable."""
    a = ds.generate_for_model(model, num_samples=3, seed=5)
    out = tmp_path / "foo"
    path = a.save(out)
    assert path.suffix == ".npz"
    assert path.exists()
    # The data file and metadata sidecar agree, so the returned path loads back.
    loaded = ds.Dataset.load(path)
    assert loaded.fingerprint() == a.fingerprint()
    # Loading via the original (suffix-less) path also resolves.
    assert ds.Dataset.load(out).fingerprint() == a.fingerprint()


def test_concrete_shape_preserves_scalar():
    from onnx_grpc_benchmark.server.runtime import TensorSpec

    scalar = TensorSpec(name="s", dtype="float32", shape=())
    assert ds._concrete_shape(scalar, batch_size=4) == ()

    dynamic = TensorSpec(name="d", dtype="float32", shape=(-1, 8))
    assert ds._concrete_shape(dynamic, batch_size=4) == (4, 8)
