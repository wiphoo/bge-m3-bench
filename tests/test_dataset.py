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
