from __future__ import annotations

from onnx_grpc_benchmark.metadata import collect_metadata, metadata_schema


def test_metadata_required_fields():
    meta = collect_metadata()
    schema = metadata_schema()
    for field in schema["required"]:
        assert field in meta, f"missing required metadata field: {field}"


def test_metadata_is_json_serialisable():
    import json

    meta = collect_metadata(extra={"note": "test"})
    json.dumps(meta, default=str)
    assert meta["extra"]["note"] == "test"


def test_container_detection_keys():
    meta = collect_metadata()
    assert "in_docker" in meta["container"]
    assert "in_kubernetes" in meta["container"]
