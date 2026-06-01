from __future__ import annotations

import json

from ingest import build_index_manifest, write_index_manifest


def test_build_index_manifest_is_dependency_light_and_contains_index_evidence(tmp_path):
    manifest = build_index_manifest(
        data_dir=tmp_path / "data",
        collection_name="unit_collection",
        dm_count=3,
        parse_success_count=2,
        parse_error_count=1,
        chunk_count=7,
        dmcs=["DMC-A", "DMC-B", "DMC-C"],
        parse_errors=["DMC-C: bad xml"],
        created_at="2026-06-01T00:00:00+00:00",
        git_commit="abc1234",
    )

    assert manifest["data_dir"] == str(tmp_path / "data")
    assert manifest["collection_name"] == "unit_collection"
    assert manifest["dm_count"] == 3
    assert manifest["parse_success_count"] == 2
    assert manifest["parse_error_count"] == 1
    assert manifest["chunk_count"] == 7
    assert manifest["created_at"] == "2026-06-01T00:00:00+00:00"
    assert manifest["text_model_profile"]
    assert manifest["embedding_model"]
    assert manifest["reranker_model"]
    assert manifest["git_commit"] == "abc1234"
    assert manifest["sample_dmcs"] == ["DMC-A", "DMC-B", "DMC-C"]
    assert manifest["sample_errors"] == ["DMC-C: bad xml"]


def test_write_index_manifest_writes_manifest_json(tmp_path):
    manifest = {"collection_name": "unit_collection", "chunk_count": 1}

    path = write_index_manifest(manifest, tmp_path / "chroma")

    assert path == tmp_path / "chroma" / "manifest.json"
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
