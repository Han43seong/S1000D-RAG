"""API/UI evidence serialization for fused text and visual-caption records."""

from __future__ import annotations

import sys

from src.rag.evidence import evidence_from_fused_record, evidences_from_fused_records
from src.rag.query_router import fuse_ranked_candidates, route_query

HEAVY_MODULES = ("chromadb", "langchain_core", "sentence_transformers", "llama_cpp")


def _text_candidate(content="Text procedure", score=0.8, **metadata):
    base_metadata = {
        "modality": "text",
        "dmc": "DMC-TEXT",
        "chunk_id": "chunk-7",
        "chunk_index": "7",
        "dm_type": "procedure",
    }
    base_metadata.update(metadata)
    return {"page_content": content, "metadata": base_metadata, "score": score}


def _image_candidate(content="Caption summary", score=0.8, **metadata):
    base_metadata = {
        "modality": "image",
        "content_role": "visual_caption",
        "dmc": "DMC-IMG",
        "asset_key": "DMC-IMG:figure:fig-1",
        "asset_path": "ICN-BRAKE-1.svg",
        "caption_path": "captions/fig-1.json",
        "title": "Brake warning diagram",
        "kind": "figure",
        "ref_id": "fig-1",
    }
    base_metadata.update(metadata)
    return {"page_content": content, "metadata": base_metadata, "score": score}


def test_text_fused_record_serializes_backward_compatible_fields():
    fused = fuse_ranked_candidates([_text_candidate(score=0.72)], route_query("remove procedure"))

    evidence = evidence_from_fused_record(fused[0])

    assert evidence["dmc"] == "DMC-TEXT"
    assert evidence["chunk_id"] == "chunk-7"
    assert evidence["chunk_index"] == "7"
    assert evidence["score"] == 0.72
    assert evidence["final_score"] == fused[0]["final_score"]
    assert evidence["modality"] == "text"
    assert evidence["content_role"] == "text"
    assert evidence["display_label"] == "DMC-TEXT · chunk chunk-7"
    assert evidence["source_label"] == "DMC-TEXT · chunk chunk-7"


def test_image_caption_fused_record_serializes_visual_metadata():
    fused = fuse_ranked_candidates([_image_candidate(score=0.9)], route_query("figure label"))

    evidence = evidence_from_fused_record(fused[0])

    assert evidence["modality"] == "image"
    assert evidence["content_role"] == "visual_caption"
    assert evidence["dmc"] == "DMC-IMG"
    assert evidence["asset_key"] == "DMC-IMG:figure:fig-1"
    assert evidence["asset_path"] == "ICN-BRAKE-1.svg"
    assert evidence["caption_path"] == "captions/fig-1.json"
    assert evidence["title"] == "Brake warning diagram"
    assert evidence["kind"] == "figure"
    assert evidence["ref_id"] == "fig-1"
    assert evidence["display_label"] == "Brake warning diagram"
    assert evidence["source_label"] == "DMC-IMG · DMC-IMG:figure:fig-1"


def test_mixed_fused_records_preserve_rank_order_and_final_score():
    route = route_query("figure label")
    fused = fuse_ranked_candidates(
        [
            _text_candidate("Text", score=0.95, dmc="DMC-A", chunk_id="chunk-a"),
            _image_candidate("Image", score=0.95, dmc="DMC-B", asset_key="asset-b"),
        ],
        route,
    )

    evidences = evidences_from_fused_records(fused)

    assert [item["rank"] for item in evidences] == [1, 2]
    assert [item["dmc"] for item in evidences] == [item["metadata"]["dmc"] for item in fused]
    assert [item["final_score"] for item in evidences] == [item["final_score"] for item in fused]
    assert evidences[0]["final_score"] >= evidences[1]["final_score"]


def test_importing_evidence_helper_does_not_import_heavy_dependencies():
    for module_name in ["src.rag.evidence", *HEAVY_MODULES]:
        sys.modules.pop(module_name, None)

    import src.rag.evidence  # noqa: F401

    for module_name in HEAVY_MODULES:
        assert module_name not in sys.modules
