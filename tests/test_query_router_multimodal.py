"""Dependency-light multimodal query routing/fusion tests."""

from __future__ import annotations

import sys

from src.rag.query_router import (
    candidate_modality,
    format_fused_context,
    fuse_ranked_candidates,
    route_query,
)


FORBIDDEN_IMPORTS = ("chromadb", "langchain_core", "sentence_transformers", "llama_cpp")


def text_candidate(content="text", score=0.8, **metadata):
    base_metadata = {"modality": "text", "dmc": "DMC-TEXT", "chunk_index": "1"}
    base_metadata.update(metadata)
    return {"page_content": content, "metadata": base_metadata, "score": score}


def image_candidate(content="image caption", score=0.8, **metadata):
    base_metadata = {
        "modality": "image",
        "content_role": "visual_caption",
        "dmc": "DMC-IMG",
        "asset_key": "fig-1",
        "caption_path": "captions/fig-1.json",
    }
    base_metadata.update(metadata)
    return {"page_content": content, "metadata": base_metadata, "score": score}


def test_korean_visual_intent_detection():
    route = route_query("브레이크 회로 도면의 라벨 위치를 보여줘")

    assert route.visual_intent is True
    assert route.text_intent is True
    assert "도면" in route.matched_terms
    assert "라벨" in route.matched_terms
    assert route.visual_weight > route.text_weight


def test_english_visual_intent_detection():
    route = route_query("Which label is shown in the wiring diagram?")

    assert route.visual_intent is True
    assert route.text_intent is True
    assert "diagram" in route.matched_terms
    assert "label" in route.matched_terms
    assert route.visual_weight > route.text_weight


def test_procedural_query_remains_text_first_but_visual_capable():
    route = route_query("브레이크 패드 교체 절차")

    assert route.visual_intent is False
    assert route.text_intent is True
    assert route.text_weight > route.visual_weight


def test_table_as_output_format_is_not_visual_intent():
    route = route_query("브레이크 패드 청소 절차를 표처럼 정리해줘")

    assert route.visual_intent is False
    assert route.text_intent is True
    assert route.text_weight > route.visual_weight


def test_visual_route_boosts_image_caption_above_comparable_text():
    route = route_query("figure label position")
    fused = fuse_ranked_candidates([text_candidate(score=0.8), image_candidate(score=0.8)], route)

    assert fused[0]["modality"] == "image_caption"
    assert fused[0]["final_score"] > fused[1]["final_score"]
    assert fused[0]["modality_boost"] == route.visual_weight


def test_procedural_route_keeps_text_candidate_above_image_candidate():
    route = route_query("remove and install procedure")
    fused = fuse_ranked_candidates([image_candidate(score=0.8), text_candidate(score=0.8)], route)

    assert fused[0]["modality"] == "text"
    assert fused[0]["final_score"] > fused[1]["final_score"]
    assert fused[0]["modality_boost"] == route.text_weight


def test_deduplicates_by_asset_key_and_text_chunk_preserving_best_score():
    route = route_query("diagram")
    candidates = [
        image_candidate("old caption", score=0.4, asset_key="same-asset", caption_path="a.json"),
        image_candidate("best caption", score=0.7, asset_key="same-asset", caption_path="b.json"),
        text_candidate("old text", score=0.5, dmc="DMC-A", chunk_index="7"),
        text_candidate("best text", score=0.9, dmc="DMC-A", chunk_index="7"),
    ]

    fused = fuse_ranked_candidates(candidates, route)

    assert len(fused) == 2
    assert {item["page_content"] for item in fused} == {"best caption", "best text"}
    assert all(item["rank"] in (1, 2) for item in fused)


def test_context_formatting_includes_modality_headers_and_metadata():
    route = route_query("table image")
    fused = fuse_ranked_candidates(
        [
            text_candidate("Text evidence", score=0.9, dmc="DMC-001", chunk_index="3"),
            image_candidate(
                "Caption evidence",
                score=0.9,
                dmc="DMC-002",
                asset_key="asset-9",
                asset_path="ICN-ASSET-9.CGM",
                caption_path="caps/asset-9.json",
            ),
        ],
        route,
    )

    context = format_fused_context(fused)

    assert "[TEXT DMC=DMC-001 CHUNK=3]" in context
    assert "[IMAGE_CAPTION DMC=DMC-002 ASSET=asset-9 PATH=ICN-ASSET-9.CGM]" in context
    assert "Text evidence" in context
    assert "Caption evidence" in context


def test_candidate_modality_detects_visual_caption_role():
    assert candidate_modality({"metadata": {"content_role": "visual_caption"}}) == "image_caption"


def test_importing_query_router_does_not_import_heavy_dependencies():
    for module_name in FORBIDDEN_IMPORTS:
        sys.modules.pop(module_name, None)

    import src.rag.query_router  # noqa: F401

    for module_name in FORBIDDEN_IMPORTS:
        assert module_name not in sys.modules
