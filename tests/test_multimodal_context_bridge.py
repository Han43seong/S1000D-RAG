"""Offline bridge tests for visual captions in multimodal RAG context."""

from __future__ import annotations

import json
import subprocess
import sys

from src.rag.multimodal_context import build_multimodal_context, load_caption_candidates

HEAVY_MODULES = ("chromadb", "langchain_core", "sentence_transformers", "llama_cpp")


def _caption(asset_key="DMC-UNIT:figure:fig-1", summary="Brake wiring diagram with label A"):
    return {
        "asset_key": asset_key,
        "asset_path": "ICN-BRAKE-1.svg",
        "status": "mock_captioned",
        "summary": summary,
        "ocr_text": "LABEL A",
        "components": ["connector", "wire"],
        "safety_notes": ["verify power is isolated"],
        "keywords": ["figure", "wiring", "label"],
        "backend": "mock",
        "model_profile": "mock-vlm-captioner",
        "prompt_profile": "s1000d-technical-manual-v1",
        "dmc": "DMC-UNIT",
        "structure_path": "/content/description/figure[1]",
        "ref_id": "fig-1",
        "kind": "figure",
        "title": "Brake warning diagram",
    }


def test_temp_caption_json_files_load_into_visual_candidates(tmp_path):
    captions_dir = tmp_path / "captions"
    captions_dir.mkdir()
    (captions_dir / "caption.json").write_text(json.dumps(_caption()), encoding="utf-8")

    candidates = load_caption_candidates(captions_dir)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert "Brake wiring diagram" in candidate["page_content"]
    assert "OCR text: LABEL A" in candidate["page_content"]
    assert candidate["score"] == 1.0
    metadata = candidate["metadata"]
    assert metadata["source_lane"] == "visual"
    assert metadata["modality"] == "image"
    assert metadata["content_role"] == "visual_caption"
    assert metadata["caption_path"] == str(captions_dir / "caption.json")


def test_route_fusion_formatting_includes_image_caption_headers(tmp_path):
    caption_path = tmp_path / "caption.json"
    caption_path.write_text(json.dumps(_caption()), encoding="utf-8")
    candidates = load_caption_candidates(tmp_path)

    route, fused, context = build_multimodal_context(
        query="Which label is shown in the wiring diagram?",
        caption_candidates=candidates,
        limit=3,
    )

    assert route.visual_intent is True
    assert fused[0]["modality"] == "image_caption"
    assert "[IMAGE_CAPTION DMC=DMC-UNIT ASSET=DMC-UNIT:figure:fig-1 PATH=ICN-BRAKE-1.svg]" in context
    assert "Brake wiring diagram with label A" in context


def test_cli_preview_works_in_mock_offline_mode(tmp_path):
    captions_dir = tmp_path / "captions"
    captions_dir.mkdir()
    (captions_dir / "one.json").write_text(json.dumps(_caption(summary="Hydraulic diagram caption")), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/preview_multimodal_context.py",
            "--captions-dir",
            str(captions_dir),
            "--query",
            "diagram label",
            "--limit",
            "2",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Route summary:" in result.stdout
    assert "Loaded visual caption candidates: 1" in result.stdout
    assert "[IMAGE_CAPTION" in result.stdout
    assert "Hydraulic diagram caption" in result.stdout


def test_multimodal_context_bridge_avoids_heavy_imports():
    for module_name in ["src.rag.multimodal_context", *HEAVY_MODULES]:
        sys.modules.pop(module_name, None)

    import src.rag.multimodal_context  # noqa: F401

    for module_name in HEAVY_MODULES:
        assert module_name not in sys.modules
