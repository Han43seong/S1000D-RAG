from __future__ import annotations

import json
import subprocess
import sys

from src.vlm.captioner import MockVisualCaptioner
from src.vlm.documents import caption_to_document
from src.vlm.prompts import build_technical_manual_caption_prompt


HEAVY_MODULES = [
    "llama_cpp",
    "chromadb",
    "langchain_core",
    "sentence_transformers",
]


def _asset(status="found"):
    return {
        "key": "DMC-UNIT:figure:fig-1",
        "kind": "figure",
        "dmc": "DMC-UNIT",
        "ref_id": "fig-1",
        "title": "Brake warning diagram",
        "info_entity_ident": "ICN-BRAKE-1",
        "structure_path": "/content/description/figure[1]",
        "asset_path": "ICN-BRAKE-1.svg",
        "status": status,
        "metadata_only": status == "metadata_only",
    }


def test_caption_prompt_requests_strict_json_fields():
    prompt = build_technical_manual_caption_prompt(_asset())

    assert "Return only valid JSON" in prompt
    for field in ["summary", "ocr_text", "components", "safety_notes", "keywords"]:
        assert f'"{field}"' in prompt
    assert "DMC-UNIT:figure:fig-1" in prompt
    assert "Do not invent unreadable label text" in prompt


def test_mock_captioner_uses_manifest_metadata_without_image_bytes(tmp_path):
    asset = _asset()
    caption = MockVisualCaptioner().caption_asset(asset, data_dir=tmp_path)

    assert caption.asset_key == asset["key"]
    assert caption.asset_path == str(tmp_path / "ICN-BRAKE-1.svg")
    assert caption.status == "mock_captioned"
    assert "Brake warning diagram" in caption.summary
    assert caption.ocr_text == ""
    assert "figure" in caption.keywords
    assert caption.safety_notes
    assert caption.metadata["manifest_status"] == "found"


def test_caption_cli_writes_mock_caption_json_from_temp_manifest(tmp_path):
    manifest_path = tmp_path / "assets_manifest.json"
    output_dir = tmp_path / "captions"
    manifest_path.write_text(
        json.dumps(
            {
                "data_dir": str(tmp_path),
                "assets": [
                    _asset("found"),
                    {**_asset("metadata_only"), "key": "DMC-UNIT:table:tab-1", "asset_path": None},
                    {**_asset("missing"), "key": "DMC-UNIT:figure:missing", "asset_path": None},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/caption_assets.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--mock",
            "--overwrite",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "seen=3" in result.stdout
    assert "written=2" in result.stdout
    assert "skipped_missing=1" in result.stdout
    captions = sorted(output_dir.glob("*.json"))
    assert len(captions) == 2
    first = json.loads(captions[0].read_text(encoding="utf-8"))
    assert first["backend"] == "mock"
    assert first["summary"]


def test_caption_document_conversion_has_visual_metadata(tmp_path):
    caption = MockVisualCaptioner().caption_asset(_asset(), data_dir=tmp_path)
    doc = caption_to_document(caption, caption_path=tmp_path / "caption.json")

    assert "Brake warning diagram" in doc["page_content"]
    metadata = doc["metadata"]
    assert metadata["modality"] == "image"
    assert metadata["asset_key"] == caption.asset_key
    assert metadata["asset_path"] == caption.asset_path
    assert metadata["dmc"] == "DMC-UNIT"
    assert metadata["structure_path"] == "/content/description/figure[1]"
    assert metadata["caption_path"] == str(tmp_path / "caption.json")
    assert metadata["content_role"] == "visual_caption"


def test_vlm_caption_modules_are_dependency_light():
    for module_name in ["src.vlm.captioner", "src.vlm.documents", "src.vlm.prompts", "src.vlm.types", *HEAVY_MODULES]:
        sys.modules.pop(module_name, None)

    import src.vlm.captioner  # noqa: F401
    import src.vlm.documents  # noqa: F401
    import src.vlm.prompts  # noqa: F401
    import src.vlm.types  # noqa: F401

    for module_name in HEAVY_MODULES:
        assert module_name not in sys.modules
