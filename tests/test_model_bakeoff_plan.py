from __future__ import annotations

import json
import subprocess
import sys

from scripts import model_bakeoff_plan


HEAVY_MODULES = [
    "llama_cpp",
    "langchain",
    "langchain_community",
    "langchain_huggingface",
    "sentence_transformers",
    "chromadb",
    "torch",
    "transformers",
]


def test_json_output_parses_and_contains_required_profiles():
    output = subprocess.check_output(
        [
            sys.executable,
            "scripts/model_bakeoff_plan.py",
            "--format",
            "json",
            "--hardware-profile",
            "rtx4080super_16gb",
        ],
        text=True,
    )
    data = json.loads(output)

    assert data["hardware_constraints"]["display_name"] == "RTX 4080 SUPER 16GB"
    assert data["measurement_default"] == "pending_measurement"
    assert data["no_downloads_or_model_loads"] is True

    text_profiles = {item["profile"] for item in data["candidates"]["text_llms"]}
    vlm_profiles = {item["profile"] for item in data["candidates"]["vlms"]}
    embedding_models = {item["model"] for item in data["candidates"]["embeddings"]}
    reranker_models = {item["model"] for item in data["candidates"]["rerankers"]}

    assert {"qwen36_27b_iq4", "gemma4_26b_iq4"} <= text_profiles
    assert "qwen3_vl_8b_q4" in vlm_profiles
    assert "BAAI/bge-m3" in embedding_models
    assert "BAAI/bge-reranker-v2-m3" in reranker_models
    assert data["model_registry"]["available"] is True


def test_markdown_output_contains_required_sections_and_pending_markers():
    plan = model_bakeoff_plan.build_plan(hardware_profile="rtx4080super_16gb")
    markdown = model_bakeoff_plan.render_markdown(plan)

    for section in [
        "## Hardware Constraints",
        "## Text LLM Candidates",
        "## VLM Candidates",
        "## Embedding/Reranker Candidates",
        "## Evaluation Dimensions",
        "## Commands To Run Later (Not Executed Now)",
        "## Decision Table",
    ]:
        assert section in markdown

    for phrase in [
        "RTX 4080 SUPER 16GB",
        "Qwen3.6 27B IQ4",
        "Gemma 4 26B IQ4_NL",
        "Qwen3-VL 8B Q4",
        "BGE-M3",
        "BGE reranker v2 m3",
        "pending_measurement",
        "answer_correctness",
        "citation_grounding",
        "visual_grounding",
        "latency",
        "VRAM fit",
        "offline_usability",
    ]:
        assert phrase in markdown


def test_scaffold_import_and_build_do_not_import_heavy_deps(monkeypatch):
    for module_name in ["scripts.model_bakeoff_plan", "src.runtime.model_registry", *HEAVY_MODULES]:
        sys.modules.pop(module_name, None)

    import scripts.model_bakeoff_plan as scaffold

    plan = scaffold.build_plan(hardware_profile="rtx4080super_16gb")
    assert plan["status"] == "planned_not_executed"

    for module_name in HEAVY_MODULES:
        assert module_name not in sys.modules
