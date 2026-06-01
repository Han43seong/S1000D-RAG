from __future__ import annotations

import sys

import pytest


HEAVY_MODULES = [
    "llama_cpp",
    "langchain",
    "langchain_community",
    "langchain_huggingface",
    "sentence_transformers",
    "chromadb",
]


def test_vlm_import_is_dependency_light(monkeypatch):
    for module_name in ["src.rag.vlm", *HEAVY_MODULES]:
        sys.modules.pop(module_name, None)

    import src.rag.vlm as vlm

    assert vlm.VisualEvidenceRequest(prompt="read this").prompt == "read this"
    for module_name in HEAVY_MODULES:
        assert module_name not in sys.modules


def test_get_vlm_client_disabled_backend_raises_clear_error(monkeypatch):
    monkeypatch.delenv("S1000D_VLM_MODEL_PATH", raising=False)
    monkeypatch.delenv("S1000D_VLM_MMPROJ_PATH", raising=False)

    from src.rag.vlm import VisualEvidenceRequest, get_vlm_client

    client = get_vlm_client()
    assert client.model_profile == "qwen3_vl_8b_q4"

    with pytest.raises(RuntimeError) as excinfo:
        client.generate(VisualEvidenceRequest(prompt="Describe figure", image_paths=("figure.png",), max_tokens=64))

    message = str(excinfo.value)
    assert "VLM inference is not enabled" in message
    assert "S1000D_VLM_MODEL_PATH" in message
    assert "S1000D_VLM_MMPROJ_PATH" in message
