from __future__ import annotations

import importlib
import sys

import pytest


S1000D_ENV_VARS = [
    "S1000D_TEXT_MODEL_PROFILE",
    "S1000D_VLM_MODEL_PROFILE",
    "S1000D_MODEL_BACKEND",
    "S1000D_TEXT_MODEL_PATH",
    "S1000D_VLM_MODEL_PATH",
    "S1000D_VLM_MMPROJ_PATH",
    "S1000D_EMBEDDING_MODEL",
    "S1000D_RERANKER_MODEL",
    "GGUF_MODEL_PATH",
    "EMBEDDING_MODEL_PATH",
    "RERANKER_MODEL_PATH",
]


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in S1000D_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_registry_import_does_not_import_heavy_ml_libraries(monkeypatch):
    _clear_env(monkeypatch)
    for module_name in [
        "src.runtime.model_registry",
        "llama_cpp",
        "langchain",
        "langchain_community",
        "langchain_huggingface",
        "sentence_transformers",
        "chromadb",
    ]:
        sys.modules.pop(module_name, None)

    import src.runtime.model_registry as registry

    assert registry.get_text_model_profile().name == "qwen36_27b_iq4"
    assert "llama_cpp" not in sys.modules
    assert "langchain_community" not in sys.modules
    assert "langchain_huggingface" not in sys.modules
    assert "sentence_transformers" not in sys.modules
    assert "chromadb" not in sys.modules


def test_registry_defaults(monkeypatch):
    _clear_env(monkeypatch)
    from src.runtime import model_registry as registry

    cfg = registry.get_model_runtime_config()

    assert cfg.backend == "llama_cpp_python"
    assert cfg.text_profile.name == "qwen36_27b_iq4"
    assert cfg.text_profile.repo_id == "unsloth/Qwen3.6-27B-GGUF"
    assert cfg.text_profile.recommended_first_quant == "IQ4_NL"
    assert cfg.vlm_profile.name == "qwen3_vl_8b_q4"
    assert cfg.embedding.model == "BAAI/bge-m3"
    assert cfg.reranker.model == "BAAI/bge-reranker-v2-m3"
    assert registry.list_model_profiles()["text"] == [
        "gemma4_26b_iq4",
        "light_qwen35_9b",
        "qwen36_27b_iq4",
    ]


def test_env_profile_and_path_overrides(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("S1000D_TEXT_MODEL_PROFILE", "gemma4_26b_iq4")
    monkeypatch.setenv("S1000D_VLM_MODEL_PROFILE", "qwen3_vl_8b_q4")
    monkeypatch.setenv("S1000D_TEXT_MODEL_PATH", "/models/gemma.gguf")
    monkeypatch.setenv("S1000D_VLM_MODEL_PATH", "/models/qwen-vl.gguf")
    monkeypatch.setenv("S1000D_VLM_MMPROJ_PATH", "/models/mmproj.gguf")
    monkeypatch.setenv("S1000D_EMBEDDING_MODEL", "/models/bge-m3")
    monkeypatch.setenv("S1000D_RERANKER_MODEL", "/models/reranker")

    from src.runtime import model_registry as registry

    cfg = registry.get_model_runtime_config()

    assert cfg.text_profile.name == "gemma4_26b_iq4"
    assert cfg.text_profile.recommended_first_quant == "UD-IQ4_NL"
    assert cfg.text_model_path == "/models/gemma.gguf"
    assert cfg.vlm_model_path == "/models/qwen-vl.gguf"
    assert cfg.vlm_mmproj_path == "/models/mmproj.gguf"
    assert cfg.embedding.model == "/models/bge-m3"
    assert cfg.reranker.model == "/models/reranker"


def test_legacy_path_overrides_are_preserved(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("GGUF_MODEL_PATH", "/legacy/model.gguf")
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "/legacy/embeddings")
    monkeypatch.setenv("RERANKER_MODEL_PATH", "/legacy/reranker")

    from src.runtime import model_registry as registry

    cfg = registry.get_model_runtime_config()

    assert cfg.text_model_path == "/legacy/model.gguf"
    assert cfg.embedding.model == "/legacy/embeddings"
    assert cfg.reranker.model == "/legacy/reranker"


def test_s1000d_path_overrides_take_precedence_over_legacy(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("S1000D_TEXT_MODEL_PATH", "/new/model.gguf")
    monkeypatch.setenv("GGUF_MODEL_PATH", "/legacy/model.gguf")
    monkeypatch.setenv("S1000D_EMBEDDING_MODEL", "/new/embeddings")
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", "/legacy/embeddings")
    monkeypatch.setenv("S1000D_RERANKER_MODEL", "/new/reranker")
    monkeypatch.setenv("RERANKER_MODEL_PATH", "/legacy/reranker")

    from src.runtime import model_registry as registry

    cfg = registry.get_model_runtime_config()

    assert cfg.text_model_path == "/new/model.gguf"
    assert cfg.embedding.model == "/new/embeddings"
    assert cfg.reranker.model == "/new/reranker"


def test_unknown_profiles_and_backend_raise_clear_errors(monkeypatch):
    _clear_env(monkeypatch)
    from src.runtime import model_registry as registry

    with pytest.raises(ValueError, match="Unknown text model profile"):
        registry.get_text_model_profile("not-a-profile")

    with pytest.raises(ValueError, match="Unknown VLM model profile"):
        registry.get_vlm_model_profile("not-a-vlm-profile")

    monkeypatch.setenv("S1000D_MODEL_BACKEND", "not-a-backend")
    with pytest.raises(ValueError, match="Unknown model backend"):
        registry.get_model_backend()


def test_config_exports_new_env_names_and_keeps_legacy_aliases(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("S1000D_TEXT_MODEL_PROFILE", "light_qwen35_9b")
    monkeypatch.setenv("S1000D_TEXT_MODEL_PATH", "/new/qwen.gguf")
    monkeypatch.setenv("S1000D_EMBEDDING_MODEL", "/new/embedding")
    monkeypatch.setenv("S1000D_RERANKER_MODEL", "/new/reranker")

    import src.config as config

    config = importlib.reload(config)

    assert config.S1000D_TEXT_MODEL_PROFILE == "light_qwen35_9b"
    assert config.S1000D_TEXT_MODEL_PATH == "/new/qwen.gguf"
    assert config.S1000D_EMBEDDING_MODEL == "/new/embedding"
    assert config.S1000D_RERANKER_MODEL == "/new/reranker"
    assert config.GGUF_MODEL_PATH == "/new/qwen.gguf"
    assert config.EMBEDDING_MODEL_PATH == "/new/embedding"
    assert config.RERANKER_MODEL_PATH == "/new/reranker"
