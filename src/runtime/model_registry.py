"""Lightweight model profile registry for S1000D-RAG.

This module is intentionally dependency-free: importing it must not import or load
llama.cpp, LangChain, SentenceTransformers, ChromaDB, or any model files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

TEXT_PROFILE_ENV = "S1000D_TEXT_MODEL_PROFILE"
VLM_PROFILE_ENV = "S1000D_VLM_MODEL_PROFILE"
MODEL_BACKEND_ENV = "S1000D_MODEL_BACKEND"
TEXT_MODEL_PATH_ENV = "S1000D_TEXT_MODEL_PATH"
VLM_MODEL_PATH_ENV = "S1000D_VLM_MODEL_PATH"
VLM_MMPROJ_PATH_ENV = "S1000D_VLM_MMPROJ_PATH"
EMBEDDING_MODEL_ENV = "S1000D_EMBEDDING_MODEL"
RERANKER_MODEL_ENV = "S1000D_RERANKER_MODEL"

LEGACY_TEXT_MODEL_PATH_ENV = "GGUF_MODEL_PATH"
LEGACY_EMBEDDING_MODEL_ENV = "EMBEDDING_MODEL_PATH"
LEGACY_RERANKER_MODEL_ENV = "RERANKER_MODEL_PATH"

DEFAULT_TEXT_PROFILE = "qwen36_27b_iq4"
DEFAULT_VLM_PROFILE = "qwen3_vl_8b_q4"
DEFAULT_MODEL_BACKEND = "llama_cpp_python"
SUPPORTED_BACKENDS = ("llama_cpp_python", "llama_server")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class TextModelProfile:
    """A selectable text LLM profile."""

    name: str
    repo_id: str
    quantization_candidates: tuple[str, ...]
    recommended_first_quant: str
    display_name: str
    context_window: int = 8192
    gpu_layers: int = -1
    path_env_var: str = TEXT_MODEL_PATH_ENV
    legacy_path_env_var: str = LEGACY_TEXT_MODEL_PATH_ENV
    path_required: bool = True
    notes: str = ""


@dataclass(frozen=True)
class VlmModelProfile:
    """A selectable vision-language model profile."""

    name: str
    repo_id: str
    quantization_candidates: tuple[str, ...]
    recommended_first_quant: str
    display_name: str
    model_path_env_var: str = VLM_MODEL_PATH_ENV
    mmproj_path_env_var: str = VLM_MMPROJ_PATH_ENV
    path_required: bool = True
    notes: str = ""


@dataclass(frozen=True)
class EmbeddingModelConfig:
    model: str
    env_var: str = EMBEDDING_MODEL_ENV
    legacy_env_var: str = LEGACY_EMBEDDING_MODEL_ENV
    normalize_embeddings: bool = True


@dataclass(frozen=True)
class RerankerModelConfig:
    model: str
    env_var: str = RERANKER_MODEL_ENV
    legacy_env_var: str = LEGACY_RERANKER_MODEL_ENV


@dataclass(frozen=True)
class ModelRuntimeConfig:
    backend: str
    text_profile: TextModelProfile
    vlm_profile: VlmModelProfile
    text_model_path: str | None
    vlm_model_path: str | None
    vlm_mmproj_path: str | None
    embedding: EmbeddingModelConfig
    reranker: RerankerModelConfig


TEXT_MODEL_PROFILES: dict[str, TextModelProfile] = {
    "qwen36_27b_iq4": TextModelProfile(
        name="qwen36_27b_iq4",
        repo_id="unsloth/Qwen3.6-27B-GGUF",
        quantization_candidates=("IQ4_NL", "IQ4_XS"),
        recommended_first_quant="IQ4_NL",
        display_name="Qwen3.6 27B IQ4",
        notes="Primary 16GB VRAM candidate; set S1000D_TEXT_MODEL_PATH to the local GGUF file.",
    ),
    "gemma4_26b_iq4": TextModelProfile(
        name="gemma4_26b_iq4",
        repo_id="unsloth/gemma-4-26B-A4B-it-GGUF",
        quantization_candidates=("UD-IQ4_NL",),
        recommended_first_quant="UD-IQ4_NL",
        display_name="Gemma 4 26B A4B IQ4",
        notes="Alternative 16GB VRAM candidate; set S1000D_TEXT_MODEL_PATH to the local GGUF file.",
    ),
    "light_qwen35_9b": TextModelProfile(
        name="light_qwen35_9b",
        repo_id="unsloth/Qwen3.5-9B-GGUF",
        quantization_candidates=("Q4_K_M", "Q5_K_M"),
        recommended_first_quant="Q4_K_M",
        display_name="Qwen3.5 9B lightweight",
        context_window=8192,
        notes="Lightweight fallback when larger profiles exceed local memory/latency targets.",
    ),
}

VLM_MODEL_PROFILES: dict[str, VlmModelProfile] = {
    "qwen3_vl_8b_q4": VlmModelProfile(
        name="qwen3_vl_8b_q4",
        repo_id="Qwen/Qwen3-VL-8B-Instruct-GGUF",
        quantization_candidates=("Q4_K_M",),
        recommended_first_quant="Q4_K_M",
        display_name="Qwen3-VL 8B Q4",
        notes="Set S1000D_VLM_MODEL_PATH and S1000D_VLM_MMPROJ_PATH to local files before enabling VLM use.",
    )
}


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return value


def _legacy_text_default_path() -> str:
    return str(MODELS_DIR / "Qwen3-14B-Q4_K_M.gguf")


def get_text_model_profile(name: str | None = None) -> TextModelProfile:
    selected = name or _env(TEXT_PROFILE_ENV) or DEFAULT_TEXT_PROFILE
    try:
        return TEXT_MODEL_PROFILES[selected]
    except KeyError as exc:
        raise ValueError(
            f"Unknown text model profile '{selected}'. Available: {', '.join(sorted(TEXT_MODEL_PROFILES))}"
        ) from exc


def get_vlm_model_profile(name: str | None = None) -> VlmModelProfile:
    selected = name or _env(VLM_PROFILE_ENV) or DEFAULT_VLM_PROFILE
    try:
        return VLM_MODEL_PROFILES[selected]
    except KeyError as exc:
        raise ValueError(
            f"Unknown VLM model profile '{selected}'. Available: {', '.join(sorted(VLM_MODEL_PROFILES))}"
        ) from exc


def get_embedding_config() -> EmbeddingModelConfig:
    return EmbeddingModelConfig(
        model=_env(EMBEDDING_MODEL_ENV) or _env(LEGACY_EMBEDDING_MODEL_ENV) or DEFAULT_EMBEDDING_MODEL
    )


def get_reranker_config() -> RerankerModelConfig:
    return RerankerModelConfig(
        model=_env(RERANKER_MODEL_ENV) or _env(LEGACY_RERANKER_MODEL_ENV) or DEFAULT_RERANKER_MODEL
    )


def get_model_backend() -> str:
    backend = _env(MODEL_BACKEND_ENV) or DEFAULT_MODEL_BACKEND
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unknown model backend '{backend}'. Available: {', '.join(SUPPORTED_BACKENDS)}")
    return backend


def resolve_text_model_path() -> str | None:
    return _env(TEXT_MODEL_PATH_ENV) or _env(LEGACY_TEXT_MODEL_PATH_ENV) or _legacy_text_default_path()


def resolve_vlm_model_path() -> str | None:
    return _env(VLM_MODEL_PATH_ENV)


def resolve_vlm_mmproj_path() -> str | None:
    return _env(VLM_MMPROJ_PATH_ENV)


def get_model_runtime_config() -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        backend=get_model_backend(),
        text_profile=get_text_model_profile(),
        vlm_profile=get_vlm_model_profile(),
        text_model_path=resolve_text_model_path(),
        vlm_model_path=resolve_vlm_model_path(),
        vlm_mmproj_path=resolve_vlm_mmproj_path(),
        embedding=get_embedding_config(),
        reranker=get_reranker_config(),
    )


def list_model_profiles() -> dict[str, list[str]]:
    return {
        "text": sorted(TEXT_MODEL_PROFILES),
        "vlm": sorted(VLM_MODEL_PROFILES),
    }
