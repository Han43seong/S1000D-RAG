"""LLM / Embedding / Reranker 싱글턴 관리.

초기화 비용이 큰 모델을 lazy singleton으로 관리한다.
각 getter 함수는 최초 호출 시에만 모델을 로드하고 이후 캐시된 인스턴스를 반환.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import (
    LLM_MAX_TOKENS,
    LLM_N_CTX,
    LLM_REPEAT_PENALTY,
    LLM_TEMPERATURE,
    LLM_TOP_P,
)
from src.runtime.model_registry import get_model_runtime_config

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseLLM
    from sentence_transformers import CrossEncoder

# ── 싱글턴 캐시 ──
_llm_instance: BaseLLM | None = None
_embeddings_instance: Embeddings | None = None
_reranker_instance: CrossEncoder | None = None


def get_llm(
    model_path: str | None = None,
    n_ctx: int = LLM_N_CTX,
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = LLM_TEMPERATURE,
    top_p: float = LLM_TOP_P,
    repeat_penalty: float = LLM_REPEAT_PENALTY,
    n_gpu_layers: int = -1,
) -> BaseLLM:
    """LlamaCpp LLM 싱글턴 반환.

    Args:
        model_path: Optional local GGUF path. Defaults to the selected registry config.
        n_gpu_layers: GPU에 오프로드할 레이어 수. -1이면 전체 레이어 GPU 사용.
    """
    global _llm_instance
    if _llm_instance is None:
        runtime_config = get_model_runtime_config()
        if runtime_config.backend != "llama_cpp_python":
            raise ValueError(
                "get_llm() currently supports S1000D_MODEL_BACKEND=llama_cpp_python only; "
                f"selected backend is {runtime_config.backend!r}."
            )
        resolved_model_path = model_path or runtime_config.text_model_path
        if not resolved_model_path:
            raise ValueError("No text model path configured. Set S1000D_TEXT_MODEL_PATH or GGUF_MODEL_PATH.")

        from langchain_community.llms import LlamaCpp

        _llm_instance = LlamaCpp(
            model_path=resolved_model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=["[|endofturn|]", "[|user|]"],
            verbose=False,
        )
    return _llm_instance


def get_embeddings(model_path: str | None = None) -> Embeddings:
    """HuggingFace 임베딩 싱글턴 반환."""
    global _embeddings_instance
    if _embeddings_instance is None:
        runtime_config = get_model_runtime_config()
        resolved_model = model_path or runtime_config.embedding.model

        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=resolved_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": runtime_config.embedding.normalize_embeddings},
        )
    return _embeddings_instance


def get_reranker(model_path: str | None = None) -> CrossEncoder:
    """CrossEncoder 리랭커 싱글턴 반환."""
    global _reranker_instance
    if _reranker_instance is None:
        runtime_config = get_model_runtime_config()
        resolved_model = model_path or runtime_config.reranker.model

        from sentence_transformers import CrossEncoder

        _reranker_instance = CrossEncoder(resolved_model)
    return _reranker_instance


def reset_singletons() -> None:
    """테스트용: 모든 싱글턴 초기화."""
    global _llm_instance, _embeddings_instance, _reranker_instance
    _llm_instance = None
    _embeddings_instance = None
    _reranker_instance = None
