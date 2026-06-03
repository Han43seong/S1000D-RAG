from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
S1000D_DATA_DIR = Path(os.getenv(
    "S1000D_DATA_DIR",
    str(PROJECT_ROOT / "docs" / "S1000D Issue 6" / "Bike Data Set for Release number 6 R2"),
))

# --- Model registry / runtime selection ---
S1000D_TEXT_MODEL_PROFILE = os.getenv("S1000D_TEXT_MODEL_PROFILE", "qwen3_8b_q5")
S1000D_VLM_MODEL_PROFILE = os.getenv("S1000D_VLM_MODEL_PROFILE", "qwen3_vl_8b_q4")
S1000D_MODEL_BACKEND = os.getenv("S1000D_MODEL_BACKEND", "llama_cpp_python")

# New S1000D_* variables take precedence; legacy names are kept for callers and local setups.
S1000D_TEXT_MODEL_PATH = os.getenv(
    "S1000D_TEXT_MODEL_PATH",
    os.getenv("GGUF_MODEL_PATH", str(MODELS_DIR / "llm" / "qwen3-8b" / "Qwen3-8B-Q5_K_M.gguf")),
)
S1000D_VLM_MODEL_PATH = os.getenv("S1000D_VLM_MODEL_PATH", "")
S1000D_VLM_MMPROJ_PATH = os.getenv("S1000D_VLM_MMPROJ_PATH", "")
S1000D_EMBEDDING_MODEL = os.getenv("S1000D_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-m3"))
S1000D_RERANKER_MODEL = os.getenv("S1000D_RERANKER_MODEL", os.getenv("RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3"))

GGUF_MODEL_PATH = S1000D_TEXT_MODEL_PATH
EMBEDDING_MODEL_PATH = S1000D_EMBEDDING_MODEL
RERANKER_MODEL_PATH = S1000D_RERANKER_MODEL
NLI_MODEL_PATH = os.getenv("NLI_MODEL_PATH", str(MODELS_DIR / "mDeBERTa-v3-base-nli"))

def _env_int(name: str, default: int, *, legacy_name: str | None = None) -> int:
    raw_value = os.getenv(name)
    if raw_value is None and legacy_name:
        raw_value = os.getenv(legacy_name)
    if raw_value is None or raw_value == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw_value!r}") from exc


# --- LLM ---
LLM_N_CTX = _env_int("S1000D_LLM_N_CTX", 8192, legacy_name="S1000D_LLM_CONTEXT_WINDOW")
LLM_MAX_TOKENS = _env_int("S1000D_LLM_MAX_TOKENS", 1024)
LLM_TEMPERATURE = _env_float("S1000D_LLM_TEMPERATURE", 0.2)
LLM_TOP_P = _env_float("S1000D_LLM_TOP_P", 0.9)
LLM_REPEAT_PENALTY = _env_float("S1000D_LLM_REPEAT_PENALTY", 1.05)

# --- Chunking ---
CHUNK_BLOCK_COUNT = 5
CHUNK_MAX_SIZE = 1500
CHUNK_OVERLAP = 1

# --- Retrieval ---
VECTOR_CANDIDATE_K = 10
RERANK_TOP_K = 3
RELEVANCE_THRESHOLD = 0.3
MAX_CONTEXT_CHARS = 10000

# --- Conversation History ---
MAX_CONVERSATION_HISTORY_TURNS = 2

# --- ChromaDB ---
# Default to the full-corpus index.  Operators can still point the app/CLI back
# to the smaller smoke index with S1000D_CHROMA_* env vars when needed.
CHROMA_PERSIST_DIR = os.getenv("S1000D_CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db_full"))
CHROMA_COLLECTION_NAME = os.getenv("S1000D_CHROMA_COLLECTION_NAME", "s1000d_chunks_full")

# --- Graph-first retrieval ---
S1000D_GRAPH_MANIFEST_PATH = os.getenv(
    "S1000D_GRAPH_MANIFEST_PATH",
    str(Path(CHROMA_PERSIST_DIR) / "s1000d_graph_manifest.json"),
)
