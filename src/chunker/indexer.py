"""S1000DChunk → LangChain Document 변환 + 벡터 인덱스 저장.

벡터스토어는 추상화하되, 초기 구현은 ChromaDB 사용.
임베딩 모델 로딩은 rag/models.py에서 관리하므로 여기서는
Document 변환과 인덱스 저장/로드 인터페이스만 담당한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from src.types.chunk import S1000DChunk

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore


@dataclass(frozen=True)
class LightweightDocument:
    """Small LangChain-compatible stand-in for dependency-light tests.

    Production indexing environments should have ``langchain_core`` installed,
    in which case :func:`chunks_to_documents` returns real LangChain Documents.
    """

    page_content: str
    metadata: dict[str, Any]


def _extract_sns_code(dmc: str) -> str:
    """DMC에서 SNS 코드 추출.

    예: "DMC-S1000DBIKE-AAA-DA1-00-00-00AA-041A-A" → "DA1"
        "S1000DBIKE-AAA-DA1-00-00-00AA-041A-A" → "DA1"
        "BRAKE-AAA-DA1-00-00-00AA-341A-A" → "DA1"
    """
    parts = dmc.split("-")
    if parts and parts[0].casefold() == "dmc":
        return parts[3] if len(parts) >= 4 else ""
    return parts[2] if len(parts) >= 3 else ""


def _primitive_metadata_value(value: Any) -> str | int | float | bool:
    """Return a Chroma-compatible primitive metadata value.

    Chroma metadata values must be primitive scalars. S1000D evidence fields
    such as block IDs and role distributions are still valuable, so structured
    values are JSON encoded deterministically instead of dropped.
    """
    if value is None:
        return ""
    if isinstance(value, (bool, int, float, str)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_chunk_metadata(chunk: S1000DChunk) -> dict[str, str | int | float | bool]:
    """Build normalized, Chroma-compatible S1000D text chunk metadata."""
    raw_metadata: dict[str, Any] = {
        "dmc": chunk.dmc,
        "chunk_id": chunk.chunk_id,
        "dm_type": chunk.dm_type.value,
        "security": chunk.security,
        "applicability": chunk.applicability,
        "sns_code": chunk.metadata.get("sns_code") or _extract_sns_code(chunk.dmc),
        "issue": chunk.metadata.get("issue", ""),
        "language": chunk.metadata.get("language", ""),
        "title": chunk.metadata.get("title", ""),
        "structure_path_range": chunk.structure_path_range,
        "block_ids": chunk.metadata.get("block_ids", []),
        "role_distribution": chunk.metadata.get("role_distribution", {}),
        "source_file": chunk.metadata.get("source_file") or chunk.metadata.get("source_path", ""),
        "source_path": chunk.metadata.get("source_path") or chunk.metadata.get("source_file", ""),
        "modality": "text",
    }
    if "block_count" in chunk.metadata:
        raw_metadata["block_count"] = chunk.metadata["block_count"]
    return {key: _primitive_metadata_value(value) for key, value in raw_metadata.items()}


def _document_class() -> type:
    try:
        from langchain_core.documents import Document

        return Document
    except ImportError:
        return LightweightDocument


def chunks_to_documents(chunks: list[S1000DChunk]) -> list["Document"]:
    """S1000DChunk 리스트를 LangChain Document 리스트로 변환.

    Document.metadata에 벡터 검색/표시에 필요한 S1000D 근거 필드를 포함한다.
    LangChain이 설치되지 않은 테스트 환경에서는 동일 속성을 가진
    LightweightDocument를 반환한다.
    """
    document_cls = _document_class()
    docs: list[Document] = []
    for chunk in chunks:
        docs.append(document_cls(
            page_content=chunk.text,
            metadata=build_chunk_metadata(chunk),
        ))
    return docs


def build_chroma_index(
    documents: list[Document],
    embedding_fn: Embeddings,
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> VectorStore:
    """LangChain Document 리스트를 ChromaDB에 인덱싱.

    Args:
        documents: 인덱싱할 Document 리스트.
        embedding_fn: LangChain 호환 임베딩 함수.
        persist_directory: ChromaDB 영속화 디렉터리.
        collection_name: ChromaDB 컬렉션 이름.

    Returns:
        초기화된 VectorStore 인스턴스.
    """
    from langchain_chroma import Chroma

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_fn,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return vectorstore


def load_chroma_index(
    embedding_fn: Embeddings,
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> VectorStore:
    """기존 ChromaDB 인덱스 로드.

    Args:
        embedding_fn: LangChain 호환 임베딩 함수.
        persist_directory: ChromaDB 영속화 디렉터리.
        collection_name: ChromaDB 컬렉션 이름.

    Returns:
        로드된 VectorStore 인스턴스.
    """
    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_fn,
    )
