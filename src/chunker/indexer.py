"""S1000DChunk → LangChain Document 변환 + 벡터 인덱스 저장.

벡터스토어는 추상화하되, 초기 구현은 ChromaDB 사용.
임베딩 모델 로딩은 rag/models.py에서 관리하므로 여기서는
Document 변환과 인덱스 저장/로드 인터페이스만 담당한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document

from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from src.types.chunk import S1000DChunk

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore


def _extract_sns_code(dmc: str) -> str:
    """DMC에서 SNS 코드 추출.

    예: "S1000DBIKE-AAA-DA1-00-00-00AA-041A-A" → "DA1"
        "BRAKE-AAA-DA1-00-00-00AA-341A-A" → "DA1"
    """
    parts = dmc.split("-")
    return parts[2] if len(parts) >= 3 else ""


def chunks_to_documents(chunks: list[S1000DChunk]) -> list[Document]:
    """S1000DChunk 리스트를 LangChain Document 리스트로 변환.

    Document.metadata에 벡터 검색 시 필터링 가능한 필드를 포함:
    - dmc, chunk_id (PK)
    - dm_type, security, applicability, sns_code (필터)
    - structure_path_range, title, issue, language (보조 정보)
    """
    docs: list[Document] = []
    for chunk in chunks:
        metadata = {
            "dmc": chunk.dmc,
            "chunk_id": chunk.chunk_id,
            "dm_type": chunk.dm_type.value,
            "security": chunk.security,
            "applicability": chunk.applicability,
            "structure_path_range": chunk.structure_path_range,
            "sns_code": _extract_sns_code(chunk.dmc),
        }
        # chunk.metadata에서 추가 필드 병합
        for key in ("title", "issue", "language", "block_count", "role_distribution"):
            if key in chunk.metadata:
                val = chunk.metadata[key]
                # ChromaDB는 dict 값을 지원하지 않으므로 문자열로 변환
                if isinstance(val, (dict, list)):
                    import json
                    metadata[key] = json.dumps(val, ensure_ascii=False)
                else:
                    metadata[key] = val

        docs.append(Document(
            page_content=chunk.text,
            metadata=metadata,
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
