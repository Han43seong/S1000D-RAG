"""벡터 검색 + 메타데이터 필터링 래퍼.

ChromaDB 벡터스토어에서 유사도 검색을 수행하고,
security / dm_type 등의 메타 필터를 적용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from src.tracing import traceable

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore


@dataclass
class MetaFilter:
    """벡터 검색 시 적용할 메타데이터 필터."""

    security: str | None = None
    dm_type: str | None = None
    dmc: str | None = None
    sns_code: str | None = None


@traceable(run_type="retriever", name="vector_search")
def retrieve(
    vectorstore: VectorStore,
    query: str,
    top_k: int = 10,
    meta_filter: MetaFilter | None = None,
) -> list[tuple[Document, float]]:
    """벡터스토어에서 유사도 검색 + 메타 필터.

    Args:
        vectorstore: LangChain VectorStore 인스턴스.
        query: 검색 쿼리 문자열.
        top_k: 반환할 상위 결과 수.
        meta_filter: 메타데이터 필터 (선택).

    Returns:
        (Document, score) 튜플 리스트. score는 유사도 점수.
    """
    where_filter = _build_where_filter(meta_filter)

    kwargs: dict = {"k": top_k}
    if where_filter:
        kwargs["filter"] = where_filter

    results = vectorstore.similarity_search_with_relevance_scores(
        query, **kwargs
    )
    return results


def _build_where_filter(meta_filter: MetaFilter | None) -> dict | None:
    """MetaFilter → ChromaDB where 절 변환."""
    if meta_filter is None:
        return None

    conditions: list[dict] = []

    if meta_filter.security:
        conditions.append({"security": meta_filter.security})
    if meta_filter.dm_type:
        conditions.append({"dm_type": meta_filter.dm_type})
    if meta_filter.dmc:
        conditions.append({"dmc": meta_filter.dmc})
    if meta_filter.sns_code:
        conditions.append({"sns_code": meta_filter.sns_code})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


@traceable(run_type="retriever", name="two_stage_search")
def retrieve_two_stage(
    vectorstore: VectorStore,
    query: str,
    top_k: int = 10,
    meta_filter: MetaFilter | None = None,
    sns_code: str | None = None,
) -> list[tuple[Document, float]]:
    """SNS 코드 기반 2단계 벡터 검색.

    sns_code가 None이면 기존 retrieve()와 동일.
    있으면:
      1단계 - SNS 필터 검색 (해당 시스템 청크 우선)
      2단계 - 글로벌 검색으로 보충 (중복 제거)

    Args:
        vectorstore: 벡터스토어 인스턴스.
        query: 검색 쿼리.
        top_k: 최종 반환 결과 수.
        meta_filter: 기본 메타 필터.
        sns_code: SNS 코드 (예: "DA1").

    Returns:
        (Document, score) 튜플 리스트.
    """
    if not sns_code:
        return retrieve(vectorstore, query, top_k, meta_filter)

    # 1단계: SNS 필터 검색
    sns_filter = MetaFilter(
        security=meta_filter.security if meta_filter else None,
        dm_type=meta_filter.dm_type if meta_filter else None,
        dmc=meta_filter.dmc if meta_filter else None,
        sns_code=sns_code,
    )
    sns_results = retrieve(vectorstore, query, top_k, sns_filter)

    if len(sns_results) >= top_k:
        return sns_results[:top_k]

    # 2단계: 글로벌 검색으로 보충
    global_results = retrieve(vectorstore, query, top_k, meta_filter)

    # 중복 제거 (chunk_id 기준)
    seen_ids: set[str] = {
        doc.metadata.get("chunk_id", "") for doc, _ in sns_results
    }
    combined = list(sns_results)

    for doc, score in global_results:
        chunk_id = doc.metadata.get("chunk_id", "")
        if chunk_id not in seen_ids:
            combined.append((doc, score))
            seen_ids.add(chunk_id)
            if len(combined) >= top_k:
                break

    return combined[:top_k]
