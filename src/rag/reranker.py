"""CrossEncoder 기반 리랭커 hook.

벡터 검색 결과를 CrossEncoder로 재정렬하여 정밀도를 높인다.
on/off 가능한 hook 구조: RerankOptions.enabled=False이면 패스스루.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langsmith import traceable

from src.types.rag import RerankOptions

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@traceable(run_type="chain", name="rerank")
def rerank(
    query: str,
    doc_score_pairs: list[tuple[Document, float]],
    options: RerankOptions,
    cross_encoder: CrossEncoder | None = None,
) -> list[tuple[Document, float]]:
    """검색 결과를 리랭킹하여 상위 top_k 반환.

    Args:
        query: 원본 쿼리.
        doc_score_pairs: (Document, 벡터 유사도 점수) 리스트.
        options: 리랭커 설정 (enabled, top_k, model_path).
        cross_encoder: 주입할 CrossEncoder 인스턴스 (None이면 자동 로드).

    Returns:
        리랭킹된 (Document, rerank_score) 리스트 (상위 top_k개).
    """
    if not options.enabled or not doc_score_pairs:
        return doc_score_pairs[:options.top_k]

    if cross_encoder is None:
        from src.rag.models import get_reranker
        cross_encoder = get_reranker(options.model_path)

    # CrossEncoder 입력 준비: (query, document_text) 쌍
    pairs = [(query, doc.page_content) for doc, _ in doc_score_pairs]
    scores = cross_encoder.predict(pairs)

    # score와 함께 정렬
    scored = list(zip(doc_score_pairs, scores))
    scored.sort(key=lambda x: float(x[1]), reverse=True)

    # 상위 top_k 반환, rerank score로 교체
    return [
        (doc, float(rerank_score))
        for (doc, _), rerank_score in scored[:options.top_k]
    ]
