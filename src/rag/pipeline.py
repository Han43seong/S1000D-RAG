"""RAG 파이프라인 메인 함수: run_rag_query().

Retriever → (Reranker) → LLM 컨텍스트 구성 → LLM 호출 → RagResult 반환.
LangGraph 노드로 감쌀 수 있도록 I/O를 명확히 정의한 순수 함수 인터페이스.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langsmith import traceable

from src.config import MAX_CONTEXT_CHARS, RERANK_TOP_K, VECTOR_CANDIDATE_K
from src.types.rag import Evidence, RagOptions, RagResult, RerankOptions, SessionMeta
from .prompt import build_prompt
from .query_enhancer import enhance_query
from .reranker import rerank
from .retriever import MetaFilter, retrieve, retrieve_two_stage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLLM
    from langchain_core.vectorstores import VectorStore
    from sentence_transformers import CrossEncoder


@traceable(run_type="chain", name="rag_pipeline_async")
async def run_rag_query(
    query: str,
    vectorstore: VectorStore,
    llm: BaseLLM,
    session_meta: SessionMeta | None = None,
    options: RagOptions | None = None,
    cross_encoder: CrossEncoder | None = None,
    conversation_history: list[tuple[str, str]] | None = None,
) -> RagResult:
    """RAG 파이프라인 메인 함수.

    Args:
        query: 사용자 질의.
        vectorstore: 초기화된 벡터스토어.
        llm: LangChain 호환 LLM.
        session_meta: 세션 메타데이터 (권한, 언어 등).
        options: RAG 옵션 (top_k, rerank, max_context_chars 등).
        cross_encoder: 리랭커 인스턴스 (None이면 자동 로드 또는 스킵).
        conversation_history: 최근 대화 이력 [(user, assistant), ...].

    Returns:
        RagResult(answer, evidences).
    """
    opts = options or RagOptions()

    # ── 1. 쿼리 확장 + SNS 코드 추출 ──
    enhanced = enhance_query(query) if opts.expand_query else None
    search_query = enhanced.expanded if enhanced else query
    sns_code = enhanced.sns_code if (enhanced and opts.sns_filter) else None

    # ── 2. 메타 필터 구성 ──
    meta_filter = _build_meta_filter(session_meta)

    # ── 3. 벡터 검색 (2단계 또는 기존) ──
    if sns_code:
        candidates = retrieve_two_stage(
            vectorstore=vectorstore,
            query=search_query,
            top_k=opts.top_k,
            meta_filter=meta_filter,
            sns_code=sns_code,
        )
    else:
        candidates = retrieve(
            vectorstore=vectorstore,
            query=search_query,
            top_k=opts.top_k,
            meta_filter=meta_filter,
        )

    if not candidates:
        return RagResult(
            answer="제공된 문서에서 해당 정보를 찾을 수 없습니다.",
            evidences=[],
        )

    # ── 4. 리랭킹 (원본 쿼리 사용 → precision 유지) ──
    ranked = rerank(
        query=query,
        doc_score_pairs=candidates,
        options=opts.rerank,
        cross_encoder=cross_encoder,
    )

    # ── 5. 관련도 임계값 필터링 (리랭크 이후) ──
    ranked = _apply_threshold_with_fallback(ranked, candidates, opts)

    if not ranked:
        return RagResult(
            answer="제공된 문서에서 해당 정보를 찾을 수 없습니다.",
            evidences=[],
        )

    # ── 6. 컨텍스트 구성 ──
    context, evidences = _build_context(ranked, opts.max_context_chars)

    # ── 7. LLM 호출 (원본 쿼리 + 대화 이력) ──
    prompt_text = build_prompt(
        question=query,
        context=context,
        conversation_history=conversation_history,
    )
    answer = await llm.ainvoke(prompt_text)

    if hasattr(answer, "content"):
        answer_text = answer.content
    else:
        answer_text = str(answer)

    answer_text = _strip_think_tags(answer_text)

    return RagResult(
        answer=answer_text.strip(),
        evidences=evidences,
    )


@traceable(run_type="chain", name="rag_pipeline")
def run_rag_query_sync(
    query: str,
    vectorstore: VectorStore,
    llm: BaseLLM,
    session_meta: SessionMeta | None = None,
    options: RagOptions | None = None,
    cross_encoder: CrossEncoder | None = None,
    conversation_history: list[tuple[str, str]] | None = None,
) -> RagResult:
    """동기 버전 RAG 파이프라인."""
    opts = options or RagOptions()

    # ── 1. 쿼리 확장 + SNS 코드 추출 ──
    enhanced = enhance_query(query) if opts.expand_query else None
    search_query = enhanced.expanded if enhanced else query
    sns_code = enhanced.sns_code if (enhanced and opts.sns_filter) else None

    # ── 2. 메타 필터 구성 ──
    meta_filter = _build_meta_filter(session_meta)

    # ── 3. 벡터 검색 (2단계 또는 기존) ──
    if sns_code:
        candidates = retrieve_two_stage(
            vectorstore=vectorstore,
            query=search_query,
            top_k=opts.top_k,
            meta_filter=meta_filter,
            sns_code=sns_code,
        )
    else:
        candidates = retrieve(
            vectorstore=vectorstore,
            query=search_query,
            top_k=opts.top_k,
            meta_filter=meta_filter,
        )

    if not candidates:
        return RagResult(
            answer="제공된 문서에서 해당 정보를 찾을 수 없습니다.",
            evidences=[],
        )

    # ── 4. 리랭킹 (원본 쿼리 사용) ──
    ranked = rerank(
        query=query,
        doc_score_pairs=candidates,
        options=opts.rerank,
        cross_encoder=cross_encoder,
    )

    # 관련도 임계값 필터링 (리랭크 이후, 벡터 점수 폴백)
    ranked = _apply_threshold_with_fallback(ranked, candidates, opts)

    if not ranked:
        return RagResult(
            answer="제공된 문서에서 해당 정보를 찾을 수 없습니다.",
            evidences=[],
        )

    context, evidences = _build_context(ranked, opts.max_context_chars)

    # ── 5. LLM 호출 (원본 쿼리 + 대화 이력) ──
    prompt_text = build_prompt(
        question=query,
        context=context,
        conversation_history=conversation_history,
    )
    answer = llm.invoke(prompt_text)

    if hasattr(answer, "content"):
        answer_text = answer.content
    else:
        answer_text = str(answer)

    answer_text = _strip_think_tags(answer_text)

    return RagResult(
        answer=answer_text.strip(),
        evidences=evidences,
    )


# ═══════════════════════════════════════════════════════════════════════
# 내부 헬퍼
# ═══════════════════════════════════════════════════════════════════════


def _strip_think_tags(text: str) -> str:
    """LLM 출력 후처리 (모델 공통)."""
    # 1. <think>...</think> 블록 제거 (Qwen3 호환). max_tokens로 닫히지 않은
    #    사고 과정도 최종 답변에 노출하지 않도록 제거한다.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    # 2. EXAONE 특수 토큰 제거
    text = re.sub(r"\[?\|endofturn\|]?", "", text)
    text = re.sub(r"\[?\|assistant\|]?", "", text)
    # 3. 영어 사고 과정 블록 제거 (줄의 시작이 영어 추론 패턴이면 해당 줄부터 끝까지 삭제)
    text = _truncate_at_english_reasoning(text)
    # 4. 마크다운 헤더 제거
    text = re.sub(r"^#{1,3}\s+.*$", "", text, flags=re.MULTILINE)
    # 5. 메타 접두사 (Answer:, 답변: 등)
    text = re.sub(r"^(Answer|answer|Antwort|답변|정답)\s*:?\s*", "", text, flags=re.MULTILINE)
    # 6. 메타 라벨 줄 (질문:, 참고 문서: 등)
    text = re.sub(r"^(질문|참고 문서|요구 사항|참고 문서의 DMC|예시)\s*[:：].*$", "", text, flags=re.MULTILINE)
    # 7. 일본어 카타카나/히라가나/태국어 혼입 정리
    text = re.sub(r"[\u30A0-\u30FF\u3040-\u309F]", "", text)  # 가타카나+히라가나
    text = re.sub(r"[\u0E00-\u0E7F]", "", text)  # 태국어
    # 8. 모델 아티팩트 정리 (코드/템플릿 잔여물)
    text = re.sub(r"['\"]?[a-z_]+['\"]?\)\s*\}\}", "", text)  # 'brake cable') }}
    text = re.sub(r"_SAME LEVEL_", "", text)
    # 9. 남은 연속 빈줄 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 9. 반복 절삭
    text = _truncate_repetition(text)
    return text.strip()


_ENGLISH_REASONING_RE = re.compile(
    r"^(Okay|Let me|First,|I need to|Wait,|Hmm|The user|Now,|In summary)",
)


def _truncate_at_english_reasoning(text: str) -> str:
    """영어 사고 과정이 시작되면 해당 줄부터 끝까지 잘라낸다."""
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and _ENGLISH_REASONING_RE.match(stripped):
            break
        kept.append(line)
    return "\n".join(kept)


def _truncate_repetition(text: str) -> str:
    """반복된 문단을 감지하여 첫 등장까지만 유지."""
    paragraphs = text.split("\n\n")
    if len(paragraphs) <= 1:
        return text

    seen: set[str] = set()
    kept: list[str] = []
    for p in paragraphs:
        normalized = p.strip()
        if not normalized:
            continue
        # 핵심 내용 기준 (앞 20자, 공백/기호 무시)
        key = re.sub(r"[\s\-\*#>]", "", normalized)[:20]
        if key in seen:
            break
        seen.add(key)
        kept.append(p)

    return "\n\n".join(kept)


def _apply_threshold_with_fallback(
    ranked: list[tuple[Document, float]],
    candidates: list[tuple[Document, float]],
    opts: RagOptions,
) -> list[tuple[Document, float]]:
    """리랭크 결과에 threshold를 적용하되, 전수 탈락 시 벡터 점수로 폴백.

    리랭커가 cross-lingual 쿼리에서 모든 점수를 0으로 매기는 경우,
    원본 벡터 검색 결과 중 threshold 이상인 것으로 폴백한다.
    """
    if opts.relevance_threshold <= 0:
        return ranked

    filtered = [(doc, score) for doc, score in ranked if score >= opts.relevance_threshold]
    if filtered:
        return filtered

    # 리랭크 전수 탈락 → 벡터 점수로 폴백
    fallback = [(doc, score) for doc, score in candidates if score >= opts.relevance_threshold]
    if fallback:
        return fallback[:opts.rerank.top_k]

    # 일부 Chroma/embedding 조합은 relevance score 대신 음수 distance-like 값을
    # 반환한다. 이 경우 임계값 비교가 의미 없으므로 검색 순위를 신뢰해 top_k를
    # 보존한다. 전체 파이프라인 smoke/eval에서 유효 문서가 전수 탈락하는 것을
    # 방지하되, 후보가 없는 경우에는 기존처럼 no-answer로 처리된다.
    if candidates and all(score < 0 for _, score in candidates):
        return candidates[:opts.rerank.top_k]

    return []


def _build_meta_filter(session_meta: SessionMeta | None) -> MetaFilter | None:
    """SessionMeta에서 MetaFilter 생성."""
    if session_meta is None:
        return None

    return MetaFilter(
        security=session_meta.security_clearance,
    )


@traceable(run_type="chain", name="build_context")
def _build_context(
    ranked_docs: list[tuple[Document, float]],
    max_chars: int,
) -> tuple[str, list[Evidence]]:
    """리랭킹된 문서들로 LLM 컨텍스트 문자열 + Evidence 리스트 생성.

    max_chars를 초과하지 않도록 문서를 순서대로 추가.
    """
    context_parts: list[str] = []
    evidences: list[Evidence] = []
    total_chars = 0

    for doc, score in ranked_docs:
        meta = doc.metadata
        chunk_text = doc.page_content

        if total_chars + len(chunk_text) > max_chars and context_parts:
            break

        # 컨텍스트에 출처 정보 포함
        header = f"[DMC: {meta.get('dmc', '?')} | Type: {meta.get('dm_type', '?')}]"
        context_parts.append(f"{header}\n{chunk_text}")
        total_chars += len(chunk_text) + len(header) + 1

        evidences.append(Evidence(
            dmc=meta.get("dmc", ""),
            chunk_id=meta.get("chunk_id", ""),
            score=round(score, 4),
            dm_type=meta.get("dm_type"),
            security=meta.get("security"),
            applicability=meta.get("applicability"),
        ))

    context = "\n\n---\n\n".join(context_parts)
    return context, evidences
