"""Chat 페이지 — 메인 Q&A 인터페이스."""

from __future__ import annotations

import time

import streamlit as st

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    MAX_CONVERSATION_HISTORY_TURNS,
)
from src.ui.components import (
    render_empty_state,
    render_evidence_cards,
    render_perf_metrics,
    render_topbar,
)
from src.ui.session_manager import add_message, get_active_messages


# ── 모델 캐시 ──


@st.cache_resource(show_spinner="임베딩 모델 로딩 중...")
def _get_embeddings():
    from src.rag.models import get_embeddings

    return get_embeddings()


@st.cache_resource(show_spinner="LLM 로딩 중...")
def _get_llm():
    from src.rag.models import get_llm

    return get_llm()


@st.cache_resource(show_spinner="리랭커 로딩 중...")
def _get_reranker():
    from src.rag.models import get_reranker

    return get_reranker()


def _load_vectorstore():
    from src.chunker.indexer import load_chroma_index

    emb = _get_embeddings()
    return load_chroma_index(embedding_fn=emb)


def _get_chunk_count() -> int:
    try:
        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        col = client.get_collection(CHROMA_COLLECTION_NAME)
        return col.count()
    except Exception:
        return 0


# ── 설정 기본값 ──


def _get_setting(key: str, default):
    return st.session_state.get(f"setting_{key}", default)


# ── 페이지 렌더링 ──


def render() -> None:
    """Chat 페이지 렌더링."""
    chunk_count = _get_chunk_count()

    # Top bar
    render_topbar(model_name="Qwen3-14B", ready=chunk_count > 0)

    # Delete 버튼 처리
    if st.session_state.get("_delete_chat"):
        st.session_state._delete_chat = False
        from src.ui.session_manager import get_active_session

        session = get_active_session()
        if session:
            session["messages"] = []
            session["title"] = "새 대화"
        st.rerun()

    messages = get_active_messages()

    # 빈 상태
    if not messages:
        example_q = render_empty_state()
        if example_q:
            st.session_state._pending_question = example_q
            st.rerun()

    # 대화 히스토리 표시
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("evidences"):
                render_evidence_cards(msg["evidences"])
            if msg["role"] == "assistant" and (msg.get("query_ms") or msg.get("llm_sec")):
                render_perf_metrics(msg.get("query_ms"), msg.get("llm_sec"))

    # pending 질문 처리 (예시 질문 클릭 시)
    pending = st.session_state.pop("_pending_question", None)

    # 입력
    question = st.chat_input(
        "S1000D 기술 교범에 대해 질문하세요...",
        key="chat_input",
    )

    if pending:
        question = pending

    if question:
        _handle_question(question, chunk_count)


def _handle_question(question: str, chunk_count: int) -> None:
    """질문 처리 + RAG 파이프라인 실행."""
    # 사용자 메시지 추가
    add_message("user", question)
    with st.chat_message("user"):
        st.markdown(question)

    if chunk_count == 0:
        answer = "인덱스가 없습니다. `python ingest.py`로 인제스천을 먼저 실행하세요."
        add_message("assistant", answer)
        with st.chat_message("assistant"):
            st.markdown(answer)
        return

    # RAG 파이프라인 실행
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            from src.rag.pipeline_v2 import run_rag_query_sync
            from src.types.rag import RagOptions, RerankOptions

            top_k = _get_setting("top_k", 10)
            relevance_threshold = _get_setting("relevance_threshold", 0.3)
            rerank_top_k = _get_setting("rerank_top_k", 3)
            max_context = _get_setting("max_context", 10000)

            llm = _get_llm()
            vectorstore = _load_vectorstore()
            cross_encoder = _get_reranker()

            options = RagOptions(
                top_k=top_k,
                relevance_threshold=relevance_threshold,
                rerank=RerankOptions(enabled=True, top_k=rerank_top_k),
                max_context_chars=max_context,
            )

            # 대화 이력 추출
            messages = get_active_messages()
            history: list[tuple[str, str]] = []
            for i in range(len(messages) - 1, -1, -1):
                if (
                    messages[i]["role"] == "assistant"
                    and i > 0
                    and messages[i - 1]["role"] == "user"
                ):
                    history.insert(0, (messages[i - 1]["content"], messages[i]["content"]))
                    if len(history) >= MAX_CONVERSATION_HISTORY_TURNS:
                        break

            t0 = time.time()
            result = run_rag_query_sync(
                query=question,
                vectorstore=vectorstore,
                llm=llm,
                options=options,
                cross_encoder=cross_encoder,
                conversation_history=history if history else None,
            )
            llm_sec = time.time() - t0

        st.markdown(result.answer)

        # Evidence 데이터 구성
        evidences_data = []
        if result.evidences:
            for ev in result.evidences:
                ev_dict = {
                    "dmc": ev.dmc,
                    "score": ev.score,
                    "dm_type": ev.dm_type.value if ev.dm_type else None,
                    "security": ev.security,
                    "title": getattr(ev, "title", ""),
                    "text": getattr(ev, "text", ""),
                }
                evidences_data.append(ev_dict)
            render_evidence_cards(evidences_data)

        render_perf_metrics(llm_sec=llm_sec)

        add_message(
            "assistant",
            result.answer,
            evidences=evidences_data,
            llm_sec=llm_sec,
        )
