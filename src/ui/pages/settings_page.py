"""Settings 페이지 — RAG/LLM 파라미터 + 시스템 정보."""

from __future__ import annotations

import streamlit as st

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    GGUF_MODEL_PATH,
    LLM_MAX_TOKENS,
    LLM_REPEAT_PENALTY,
    LLM_TEMPERATURE,
    MAX_CONTEXT_CHARS,
    RERANK_TOP_K,
    RELEVANCE_THRESHOLD,
    VECTOR_CANDIDATE_K,
)
from src.ui.components import placeholder_toast, render_info_card


def _get_chunk_count() -> int:
    try:
        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        col = client.get_collection(CHROMA_COLLECTION_NAME)
        return col.count()
    except Exception:
        return 0


def render() -> None:
    """Settings 페이지 렌더링."""
    # 헤더
    st.markdown(
        '<div style="font-size:24px;font-weight:800;color:#191c1e;margin-bottom:4px;">'
        "Settings</div>"
        '<div style="font-size:14px;color:#434655;margin-bottom:20px;">'
        "RAG 파이프라인 파라미터를 조정합니다.</div>",
        unsafe_allow_html=True,
    )

    # 카테고리 + 설정 패널 2열 레이아웃
    cat_col, main_col = st.columns([1, 3], gap="large")

    with cat_col:
        st.markdown('<div style="font-size:12px;font-weight:700;color:#737686;'
                    'text-transform:uppercase;letter-spacing:0.5px;padding:8px 0;">'
                    'CATEGORIES</div>', unsafe_allow_html=True)

        categories = [
            ("palette", "Appearance", True),
            ("person", "Account", False),
            ("shield", "Security", False),
            ("notifications", "Notifications", False),
            ("extension", "Integrations", False),
        ]

        active_cat = st.session_state.get("settings_category", "Appearance")

        for icon, label, available in categories:
            cls = "settings-category active" if label == active_cat else "settings-category"
            if st.button(
                label,
                key=f"cat_{label}",
                use_container_width=True,
                icon=f":material/{icon}:",
            ):
                if available:
                    st.session_state.settings_category = label
                    st.rerun()
                else:
                    placeholder_toast(label)

    with main_col:
        if active_cat == "Appearance":
            _render_appearance_settings()

    # 하단 버전 배지
    st.markdown("")
    st.markdown(
        '<div style="text-align:center;padding:16px 0;">'
        '<span class="version-badge">WinneAI v1.0.0</span></div>',
        unsafe_allow_html=True,
    )


def _render_appearance_settings() -> None:
    """Appearance 설정 패널."""
    # Interface Customization
    st.markdown(
        '<div style="font-size:18px;font-weight:700;color:#191c1e;margin-bottom:4px;">'
        "Interface Customization</div>"
        '<div style="font-size:13px;color:#434655;margin-bottom:16px;">'
        "인터페이스 설정을 조정합니다.</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        dark_mode = st.toggle("Dark Mode", value=False, key="setting_dark_mode", disabled=True)
        st.caption("준비 중 (Recommended)")
    with col2:
        pass

    st.divider()

    # ── Search (Retrieval) Parameters ──
    st.markdown(
        '<div style="font-size:16px;font-weight:700;color:#191c1e;margin-bottom:12px;">'
        ":material/search: Search (Retrieval)</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=30,
            value=st.session_state.get("setting_top_k", VECTOR_CANDIDATE_K),
            key="setting_top_k",
            help="벡터 검색 후보 수",
        )
    with col2:
        threshold = st.slider(
            "Relevance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("setting_relevance_threshold", float(RELEVANCE_THRESHOLD)),
            step=0.05,
            key="setting_relevance_threshold",
            help="이 값 미만의 검색 결과는 제외됩니다.",
        )

    col1, col2 = st.columns(2)
    with col1:
        rerank_k = st.slider(
            "Rerank Top-K",
            min_value=1,
            max_value=10,
            value=st.session_state.get("setting_rerank_top_k", RERANK_TOP_K),
            key="setting_rerank_top_k",
            help="리랭킹 후 상위 K개 선택",
        )
    with col2:
        max_ctx = st.slider(
            "Max Context (chars)",
            min_value=1000,
            max_value=30000,
            value=st.session_state.get("setting_max_context", MAX_CONTEXT_CHARS),
            step=1000,
            key="setting_max_context",
            help="LLM에 전달할 최대 컨텍스트 길이",
        )

    st.divider()

    # ── LLM Generation Parameters ──
    st.markdown(
        '<div style="font-size:16px;font-weight:700;color:#191c1e;margin-bottom:12px;">'
        ":material/psychology: LLM Generation</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        max_tokens = st.slider(
            "Max Tokens",
            min_value=64,
            max_value=2048,
            value=st.session_state.get("setting_max_tokens", LLM_MAX_TOKENS),
            step=64,
            key="setting_max_tokens",
            help="응답 최대 토큰 수",
        )
    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get("setting_temperature", float(LLM_TEMPERATURE)),
            step=0.05,
            key="setting_temperature",
            help="낮을수록 결정적, 높을수록 창의적",
        )

    repeat_penalty = st.slider(
        "Repeat Penalty",
        min_value=1.0,
        max_value=1.5,
        value=st.session_state.get("setting_repeat_penalty", float(LLM_REPEAT_PENALTY)),
        step=0.05,
        key="setting_repeat_penalty",
        help="반복 토큰 억제 강도",
    )

    st.divider()

    # ── System Info ──
    st.markdown(
        '<div style="font-size:16px;font-weight:700;color:#191c1e;margin-bottom:12px;">'
        ":material/info: System Info</div>",
        unsafe_allow_html=True,
    )

    from pathlib import Path

    model_name = Path(GGUF_MODEL_PATH).stem
    chunk_count = _get_chunk_count()

    col1, col2, col3 = st.columns(3)
    with col1:
        render_info_card("LLM Model", model_name)
    with col2:
        render_info_card("Embedding", "BGE-m3-ko")
    with col3:
        render_info_card("Index", f"{chunk_count} chunks")
