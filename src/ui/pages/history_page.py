"""History 페이지 — 대화 이력 브라우저."""

from __future__ import annotations

import streamlit as st

from src.ui.components import placeholder_toast, render_evidence_cards
from src.ui.session_manager import (
    get_session_preview,
    get_session_relative_time,
    get_sessions,
    group_sessions_by_date,
    switch_session,
)


def render() -> None:
    """History 페이지 렌더링."""
    st.markdown(
        '<div style="font-size:24px;font-weight:800;color:#191c1e;margin-bottom:4px;">'
        "History</div>"
        '<div style="font-size:14px;color:#434655;margin-bottom:20px;">'
        "대화 이력을 탐색합니다.</div>",
        unsafe_allow_html=True,
    )

    # 세션 목록 + 상세 뷰 2열 레이아웃
    all_sessions = get_sessions(archived=False) + get_sessions(archived=True)

    if not all_sessions:
        st.info("대화 이력이 없습니다. 채팅을 시작하세요.")
        return

    list_col, detail_col = st.columns([1, 2], gap="large")

    with list_col:
        # 필터/검색 바
        search_col, action_col = st.columns([3, 1])
        with search_col:
            search = st.text_input(
                "검색",
                placeholder="Search sessions...",
                label_visibility="collapsed",
                key="history_search",
            )
        with action_col:
            if st.button("", icon=":material/filter_list:", key="history_filter"):
                placeholder_toast("필터")

        # 날짜별 그룹핑
        filtered = all_sessions
        if search:
            filtered = [
                s for s in filtered if search.lower() in s["title"].lower()
            ]

        groups = group_sessions_by_date(filtered)

        for date_label, sessions in groups.items():
            st.markdown(
                f'<div class="history-date-group">{date_label}</div>',
                unsafe_allow_html=True,
            )
            for s in sessions:
                selected = st.session_state.get("history_selected") == s["id"]
                tag = "Q&A"
                if s.get("is_archived"):
                    tag = "Archived"

                # 세션 엔트리
                cls = "history-session-entry active" if selected else "history-session-entry"
                st.markdown(
                    f"""
                    <div class="{cls}">
                        <div class="history-session-meta">
                            <span class="session-card-tag">{tag}</span>
                            <span>{get_session_relative_time(s)}</span>
                        </div>
                        <div class="history-session-title">{s['title']}</div>
                        <div class="history-session-desc">{get_session_preview(s)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                btn_cols = st.columns(4)
                with btn_cols[0]:
                    if st.button("", key=f"hsel_{s['id']}", icon=":material/visibility:", help="보기"):
                        st.session_state.history_selected = s["id"]
                        st.rerun()
                with btn_cols[1]:
                    if st.button("", key=f"hresume_{s['id']}", icon=":material/chat:", help="이어서 대화"):
                        switch_session(s["id"])
                        st.session_state.current_page = "chat"
                        st.rerun()
                with btn_cols[2]:
                    if st.button("", key=f"hstar_{s['id']}", icon=":material/star:", help="즐겨찾기"):
                        placeholder_toast("즐겨찾기")
                with btn_cols[3]:
                    if st.button("", key=f"hdl_{s['id']}", icon=":material/download:", help="다운로드"):
                        placeholder_toast("다운로드")

    with detail_col:
        # 선택된 세션 상세 뷰
        selected_id = st.session_state.get("history_selected")
        if selected_id and selected_id in st.session_state.get("sessions", {}):
            session = st.session_state.sessions[selected_id]

            st.markdown(
                f'<div style="font-size:18px;font-weight:700;color:#191c1e;margin-bottom:12px;">'
                f'{session["title"]}</div>',
                unsafe_allow_html=True,
            )

            # 메시지 표시
            for msg in session.get("messages", []):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("evidences"):
                        render_evidence_cards(msg["evidences"])
        else:
            st.markdown(
                '<div style="text-align:center;padding:80px 20px;color:#737686;">'
                '<span class="material-symbols-outlined" style="font-size:48px;display:block;margin-bottom:12px;">'
                "history</span>"
                "왼쪽에서 세션을 선택하세요.</div>",
                unsafe_allow_html=True,
            )
