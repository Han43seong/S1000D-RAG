"""WinneAI — S1000D 기술 교범 RAG Streamlit UI.

Stitch 'WinneAI PC Chat (Light Mode)' 디자인 기반.

사전 준비:
    python ingest.py  # 인제스천은 CLI로 미리 실행

실행:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.ui.components import (
    inject_css,
    placeholder_toast,
    render_brand,
    render_footer,
    render_session_card,
)
from src.ui.session_manager import (
    archive_session,
    create_session,
    delete_session,
    get_active_session,
    get_session_preview,
    get_session_relative_time,
    get_sessions,
    init_sessions,
    switch_session,
)

# ══════════════════════════════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="WinneAI",
    page_icon=":material/smart_toy:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ══════════════════════════════════════════════════════════════════════
# 세션 초기화
# ══════════════════════════════════════════════════════════════════════

init_sessions()

if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"

# ══════════════════════════════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    render_brand()

    st.markdown("")

    # ── New Chat 버튼 ──
    if st.button(
        "New Chat",
        key="new_chat",
        use_container_width=True,
        type="primary",
        icon=":material/add:",
    ):
        create_session()
        st.session_state.current_page = "chat"
        st.rerun()

    st.markdown("")

    # ── 네비게이션 ──
    current_page = st.session_state.current_page

    nav_items = [
        ("chat", "Chat", "chat"),
        ("history", "History", "history"),
        ("auto_stories", "Library", None),
        ("psychology", "Models", None),
        ("extension", "Plugins", None),
    ]

    for icon, label, page_key in nav_items:
        is_active = current_page == page_key if page_key else False
        btn_type = "primary" if is_active else "secondary"
        if st.button(
            label,
            key=f"nav_{label.lower()}",
            use_container_width=True,
            icon=f":material/{icon}:",
            type=btn_type if is_active else "secondary",
        ):
            if page_key:
                st.session_state.current_page = page_key
                st.rerun()
            else:
                placeholder_toast(label)

    st.divider()

    # ── 세션 리스트 ──
    st.markdown(
        '<div class="sidebar-section-header">Recent Sessions</div>',
        unsafe_allow_html=True,
    )

    sessions = get_sessions(archived=False)
    active_session = get_active_session()
    active_id = active_session["id"] if active_session else None

    for s in sessions[:8]:  # 최대 8개 표시
        is_active = s["id"] == active_id

        col1, col2 = st.columns([5, 1])
        with col1:
            render_session_card(
                title=s["title"],
                preview=get_session_preview(s),
                time_str=get_session_relative_time(s),
                active=is_active,
            )
            if st.button(
                " ",
                key=f"switch_{s['id']}",
                use_container_width=True,
            ):
                switch_session(s["id"])
                st.session_state.current_page = "chat"
                st.rerun()
        with col2:
            if st.button(
                "",
                key=f"del_{s['id']}",
                icon=":material/close:",
                help="삭제",
            ):
                delete_session(s["id"])
                st.rerun()

    # 아카이브 세션 링크
    archived = get_sessions(archived=True)
    if archived:
        if st.button(
            f"Archived ({len(archived)})",
            key="show_archived",
            use_container_width=True,
            icon=":material/inventory_2:",
        ):
            st.session_state.current_page = "history"
            st.rerun()

    st.divider()

    # ── 하단 네비게이션 ──
    if st.button(
        "Settings",
        key="nav_settings",
        use_container_width=True,
        icon=":material/settings:",
        type="primary" if current_page == "settings" else "secondary",
    ):
        st.session_state.current_page = "settings"
        st.rerun()

    bottom_items = [("shield", "Privacy"), ("gavel", "Legal")]
    for icon, label in bottom_items:
        if st.button(
            label,
            key=f"nav_{label.lower()}",
            use_container_width=True,
            icon=f":material/{icon}:",
        ):
            placeholder_toast(label)


# ══════════════════════════════════════════════════════════════════════
# 메인 영역 — 페이지 라우팅
# ══════════════════════════════════════════════════════════════════════

page = st.session_state.current_page

if page == "chat":
    from src.ui.pages.chat_page import render as render_chat

    render_chat()

elif page == "history":
    from src.ui.pages.history_page import render as render_history

    render_history()

elif page == "settings":
    from src.ui.pages.settings_page import render as render_settings

    render_settings()

# 하단 면책 문구
render_footer()
