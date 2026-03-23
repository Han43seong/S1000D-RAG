"""WinneAI UI 컴포넌트 모듈.

Stitch 'WinneAI PC Chat (Light Mode)' 디자인 기반 재사용 가능 컴포넌트.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


_CSS_PATH = Path(__file__).resolve().parent.parent.parent / "static" / "style.css"


def inject_css() -> None:
    """CSS 테마 주입."""
    css = _CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_brand() -> None:
    """사이드바 상단 브랜드 로고."""
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">W</div>
            <div class="sidebar-brand-text">
                <span class="sidebar-brand-name">WinneAI</span>
                <span class="sidebar-brand-subtitle">Professional Assistant</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_nav_item(icon: str, label: str, key: str, active: bool = False) -> bool:
    """사이드바 네비게이션 항목. 클릭 시 True 반환."""
    cls = "nav-item active" if active else "nav-item"
    col1, col2 = st.columns([1, 0.001], gap="small")
    with col1:
        clicked = st.button(
            f":{icon}: {label}" if not icon.startswith("material") else label,
            key=f"nav_{key}",
            use_container_width=True,
        )
    return clicked


def render_topbar(model_name: str = "Qwen3-14B", ready: bool = True) -> None:
    """상단 바: 모델 상태 + 액션 버튼."""
    status_cls = "ready" if ready else "loading"
    status_text = "Ready" if ready else "Loading..."
    dot_cls = "ready" if ready else "loading"

    left_col, right_col = st.columns([3, 1])
    with left_col:
        st.markdown(
            f"""
            <div class="topbar-left">
                <span class="topbar-title">WinneAI</span>
                <span class="topbar-status {status_cls}">
                    <span class="topbar-status-dot {dot_cls}"></span>
                    Model: {model_name} &middot; {status_text}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_col:
        btn_cols = st.columns(4)
        with btn_cols[0]:
            if st.button("", key="btn_share", help="Share", icon=":material/share:"):
                st.toast("공유 기능은 준비 중입니다.", icon=":material/info:")
        with btn_cols[1]:
            if st.button("", key="btn_star", help="Favorite", icon=":material/star:"):
                st.toast("즐겨찾기 기능은 준비 중입니다.", icon=":material/info:")
        with btn_cols[2]:
            if st.button("", key="btn_delete", help="Clear chat", icon=":material/delete:"):
                return  # handled by caller
        with btn_cols[3]:
            if st.button("", key="btn_more", help="More", icon=":material/more_vert:"):
                st.toast("추가 옵션은 준비 중입니다.", icon=":material/info:")


def render_empty_state() -> str | None:
    """빈 상태 환영 카드 + 예시 질문. 클릭된 질문 반환."""
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">W</div>
            <div class="empty-state-title">WinneAI</div>
            <div class="empty-state-subtitle">
                S1000D 기술 교범 데이터 모듈에 대해 질문해 보세요.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    examples = [
        ("자전거의 주요 구성품은 무엇인가요?", "auto_awesome"),
        ("브레이크 시스템 원리에 대해 알려줘", "psychology"),
        ("앞바퀴 설치 절차를 알려줘", "build"),
    ]

    cols = st.columns(len(examples))
    for i, (q, icon) in enumerate(examples):
        with cols[i]:
            if st.button(
                q,
                key=f"example_q_{i}",
                use_container_width=True,
                icon=f":material/{icon}:",
            ):
                return q
    return None


def render_evidence_cards(evidences: list[dict]) -> None:
    """Evidence 카드 목록 렌더링."""
    if not evidences:
        return

    with st.expander(f":material/description: 참고 문서 ({len(evidences)}건)", expanded=False):
        for i, ev in enumerate(evidences, 1):
            score = ev.get("score", 0)
            pct = f"{score * 100:.0f}%"
            if score >= 0.6:
                score_cls = "high"
            elif score >= 0.45:
                score_cls = "medium"
            else:
                score_cls = "low"

            dmc = ev.get("dmc", "-")
            dm_type = ev.get("dm_type", "-") or "-"
            title = ev.get("title", "")
            text_preview = ev.get("text", "")[:150]

            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-card-header">
                        <span class="evidence-rank">#{i}</span>
                        <span class="evidence-dmc">{dmc}</span>
                        <span class="evidence-score {score_cls}">{pct}</span>
                        <span class="evidence-type-badge">{dm_type}</span>
                    </div>
                    {f'<div class="evidence-preview">{text_preview}...</div>' if text_preview else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_perf_metrics(query_ms: float | None = None, llm_sec: float | None = None) -> None:
    """성능 메트릭 표시."""
    parts = []
    if query_ms is not None:
        parts.append(f'<span class="material-symbols-outlined">search</span> 검색 {query_ms:.0f}ms')
    if llm_sec is not None:
        parts.append(f'<span class="material-symbols-outlined">psychology</span> 추론 {llm_sec:.1f}s')
    if parts:
        st.markdown(
            f'<div class="perf-metrics">{" &middot; ".join(parts)}</div>',
            unsafe_allow_html=True,
        )


def render_session_card(
    title: str,
    preview: str,
    time_str: str,
    active: bool = False,
    tag: str = "Q&A",
) -> None:
    """세션 카드 HTML 렌더링."""
    cls = "session-card active" if active else "session-card"
    st.markdown(
        f"""
        <div class="{cls}">
            <div class="session-card-header">
                <span class="session-card-tag">{tag}</span>
                <span class="session-card-time">{time_str}</span>
            </div>
            <div class="session-card-title">{title}</div>
            <div class="session-card-preview">{preview}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(label: str, value: str) -> None:
    """시스템 정보 카드 (Settings 페이지용)."""
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-label">{label}</div>
            <div class="info-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """하단 면책 문구."""
    st.markdown(
        '<div class="footer-disclaimer">'
        "WinneAI can make mistakes. Verify important information."
        "</div>",
        unsafe_allow_html=True,
    )


def placeholder_toast(feature: str) -> None:
    """미구현 기능 토스트."""
    st.toast(f"{feature} 기능은 준비 중입니다.", icon=":material/info:")
