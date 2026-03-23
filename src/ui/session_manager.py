"""세션 관리 모듈.

session_state 기반 멀티 세션 CRUD.
향후 SQLite 영구 저장 확장 가능.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import streamlit as st


def _now() -> datetime:
    return datetime.now()


def _relative_time(dt: datetime) -> str:
    """상대 시간 문자열 반환."""
    diff = _now() - dt
    if diff < timedelta(minutes=1):
        return "방금 전"
    if diff < timedelta(hours=1):
        return f"{int(diff.total_seconds() // 60)}분 전"
    if diff < timedelta(days=1):
        return f"{int(diff.total_seconds() // 3600)}시간 전"
    if diff < timedelta(days=2):
        return "어제"
    if diff < timedelta(days=7):
        return f"{diff.days}일 전"
    return dt.strftime("%m/%d")


def init_sessions() -> None:
    """세션 상태 초기화."""
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    if "active_session_id" not in st.session_state:
        # 첫 세션 자동 생성
        create_session()


def create_session() -> str:
    """새 세션 생성. 세션 ID 반환."""
    session_id = str(uuid.uuid4())[:8]
    now = _now()
    st.session_state.sessions[session_id] = {
        "id": session_id,
        "title": "새 대화",
        "messages": [],
        "created_at": now,
        "updated_at": now,
        "is_archived": False,
    }
    st.session_state.active_session_id = session_id
    return session_id


def get_active_session() -> dict | None:
    """현재 활성 세션 반환."""
    sid = st.session_state.get("active_session_id")
    if sid and sid in st.session_state.get("sessions", {}):
        return st.session_state.sessions[sid]
    return None


def get_active_messages() -> list[dict]:
    """활성 세션의 메시지 리스트 반환."""
    session = get_active_session()
    return session["messages"] if session else []


def add_message(role: str, content: str, **kwargs) -> None:
    """활성 세션에 메시지 추가."""
    session = get_active_session()
    if not session:
        return
    msg = {"role": role, "content": content, **kwargs}
    session["messages"].append(msg)
    session["updated_at"] = _now()
    # 첫 사용자 메시지로 세션 제목 자동 설정
    if role == "user" and session["title"] == "새 대화":
        session["title"] = content[:30] + ("..." if len(content) > 30 else "")


def switch_session(session_id: str) -> None:
    """세션 전환. 빈 세션 자동 정리."""
    _cleanup_empty_sessions(exclude=session_id)
    if session_id in st.session_state.sessions:
        st.session_state.active_session_id = session_id


def delete_session(session_id: str) -> None:
    """세션 삭제."""
    if session_id in st.session_state.sessions:
        del st.session_state.sessions[session_id]
        if st.session_state.get("active_session_id") == session_id:
            remaining = [
                s for s in st.session_state.sessions
                if not st.session_state.sessions[s]["is_archived"]
            ]
            if remaining:
                st.session_state.active_session_id = remaining[0]
            else:
                create_session()


def archive_session(session_id: str) -> None:
    """세션 아카이브."""
    if session_id in st.session_state.sessions:
        st.session_state.sessions[session_id]["is_archived"] = True


def unarchive_session(session_id: str) -> None:
    """세션 아카이브 해제."""
    if session_id in st.session_state.sessions:
        st.session_state.sessions[session_id]["is_archived"] = False


def get_sessions(archived: bool = False) -> list[dict]:
    """세션 목록 반환 (최신순 정렬)."""
    sessions = [
        s for s in st.session_state.get("sessions", {}).values()
        if s["is_archived"] == archived
    ]
    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return sessions


def get_session_relative_time(session: dict) -> str:
    """세션 상대 시간 반환."""
    return _relative_time(session["updated_at"])


def get_session_preview(session: dict) -> str:
    """세션 미리보기 텍스트 반환."""
    msgs = session.get("messages", [])
    if not msgs:
        return "대화를 시작하세요..."
    last = msgs[-1]
    text = last.get("content", "")
    return text[:50] + ("..." if len(text) > 50 else "")


def _cleanup_empty_sessions(exclude: str | None = None) -> None:
    """메시지가 없는 세션 자동 삭제."""
    current = st.session_state.get("active_session_id")
    to_delete = []
    for sid, session in st.session_state.sessions.items():
        if sid == exclude or sid == current:
            continue
        if not session["messages"] and not session["is_archived"]:
            to_delete.append(sid)
    for sid in to_delete:
        del st.session_state.sessions[sid]


def group_sessions_by_date(sessions: list[dict]) -> dict[str, list[dict]]:
    """세션을 날짜별로 그룹핑."""
    groups: dict[str, list[dict]] = {}
    today = _now().date()
    yesterday = today - timedelta(days=1)

    for s in sessions:
        d = s["updated_at"].date()
        if d == today:
            key = "Today"
        elif d == yesterday:
            key = "Yesterday"
        else:
            key = d.strftime("%m월 %d일")
        groups.setdefault(key, []).append(s)
    return groups
