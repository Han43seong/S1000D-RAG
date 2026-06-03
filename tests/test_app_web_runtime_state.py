from __future__ import annotations

import time
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import app_web
from src.types.rag import RagResult


class _LockedChat:
    def locked(self) -> bool:
        return True


class _UnlockedChat:
    def locked(self) -> bool:
        return False


class _Profile:
    name = "unit_profile"
    repo_id = "unit/repo"
    display_name = "Unit Model"


class _Embedding:
    model = "unit-embedding"


class _Reranker:
    model = "unit-reranker"


class _RuntimeConfig:
    text_model_path = None
    text_profile = _Profile()
    vlm_profile = _Profile()
    embedding = _Embedding()
    reranker = _Reranker()
    backend = "llama_cpp_python"


def test_status_reports_busy_state_without_loading_models(monkeypatch):
    monkeypatch.setattr(app_web, "_get_chunk_count", lambda: 7)
    monkeypatch.setattr(
        "src.runtime.model_registry.get_model_runtime_config",
        lambda: _RuntimeConfig(),
    )
    started = time.time() - 12
    monkeypatch.setitem(app_web._chat_state, "busy", True)
    monkeypatch.setitem(app_web._chat_state, "started_at", started)

    client = TestClient(app_web.app)
    data = client.get("/api/status").json()

    assert data["ready"] is True
    assert data["busy"] is True
    assert data["busy_for_sec"] >= 10
    assert data["model_name"] == "Unit Model"


def test_chat_retries_once_on_transient_llama_decode_error(monkeypatch):
    calls = {"count": 0}

    def flaky_run_rag_query_sync(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("llama_decode returned 1")
        return RagResult(answer="재시도 후 성공", evidences=[])

    monkeypatch.setattr(app_web, "_get_models", lambda: {"vectorstore": object(), "llm": MagicMock(), "reranker": None})
    monkeypatch.setattr("src.rag.pipeline.run_rag_query_sync", flaky_run_rag_query_sync)

    response = app_web._chat_sync(app_web.ChatRequest(session_id="retry-unit", question="브레이크 설명"))

    assert calls["count"] == 2
    assert response.answer == "재시도 후 성공"
    assert app_web.sessions_db["retry-unit"]["messages"][-1]["content"] == "재시도 후 성공"


def test_chat_returns_409_when_model_is_busy(monkeypatch):
    monkeypatch.setattr(app_web, "_chat_lock", _LockedChat())

    client = TestClient(app_web.app)
    response = client.post(
        "/api/chat",
        json={"session_id": "unit", "question": "hello"},
    )

    assert response.status_code == 409
    assert "이미 답변을 생성 중" in response.json()["detail"]
