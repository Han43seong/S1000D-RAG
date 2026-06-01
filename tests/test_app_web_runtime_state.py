from __future__ import annotations

import time

from fastapi.testclient import TestClient

import app_web


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


def test_chat_returns_409_when_model_is_busy(monkeypatch):
    monkeypatch.setattr(app_web, "_chat_lock", _LockedChat())

    client = TestClient(app_web.app)
    response = client.post(
        "/api/chat",
        json={"session_id": "unit", "question": "hello"},
    )

    assert response.status_code == 409
    assert "이미 답변을 생성 중" in response.json()["detail"]
