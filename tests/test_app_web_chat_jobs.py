from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

import app_web


def _reset_jobs(monkeypatch):
    monkeypatch.setattr(app_web, "chat_jobs", {})
    monkeypatch.setattr(app_web, "_active_job_id", None)
    monkeypatch.setattr(app_web, "_schedule_chat_job", lambda job_id: None)


def test_create_chat_job_returns_job_id_without_running_llm(monkeypatch):
    _reset_jobs(monkeypatch)

    client = TestClient(app_web.app)
    response = client.post(
        "/api/chat/jobs",
        json={"session_id": "unit-session", "question": "브레이크 절차는?"},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["job_id"]
    assert data["status"] == "queued"
    assert data["progress"] == "queued"
    assert data["session_id"] == "unit-session"
    assert data["question"] == "브레이크 절차는?"
    assert data["answer"] is None
    assert data["error"] is None


def test_get_chat_job_status_returns_existing_job(monkeypatch):
    _reset_jobs(monkeypatch)
    client = TestClient(app_web.app)
    created = client.post(
        "/api/chat/jobs",
        json={"session_id": "unit-session", "question": "DMC가 뭐야?"},
    ).json()

    response = client.get(f"/api/chat/jobs/{created['job_id']}")

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == created["job_id"]
    assert data["status"] == "queued"
    assert data["progress"] == "queued"


def test_cancel_queued_chat_job_marks_cancelled(monkeypatch):
    _reset_jobs(monkeypatch)
    client = TestClient(app_web.app)
    created = client.post(
        "/api/chat/jobs",
        json={"session_id": "unit-session", "question": "취소 테스트"},
    ).json()

    response = client.delete(f"/api/chat/jobs/{created['job_id']}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cancelled"
    assert data["progress"] == "cancelled"


def test_create_chat_job_rejects_when_another_job_is_active(monkeypatch):
    _reset_jobs(monkeypatch)
    client = TestClient(app_web.app)
    created = client.post(
        "/api/chat/jobs",
        json={"session_id": "unit-session", "question": "첫 질문"},
    ).json()
    monkeypatch.setattr(app_web, "_active_job_id", created["job_id"])

    response = client.post(
        "/api/chat/jobs",
        json={"session_id": "unit-session", "question": "두 번째 질문"},
    )

    assert response.status_code == 409
    assert "이미 답변을 생성 중" in response.json()["detail"]


def test_run_chat_job_records_done_result(monkeypatch):
    monkeypatch.setattr(app_web, "chat_jobs", {})
    monkeypatch.setattr(app_web, "_active_job_id", None)

    async def fake_chat(req):
        return app_web.ChatResponse(answer=f"answer:{req.question}", evidences=[], llm_sec=1.25)

    monkeypatch.setattr(app_web, "chat", fake_chat)
    now = "2026-06-01T00:00:00"
    app_web.chat_jobs["job1"] = {
        "job_id": "job1",
        "session_id": "unit-session",
        "question": "테스트 질문",
        "request": {"session_id": "unit-session", "question": "테스트 질문"},
        "status": "queued",
        "progress": "queued",
        "answer": None,
        "evidences": [],
        "llm_sec": 0,
        "error": None,
        "cancel_requested": False,
        "created_at": now,
        "updated_at": now,
    }

    asyncio.run(app_web._run_chat_job("job1"))

    job = app_web.chat_jobs["job1"]
    assert job["status"] == "done"
    assert job["progress"] == "done"
    assert job["answer"] == "answer:테스트 질문"
    assert job["llm_sec"] == 1.25
    assert app_web._active_job_id is None
