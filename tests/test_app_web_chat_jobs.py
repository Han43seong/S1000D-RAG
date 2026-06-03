from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

import app_web


def _reset_jobs(monkeypatch):
    monkeypatch.setattr(app_web, "chat_jobs", {})
    monkeypatch.setattr(app_web, "_active_job_id", None)
    monkeypatch.setattr(app_web, "_schedule_chat_job", lambda job_id: None)


def test_format_answer_for_display_removes_trailing_evidence_line():
    raw = "브레이크 케이블은 레버 힘을 브레이크 패드로 전달합니다.\n근거: BRAKE-AAA-DA1-00-00-00AA-041A-A"

    answer = app_web._format_answer_for_display(raw)

    assert answer == "브레이크 케이블은 레버 힘을 브레이크 패드로 전달합니다."


def test_format_answer_for_display_removes_trailing_reference_document_line():
    raw = "브레이크 시스템 구성품은 레버, 케이블, 암, 클램프, 패드입니다.\n참고 문서: BRAKE-AAA-DA1-00-00-00AA-041A-A"

    answer = app_web._format_answer_for_display(raw)

    assert answer == "브레이크 시스템 구성품은 레버, 케이블, 암, 클램프, 패드입니다."


def test_format_answer_for_display_keeps_reference_document_words_in_body_text():
    raw = "정비자는 관련 참고 문서를 확인한 뒤 절차를 수행해야 합니다."

    answer = app_web._format_answer_for_display(raw)

    assert answer == raw


def test_format_answer_for_display_removes_multiple_trailing_metadata_lines():
    raw = "브레이크 패드는 림을 눌러 감속합니다.\n근거: DMC-000\n참고 문서： BRAKE-AAA-DA1-00-00-00AA-041A-A"

    answer = app_web._format_answer_for_display(raw)

    assert answer == "브레이크 패드는 림을 눌러 감속합니다."


def test_format_answer_for_display_keeps_non_trailing_reference_metadata_like_text():
    raw = "참고 문서: BRAKE-AAA-DA1-00-00-00AA-041A-A\n이 문서는 브레이크 시스템 구성품을 설명합니다."

    answer = app_web._format_answer_for_display(raw)

    assert answer == raw


def test_format_answer_for_display_removes_truncated_table_tail():
    raw = """브레이크 패드 청소 절차는 다음과 같이 표로 정리할 수 있습니다:

| 단계 | 작업 내용 |
|------|-------------|
| 1. 시각 점검 | 사전 탑승 점검에 따라 브레이크를 시각적으로 점검합니다. |
| 2. 청소 | 각 브레이크 패드를 찾아 청소합니다. |
| 3. 표면 마찰 | 표면을 문지르면서 특정 물질을 패드 전체 표면에 적용합니다. |

| 3. 표면 마찰 | 표면을 문지르면서 특정 물질을 패드 전체 表面上에 적용합니다. |

| 1. 시각 점검 | 사전 탑승 점검에 따라 브레이크를 시각
근거: BRAKE-AAA-DA1-10-00-00AA-251A-A"""

    answer = app_web._format_answer_for_display(raw)

    assert "근거:" not in answer
    assert "表面上" not in answer
    assert not answer.endswith("시각")
    assert answer.endswith("| 3. 표면 마찰 | 표면을 문지르면서 특정 물질을 패드 전체 표면에 적용합니다. |")


def test_format_answer_for_display_removes_restarted_answer_tail():
    raw = (
        "브레이크 케이블은 브레이크 레버를 당기면 케이블이 당겨져 브레이크의 두 레버를 함께 당깁니다. "
        "이로 인해 브레이크 패드가 바퀴의 외곽 휠 림에 마찰력을 발생시켜 자전거의 속도를 줄입니다.\n"
        "브레이크 케이블은 브레이크 레버를 당기면 케이블이 당겨져 브레이크의 두 레"
    )

    answer = app_web._format_answer_for_display(raw)

    assert answer == (
        "브레이크 케이블은 브레이크 레버를 당기면 케이블이 당겨져 브레이크의 두 레버를 함께 당깁니다. "
        "이로 인해 브레이크 패드가 바퀴의 외곽 휠 림에 마찰력을 발생시켜 자전거의 속도를 줄입니다."
    )


def test_format_answer_for_display_removes_short_incomplete_tail():
    raw = "브레이크 패드 청소 절차는 1) 점검 2) 청소 3) 표면 마찰 순서입니다.\n브"

    answer = app_web._format_answer_for_display(raw)

    assert answer == "브레이크 패드 청소 절차는 1) 점검 2) 청소 3) 표면 마찰 순서입니다."


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


def test_chat_response_serializes_reference_materials_separately():
    response = app_web.ChatResponse(
        answer="자연어 답변만",
        evidences=[],
        reference_materials={
            "warnings": [
                {"id": "warning:1", "label": "warning", "type": "Warning", "text": "Wear goggles"}
            ],
            "figures": [
                {"id": "figure:1", "label": "Brake figure", "title": "Brake figure", "type": "Figure"}
            ],
        },
    )

    payload = response.model_dump()

    assert payload["answer"] == "자연어 답변만"
    assert payload["reference_materials"]["warnings"][0]["text"] == "Wear goggles"
    assert payload["reference_materials"]["figures"][0]["title"] == "Brake figure"


def test_run_chat_job_records_reference_materials(monkeypatch):
    monkeypatch.setattr(app_web, "chat_jobs", {})
    monkeypatch.setattr(app_web, "_active_job_id", None)

    async def fake_chat(req):
        return app_web.ChatResponse(
            answer=f"answer:{req.question}",
            evidences=[],
            reference_materials={"warnings": [{"id": "warning:1", "label": "warning", "type": "Warning"}]},
            llm_sec=1.25,
        )

    monkeypatch.setattr(app_web, "chat", fake_chat)
    now = "2026-06-01T00:00:00"
    app_web.chat_jobs["job-ref"] = {
        "job_id": "job-ref",
        "session_id": "unit-session",
        "question": "테스트 질문",
        "request": {"session_id": "unit-session", "question": "테스트 질문"},
        "status": "queued",
        "progress": "queued",
        "answer": None,
        "evidences": [],
        "reference_materials": {},
        "llm_sec": 0,
        "error": None,
        "cancel_requested": False,
        "created_at": now,
        "updated_at": now,
    }

    asyncio.run(app_web._run_chat_job("job-ref"))

    job = app_web.chat_jobs["job-ref"]
    assert job["status"] == "done"
    assert job["reference_materials"]["warnings"][0]["id"] == "warning:1"


def test_static_app_renders_reference_materials_panel():
    js = open("static/app.js", encoding="utf-8").read()

    assert "referenceMaterials" in js
    assert "참고자료" in js
    assert "안전주의" in js
    assert "관련 그림" in js


def test_chat_sync_persists_reference_materials_in_session(monkeypatch):
    from src.types.rag import RagResult, ReferenceMaterials, ReferenceMaterialItem
    import src.rag.pipeline as pipeline

    monkeypatch.setattr(app_web, "sessions_db", {})
    monkeypatch.setattr(app_web, "_get_models", lambda: {"vectorstore": object(), "llm": object(), "reranker": None})

    def fake_run_rag_query_sync(**kwargs):
        return RagResult(
            answer="자연어 답변만",
            evidences=[],
            reference_materials=ReferenceMaterials(
                warnings=[ReferenceMaterialItem(id="warning:1", label="warning", type="Warning", text="Wear goggles")]
            ),
        )

    monkeypatch.setattr(pipeline, "run_rag_query_sync", fake_run_rag_query_sync)

    response = app_web._chat_sync(app_web.ChatRequest(session_id="session-ref", question="질문"))

    assert response.reference_materials.warnings[0].id == "warning:1"
    message = app_web.sessions_db["session-ref"]["messages"][-1]
    assert message["content"] == "자연어 답변만"
    assert message["reference_materials"]["warnings"][0]["id"] == "warning:1"
