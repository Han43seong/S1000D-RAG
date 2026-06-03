"""WinneAI — FastAPI 백엔드 서버.

Stitch 'WinneAI PC Chat (Light Mode)' HTML을 정적 파일로 서빙하고,
RAG 파이프라인 API를 제공합니다.

실행:
    python app_web.py
    또는
    uvicorn app_web:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.types.rag import ReferenceMaterials

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    LLM_MAX_TOKENS,
    LLM_N_CTX,
    LLM_REPEAT_PENALTY,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    MAX_CONTEXT_CHARS,
    MAX_CONVERSATION_HISTORY_TURNS,
    RERANK_TOP_K,
    RELEVANCE_THRESHOLD,
    VECTOR_CANDIDATE_K,
)

# ══════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════

logger = logging.getLogger("winneai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="WinneAI", version="1.0.0")
_chat_lock = asyncio.Lock()
_chat_state = {"busy": False, "started_at": None, "question": None}
chat_jobs: dict[str, dict] = {}
_active_job_id: str | None = None


@app.on_event("startup")
async def _log_model_config():
    """서버 시작 시 모델을 로드하지 않고 선택된 프로필만 로깅."""
    from src.runtime.model_registry import get_model_runtime_config

    cfg = get_model_runtime_config()
    logger.info(
        "Model config selected: backend=%s text_profile=%s vlm_profile=%s",
        cfg.backend,
        cfg.text_profile.name,
        cfg.vlm_profile.name,
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ══════════════════════════════════════════════════════════════════════
# Models (lazy init)
# ══════════════════════════════════════════════════════════════════════

_models: dict = {}


def _get_models():
    """모델 싱글턴 로딩."""
    if "llm" not in _models:
        from src.rag.models import get_embeddings, get_llm, get_reranker

        _models["llm"] = get_llm()
        _models["embeddings"] = get_embeddings()
        _models["reranker"] = get_reranker()

        from src.chunker.indexer import load_chroma_index

        _models["vectorstore"] = load_chroma_index(embedding_fn=_models["embeddings"])
    return _models


def _get_chunk_count() -> int:
    try:
        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        col = client.get_collection(CHROMA_COLLECTION_NAME)
        return col.count()
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════
# In-memory sessions
# ══════════════════════════════════════════════════════════════════════

sessions_db: dict[str, dict] = {}


def _create_session() -> dict:
    sid = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    session = {
        "id": sid,
        "title": "새 대화",
        "messages": [],
        "created_at": now,
        "updated_at": now,
        "is_archived": False,
    }
    sessions_db[sid] = session
    return session


def _relative_time(iso_str: str) -> str:
    dt = datetime.fromisoformat(iso_str)
    diff = datetime.now() - dt
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


# ══════════════════════════════════════════════════════════════════════
# API Schemas
# ══════════════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = VECTOR_CANDIDATE_K
    relevance_threshold: float = float(RELEVANCE_THRESHOLD)
    rerank_top_k: int = RERANK_TOP_K
    max_context: int = MAX_CONTEXT_CHARS


class EvidenceResponse(BaseModel):
    rank: int
    dmc: str
    score: float
    dm_type: str | None = None
    text: str = ""
    modality: str | None = None
    content_role: str | None = None
    asset_key: str | None = None
    asset_path: str | None = None
    caption_path: str | None = None
    title: str | None = None
    kind: str | None = None
    ref_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    evidences: list[EvidenceResponse] = Field(default_factory=list)
    reference_materials: ReferenceMaterials = Field(default_factory=ReferenceMaterials)
    llm_sec: float = 0


def _format_answer_for_display(answer: str) -> str:
    """Remove trailing evidence metadata and obvious generation tails.

    The internal RAG/LLM answer may carry source metadata for tracing and QA, but
    the chat bubble should show only the answer text.  Source DMCs and ontology
    evidence are rendered separately below the answer via the existing reference
    dropdowns.  Local GGUF outputs can also end with a partially generated
    markdown table row; remove those display-only tails rather than showing a
    broken final line to the user.
    """
    lines = answer.rstrip().split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and _is_trailing_answer_metadata_line(lines[-1]):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()

    cleaned: list[str] = []
    seen_nonempty: set[str] = set()
    has_korean = bool(re.search(r"[가-힣]", answer))
    for line in lines:
        stripped = line.strip()
        if has_korean and re.search(r"[\u4E00-\u9FFF]", stripped):
            continue
        if stripped.startswith("|") and not stripped.endswith("|"):
            continue
        normalized = re.sub(r"\s+", " ", stripped)
        if normalized and normalized in seen_nonempty:
            continue
        if normalized:
            seen_nonempty.add(normalized)
        cleaned.append(line.rstrip())

    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return _strip_restarted_answer_tail("\n".join(cleaned).strip())


def _is_trailing_answer_metadata_line(line: str) -> bool:
    """Return True for source metadata lines that belong in dropdowns, not UI text."""
    stripped = line.strip()
    return bool(re.match(r"^(근거|참고 문서)\s*[:：]\s*.*$", stripped))


def _strip_restarted_answer_tail(text: str) -> str:
    """Remove a trailing fragment that restarts the answer from the beginning."""
    current = text.strip()
    while True:
        lines = current.split("\n")
        last_index = next((idx for idx in range(len(lines) - 1, -1, -1) if lines[idx].strip()), None)
        if last_index is None:
            return ""
        tail = lines[last_index].strip()
        prior = "\n".join(lines[:last_index]).strip()
        if not prior:
            return current
        normalized_tail = re.sub(r"\s+", " ", tail)
        normalized_prior = re.sub(r"\s+", " ", prior)
        # The local model sometimes starts the whole answer again and then hits
        # max_tokens. If the final non-empty line is a prefix of earlier answer
        # text, drop that line and keep the completed earlier answer.
        if normalized_tail[: min(24, len(normalized_tail))] in normalized_prior:
            lines.pop(last_index)
            while lines and not lines[-1].strip():
                lines.pop()
            current = "\n".join(lines).strip()
            continue
        if len(normalized_tail) < 8 and not re.search(r"[.!?。！？]|[가-힣]다\.?$", normalized_tail):
            lines.pop(last_index)
            while lines and not lines[-1].strip():
                lines.pop()
            current = "\n".join(lines).strip()
            continue
        return current


class ChatJobResponse(BaseModel):
    job_id: str
    session_id: str
    question: str
    status: str
    progress: str
    answer: str | None = None
    evidences: list[EvidenceResponse] = Field(default_factory=list)
    reference_materials: ReferenceMaterials = Field(default_factory=ReferenceMaterials)
    llm_sec: float = 0
    error: str | None = None
    created_at: str
    updated_at: str


class SessionResponse(BaseModel):
    id: str
    title: str
    messages: list[dict]
    created_at: str
    updated_at: str
    is_archived: bool
    relative_time: str = ""
    preview: str = ""


class StatusResponse(BaseModel):
    model_name: str
    embedding_model: str
    chunk_count: int
    ready: bool
    busy: bool = False
    busy_for_sec: float = 0
    backend: str
    text_profile: str
    text_repo_id: str
    vlm_profile: str
    vlm_repo_id: str
    reranker_model: str


class ConfigResponse(BaseModel):
    temperature: float
    top_p: float
    repeat_penalty: float
    max_tokens: int
    n_ctx: int
    top_k: int
    relevance_threshold: float
    rerank_top_k: int
    max_context_chars: int


# ══════════════════════════════════════════════════════════════════════
# Routes — Pages
# ══════════════════════════════════════════════════════════════════════


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# ══════════════════════════════════════════════════════════════════════
# Routes — API
# ══════════════════════════════════════════════════════════════════════


@app.get("/api/status")
async def get_status() -> StatusResponse:
    from src.runtime.model_registry import get_model_runtime_config

    cfg = get_model_runtime_config()
    model_name = Path(cfg.text_model_path).stem if cfg.text_model_path else cfg.text_profile.display_name
    chunk_count = _get_chunk_count()
    started_at = _chat_state.get("started_at")
    busy_for_sec = time.time() - started_at if _chat_state.get("busy") and started_at else 0
    return StatusResponse(
        model_name=model_name,
        embedding_model=cfg.embedding.model,
        chunk_count=chunk_count,
        ready=chunk_count > 0,
        busy=bool(_chat_state.get("busy")),
        busy_for_sec=busy_for_sec,
        backend=cfg.backend,
        text_profile=cfg.text_profile.name,
        text_repo_id=cfg.text_profile.repo_id,
        vlm_profile=cfg.vlm_profile.name,
        vlm_repo_id=cfg.vlm_profile.repo_id,
        reranker_model=cfg.reranker.model,
    )


@app.get("/api/config")
async def get_config() -> ConfigResponse:
    return ConfigResponse(
        temperature=LLM_TEMPERATURE,
        top_p=LLM_TOP_P,
        repeat_penalty=LLM_REPEAT_PENALTY,
        max_tokens=LLM_MAX_TOKENS,
        n_ctx=LLM_N_CTX,
        top_k=VECTOR_CANDIDATE_K,
        relevance_threshold=RELEVANCE_THRESHOLD,
        rerank_top_k=RERANK_TOP_K,
        max_context_chars=MAX_CONTEXT_CHARS,
    )


@app.post("/api/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    if _chat_lock.locked():
        raise HTTPException(
            status_code=409,
            detail="모델이 이미 답변을 생성 중입니다. 현재 CPU-only 27B 실행은 1~2분 이상 걸릴 수 있습니다.",
        )

    async with _chat_lock:
        _chat_state.update({"busy": True, "started_at": time.time(), "question": req.question[:120]})
        try:
            return await asyncio.to_thread(_chat_sync, req)
        finally:
            _chat_state.update({"busy": False, "started_at": None, "question": None})


def _chat_sync(req: ChatRequest) -> ChatResponse:
    # 세션 확인
    session = sessions_db.get(req.session_id)
    if not session:
        now = datetime.now().isoformat()
        session = {
            "id": req.session_id,
            "title": "새 대화",
            "messages": [],
            "created_at": now,
            "updated_at": now,
            "is_archived": False,
        }
        sessions_db[req.session_id] = session

    # 사용자 메시지 추가
    session["messages"].append({"role": "user", "content": req.question})
    session["updated_at"] = datetime.now().isoformat()
    if session["title"] == "새 대화":
        session["title"] = req.question[:30] + ("..." if len(req.question) > 30 else "")

    # 모델 로딩
    models = _get_models()

    from src.rag.pipeline import run_rag_query_sync
    from src.types.rag import RagOptions, RerankOptions

    options = RagOptions(
        top_k=req.top_k,
        relevance_threshold=req.relevance_threshold,
        rerank=RerankOptions(enabled=True, top_k=req.rerank_top_k),
        max_context_chars=req.max_context,
    )

    # 대화 이력 추출
    history: list[tuple[str, str]] = []
    msgs = session["messages"]
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "assistant" and i > 0 and msgs[i - 1]["role"] == "user":
            history.insert(0, (msgs[i - 1]["content"], msgs[i]["content"]))
            if len(history) >= MAX_CONVERSATION_HISTORY_TURNS:
                break

    logger.info("Query: %s | session=%s", req.question[:60], req.session_id)
    t0 = time.time()
    result = run_rag_query_sync(
        query=req.question,
        vectorstore=models["vectorstore"],
        llm=models["llm"],
        options=options,
        cross_encoder=models["reranker"],
        conversation_history=history if history else None,
    )
    llm_sec = time.time() - t0
    logger.info("Done: %.1fs | evidences=%d | answer_len=%d", llm_sec, len(result.evidences) if result.evidences else 0, len(result.answer))

    # Evidence 구성
    evidences = []
    if result.evidences:
        for i, ev in enumerate(result.evidences, 1):
            evidences.append(
                EvidenceResponse(
                    rank=i,
                    dmc=ev.dmc,
                    score=ev.score,
                    dm_type=ev.dm_type.value if ev.dm_type else None,
                    text=(ev.text or "")[:200],
                    modality=ev.modality,
                    content_role=ev.content_role,
                    asset_key=ev.asset_key,
                    asset_path=ev.asset_path,
                    caption_path=ev.caption_path,
                    title=ev.title,
                    kind=ev.kind,
                    ref_id=ev.ref_id,
                )
            )

    display_answer = _format_answer_for_display(result.answer)

    # 어시스턴트 메시지 추가
    session["messages"].append({
        "role": "assistant",
        "content": display_answer,
        "evidences": [e.model_dump() for e in evidences],
        "reference_materials": result.reference_materials.model_dump(),
        "llm_sec": llm_sec,
    })
    session["updated_at"] = datetime.now().isoformat()

    return ChatResponse(answer=display_answer, evidences=evidences, reference_materials=result.reference_materials, llm_sec=llm_sec)


def _job_response(job: dict) -> ChatJobResponse:
    return ChatJobResponse(
        job_id=job["job_id"],
        session_id=job["session_id"],
        question=job["question"],
        status=job["status"],
        progress=job["progress"],
        answer=job.get("answer"),
        evidences=job.get("evidences") or [],
        reference_materials=job.get("reference_materials") or ReferenceMaterials(),
        llm_sec=job.get("llm_sec") or 0,
        error=job.get("error"),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


def _schedule_chat_job(job_id: str) -> None:
    asyncio.create_task(_run_chat_job(job_id))


async def _run_chat_job(job_id: str) -> None:
    global _active_job_id
    job = chat_jobs[job_id]
    _active_job_id = job_id
    job["status"] = "running"
    job["progress"] = "generating"
    job["updated_at"] = datetime.now().isoformat()
    try:
        if job.get("cancel_requested"):
            job["status"] = "cancelled"
            job["progress"] = "cancelled"
            return
        result = await chat(ChatRequest(**job["request"]))
        if job.get("cancel_requested"):
            job["status"] = "cancelled"
            job["progress"] = "cancelled"
            return
        job["status"] = "done"
        job["progress"] = "done"
        job["answer"] = result.answer
        job["evidences"] = result.evidences
        job["reference_materials"] = result.reference_materials.model_dump()
        job["llm_sec"] = result.llm_sec
    except Exception as exc:  # pragma: no cover - defensive runtime path
        logger.exception("Chat job failed: %s", job_id)
        job["status"] = "error"
        job["progress"] = "error"
        job["error"] = str(exc)
    finally:
        job["updated_at"] = datetime.now().isoformat()
        if _active_job_id == job_id:
            _active_job_id = None


@app.post("/api/chat/jobs", status_code=202)
async def create_chat_job(req: ChatRequest) -> ChatJobResponse:
    global _active_job_id
    if _active_job_id is not None:
        raise HTTPException(
            status_code=409,
            detail="모델이 이미 답변을 생성 중입니다. 현재 작업이 끝난 뒤 다시 질문하세요.",
        )
    job_id = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    job = {
        "job_id": job_id,
        "session_id": req.session_id,
        "question": req.question,
        "request": req.model_dump(),
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
    chat_jobs[job_id] = job
    _active_job_id = job_id
    _schedule_chat_job(job_id)
    return _job_response(job)


@app.get("/api/chat/jobs/{job_id}")
async def get_chat_job(job_id: str) -> ChatJobResponse:
    job = chat_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Chat job not found")
    return _job_response(job)


@app.delete("/api/chat/jobs/{job_id}")
async def cancel_chat_job(job_id: str) -> ChatJobResponse:
    job = chat_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Chat job not found")
    if job["status"] in {"queued", "running"}:
        job["cancel_requested"] = True
        job["status"] = "cancelled"
        job["progress"] = "cancelled"
        job["updated_at"] = datetime.now().isoformat()
        global _active_job_id
        if _active_job_id == job_id:
            _active_job_id = None
    return _job_response(job)


# ── Sessions API ──


@app.get("/api/sessions")
async def list_sessions() -> list[SessionResponse]:
    result = []
    for s in sorted(sessions_db.values(), key=lambda x: x["updated_at"], reverse=True):
        msgs = s.get("messages", [])
        if not msgs:  # Skip empty sessions
            continue
        last = msgs[-1]
        text = last.get("content", "")
        preview = text[:50] + ("..." if len(text) > 50 else "")

        result.append(
            SessionResponse(
                id=s["id"],
                title=s["title"],
                messages=s["messages"],
                created_at=s["created_at"],
                updated_at=s["updated_at"],
                is_archived=s["is_archived"],
                relative_time=_relative_time(s["updated_at"]),
                preview=preview,
            )
        )
    return result


@app.post("/api/sessions")
async def create_session_api() -> SessionResponse:
    s = _create_session()
    return SessionResponse(
        id=s["id"],
        title=s["title"],
        messages=[],
        created_at=s["created_at"],
        updated_at=s["updated_at"],
        is_archived=False,
        relative_time="방금 전",
        preview="대화를 시작하세요...",
    )


@app.delete("/api/sessions/{session_id}")
async def delete_session_api(session_id: str):
    # dict 키로 직접 삭제
    if session_id in sessions_db:
        del sessions_db[session_id]
        return {"ok": True}
    # session["id"]로 검색하여 삭제 (키 불일치 대응)
    for key, s in list(sessions_db.items()):
        if s.get("id") == session_id:
            del sessions_db[key]
            return {"ok": True}
    return {"ok": True}


@app.delete("/api/sessions")
async def delete_all_sessions():
    """모든 세션 일괄 삭제."""
    count = len(sessions_db)
    sessions_db.clear()
    return {"ok": True, "deleted": count}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> SessionResponse:
    s = sessions_db.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    msgs = s.get("messages", [])
    preview = ""
    if msgs:
        last = msgs[-1]
        text = last.get("content", "")
        preview = text[:50] + ("..." if len(text) > 50 else "")

    return SessionResponse(
        id=s["id"],
        title=s["title"],
        messages=s["messages"],
        created_at=s["created_at"],
        updated_at=s["updated_at"],
        is_archived=s["is_archived"],
        relative_time=_relative_time(s["updated_at"]),
        preview=preview,
    )


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_web:app", host="0.0.0.0", port=8000, reload=True)
