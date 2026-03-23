"""WinneAI — FastAPI 백엔드 서버.

Stitch 'WinneAI PC Chat (Light Mode)' HTML을 정적 파일로 서빙하고,
RAG 파이프라인 API를 제공합니다.

실행:
    python app_web.py
    또는
    uvicorn app_web:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    GGUF_MODEL_PATH,
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


@app.on_event("startup")
async def _preload_models():
    """서버 시작 시 모델 사전 로드."""
    logger.info("Preloading models...")
    _get_models()
    logger.info("Models ready.")


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


class ChatResponse(BaseModel):
    answer: str
    evidences: list[EvidenceResponse] = []
    llm_sec: float = 0


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
    model_name = Path(GGUF_MODEL_PATH).stem
    chunk_count = _get_chunk_count()
    return StatusResponse(
        model_name=model_name,
        embedding_model="BGE-m3-ko",
        chunk_count=chunk_count,
        ready=chunk_count > 0,
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
                    text=getattr(ev, "text", "")[:200],
                )
            )

    # 어시스턴트 메시지 추가
    session["messages"].append({
        "role": "assistant",
        "content": result.answer,
        "evidences": [e.model_dump() for e in evidences],
        "llm_sec": llm_sec,
    })
    session["updated_at"] = datetime.now().isoformat()

    return ChatResponse(answer=result.answer, evidences=evidences, llm_sec=llm_sec)


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
