#!/usr/bin/env python3
"""Run a small web/API RAG smoke and confirm LangSmith traces are visible.

This script intentionally talks to the running FastAPI server instead of loading
models in-process. Start the server first, for example:

    eval "$(python scripts/local_model_env.py)"
    uvicorn app_web:app --host 127.0.0.1 --port 8000

The server imports src.config, which loads .env and enables LangSmith tracing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "eval" / "questions" / "langsmith_smoke.json"
DEFAULT_ENV = PROJECT_ROOT / ".env"


def _answer_quality_error(answer: str) -> str | None:
    if not answer:
        return "empty_answer"
    if "근거:" in answer:
        return "evidence_leaked_into_answer"
    if re.search(r"(브레이KE|레이KE|Final Answer|</think>|<think)", answer, re.IGNORECASE):
        return "answer_artifact"
    if re.search(r"[\u4E00-\u9FFF]", answer) and re.search(r"[가-힣]", answer):
        return "cjk_leaked_into_korean_answer"
    lines = [line.strip() for line in answer.split("\n") if line.strip()]
    if len(lines) >= 2:
        tail = re.sub(r"\s+", " ", lines[-1])
        prior = re.sub(r"\s+", " ", "\n".join(lines[:-1]))
        if len(tail) >= 8 and tail[: min(24, len(tail))] in prior:
            return "restarted_answer_tail"
    return None


@dataclass
class SmokeResult:
    id: str
    question: str
    status: str
    llm_sec: float
    evidence_count: int
    expected_dmc_hit: bool
    expected_modality_hit: bool
    answer_prefix: str
    error: str | None = None


def _load_questions(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"Question file must contain a non-empty 'questions' list: {path}")
    if limit is not None:
        questions = questions[:limit]
    return questions


def _request_json(method: str, base_url: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 10) -> Any:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {path}: {body}") from exc


def _wait_ready(base_url: str, timeout_sec: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status = _request_json("GET", base_url, "/api/status", timeout=3)
            if status.get("ready"):
                return status
            last_error = RuntimeError(f"server not ready: {status}")
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            last_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"Server did not become ready within {timeout_sec}s: {last_error}")


def _run_job(base_url: str, item: dict[str, Any], session_id: str, args: argparse.Namespace) -> SmokeResult:
    request_payload = {
        "session_id": session_id,
        "question": item["question"],
        "top_k": args.top_k,
        "rerank_top_k": args.rerank_top_k,
        "max_context": args.max_context,
    }
    created = _request_json("POST", base_url, "/api/chat/jobs", request_payload, timeout=10)
    job_id = created["job_id"]

    deadline = time.monotonic() + args.job_timeout_sec
    job: dict[str, Any] = created
    while time.monotonic() < deadline:
        job = _request_json("GET", base_url, f"/api/chat/jobs/{job_id}", timeout=10)
        if job.get("status") in {"done", "error", "cancelled"}:
            break
        time.sleep(args.poll_interval_sec)
    else:
        raise TimeoutError(f"Job {job_id} did not finish within {args.job_timeout_sec}s")

    evidences = job.get("evidences") or []
    expected_substrings = item.get("expected_dmc_substrings") or []
    dmc_blob = "\n".join(str(ev.get("dmc", "")) for ev in evidences)
    expected_dmc_hit = not expected_substrings or any(expected in dmc_blob for expected in expected_substrings)
    expected_modalities = item.get("expected_modalities") or []
    modality_blob = "\n".join(
        " ".join(str(ev.get(key, "")) for key in ("modality", "content_role"))
        for ev in evidences
    )
    expected_modality_hit = not expected_modalities or any(expected in modality_blob for expected in expected_modalities)
    answer = (job.get("answer") or "").strip()
    return SmokeResult(
        id=item["id"],
        question=item["question"],
        status=job.get("status", "unknown"),
        llm_sec=float(job.get("llm_sec") or 0),
        evidence_count=len(evidences),
        expected_dmc_hit=expected_dmc_hit,
        expected_modality_hit=expected_modality_hit,
        answer_prefix=answer[:160].replace("\n", " "),
        error=job.get("error") or _answer_quality_error(answer),
    )


def _check_langsmith(project_name: str, start_time: datetime, wait_sec: float) -> dict[str, Any]:
    from langsmith import Client

    client = Client()
    deadline = time.monotonic() + wait_sec
    last_runs: list[Any] = []
    while time.monotonic() < deadline:
        # Allow a small clock-skew window because server and checker are separate processes.
        runs = list(
            client.list_runs(
                project_name=project_name,
                start_time=start_time - timedelta(seconds=15),
                limit=20,
            )
        )
        last_runs = runs
        names = {getattr(run, "name", "") for run in runs}
        if {"rag_pipeline", "vector_search"} & names:
            return {
                "ok": True,
                "project": project_name,
                "run_count": len(runs),
                "sample_names": sorted(name for name in names if name)[:12],
            }
        time.sleep(2)
    return {
        "ok": False,
        "project": project_name,
        "run_count": len(last_runs),
        "sample_names": sorted({getattr(run, "name", "") for run in last_runs if getattr(run, "name", "")})[:12],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run web/API RAG smoke and verify LangSmith traces")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Running FastAPI server URL")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="Question JSON file")
    parser.add_argument("--limit", type=int, default=1, help="Limit number of questions; default is 1 stable gate question")
    parser.add_argument("--session-prefix", default="langsmith-smoke", help="Session ID prefix")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rerank-top-k", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=3000)
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("--job-timeout-sec", type=float, default=180.0)
    parser.add_argument("--ready-timeout-sec", type=float, default=20.0)
    parser.add_argument("--langsmith-wait-sec", type=float, default=45.0)
    parser.add_argument("--skip-langsmith", action="store_true", help="Run API smoke only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(DEFAULT_ENV)
    project_name = (
        __import__("os").getenv("LANGCHAIN_PROJECT")
        or __import__("os").getenv("LANGSMITH_PROJECT")
        or "default"
    )

    status = _wait_ready(args.base_url, args.ready_timeout_sec)
    print("SERVER_READY", json.dumps({k: status.get(k) for k in ["model_name", "backend", "chunk_count", "ready", "busy"]}, ensure_ascii=False))

    questions = _load_questions(args.questions, args.limit)
    started_at = datetime.now(UTC)
    results: list[SmokeResult] = []
    for idx, item in enumerate(questions, start=1):
        session_id = f"{args.session_prefix}-{int(time.time())}-{idx}"
        result = _run_job(args.base_url, item, session_id, args)
        results.append(result)
        print(
            "RESULT",
            json.dumps(
                {
                    "id": result.id,
                    "status": result.status,
                    "llm_sec": round(result.llm_sec, 3),
                    "evidence_count": result.evidence_count,
                    "expected_dmc_hit": result.expected_dmc_hit,
                    "expected_modality_hit": result.expected_modality_hit,
                    "answer_prefix": result.answer_prefix,
                    "error": result.error,
                },
                ensure_ascii=False,
            ),
        )

    langsmith = {"ok": None, "skipped": True, "project": project_name}
    if not args.skip_langsmith:
        langsmith = _check_langsmith(project_name, started_at, args.langsmith_wait_sec)
        print("LANGSMITH", json.dumps(langsmith, ensure_ascii=False))

    failures = [
        r
        for r in results
        if r.status != "done"
        or r.error
        or r.evidence_count == 0
        or not r.expected_dmc_hit
        or not r.expected_modality_hit
        or not r.answer_prefix
    ]
    if failures:
        print("SMOKE_FAILED", json.dumps([r.id for r in failures], ensure_ascii=False), file=sys.stderr)
        return 1
    if not args.skip_langsmith and not langsmith.get("ok"):
        print("LANGSMITH_TRACE_CHECK_FAILED", file=sys.stderr)
        return 2
    print("SMOKE_PASSED", json.dumps({"questions": len(results), "langsmith_ok": langsmith.get("ok")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
