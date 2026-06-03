#!/usr/bin/env python3
"""Autonomous 500-case QA → fix → reverify loop for S1000D-RAG.

This long-running harness is intentionally conservative:
- It runs one QA case at a time against the FastAPI server.
- On pass, it advances to the next case.
- On fail/error, it writes a failure artifact and invokes a bounded Hermes fixer.
- The same case is re-run after each fix attempt.
- It reports start/progress/failure/fix/completion to Telegram via `hermes send`.
- It does not push, deploy, sudo, install packages, edit secrets, or delete files.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_quality_qa_loop import QaCase, build_cases, classify, post_chat  # noqa: E402

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "eval" / "results" / "autonomous-500"
SAFETY_BANNER = "sudo/삭제/push/deploy/secrets/install 금지; 필요 시 Telegram 승인요청 후 중단"


def send_telegram(message: str) -> None:
    # Hermes CLI `send` can block when invoked from a Hermes-managed background
    # process in this WSL session.  Keep the harness non-blocking by recording the
    # exact message to stdout; the supervising Hermes session forwards progress
    # via the send_message tool when it polls this process/log.
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"TELEGRAM_EVENT {timestamp} {message.replace(chr(10), ' | ')}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def request_json(method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        DEFAULT_BASE_URL.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_ready(timeout_sec: float = 90.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status = request_json("GET", "/api/status", timeout=5)
            if status.get("ready") and not status.get("busy"):
                return status
            last_error = RuntimeError(f"not ready/busy: {status}")
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"server not ready after {timeout_sec}s: {last_error}")


def find_uvicorn_pids() -> list[int]:
    proc = subprocess.run(
        ["pgrep", "-af", "uvicorn app_web:app"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    pids: list[int] = []
    for line in proc.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            pids.append(int(parts[0]))
        except ValueError:
            continue
    return pids


def start_server(log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = 'eval "$(python scripts/local_model_env.py)"\nuvicorn app_web:app --host 127.0.0.1 --port 8000'
    log_file = log_path.open("a", encoding="utf-8")
    return subprocess.Popen(
        ["bash", "-lc", command],
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def restart_server(run_dir: Path) -> None:
    for pid in find_uvicorn_pids():
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(3)
    start_server(run_dir / "uvicorn.log")
    wait_ready(120)


def ensure_server(run_dir: Path) -> dict[str, Any]:
    try:
        return wait_ready(15)
    except Exception:
        start_server(run_dir / "uvicorn.log")
        return wait_ready(120)


def run_case(case: QaCase, timeout: int) -> dict[str, Any]:
    try:
        response = post_chat(DEFAULT_BASE_URL, case, timeout)
        status, issues = classify(case, response)
        return {
            "case": asdict(case),
            "status": status,
            "issues": issues,
            "answer": response.get("answer"),
            "llm_sec": response.get("llm_sec"),
            "wall_sec": response.get("wall_sec"),
            "evidences": response.get("evidences") or [],
        }
    except (urllib.error.URLError, TimeoutError, Exception) as exc:  # noqa: BLE001
        return {
            "case": asdict(case),
            "status": "error",
            "issues": ["request_error"],
            "error": repr(exc),
        }


def run_static_verification() -> tuple[bool, str]:
    cmd = (
        "python -m py_compile src/rag/ontology.py src/rag/graph_retrieval.py src/rag/pipeline.py "
        "scripts/build_ontology_exports.py scripts/build_graph_manifest.py && "
        "python -m pytest tests/test_s1000d_ontology.py tests/test_graph_retrieval.py tests/test_rag.py "
        "tests/test_app_web_chat_jobs.py tests/test_query_enhancer.py tests/test_model_config.py tests/test_local_model_env.py -q"
    )
    proc = subprocess.run(["bash", "-lc", cmd], cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=300)
    output = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode == 0, output[-4000:]


def run_fix_agent(run_dir: Path, iteration: int, attempt: int, failure_path: Path) -> dict[str, Any]:
    prompt = f"""
당신은 S1000D-RAG repo의 bounded fixer입니다.
작업 디렉터리: {PROJECT_ROOT}
실패 artifact: {failure_path}
현재 요청: 500회 QA 루프 중 {iteration}번째 질문이 실패했습니다. 실패 JSON을 읽고, 테스트를 먼저 추가/보강한 뒤 최소 수정으로 해당 오류를 해결하세요.

엄격한 제한:
- work only inside {PROJECT_ROOT}
- do not push, deploy, sudo, install packages, edit secrets, edit .env/auth files, or delete files
- do not run rm -rf, git reset --hard, git clean, force operations
- broad rewrite 금지; 실패 원인에 필요한 좁은 수정만
- 승인 필요한 작업이 있으면 코드 수정하지 말고 최종 응답 맨 앞에 APPROVAL_REQUIRED: 라고 쓰고 필요한 승인 문구를 한국어로 제시

필수 절차:
1. 실패 JSON과 관련 코드/테스트를 읽으세요.
2. 가능하면 RED 테스트를 먼저 추가하거나 기존 테스트를 보강하고 focused test 실패를 확인하세요.
3. 최소 수정으로 GREEN을 만드세요.
4. 다음 검증을 실행하세요:
   python -m py_compile src/rag/ontology.py src/rag/graph_retrieval.py src/rag/pipeline.py scripts/build_ontology_exports.py scripts/build_graph_manifest.py
   python -m pytest tests/test_s1000d_ontology.py tests/test_graph_retrieval.py tests/test_rag.py tests/test_query_enhancer.py tests/test_model_config.py tests/test_local_model_env.py -q
5. 최종 응답은 다음 형식으로만:
   PASS/REQUEST_CHANGES/APPROVAL_REQUIRED
   changed_files: ...
   tests_run: ...
   remaining_risks: ...
""".strip()
    log_path = run_dir / "fixer" / f"fix-{iteration:03d}-attempt-{attempt}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["hermes", "-z", prompt, "--skills", "test-driven-development"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=1800,
    )
    output = (proc.stdout + "\n" + proc.stderr).strip()
    log_path.write_text(output, encoding="utf-8")
    return {"returncode": proc.returncode, "output": output, "log_path": str(log_path)}


def expanded_cases(total: int) -> list[QaCase]:
    base = build_cases()
    cases: list[QaCase] = []
    for index in range(total):
        source = base[index % len(base)]
        cycle = index // len(base) + 1
        cases.append(QaCase(
            id=f"qa500-{index + 1:03d}-cycle{cycle}-{source.id}",
            question=source.question,
            expected=source.expected,
            notes=f"{source.notes}; repeated-cycle={cycle}; source={source.id}",
        ))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-fix-attempts", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    run_id = datetime.now().strftime("autonomous-500-%Y%m%d-%H%M%S")
    run_dir = DEFAULT_OUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "records.jsonl"
    summary_path = run_dir / "summary.json"

    print(f"RUN_DIR={run_dir}", flush=True)
    send_telegram(f"[S1000D-RAG] 500회 QA 루프 시작: run_id={run_id}\n{SAFETY_BANNER}")

    try:
        server_status = ensure_server(run_dir)
    except Exception as exc:
        msg = f"[S1000D-RAG] 승인/조치 필요: 서버 기동 실패\nrun_id={run_id}\nerror={exc!r}"
        send_telegram(msg)
        print(msg, flush=True)
        return 2
    print("SERVER_READY", json.dumps(server_status, ensure_ascii=False), flush=True)

    cases = expanded_cases(args.total)
    pass_count = 0
    failure_count = 0
    fixed_count = 0
    started = time.time()
    final_records: list[dict[str, Any]] = []

    for index, case in enumerate(cases, 1):
        print(f"[{index:03d}/{args.total:03d}] {case.id}: {case.question}", flush=True)
        record = run_case(case, args.timeout)
        attempt_records = [record]

        attempt = 0
        while record.get("status") != "pass" and attempt < args.max_fix_attempts:
            attempt += 1
            failure_path = run_dir / "failures" / f"case-{index:03d}-attempt-{attempt}.json"
            write_json(failure_path, record)
            send_telegram(
                f"[S1000D-RAG] QA 실패 감지 → 수정 시도 {attempt}/{args.max_fix_attempts}\n"
                f"run_id={run_id}\ncase={index}/{args.total} {case.id}\nissues={record.get('issues')}\nartifact={failure_path}"
            )
            fix_result = run_fix_agent(run_dir, index, attempt, failure_path)
            write_json(run_dir / "fixer" / f"fix-{index:03d}-attempt-{attempt}.json", fix_result)
            output = fix_result.get("output", "")
            if fix_result.get("returncode") != 0 or output.startswith("APPROVAL_REQUIRED") or "APPROVAL_REQUIRED" in output[:500]:
                msg = (
                    f"[S1000D-RAG] 승인필요/수정중단\nrun_id={run_id}\ncase={index}/{args.total} {case.id}\n"
                    f"fix_log={fix_result.get('log_path')}\n요약={output[:900]}"
                )
                send_telegram(msg)
                write_json(summary_path, {
                    "run_id": run_id,
                    "status": "approval_required_or_fixer_failed",
                    "case_index": index,
                    "case": asdict(case),
                    "fix_result": fix_result,
                    "run_dir": str(run_dir),
                })
                print(msg, flush=True)
                return 3

            ok, verify_output = run_static_verification()
            write_json(run_dir / "verification" / f"verify-{index:03d}-attempt-{attempt}.json", {"ok": ok, "output": verify_output})
            if not ok:
                record = {
                    "case": asdict(case),
                    "status": "error",
                    "issues": ["post_fix_static_verification_failed"],
                    "verification_output": verify_output,
                }
                attempt_records.append(record)
                continue

            restart_server(run_dir)
            record = run_case(case, args.timeout)
            attempt_records.append(record)
            if record.get("status") == "pass":
                fixed_count += 1
                send_telegram(
                    f"[S1000D-RAG] 수정 후 재검증 통과\nrun_id={run_id}\ncase={index}/{args.total} {case.id}\nattempt={attempt}"
                )

        if record.get("status") == "pass":
            pass_count += 1
        else:
            failure_count += 1
            failure_path = run_dir / "failures" / f"case-{index:03d}-final.json"
            write_json(failure_path, {"case": asdict(case), "attempt_records": attempt_records})
            msg = (
                f"[S1000D-RAG] QA 루프 중단: 최대 수정 시도 후에도 실패\nrun_id={run_id}\n"
                f"case={index}/{args.total} {case.id}\nissues={record.get('issues')}\nartifact={failure_path}"
            )
            send_telegram(msg)
            write_json(summary_path, {
                "run_id": run_id,
                "status": "failed_after_fix_attempts",
                "completed_cases": index - 1,
                "pass_count": pass_count,
                "failure_count": failure_count,
                "fixed_count": fixed_count,
                "run_dir": str(run_dir),
            })
            print(msg, flush=True)
            return 4

        final_record = {"case_index": index, "final": record, "attempts": attempt_records}
        final_records.append(final_record)
        with records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(final_record, ensure_ascii=False) + "\n")

        if index % args.progress_every == 0:
            elapsed_min = (time.time() - started) / 60
            send_telegram(
                f"[S1000D-RAG] 500회 QA 진행상황\nrun_id={run_id}\nprogress={index}/{args.total}\n"
                f"pass={pass_count}, fixed={fixed_count}, fail={failure_count}\nelapsed_min={elapsed_min:.1f}\nrun_dir={run_dir}"
            )

    ok, verify_output = run_static_verification()
    write_json(run_dir / "verification" / "final-static-verification.json", {"ok": ok, "output": verify_output})
    status = "passed" if ok else "static_verification_failed"
    summary = {
        "run_id": run_id,
        "status": status,
        "total": args.total,
        "pass_count": pass_count,
        "failure_count": failure_count,
        "fixed_count": fixed_count,
        "elapsed_sec": round(time.time() - started, 3),
        "run_dir": str(run_dir),
        "records_path": str(records_path),
        "final_verification_ok": ok,
        "final_verification_output": verify_output,
    }
    write_json(summary_path, summary)
    send_telegram(
        f"[S1000D-RAG] 500회 QA 루프 완료\nrun_id={run_id}\nstatus={status}\n"
        f"pass={pass_count}/{args.total}, fixed={fixed_count}, fail={failure_count}\nrun_dir={run_dir}"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
