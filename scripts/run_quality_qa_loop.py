#!/usr/bin/env python3
"""Run a deterministic 100-question QA loop against the web RAG API.

The loop is intentionally lightweight: it generates expected demo questions,
calls /api/chat sequentially, classifies response issues, and writes JSON + MD
reports under eval/results/.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

KOREAN_RE = re.compile(r"[가-힣]")
LATIN_RE = re.compile(r"[A-Za-z]")
CONTEXT_LEAK_RE = re.compile(r"\[DMC:|\| Type:|(^|\n)---($|\n)|\[Figure:|Context / 참고 문서|Question / 질문", re.I)
NOANSWER_RE = re.compile(r"찾을 수 없습니다|확인되지 않습니다|없습니다")
DMC_RE = re.compile(r"[A-Z]+-[A-Z]+-[A-Z0-9]+-[A-Z0-9-]+")
PROCEDURE_RE = re.compile(r"(방법|절차|교체|설치|장착|탈거|제거|분해|조립|청소|점검|검사|시험|테스트)")
UI_METADATA_TRAILING_RE = re.compile(r"(^|\n)\s*(참고\s*문서|근거)\s*[:：]\s*\S.*\s*$")
ALLOWED_PREVIEW_STATUSES = {"available", "unsupported_cgm", "missing"}
REFERENCE_CATEGORIES = (
    "data_modules",
    "procedures",
    "faults",
    "references",
    "warnings",
    "cautions",
    "figures",
    "graphic_assets",
    "hotspots",
)


@dataclass(frozen=True)
class QaCase:
    id: str
    question: str
    expected: str  # supported | unsupported | broad
    notes: str = ""
    required_reference_categories: tuple[str, ...] = ()
    optional_reference_categories: tuple[str, ...] = ()
    require_reference_materials_when_evidence: bool = False
    require_visual_preview_status: bool = False
    require_clean_display_answer: bool = False
    expected_dmc_substrings: tuple[str, ...] = ()


def build_cases() -> list[QaCase]:
    supported_desc = [
        "브레이크 시스템의 주요 구성품은 무엇입니까?",
        "브레이크 케이블의 역할을 설명해줘",
        "브레이크 패드는 어떤 역할을 하나요?",
        "브레이크 레버와 케이블 관계를 알려줘",
        "브레이크 시스템은 왜 중요한가요?",
        "브레이크 케이블 장력은 무엇으로 조정하나요?",
        "브레이크 패드는 어디에 있나요?",
        "브레이크가 바퀴 속도를 줄이는 원리를 설명해줘",
        "브레이크 클램프는 무엇인가요?",
        "브레이크 암에 대해 알려줘",
        "브레이크 시스템 구성품을 한 문단으로 정리해줘",
        "브레이크 케이블을 조금 더 자세히 설명해줘",
        "브레이크 패드 재질은 무엇인가요?",
        "브레이크 레버를 작동하면 어떤 일이 일어나나요?",
        "브레이크 시스템 유지보수가 중요한 이유는?",
        "브레이크 케이블은 어디에 고정되나요?",
        "브레이크 케이블 조절 잠금 너트의 기능은?",
        "브레이크 패드는 몇 개가 있나요?",
        "앞/뒤 바퀴 브레이크 패드 위치를 알려줘",
        "자전거 브레이크 관련 확인 가능한 구성품만 알려줘",
    ]
    supported_proc = [
        "브레이크 패드 청소 절차를 알려줘",
        "브레이크 패드 청소 방법은?",
        "브레이크 패드를 어떻게 청소하나요?",
        "브레이크 패드 청소 시 주의할 점은?",
        "브레이크 패드 청소 작업을 단계별로 설명해줘",
        "브레이크 수동 테스트 절차를 알려줘",
        "브레이크 시험 방법은?",
        "브레이크 테스트할 때 어떤 순서로 하나요?",
        "자전거를 세우고 브레이크를 시험하는 절차는?",
        "브레이크가 제대로 작동하는지 확인하는 테스트 방법은?",
    ]
    unsupported_proc = [
        "브레이크 케이블 교체 방법은?",
        "브레이크 케이블 설치 절차를 알려줘",
        "브레이크 케이블 탈거 절차는?",
        "앞바퀴 설치 절차를 알려줘",
        "앞바퀴 교체 방법은?",
        "뒷바퀴 탈거 방법은?",
        "브레이크 레버 교체 절차는?",
        "브레이크 암 분해 방법은?",
        "타이어 교체 절차를 알려줘",
        "체인 장착 방법은?",
        "핸들바 교체 방법은?",
        "페달 설치 절차는?",
        "브레이크 케이블을 새 부품으로 바꾸는 순서를 알려줘",
        "브레이크 패드 교체 절차를 알려줘",
        "브레이크 클램프 교체 방법은?",
        "브레이크 케이블 제거 후 다시 설치하는 방법은?",
        "휠 베어링 교체 절차는?",
        "프론트 휠 인스톨 절차를 알려줘",
        "rear wheel removal procedure를 한국어로 알려줘",
        "brake cable replacement procedure를 알려줘",
        "브레이크 케이블 정비 절차 전체를 알려줘",
        "바퀴 설치 후 브레이크 조정 절차는?",
        "앞바퀴를 분리하고 장착하는 순서는?",
        "브레이크 라인 교체 방법은?",
        "브레이크 케이블 윤활 절차는?",
        "브레이크 암 장착 방법은?",
        "브레이크 레버 설치 방법은?",
        "브레이크 패드 탈거 방법은?",
        "브레이크 패드 새것으로 교환하는 절차는?",
        "브레이크 시스템 전체 오버홀 절차는?",
    ]
    broad = [
        "자전거의 주요 구성품은 무엇인가요?",
        "자전거 전체 정비 순서를 알려줘",
        "자전거의 모든 부품을 설명해줘",
        "자전거 유지보수 계획을 만들어줘",
        "산악자전거 정비 매뉴얼 전체를 요약해줘",
        "군용 장비 정비 절차처럼 상세 체크리스트를 만들어줘",
        "브레이크 외에 변속기 정보도 알려줘",
        "타이어 공기압 기준은 얼마인가요?",
        "체인 장력 조정 기준은?",
        "핸들바 토크값은 얼마인가요?",
    ]
    mixed = [
        "브레이크 케이블이 끊어졌을 때 문서에서 확인 가능한 조치는?",
        "문서에 브레이크 케이블 교체 절차가 있나요?",
        "문서에 앞바퀴 설치 절차가 있으면 알려줘",
        "브레이크 패드 청소와 교체 중 문서에 있는 작업만 알려줘",
        "브레이크 시스템 설명과 절차를 구분해서 알려줘",
        "브레이크 패드 청소 문서의 DMC를 알려줘",
        "브레이크 수동 테스트 문서의 DMC는?",
        "브레이크 케이블 설명 문서의 DMC는?",
        "근거 문서 기준으로만 브레이크 케이블을 설명해줘",
        "근거가 없으면 없다고 말하고 앞바퀴 설치 절차를 알려줘",
        "브레이크 패드 청소 절차와 근거 DMC를 같이 알려줘",
        "브레이크 수동 테스트를 한 문단으로 요약해줘",
        "브레이크 패드 청소 절차를 표처럼 정리해줘",
        "브레이크 케이블 교체 절차가 없다면 관련 설명만 알려줘",
        "제공 문서에서 확인 가능한 브레이크 정비 항목은?",
        "제공 문서에서 확인되지 않는 정비 항목은 무엇인가요?",
        "브레이크 관련 절차 문서에는 어떤 작업이 있나요?",
        "브레이크 관련 설명 문서에는 어떤 내용이 있나요?",
        "문서 기준으로 자전거 안전과 관련된 내용을 알려줘",
        "브레이크 시스템을 초보자에게 설명해줘",
        "브레이크 패드 청소 전후 확인해야 할 내용은?",
        "브레이크 테스트 결과 바퀴가 잠겨야 하나요?",
        "브레이크 케이블 장력 조정 설명은 어디에 나오나요?",
        "브레이크 패드가 림을 누르는 원리를 설명해줘",
        "브레이크 케이블과 브레이크 패드 차이를 알려줘",
        "브레이크 패드 청소와 수동 테스트를 비교해줘",
        "브레이크 케이블 교체 절차가 없으면 그렇게 답해줘",
        "앞바퀴 설치 문서가 없으면 그렇게 답해줘",
        "브레이크 패드 청소 문서 내용을 한국어로 번역 요약해줘",
        "브레이크 케이블 설명을 한국어로만 답해줘",
    ]
    cases: list[QaCase] = []
    for group, expected, notes in (
        (supported_desc, "supported", "known descriptive brake content"),
        (supported_proc, "supported", "known procedural brake content"),
        (unsupported_proc, "unsupported", "unsupported procedure should not hallucinate"),
        (broad, "broad", "scope should be limited to provided docs"),
        (mixed, "supported", "mixed/guardrail questions"),
    ):
        for q in group:
            cases.append(QaCase(f"q{len(cases)+1:03d}", q, expected, notes))
    annotated: list[QaCase] = []
    for case in cases:
        kwargs: dict[str, Any] = {}
        if case.expected == "supported" and ("브레이크" in case.question):
            kwargs["require_reference_materials_when_evidence"] = True
            kwargs["require_clean_display_answer"] = True
        if case.expected == "supported" and any(term in case.question for term in ("시스템", "케이블", "패드", "수동 테스트", "청소", "DMC", "문서")):
            kwargs["required_reference_categories"] = ("data_modules",)
        if any(term in case.question for term in ("구성품", "위치", "어디", "원리", "설명")):
            kwargs["optional_reference_categories"] = ("figures", "graphic_assets")
        if any(term in case.question for term in ("구성품", "위치", "어디")):
            kwargs["require_visual_preview_status"] = True
        if "청소" in case.question and case.expected == "supported" and ("DMC" in case.question or "근거" in case.question):
            kwargs["expected_dmc_substrings"] = ("BRAKE",)
        if kwargs:
            annotated.append(replace(case, **kwargs))
        else:
            annotated.append(case)
    cases = annotated
    if len(cases) != 100:
        raise RuntimeError(f"expected 100 cases, got {len(cases)}")
    return cases


def post_chat(base_url: str, case: QaCase, timeout: int) -> dict[str, Any]:
    payload = json.dumps({"session_id": f"qa-loop-{case.id}-{int(time.time())}", "question": case.question}).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    data["wall_sec"] = round(time.time() - started, 3)
    return data


def _check(status_issues: list[str]) -> dict[str, Any]:
    return {"status": "pass" if not status_issues else "fail", "issues": status_issues}


def _reference_materials_has_content(reference_materials: dict[str, Any]) -> bool:
    return any(bool(reference_materials.get(category)) for category in REFERENCE_CATEGORIES)


def _collect_reference_dmc_text(reference_materials: dict[str, Any]) -> str:
    parts: list[str] = []
    for value in reference_materials.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for key in ("dmc", "dmc_code", "dmcode", "id", "title", "source"):
                        raw = item.get(key)
                        if raw is not None:
                            parts.append(str(raw))
                elif item is not None:
                    parts.append(str(item))
    return "\n".join(parts)


def classify_detailed(case: QaCase, response: dict[str, Any]) -> dict[str, Any]:
    """Classify a QA response with answer, evidence, ontology, visual, and UI checks."""
    answer = (response.get("answer") or "").strip()
    evidences = response.get("evidences") or []
    raw_reference_materials = response.get("reference_materials")

    answer_issues: list[str] = []
    if not answer:
        answer_issues.append("empty_answer")
    if CONTEXT_LEAK_RE.search(answer):
        answer_issues.append("context_leak")
    answer_for_lang = DMC_RE.sub("", answer)
    korean = len(KOREAN_RE.findall(answer_for_lang))
    latin = len(LATIN_RE.findall(answer_for_lang))
    if latin > max(40, korean):
        answer_issues.append("too_much_english")
    if "<think" in answer.lower() or "reasoning" in answer.lower():
        answer_issues.append("thinking_leak")
    if re.search(r"(Final Answer|하기 전에|질문을 분석|답변을 작성하세요|브레이KE)", answer, re.IGNORECASE):
        answer_issues.append("answer_artifact")

    is_noanswer = bool(NOANSWER_RE.search(answer))
    has_supported_info = bool(re.search(r"(확인되는 작업|다만 관련|관련 설명|근거:|참고 문서:)", answer))
    is_pure_noanswer = is_noanswer and not has_supported_info
    is_proc = bool(PROCEDURE_RE.search(case.question))
    if case.expected == "unsupported" and not is_noanswer:
        answer_issues.append("unsupported_not_rejected")
    noanswer_ok_supported = bool(re.search(r"(없으면|없다면|있나요|있는지|있으면|문서에|문서 기준|확인 가능한|확인되지 않는)", case.question))
    if case.expected == "supported" and is_pure_noanswer and not noanswer_ok_supported:
        answer_issues.append("supported_rejected")
    if case.expected == "broad" and not evidences and not ("제공" in answer or "문서" in answer or is_noanswer):
        answer_issues.append("scope_not_limited")
    if is_proc and case.expected == "unsupported" and not is_noanswer:
        answer_issues.append("procedure_hallucination_risk")

    evidence_issues: list[str] = []
    if any(not (ev.get("text") or "").strip() for ev in evidences if isinstance(ev, dict)):
        evidence_issues.append("empty_evidence_text")
    evidence_dmc_text = "\n".join(str(ev.get("dmc") or ev.get("dmc_code") or "") for ev in evidences if isinstance(ev, dict))

    reference_issues: list[str] = []
    reference_materials: dict[str, Any] = {}
    if raw_reference_materials is None:
        if evidences and case.require_reference_materials_when_evidence:
            reference_issues.append("missing_reference_materials")
    elif not isinstance(raw_reference_materials, dict):
        reference_issues.append("invalid_reference_materials_shape")
    else:
        reference_materials = raw_reference_materials
        for category, items in reference_materials.items():
            if category in REFERENCE_CATEGORIES and not isinstance(items, list):
                reference_issues.append("invalid_reference_materials_shape")
                break
        if evidences and case.require_reference_materials_when_evidence and not _reference_materials_has_content(reference_materials):
            reference_issues.append("missing_reference_materials")

    for category in case.required_reference_categories:
        if not isinstance(reference_materials.get(category), list) or not reference_materials.get(category):
            reference_issues.append(f"missing_reference_category:{category}")

    reference_dmc_text = _collect_reference_dmc_text(reference_materials)
    for expected_substring in case.expected_dmc_substrings:
        if expected_substring not in evidence_dmc_text:
            # Reference-material DMC text is accepted as a documented fallback for current
            # structured responses, but evidence.dmc remains the preferred source.
            if expected_substring not in reference_dmc_text and expected_substring not in answer:
                evidence_issues.append("missing_required_evidence_dmc")

    visual_issues: list[str] = []
    graphic_assets = reference_materials.get("graphic_assets") if isinstance(reference_materials, dict) else None
    if case.require_visual_preview_status and (not isinstance(graphic_assets, list) or not graphic_assets):
        visual_issues.append("missing_graphic_assets")
    if graphic_assets is not None and not isinstance(graphic_assets, list):
        visual_issues.append("invalid_reference_materials_shape")
    elif isinstance(graphic_assets, list) and (case.require_visual_preview_status or graphic_assets):
        for asset in graphic_assets:
            if not isinstance(asset, dict):
                visual_issues.append("invalid_reference_materials_shape")
                continue
            status = asset.get("preview_status")
            preview_url = (asset.get("preview_url") or "").strip()
            if not status:
                visual_issues.append("missing_visual_preview_status")
                continue
            if status not in ALLOWED_PREVIEW_STATUSES:
                visual_issues.append("invalid_visual_preview_status")
            if asset.get("preview_available") is True and not preview_url:
                visual_issues.append("missing_visual_preview_status")
            if status == "available" and not preview_url:
                visual_issues.append("missing_visual_preview_status")

    ui_issues: list[str] = []
    if case.require_clean_display_answer and UI_METADATA_TRAILING_RE.search(answer):
        ui_issues.append("ui_metadata_leak")

    checks = {
        "answer": _check(answer_issues),
        "evidence": _check(evidence_issues),
        "reference_materials": _check(reference_issues),
        "visual_preview": _check(sorted(set(visual_issues))),
        "ui_display": _check(ui_issues),
    }
    issues: list[str] = []
    for check in checks.values():
        issues.extend(check["issues"])
    # Preserve first-seen ordering while removing duplicates caused by repeated assets.
    issues = list(dict.fromkeys(issues))
    metrics = {
        "answer_chars": len(answer),
        "evidence_count": len(evidences) if isinstance(evidences, list) else 0,
        "reference_category_counts": {
            category: len(reference_materials.get(category) or [])
            for category in REFERENCE_CATEGORIES
            if isinstance(reference_materials.get(category), list)
        },
        "graphic_asset_count": len(graphic_assets) if isinstance(graphic_assets, list) else 0,
        "korean_chars": korean,
        "latin_chars": latin,
    }
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "checks": checks,
        "metrics": metrics,
    }


def classify(case: QaCase, response: dict[str, Any]) -> tuple[str, list[str]]:
    detailed = classify_detailed(case, response)
    return detailed["status"], list(detailed["issues"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out-dir", default="eval/results")
    args = parser.parse_args()

    cases = build_cases()[: args.count]
    run_id = datetime.now().strftime("quality-qa-loop-%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for index, case in enumerate(cases, 1):
        print(f"[{index:03d}/{len(cases):03d}] {case.question}", flush=True)
        try:
            response = post_chat(args.base_url, case, args.timeout)
            detailed = classify_detailed(case, response)
            record = {
                "case": asdict(case),
                "status": detailed["status"],
                "issues": detailed["issues"],
                "checks": detailed["checks"],
                "metrics": detailed["metrics"],
                "answer": response.get("answer"),
                "llm_sec": response.get("llm_sec"),
                "wall_sec": response.get("wall_sec"),
                "evidences": response.get("evidences") or [],
                "reference_materials": response.get("reference_materials") or {},
            }
        except (urllib.error.URLError, TimeoutError, Exception) as exc:
            record = {
                "case": asdict(case),
                "status": "error",
                "issues": ["request_error"],
                "error": repr(exc),
            }
        records.append(record)

    failures = [r for r in records if r["status"] != "pass"]
    issue_counts: dict[str, int] = {}
    check_group_counts: dict[str, dict[str, int]] = {}
    for r in records:
        for issue in r.get("issues", []):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        for group, check in (r.get("checks") or {}).items():
            status = check.get("status", "unknown") if isinstance(check, dict) else "unknown"
            group_counts = check_group_counts.setdefault(group, {})
            group_counts[status] = group_counts.get(status, 0) + 1
    avg_llm = sum(float(r.get("llm_sec") or 0) for r in records) / max(1, len(records))
    summary = {
        "run_id": run_id,
        "total": len(records),
        "pass": len(records) - len(failures),
        "fail": len(failures),
        "issue_counts": dict(sorted(issue_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "check_group_counts": check_group_counts,
        "avg_llm_sec": round(avg_llm, 3),
    }
    json_path = out_dir / f"{run_id}.json"
    md_path = out_dir / f"{run_id}.md"
    json_path.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {run_id}",
        "",
        f"- total: {summary['total']}",
        f"- pass: {summary['pass']}",
        f"- fail: {summary['fail']}",
        f"- avg_llm_sec: {summary['avg_llm_sec']}",
        f"- issue_counts: `{json.dumps(summary['issue_counts'], ensure_ascii=False)}`",
        f"- check_group_counts: `{json.dumps(summary['check_group_counts'], ensure_ascii=False)}`",
        "",
        "## Ontology / Reference / Visual Summary",
        "",
        f"- reference_materials: `{json.dumps(summary['check_group_counts'].get('reference_materials', {}), ensure_ascii=False)}`",
        f"- visual_preview: `{json.dumps(summary['check_group_counts'].get('visual_preview', {}), ensure_ascii=False)}`",
        f"- ui_display: `{json.dumps(summary['check_group_counts'].get('ui_display', {}), ensure_ascii=False)}`",
        "",
        "## Failures",
    ]
    for r in failures[:50]:
        c = r["case"]
        lines.extend([
            "",
            f"### {c['id']} {c['question']}",
            f"- expected: {c['expected']}",
            f"- issues: {', '.join(r.get('issues', []))}",
            f"- checks: `{json.dumps(r.get('checks') or {}, ensure_ascii=False)}`",
            f"- llm_sec: {r.get('llm_sec')}",
            "- answer:",
            "```",
            str(r.get("answer") or r.get("error") or "")[:2000],
            "```",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON={json_path}")
    print(f"MD={md_path}")
    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
