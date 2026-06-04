"""Quality gate for generated/deterministic RAG answers."""
from __future__ import annotations

import re

from .schema import QualityGateResult

FORBIDDEN_LITERALS: tuple[str, ...] = (
    "브레이크 버",
    "레이크 이블",
    "레이크 드",
    "곽",
    "도를 입니다",
    "DM0000000000",
    "<한국어 답변>",
    "<think>",
    "</think>",
)

CONTEXT_HEADER_RE = re.compile(r"\[DMC:\s*[^\]]+\|\s*Type:\s*[^\]]+\]", re.I)
DMC_REPEAT_RE = re.compile(r"(?:^|\n)\s*DMC\s*:\s*.*(?:\n\s*DMC\s*:)", re.I)
ANSWER_REPEAT_RE = re.compile(r"(?:^|\n)\s*답변\s*:\s*.*(?:\n\s*답변\s*:)", re.S)


def check_answer_quality(answer: str) -> QualityGateResult:
    reasons: list[str] = []
    for bad in FORBIDDEN_LITERALS:
        if bad in answer:
            reasons.append(f"forbidden:{bad}")
    if CONTEXT_HEADER_RE.search(answer):
        reasons.append("context_header_leakage")
    if DMC_REPEAT_RE.search(answer):
        reasons.append("repeated_dmc_blocks")
    if ANSWER_REPEAT_RE.search(answer):
        reasons.append("repeated_answer_blocks")
    return QualityGateResult(ok=not reasons, reasons=tuple(reasons))


def enforce_quality(answer: str) -> str:
    result = check_answer_quality(answer)
    if result.ok:
        return answer.strip()
    return "답변 품질 검사에서 생성 오류가 감지되어 원문 답변을 제공하지 않습니다. 관련 문서를 다시 확인해 주세요."
