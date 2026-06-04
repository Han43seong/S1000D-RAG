"""Deterministic Korean draft composer for v4 answer plans."""
from __future__ import annotations

import re

from src.rag.ontology import Intent
from .answer_plan import AnswerPlan
from .claim_normalizer import normalize_claims


def compose_korean_draft(plan: AnswerPlan) -> str:
    """Build a clean Korean user-facing draft without citations or metadata."""
    subject = subject_from_query(plan.query)
    units = _normalized_units(plan, limit=12 if plan.intent == Intent.PROCEDURE else 8)
    if plan.intent == Intent.PROCEDURE:
        lines = [f"{subject} 절차는 다음과 같습니다."]
        lines.extend(f"{idx}. {unit}" for idx, unit in enumerate(units, start=1))
        return "\n".join(lines).strip()

    lines = [f"{subject}에 대해 문서 근거 기준으로 설명하면 다음과 같습니다."]
    lines.extend(f"- {unit}" for unit in units)
    return "\n".join(lines).strip()


def subject_from_query(query: str) -> str:
    query_lower = query.lower()
    if "앞바퀴" in query or "front wheel" in query_lower:
        return "앞바퀴 설치"
    if "브레이크 시스템" in query or "brake system" in query_lower:
        return "브레이크 시스템"
    if "브레이크" in query or "brake" in query_lower:
        return "브레이크 정비"
    return "정비 작업"


def _normalized_units(plan: AnswerPlan, *, limit: int) -> list[str]:
    claims = _dedupe_claim_texts(plan)
    units = normalize_claims(claims, limit=limit)
    if units:
        return units
    return ["확인된 근거 문서가 있지만 사용자 답변으로 안전하게 재작성할 수 있는 핵심 문장이 부족합니다. 원문 근거를 함께 확인해 주세요."]


def _dedupe_claim_texts(plan: AnswerPlan) -> list[str]:
    seen: set[str] = set()
    texts: list[str] = []
    for claim in plan.claims:
        text = re.sub(r"\s+", " ", claim.text).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        texts.append(text)
    return texts[:8]
