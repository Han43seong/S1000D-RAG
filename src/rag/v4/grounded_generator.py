"""Grounded LLM answer generation for v4 AnswerPlans.

This module lets the LLM reason over a structured EvidencePacket instead of
merely polishing a deterministic Korean draft. The LLM output is still validated
against citations/leakage constraints, with KoreanComposer fallback handled by
verbalizer.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from .answer_plan import AnswerPlan


@dataclass(frozen=True)
class EvidencePacket:
    question: str
    intent: str
    support_level: str
    claims: list[dict[str, object]]
    relations: list[str]
    required_citations: list[str]
    forbidden_claims: list[str]


def build_evidence_packet(plan: AnswerPlan) -> EvidencePacket:
    return EvidencePacket(
        question=plan.query,
        intent=plan.intent.value,
        support_level=plan.support_level.value,
        claims=[
            {
                "text": re.sub(r"\s+", " ", claim.text).strip(),
                "section": claim.section,
                "citations": list(claim.evidence_dmcs),
                "titles": list(claim.source_titles),
                "support_level": claim.support_level.value,
            }
            for claim in plan.claims
        ],
        relations=list(plan.graph_paths),
        required_citations=list(plan.required_citations),
        forbidden_claims=list(plan.forbidden_claims),
    )


def generate_grounded_answer(plan: AnswerPlan, llm: Any) -> str:
    prompt = build_grounded_prompt(build_evidence_packet(plan))
    response = llm.invoke(prompt)
    text = getattr(response, "content", response)
    parsed = _parse_grounded_json(str(text))
    if parsed is None or not _grounded_output_is_safe(parsed, plan):
        return ""
    return _render_grounded_output(parsed)


def build_grounded_prompt(packet: EvidencePacket) -> str:
    packet_json = json.dumps(packet.__dict__, ensure_ascii=False, indent=2)
    return f"""당신은 S1000D 정비문서 기반 AI 정비지원 챗봇입니다.

사용자 질문과 EvidencePacket을 함께 보고 근거 기반으로 판단해서 답변하세요.
온톨로지/RAG는 관련 문서와 근거를 찾기 위한 시스템이며, 당신은 그 근거 안에서 질문에 맞는 해석과 우선 확인 항목을 제시합니다.

규칙:
- EvidencePacket의 claims/relations/required_citations 안에서만 답변하세요.
- 문서 근거만으로 특정 고장 원인을 확정할 수 없으면 그 불확실성을 명시하세요.
- 문서에 없는 부품 고장, 교체, 분해, 공구, 안전 경고를 만들지 마세요.
- 내부 JSON, 영어 원문, [DMC: ...] metadata를 사용자 답변에 노출하지 마세요.
- used_citations에는 실제 사용한 required_citations 안의 DMC만 넣으세요.

사용자 질문: {packet.question}
EvidencePacket:
{packet_json}

반드시 아래 JSON만 출력하세요:
{{
  "answer": "한국어 본문 답변",
  "check_items": ["근거 기반 우선 확인 항목"],
  "uncertainty": "확정할 수 없는 부분 또는 빈 문자열",
  "used_citations": ["DMC"]
}}
"""


def _parse_grounded_json(text: str) -> dict[str, object] | None:
    text = text.strip()
    if not text:
        return None
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _grounded_output_is_safe(parsed: dict[str, object], plan: AnswerPlan) -> bool:
    answer = str(parsed.get("answer", "")).strip()
    if not _contains_hangul(answer) or _looks_like_english_evidence_dump(answer):
        return False
    check_items = parsed.get("check_items", [])
    uncertainty = parsed.get("uncertainty", "")
    if not isinstance(check_items, list) or not all(isinstance(item, str) and item.strip() for item in check_items):
        return False
    if not isinstance(uncertainty, str):
        return False
    forbidden_markers = (
        "AnswerPlan",
        "EvidencePacket",
        "required_citations",
        "forbidden_claims",
        "[DMC:",
        "support:",
        "titles:",
        "Okay",
        "let me think",
    )
    rendered = json.dumps(parsed, ensure_ascii=False)
    if any(marker in rendered for marker in forbidden_markers):
        return False
    allowed = set(plan.required_citations)
    used = parsed.get("used_citations", [])
    if not isinstance(used, list) or not used:
        return False
    if any(str(citation) not in allowed for citation in used):
        return False
    if _contains_unsupported_overreach(rendered):
        return False
    return True


def _render_grounded_output(parsed: dict[str, object]) -> str:
    lines = [str(parsed.get("answer", "")).strip()]
    check_items = parsed.get("check_items", [])
    if isinstance(check_items, list) and check_items:
        lines.append("\n확인할 항목:")
        for idx, item in enumerate(check_items, start=1):
            lines.append(f"{idx}. {str(item).strip()}")
    uncertainty = str(parsed.get("uncertainty", "")).strip()
    if uncertainty and uncertainty not in lines[0]:
        lines.append("\n" + uncertainty)
    used = [str(citation) for citation in parsed.get("used_citations", []) if str(citation).strip()]
    if used:
        lines.append("근거 DMC: " + ", ".join(dict.fromkeys(used)))
    return "\n".join(line for line in lines if line).strip()


def _contains_hangul(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


def _looks_like_english_evidence_dump(text: str) -> bool:
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    hangul = sum(1 for ch in text if "가" <= ch <= "힣")
    return latin > max(40, hangul * 2)


def _contains_unsupported_overreach(text: str) -> bool:
    overreach_patterns = (
        r"베어링(?:이|을|은|는)?\s*(?:고장|교체|파손)",
        r"(?:즉시|반드시)\s*(?:교체|분해|수리)",
        r"(?:고장났|고장났으니|고장입니다)",
        r"\bFAKE\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in overreach_patterns)
