"""LLM verbalization for v4 answer plans."""
from __future__ import annotations

from typing import Any
import re

from .answer_plan import AnswerPlan


def verbalize_answer_plan(plan: AnswerPlan, llm: Any | None = None) -> str:
    prompt = build_verbalizer_prompt(plan)
    if llm is not None:
        response = llm.invoke(prompt)
        text = getattr(response, "content", response)
        answer = _clean_llm_answer(str(text).strip())
        if answer:
            answer = _ensure_korean_user_answer(answer, plan)
            return _ensure_citations(answer, plan.required_citations)
    return _deterministic_fallback(plan)


def build_verbalizer_prompt(plan: AnswerPlan) -> str:
    claims = "\n".join(_format_claim_for_prompt(claim) for claim in plan.claims)
    graph_paths = "\n".join(f"- {path}" for path in plan.graph_paths)
    return f"""당신은 S1000D 정비문서 기반 한국어 AI 정비지원 챗봇입니다.

반드시 아래 구조화된 AnswerPlan만 사용해서 답변하세요.
근거 없는 사실, 절차, 공구, 안전 경고를 만들지 마세요.
답변에는 required citations의 DMC를 포함하세요.
AnswerPlan을 재출력하지 마세요.
[DMC: ...; support: ...; titles: ...] 같은 내부 metadata bracket을 답변 본문에 쓰지 마세요.
최종 답변만 한국어로 작성하세요.

질문: {plan.query}
의도: {plan.intent.value}
상세도: {plan.detail_level.value}
대상 독자: {plan.audience}
지원 수준: {plan.support_level.value}
섹션: {', '.join(plan.sections)}

허용된 claims:
{claims or '- 없음'}

RDF graph paths:
{graph_paths or '- 없음'}

required citations: {', '.join(plan.required_citations) or '없음'}
forbidden claims: {', '.join(plan.forbidden_claims)}
"""


def _format_claim_for_prompt(claim) -> str:
    details = [f"DMC: {', '.join(claim.evidence_dmcs) or '없음'}", f"support: {claim.support_level.value}"]
    if claim.evidence_blocks:
        details.append(f"blocks: {', '.join(claim.evidence_blocks)}")
    if claim.source_titles:
        details.append(f"titles: {', '.join(claim.source_titles)}")
    return f"- [{claim.section}] {claim.text} [{'; '.join(details)}]"


def _deterministic_fallback(plan: AnswerPlan) -> str:
    lines = [f"구조화된 온톨로지 근거 기준으로 답변합니다. 지원 수준: {plan.support_level.value}"]
    if plan.intent.value == "procedure" and plan.support_level.value != "exact":
        support_claims = [claim for claim in plan.claims if claim.section == "지원 여부"]
        related_claims = [claim for claim in plan.claims if claim.section != "지원 여부"]
        if support_claims:
            lines.append("\n[지원 여부]")
            lines.extend(f"- {claim.text}" for claim in support_claims)
        if related_claims:
            lines.append("\n[관련 근거]")
            lines.extend(f"- {claim.text}" for claim in related_claims[:6])
        if plan.graph_paths:
            lines.append("\n[온톨로지 선택 근거]")
            lines.extend(f"- {path}" for path in plan.graph_paths)
        if plan.required_citations:
            lines.append("\n근거 DMC: " + ", ".join(plan.required_citations))
        return "\n".join(lines).strip()

    for section in plan.sections:
        lines.append(f"\n[{section}]")
        for claim in plan.claims:
            lines.append(f"- {claim.text}")
    if plan.graph_paths:
        lines.append("\n[온톨로지 선택 근거]")
        lines.extend(f"- {path}" for path in plan.graph_paths)
    if plan.required_citations:
        lines.append("\n근거 DMC: " + ", ".join(plan.required_citations))
    return "\n".join(lines).strip()


def _clean_llm_answer(answer: str) -> str:
    if not answer:
        return ""
    # Some local LLMs echo the prompt contract as `AnswerPlan: ... Answer: ...`.
    if "Answer:" in answer:
        answer = answer.rsplit("Answer:", 1)[-1]
    answer = re.sub(r"^\s*AnswerPlan\s*[:：]\s*", "", answer).strip()
    answer = re.sub(r"^\s*AnswerPlan에 따라[,，]?\s*", "", answer).strip()
    answer = re.sub(r"\[(?:DMC|support|titles|blocks|source)[^\]]*\]", "", answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r"\[\s*근거 기반 설명\s*\]", "", answer).strip()
    answer = re.sub(r"\[DMC:[^\[]*(?=\[근거 기반 설명\]|근거 DMC:|$)", "", answer, flags=re.IGNORECASE).strip()
    answer = re.split(r"\n\s*\]\s*\n|\n\s*Okay[,\s]|\n\s*Let me\b", answer, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    leaked_markers = ("required citations", "forbidden claims", "허용된 claims", "RDF graph paths")
    cleaned_lines = [line for line in answer.splitlines() if not any(marker in line for marker in leaked_markers)]
    return "\n".join(line.strip() for line in cleaned_lines if line.strip()).strip()


def _ensure_korean_user_answer(answer: str, plan: AnswerPlan) -> str:
    body = re.sub(r"근거 DMC:.*", "", answer).strip()
    if _contains_hangul(body) and not _looks_like_english_evidence_dump(body):
        return answer
    return _korean_user_fallback(plan)


def _contains_hangul(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


def _looks_like_english_evidence_dump(text: str) -> bool:
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    hangul = sum(1 for ch in text if "가" <= ch <= "힣")
    return latin > max(40, hangul * 2)


def _korean_user_fallback(plan: AnswerPlan) -> str:
    subject = _subject_from_query(plan.query)
    if plan.intent.value == "procedure":
        lines = [f"{subject} 절차는 다음과 같습니다."]
        for idx, claim in enumerate(_dedupe_claim_texts(plan), start=1):
            lines.append(f"{idx}. {_rewrite_evidence_sentence_to_korean(claim)}")
        return "\n".join(lines).strip()

    lines = [f"{subject}에 대해 문서 근거 기준으로 설명하면 다음과 같습니다."]
    for claim in _dedupe_claim_texts(plan):
        lines.append(f"- {_rewrite_evidence_sentence_to_korean(claim)}")
    return "\n".join(lines).strip()


def _subject_from_query(query: str) -> str:
    if "앞바퀴" in query or "front wheel" in query.lower():
        return "앞바퀴 설치"
    if "브레이크" in query or "brake" in query.lower():
        return "브레이크 정비"
    return "정비 작업"


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


def _rewrite_evidence_sentence_to_korean(text: str) -> str:
    original = re.sub(r"\s+", " ", text).strip().rstrip(".")
    lowered = original.lower()
    replacements = [
        ("install the fork and the brakes before installing the wheel", "바퀴를 설치하기 전에 포크와 브레이크가 먼저 장착되어 있는지 확인합니다"),
        ("install the fork before installing the wheel", "바퀴를 설치하기 전에 포크가 먼저 장착되어 있는지 확인합니다"),
        ("hold the front of the bicycle", "자전거 앞부분을 안정적으로 잡습니다"),
        ("install the wheel", "바퀴를 장착합니다"),
        ("be careful to not damage the chainring", "체인링이 손상되지 않도록 주의합니다"),
        ("close the light circuit breaker located on the handlebar", "핸들바에 있는 라이트 회로 차단기를 닫습니다"),
        ("open the light circuit breaker located on the handlebar", "핸들바에 있는 라이트 회로 차단기를 엽니다"),
        ("put the bike on the floor", "자전거를 바닥에 내려놓습니다"),
        ("lift the wheel away from the frame", "바퀴를 프레임에서 들어 올려 분리합니다"),
        ("put the frame on the floor", "프레임을 바닥에 내려놓습니다"),
        ("use specific oil if the fork do not desengage easily", "포크가 쉽게 분리되지 않으면 지정된 오일을 사용합니다"),
        ("if not available, use any oil compliant with requirements", "지정 오일이 없으면 요구사항을 만족하는 오일을 사용합니다"),
        ("disengage the fork from the chainring", "포크를 체인링에서 분리합니다"),
        ("pushing the wheel forwards and down", "바퀴를 앞으로 밀고 아래로 내려 작업합니다"),
    ]
    translated_parts = [ko for en, ko in replacements if en in lowered]
    if translated_parts:
        return " ".join(part.rstrip(".") + "." for part in dict.fromkeys(translated_parts))
    if _contains_hangul(original):
        return original + "."
    return "문서에 확인된 해당 절차 항목을 근거 문서 순서에 따라 수행합니다."


def _ensure_citations(answer: str, citations: tuple[str, ...]) -> str:
    missing = [dmc for dmc in citations if dmc not in answer]
    if missing:
        answer = answer.rstrip() + "\n근거 DMC: " + ", ".join(citations)
    return answer
