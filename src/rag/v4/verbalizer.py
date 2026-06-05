"""LLM verbalization for v4 answer plans."""
from __future__ import annotations

from typing import Any
import re

from .answer_plan import AnswerPlan
from .grounded_generator import generate_grounded_answer
from .korean_composer import compose_korean_draft


def verbalize_answer_plan(plan: AnswerPlan, llm: Any | None = None) -> str:
    draft = _korean_user_fallback(plan)
    if llm is not None:
        grounded = generate_grounded_answer(plan, llm)
        grounded = _clean_llm_answer(grounded)
        if _is_acceptable_user_answer(grounded):
            return grounded
    return _ensure_citations(draft, plan.required_citations)


def build_polish_prompt(plan: AnswerPlan, korean_draft: str) -> str:
    claims = "\n".join("- " + re.sub(r"\s+", " ", claim.text).strip() for claim in plan.claims)
    graph_paths = "\n".join(f"- {path}" for path in plan.graph_paths)
    return f"""당신은 S1000D 정비지원 답변 문장을 다듬는 한국어 편집자입니다.

아래 한국어 초안을 사용자에게 자연스럽게 보이도록 문장만 다듬으세요.
새로운 사실, 절차, 공구, 안전 경고를 추가하지 마세요.
원문 claim에 없는 내용을 만들지 마세요.
DMC 목록은 본문에 반복하지 마세요. 시스템이 마지막에 자동으로 붙입니다.
영어 원문, 내부 metadata, 내부 계획명, 대괄호 형식 근거를 출력하지 마세요.
최종 답변 본문만 한국어로 작성하세요.

질문: {plan.query}
의도: {plan.intent.value}
지원 수준: {plan.support_level.value}
금지 사항: {', '.join(plan.forbidden_claims)}
근거 DMC 목록: {', '.join(plan.required_citations) or '없음'}

한국어 초안:
{korean_draft}

허용된 원문 claim:
{claims or '- 없음'}

그래프 선택 근거:
{graph_paths or '- 없음'}
"""


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
    if _is_acceptable_user_answer(body):
        return answer
    return _korean_user_fallback(plan)


def _is_acceptable_user_answer(answer: str) -> bool:
    if not answer.strip():
        return False
    forbidden = (
        "AnswerPlan",
        "required citations",
        "forbidden claims",
        "허용된 claims",
        "RDF graph paths",
        "[DMC:",
        "support:",
        "titles:",
        "Okay",
        "let me think",
    )
    if any(marker in answer for marker in forbidden):
        return False
    return _contains_hangul(answer) and not _looks_like_english_evidence_dump(answer)


def _contains_hangul(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


def _looks_like_english_evidence_dump(text: str) -> bool:
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    hangul = sum(1 for ch in text if "가" <= ch <= "힣")
    return latin > max(40, hangul * 2)


def _korean_user_fallback(plan: AnswerPlan) -> str:
    return compose_korean_draft(plan)


def _ensure_citations(answer: str, citations: tuple[str, ...]) -> str:
    missing = [dmc for dmc in citations if dmc not in answer]
    if missing:
        answer = answer.rstrip() + "\n근거 DMC: " + ", ".join(citations)
    return answer
