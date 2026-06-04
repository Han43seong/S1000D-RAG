"""LLM verbalization for v4 answer plans."""
from __future__ import annotations

from typing import Any

from .answer_plan import AnswerPlan


def verbalize_answer_plan(plan: AnswerPlan, llm: Any | None = None) -> str:
    prompt = build_verbalizer_prompt(plan)
    if llm is not None:
        response = llm.invoke(prompt)
        text = getattr(response, "content", response)
        answer = str(text).strip()
        if answer:
            return _ensure_citations(answer, plan.required_citations)
    return _deterministic_fallback(plan)


def build_verbalizer_prompt(plan: AnswerPlan) -> str:
    claims = "\n".join(_format_claim_for_prompt(claim) for claim in plan.claims)
    graph_paths = "\n".join(f"- {path}" for path in plan.graph_paths)
    return f"""당신은 S1000D 정비문서 기반 한국어 AI 정비지원 챗봇입니다.

반드시 아래 구조화된 AnswerPlan만 사용해서 답변하세요.
근거 없는 사실, 절차, 공구, 안전 경고를 만들지 마세요.
답변에는 required citations의 DMC를 포함하세요.

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


def _ensure_citations(answer: str, citations: tuple[str, ...]) -> str:
    missing = [dmc for dmc in citations if dmc not in answer]
    if missing:
        answer = answer.rstrip() + "\n근거 DMC: " + ", ".join(citations)
    return answer
