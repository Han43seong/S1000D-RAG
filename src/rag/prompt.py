"""RAG 프롬프트 템플릿 모듈.

대화 이력 지원 + 한↔영 기술 용어 대응 가이드를 포함하는
개선된 프롬프트를 생성한다.
"""

from __future__ import annotations


_SYSTEM_TEMPLATE = """\
당신은 S1000D 기술 교범 어시스턴트입니다.
아래 참고 문서(영어)를 읽고 반드시 한국어로만 답변하세요.

답변 규칙:
1. 참고 문서에 관련 정보가 있으면 해당 내용을 근거로 답변하세요.
   - 직접적 답이 없어도 관련 수치·규격·절차가 있으면 그것을 인용하여 추론하세요.
   - 예: 정상 범위가 문서에 있고 범위 초과를 질문하면, 정상 범위를 인용한 뒤 초과 상태를 설명하세요.
2. 참고 문서에 전혀 관련 없는 내용만 있으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고만 답하세요.
3. 문서에 없는 일반 지식을 추가하지 마세요. 반드시 문서 내용만 근거로 사용하세요.
4. 답변 끝에 근거 DMC를 표기하세요.
5. 절차가 있으면 단계별로 정리하세요.
6. 답변은 한 번만 작성하세요."""


def build_prompt(
    question: str,
    context: str,
    conversation_history: list[tuple[str, str]] | None = None,
) -> str:
    """RAG 프롬프트 문자열 생성.

    Args:
        question: 사용자 질문.
        context: 검색된 문서 컨텍스트.
        conversation_history: 최근 대화 이력 [(user, assistant), ...].

    Returns:
        LLM에 전달할 프롬프트 문자열.
    """
    parts: list[str] = [_SYSTEM_TEMPLATE]

    # 대화 이력 추가
    if conversation_history:
        history_lines: list[str] = []
        for user_msg, assistant_msg in conversation_history:
            history_lines.append(f"사용자: {user_msg}")
            history_lines.append(f"어시스턴트: {assistant_msg}")
        history_text = "\n".join(history_lines)
        parts.append(f"\n이전 대화:\n{history_text}")

    parts.append(f"\n참고 문서:\n{context}")
    parts.append(f"\n질문: {question}")
    parts.append("\n답변:")

    return "\n".join(parts)
