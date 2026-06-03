"""RAG 프롬프트 템플릿 모듈.

대화 이력 지원 + 한↔영 기술 용어 대응 가이드를 포함하는
개선된 프롬프트를 생성한다.
"""

from __future__ import annotations

_SYSTEM_TEMPLATE = """\
당신은 S1000D 기술 교범 어시스턴트입니다.
아래 Context/참고 문서(영어)를 읽고 반드시 한국어로만 답변하세요.

한↔영 기술 용어 가이드:
- Remove = 탈거/제거
- Install = 장착/설치
- Inspect = 점검/검사
- Replace = 교체
- Clean = 청소
- Test = 시험/테스트
- Brake = 브레이크
- Wheel = 휠/바퀴
- Cable = 케이블
- Pad = 패드

답변 규칙:
1. 반드시 한국어 최종 답변만 작성하세요.
2. Context 원문을 그대로 복사하지 마세요. 영어 원문은 한국어로 요약/번역하세요.
3. 문서에 없는 일반 지식이나 절차를 추가하지 마세요. 반드시 문서 내용만 근거로 사용하세요.
4. 질문이 문서 범위보다 넓으면 "제공된 문서 기준으로는"이라고 범위를 제한하세요.
5. 절차/방법/교체/설치/장착/탈거/점검 질문에서 해당 작업 절차가 Context에 없으면 절차를 만들지 말고 "제공된 문서에서 해당 절차를 찾을 수 없습니다."라고 답하세요.
6. 참고 문서에 전혀 관련 없는 내용만 있으면 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고만 답하세요.
7. 답변 끝에는 참고 문서 DMC를 표기하세요.
8. 사고 과정, reasoning, <think> 블록은 절대 출력하지 말고 최종 답변만 작성하세요.
9. 답변은 한 번만 작성하세요.

출력은 반드시 다음 형식을 따르세요:
답변: <한국어 답변>
참고 문서: <DMC 목록 또는 없음>"""


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

    parts.append(f"\nContext / 참고 문서:\n{context}")
    # Keep a model-native no-thinking control token in the prompt so users do not
    # need to append it manually in the web UI.
    parts.append("\n/no_think")
    parts.append(f"\nQuestion / 질문: {question}")
    parts.append("\n답변:")

    return "\n".join(parts)
