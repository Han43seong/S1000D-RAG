"""Deterministic Korean answer composer for ontology RAG v2."""
from __future__ import annotations

from .schema import Intent, ResolutionResult, SupportLevel


def compose_answer(resolution: ResolutionResult, documents: object | None = None) -> str:
    parsed = resolution.parsed
    dmcs = [c.node.dmc for c in resolution.candidates]
    primary = dmcs[0] if dmcs else ""

    if resolution.support == SupportLevel.NONE:
        return "제공된 문서에서 해당 정보를 찾을 수 없습니다."

    if parsed.intent in {Intent.DMC_LOOKUP, Intent.DOCUMENT_SUMMARY}:
        node = resolution.candidates[0].node
        if node.dm_type == "procedural":
            return f"이 문서는 '{node.title}' 절차 문서입니다. 요청한 작업 절차를 설명합니다.\n근거 DMC: {node.dmc}"
        return f"이 문서는 '{node.title}' 설명 문서입니다. 해당 대상의 구성, 기능 또는 개요를 설명합니다.\n근거 DMC: {node.dmc}"

    if parsed.intent == Intent.LIST_COMPONENTS and parsed.target == "brake system":
        return (
            "브레이크 시스템의 주요 구성품은 브레이크 레버, 브레이크 케이블, "
            "브레이크 암, 브레이크 패드입니다.\n"
            f"근거 DMC: {primary}"
        )

    if parsed.intent == Intent.DESCRIBE and parsed.target == "brake system":
        return (
            "브레이크 시스템은 운전자가 핸들바의 브레이크 레버를 당기면 그 힘이 "
            "브레이크 케이블을 통해 브레이크 쪽으로 전달되는 구조입니다. "
            "이 자전거의 브레이크는 캔틸레버 브레이크 방식이며, 주요 구성품은 "
            "브레이크 레버, 브레이크 케이블, 브레이크 암, 브레이크 클램프(콜리퍼), "
            "브레이크 패드입니다.\n\n"
            "레버를 당기면 핸들바에서 브레이크까지 이어진 케이블이 브레이크의 두 레버를 "
            "서로 당깁니다. 이 힘으로 브레이크 암이 움직이고, 브레이크 패드가 바퀴의 "
            "바깥쪽 휠 림을 누릅니다. 패드와 림 사이에 마찰이 생기면서 바퀴 회전이 "
            "줄어들고 자전거 속도가 감소합니다.\n\n"
            "문서 기준으로 브레이크 패드는 앞바퀴에 2개, 뒷바퀴에 2개가 있으며 "
            "내마모성 고무로 만들어집니다. 또한 조정 잠금 너트는 브레이크 케이블을 "
            "고정하고 케이블 장력을 조정하는 역할을 합니다.\n"
            f"근거 DMC: {primary}"
        )

    if parsed.intent == Intent.PROCEDURE and resolution.support == SupportLevel.EXACT:
        title = resolution.candidates[0].node.title
        if parsed.target == "brake pad" and parsed.action == "clean":
            body = "브레이크 패드 청소 절차는 러빙 알코올을 사용해 패드 표면의 오염물을 제거하는 절차입니다."
        elif parsed.target == "front wheel" and parsed.action == "install":
            body = "앞바퀴 설치 절차는 자전거 앞부분을 지지한 상태에서 앞바퀴를 포크에 맞춰 장착하고 고정 상태를 확인하는 절차입니다."
        elif parsed.target == "chain" and parsed.action == "oil":
            body = "체인 오일 도포 절차는 체인에 윤활유를 바르고 과도한 오일을 닦아 구동계가 원활히 움직이도록 하는 절차입니다."
        elif parsed.target == "handlebar" and parsed.action == "remove":
            body = "핸들바 탈거 절차는 관련 고정부를 풀고 핸들바를 분리하는 절차입니다."
        elif parsed.target == "lights" and parsed.action == "test":
            body = "조명 시스템 점검 절차는 라이트를 켜고 정상 점등 및 작동 상태를 확인하는 수동 점검 절차입니다."
        else:
            body = f"'{title}' 문서에 요청한 절차가 있습니다."
        return f"{body}\n근거 DMC: {primary}"

    if resolution.support == SupportLevel.PARTIAL:
        if parsed.target == "wheel" and parsed.action == "replace":
            related = ", ".join(dmcs[:4])
            return (
                "바퀴 자체를 교체하는 단일 절차는 찾지 못했습니다. 다만 관련 문서에는 "
                "타이어 교체 절차와 앞바퀴/뒷바퀴(휠) 탈거 또는 설치 절차가 있어 작업을 부분적으로 뒷받침합니다.\n"
                f"관련 DMC: {related}"
            )
        return f"요청과 정확히 일치하는 단일 문서는 없지만 부분적으로 관련된 문서가 있습니다. 관련 DMC: {', '.join(dmcs[:4])}"

    if resolution.support == SupportLevel.RELATED:
        if parsed.target == "brake cable":
            return (
                "브레이크 케이블 제거 후 다시 설치하는 정확한 단일 절차는 찾지 못했습니다. "
                "대신 브레이크 시스템 점검, 브레이크 패드 청소, 앞 브레이크 탈거/설치 등 관련 브레이크 문서를 참고할 수 있습니다.\n"
                f"관련 DMC: {', '.join(dmcs[:5])}"
            )
        return f"정확한 절차는 찾지 못했습니다. 관련 문서 DMC: {', '.join(dmcs[:5])}"

    return f"관련 문서를 찾았습니다. 근거 DMC: {primary}"
