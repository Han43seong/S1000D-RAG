"""S1000D DM 타입 판별, DMC 문자열 빌드, 텍스트 정규화 유틸리티."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from lxml import etree

from src.types.dm import ContentBlockRole, DmType

if TYPE_CHECKING:
    pass


# ── S1000D info code → DmType 매핑 ──────────────────────────────────────
# info code 3자리 중 앞 3자리로 분류 (S1000D Issue 6 기준)
_INFO_CODE_TYPE_MAP: dict[str, DmType] = {
    # Procedural: 수리, 정비, 탈거, 장착, 시험, 세척, 교정 등
    "100": DmType.PROCEDURAL,
    "121": DmType.PROCEDURAL,
    "130": DmType.PROCEDURAL,
    "131": DmType.PROCEDURAL,
    "200": DmType.PROCEDURAL,
    "241": DmType.PROCEDURAL,
    "251": DmType.PROCEDURAL,
    "258": DmType.PROCEDURAL,
    "300": DmType.PROCEDURAL,
    "341": DmType.PROCEDURAL,
    "500": DmType.PROCEDURAL,
    "520": DmType.PROCEDURAL,
    "600": DmType.PROCEDURAL,
    "700": DmType.PROCEDURAL,
    "900": DmType.PROCEDURAL,
    "921": DmType.PROCEDURAL,
    "151": DmType.PROCEDURAL,
    # Descriptive
    "001": DmType.DESCRIPTIVE,
    "002": DmType.DESCRIPTIVE,
    "009": DmType.DESCRIPTIVE,
    "010": DmType.DESCRIPTIVE,
    "018": DmType.DESCRIPTIVE,
    "040": DmType.DESCRIPTIVE,
    "041": DmType.DESCRIPTIVE,
    "042": DmType.DESCRIPTIVE,
    "043": DmType.DESCRIPTIVE,
    # Fault
    "270": DmType.FAULT,
    "271": DmType.FAULT,
    "272": DmType.FAULT,
    # IPD (Illustrated Parts Data)
    "0A3": DmType.IPD,
    "018": DmType.DESCRIPTIVE,
    # Crew / operational
    "051": DmType.CREW,
    "052": DmType.CREW,
}


def detect_dm_type(content_el: etree._Element, info_code: str) -> DmType:
    """content 요소의 자식 구조 + infoCode로 DM 타입 판별.

    1차: content 하위 요소 태그로 직접 판별 (가장 정확)
    2차: info_code 매핑 테이블 참조
    3차: fallback → descriptive
    """
    if content_el is not None:
        child_tags = {child.tag for child in content_el}
        if "procedure" in child_tags:
            return DmType.PROCEDURAL
        if "description" in child_tags:
            return DmType.DESCRIPTIVE
        if "faultReporting" in child_tags or "faultIsolation" in child_tags:
            return DmType.FAULT
        if "illustratedPartsCatalog" in child_tags:
            return DmType.IPD
        if "crew" in child_tags:
            return DmType.CREW
        if "process" in child_tags:
            return DmType.PROCESS

    return _INFO_CODE_TYPE_MAP.get(info_code, DmType.DESCRIPTIVE)


# ── DMC 문자열 빌드 ─────────────────────────────────────────────────────

_DMC_ATTRS = [
    "modelIdentCode",
    "systemDiffCode",
    "systemCode",
    "subSystemCode",
    "subSubSystemCode",
    "assyCode",
    "disassyCode",
    "disassyCodeVariant",
    "infoCode",
    "infoCodeVariant",
    "itemLocationCode",
]


def build_dmc_string(dm_code_el: etree._Element) -> str:
    """dmCode 요소의 속성들을 조합하여 DMC 문자열 생성.

    형식: {modelIdentCode}-{systemDiffCode}-{systemCode}-
          {subSystemCode}{subSubSystemCode}-{assyCode}-
          {disassyCode}{disassyCodeVariant}-
          {infoCode}{infoCodeVariant}-{itemLocationCode}
    """
    g = {attr: (dm_code_el.get(attr) or "") for attr in _DMC_ATTRS}
    return (
        f"{g['modelIdentCode']}-{g['systemDiffCode']}-"
        f"{g['systemCode']}-{g['subSystemCode']}{g['subSubSystemCode']}-"
        f"{g['assyCode']}-{g['disassyCode']}{g['disassyCodeVariant']}-"
        f"{g['infoCode']}{g['infoCodeVariant']}-{g['itemLocationCode']}"
    )


def extract_info_code(dm_code_el: etree._Element) -> str:
    """dmCode 요소에서 infoCode 3자리 추출."""
    return dm_code_el.get("infoCode", "")


# ── 텍스트 정규화 ───────────────────────────────────────────────────────

_MULTI_SPACE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """공백 정규화, 앞뒤 trim."""
    return _MULTI_SPACE.sub(" ", raw).strip()


def extract_text_content(el: etree._Element) -> str:
    """요소와 모든 자식의 텍스트를 재귀적으로 추출하여 정규화.

    inline 마크업(emphasis, subScript, superScript 등)은 텍스트로 합침.
    internalRef, dmRef 등 참조 요소는 무시.
    """
    parts: list[str] = []
    _collect_text(el, parts)
    return clean_text("".join(parts))


def _collect_text(el: etree._Element, parts: list[str]) -> None:
    """재귀적 텍스트 수집 헬퍼."""
    # 참조/그래픽 요소는 스킵
    skip_tags = {"internalRef", "dmRef", "pmRef", "graphic", "hotspot",
                 "indexFlag", "changeInline", "reasonForUpdate"}
    if el.tag in skip_tags:
        return

    if el.text:
        parts.append(el.text)

    for child in el:
        _collect_text(child, parts)
        if child.tail:
            parts.append(child.tail)


# ── 블록 ID 생성 ────────────────────────────────────────────────────────

class BlockIdGenerator:
    """content_blocks용 순차 ID 생성기."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def next_id(self, role: ContentBlockRole) -> str:
        """role별 순차 번호가 포함된 ID 반환. 예: 'para-1', 'step-3'."""
        key = role.value
        self._counters[key] = self._counters.get(key, 0) + 1
        return f"{key}-{self._counters[key]}"
