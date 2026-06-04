"""Normalize S1000D evidence claims into Korean user-answer units."""
from __future__ import annotations

import re


def normalize_claims(claims: list[str] | tuple[str, ...], *, limit: int = 8) -> list[str]:
    """Convert raw evidence claims into deduplicated Korean answer units."""
    normalized: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        for unit in normalize_claim_text(claim):
            if unit in seen:
                continue
            seen.add(unit)
            normalized.append(unit)
            if len(normalized) >= limit:
                return normalized
    return normalized


def normalize_claim_text(text: str) -> list[str]:
    """Convert one raw claim into zero or more Korean answer units.

    Unknown English evidence is intentionally dropped instead of being exposed or
    converted into a generic placeholder. Existing Korean claims are preserved.
    """
    original = _clean_source_noise(text)
    lowered = original.lower()
    units: list[str] = []
    replacements = [
        ("has these primary components", "브레이크 시스템은 브레이크 레버, 브레이크 케이블, 브레이크 암, 브레이크 클램프(캘리퍼), 브레이크 패드로 구성됩니다."),
        ("the brake lever the brake cable the brake arm the brake clamp", "브레이크 레버, 브레이크 케이블, 브레이크 암, 브레이크 클램프가 주요 구성품입니다."),
        ("also known as callipers", "브레이크 클램프는 캘리퍼라고도 부릅니다."),
        ("a cable that goes from the brake levers on the handlebars pulls the two levers on the brakes together", "핸들바의 브레이크 레버에서 이어진 케이블이 브레이크 쪽 두 레버를 함께 당깁니다."),
        ("presses the brake pads against the outer rim of the wheel", "브레이크 패드가 바퀴의 바깥쪽 림을 누릅니다."),
        ("decreases the speed of the bicycle", "이 마찰로 자전거 속도가 줄어듭니다."),
        ("there are four brake pads", "자전거에는 브레이크 패드가 네 개 있습니다."),
        ("two are found on the front wheel and two on the rear wheel", "앞바퀴와 뒷바퀴에 각각 두 개씩 배치됩니다."),
        ("the brake pads are made out of hard wearing rubber", "브레이크 패드는 내마모성 고무로 만들어집니다."),
        ("the pads press against the rim of the wheel to cause friction", "패드가 바퀴 림을 눌러 마찰을 발생시킵니다."),
        ("install the fork and the brakes before installing the wheel", "바퀴를 설치하기 전에 포크와 브레이크가 먼저 장착되어 있는지 확인합니다."),
        ("install the fork before installing the wheel", "바퀴를 설치하기 전에 포크가 먼저 장착되어 있는지 확인합니다."),
        ("hold the front of the bicycle", "자전거 앞부분을 안정적으로 잡습니다."),
        ("install the wheel", "바퀴를 장착합니다."),
        ("be careful to not damage the chainring", "체인링이 손상되지 않도록 주의합니다."),
        ("close the light circuit breaker located on the handlebar", "핸들바에 있는 라이트 회로 차단기를 닫습니다."),
        ("open the light circuit breaker located on the handlebar", "핸들바에 있는 라이트 회로 차단기를 엽니다."),
        ("put the bike on the floor", "자전거를 바닥에 내려놓습니다."),
        ("lift the wheel away from the frame", "바퀴를 프레임에서 들어 올려 분리합니다."),
        ("put the frame on the floor", "프레임을 바닥에 내려놓습니다."),
        ("use specific oil if the fork do not desengage easily", "포크가 쉽게 분리되지 않으면 지정된 오일을 사용합니다."),
        ("if not available, use any oil compliant with requirements", "지정 오일이 없으면 요구사항을 만족하는 오일을 사용합니다."),
        ("disengage the fork from the chainring", "포크를 체인링에서 분리합니다."),
        ("pushing the wheel forwards and down", "바퀴를 앞으로 밀고 아래로 내려 작업합니다."),
    ]
    matches: list[tuple[int, str]] = []
    for english, korean in replacements:
        pos = lowered.find(english)
        if pos >= 0:
            matches.append((pos, korean))
    units = [korean for _, korean in sorted(matches, key=lambda item: item[0])]
    if units:
        return _remove_subsumed_units(_dedupe(units))
    if _contains_hangul(original):
        return [_ensure_sentence(original)]
    return []


def _clean_source_noise(text: str) -> str:
    cleaned = re.sub(r"\(refer to\s*\)", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.rstrip(".")


def _contains_hangul(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


def _ensure_sentence(text: str) -> str:
    text = text.strip().rstrip(".")
    return text + "."


def _remove_subsumed_units(items: list[str]) -> list[str]:
    result = list(items)
    full_brake_components = "브레이크 시스템은 브레이크 레버, 브레이크 케이블, 브레이크 암, 브레이크 클램프(캘리퍼), 브레이크 패드로 구성됩니다."
    partial_brake_components = "브레이크 레버, 브레이크 케이블, 브레이크 암, 브레이크 클램프가 주요 구성품입니다."
    if full_brake_components in result and partial_brake_components in result:
        result.remove(partial_brake_components)
    return result


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
