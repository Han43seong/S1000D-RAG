"""Phrase/token-safe Korean/English query parser for ontology RAG v2."""
from __future__ import annotations

import re

from .manifest_builder import ACTION_ALIASES, TARGET_ALIASES
from .schema import Intent, ParsedQuery

_DMC_RE = re.compile(r"\b[A-Z0-9]+-[A-Z0-9-]+\b", re.I)


def parse_query(query: str) -> ParsedQuery:
    normalized = normalize_query(query)
    dmc_match = _DMC_RE.search(query)
    target, target_aliases = _match_alias_group(normalized, TARGET_ALIASES)
    procedural_actions = {k: v for k, v in ACTION_ALIASES.items() if k not in {"describe", "list_components"}}
    action, action_aliases = _match_alias_group(normalized, procedural_actions)
    if action is None:
        action, action_aliases = _match_alias_group(normalized, ACTION_ALIASES)

    if dmc_match:
        intent = Intent.DMC_LOOKUP
    elif _has_any(normalized, ACTION_ALIASES["list_components"]):
        intent = Intent.LIST_COMPONENTS
        action = None
    elif action in {"clean", "install", "remove", "replace", "oil", "test"} or _has_any(normalized, ("절차", "방법", "procedure")):
        intent = Intent.PROCEDURE
    elif _has_any(normalized, ("알려준", "앞서", "이전", "그 문서", "내용")) and not target:
        intent = Intent.FOLLOW_UP
    elif target:
        intent = Intent.DESCRIBE
    else:
        intent = Intent.UNKNOWN

    confidence = 0.35 + (0.35 if target else 0.0) + (0.2 if action else 0.0) + (0.1 if intent != Intent.UNKNOWN else 0.0)
    return ParsedQuery(
        original=query,
        normalized=normalized,
        intent=intent,
        target=target,
        action=action,
        dm_type="procedural" if intent == Intent.PROCEDURE else None,
        confidence=min(confidence, 1.0),
        matched_aliases=tuple(dict.fromkeys((*target_aliases, *action_aliases))),
        follow_up=intent == Intent.FOLLOW_UP,
    )


def normalize_query(query: str) -> str:
    text = query.casefold()
    text = re.sub(r"[^0-9a-zA-Z가-힣\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _match_alias_group(normalized: str, groups: dict[str, tuple[str, ...]]) -> tuple[str | None, tuple[str, ...]]:
    matches: list[tuple[int, str, str]] = []
    for canonical, aliases in groups.items():
        for alias in aliases:
            if _contains_phrase(normalized, normalize_query(alias)):
                matches.append((len(normalize_query(alias)), canonical, alias))
    if not matches:
        return None, ()
    matches.sort(key=lambda item: item[0], reverse=True)
    canonical = matches[0][1]
    return canonical, tuple(alias for _, group, alias in matches if group == canonical)


def _has_any(normalized: str, aliases: tuple[str, ...]) -> bool:
    return any(_contains_phrase(normalized, normalize_query(alias)) for alias in aliases)


def _contains_phrase(normalized: str, phrase: str) -> bool:
    if not phrase:
        return False
    # Korean phrases have no reliable spaces; require exact phrase presence, not
    # single-syllable fragments. This prevents 시스템 -> 스템 because "스템" is
    # not equal to the multi-token alias "조명 시스템" and is only tested as its
    # own phrase.
    if re.search(r"[가-힣]", phrase):
        return phrase in normalized
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", normalized) is not None
