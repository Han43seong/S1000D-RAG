"""Lightweight S1000D graph manifest and query resolver.

This module keeps the first graph-first retrieval slice deliberately small:
procedure-oriented Data Modules are extracted from structured metadata titles,
then Korean/English maintenance queries resolve to candidate DMCs before vector
search.  It is a property-graph/ontology stepping stone, not a replacement for
Chroma.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from langsmith import traceable

from src.config import S1000D_GRAPH_MANIFEST_PATH


@dataclass(frozen=True)
class ProcedureNode:
    dmc: str
    title: str
    target: str
    action: str
    dm_type: str = "procedural"
    sns_code: str = ""


@dataclass(frozen=True)
class FaultNode:
    dmc: str
    title: str
    target: str
    dm_type: str = "fault"
    sns_code: str = ""


@dataclass(frozen=True)
class GraphCandidateResult:
    dmcs: list[str]
    reason: str = ""


@dataclass
class GraphManifest:
    procedures: list[ProcedureNode] = field(default_factory=list)
    faults: list[FaultNode] = field(default_factory=list)

    def find_procedures(self, *, target: str | None = None, action: str | None = None) -> list[ProcedureNode]:
        target_norm = _normalize_token(target or "")
        action_norm = _normalize_action(action or "")
        matches: list[ProcedureNode] = []
        for proc in self.procedures:
            if target_norm and _normalize_token(proc.target) != target_norm:
                continue
            if action_norm and _normalize_action(proc.action) != action_norm:
                continue
            matches.append(proc)
        return matches

    def find_faults(self, *, target: str | None = None) -> list[FaultNode]:
        target_norm = _normalize_token(target or "")
        matches: list[FaultNode] = []
        for fault in self.faults:
            if target_norm and _normalize_token(fault.target) != target_norm:
                continue
            matches.append(fault)
        return matches

    def to_dict(self) -> dict[str, Any]:
        return {
            "procedures": [asdict(proc) for proc in self.procedures],
            "faults": [asdict(fault) for fault in self.faults],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphManifest":
        return cls(
            procedures=[ProcedureNode(**item) for item in data.get("procedures", [])],
            faults=[FaultNode(**item) for item in data.get("faults", [])],
        )


def build_graph_from_chunk_metadata(metadata_rows: Iterable[dict[str, Any]]) -> GraphManifest:
    """Build a procedure graph from Chroma/S1000D chunk metadata rows."""
    seen: set[tuple[str, str, str, str]] = set()
    procedures: list[ProcedureNode] = []
    faults: list[FaultNode] = []
    for meta in metadata_rows:
        dm_type = str(meta.get("dm_type", "")).casefold()
        dmc = str(meta.get("dmc", "")).strip()
        title = str(meta.get("title", "")).strip()
        if not dmc or not title:
            continue
        if dm_type == "fault":
            target = _parse_fault_title(title)
            if target:
                key = (dmc, title, target, "fault")
                if key not in seen:
                    seen.add(key)
                    faults.append(FaultNode(
                        dmc=dmc,
                        title=title,
                        target=target,
                        sns_code=str(meta.get("sns_code", "")),
                    ))
            continue
        if dm_type != "procedural":
            continue
        parsed = _parse_procedure_title(title)
        if parsed is None:
            continue
        target, action = parsed
        key = (dmc, title, target, action)
        if key in seen:
            continue
        seen.add(key)
        procedures.append(ProcedureNode(
            dmc=dmc,
            title=title,
            target=target,
            action=action,
            sns_code=str(meta.get("sns_code", "")),
        ))
    return GraphManifest(procedures=procedures, faults=faults)


def save_graph_manifest(manifest: GraphManifest, path: str | Path = S1000D_GRAPH_MANIFEST_PATH) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_graph_manifest(path: str | Path = S1000D_GRAPH_MANIFEST_PATH) -> GraphManifest | None:
    target = Path(path)
    if not target.exists():
        return None
    return GraphManifest.from_dict(json.loads(target.read_text(encoding="utf-8")))


@traceable(run_type="chain", name="graph_candidate_resolve")
def resolve_graph_candidates(query: str, manifest: GraphManifest | None) -> GraphCandidateResult:
    if manifest is None:
        return GraphCandidateResult(dmcs=[], reason="no_manifest")
    target = _extract_target(query)
    action = _extract_action(query)
    if target and _is_fault_query(query):
        faults = manifest.find_faults(target=target)
        dmcs = _unique_dmcs(fault.dmc for fault in faults)
        return GraphCandidateResult(
            dmcs=dmcs,
            reason=f"target={target};intent=fault" if dmcs else "no_fault_graph_match",
        )
    if not target or not action:
        return GraphCandidateResult(dmcs=[], reason="no_target_or_action")
    matches = manifest.find_procedures(target=target, action=action)
    if not matches and action == "test":
        # Korean users often ask "정상 동작 확인/점검" for S1000D manual-test DMs.
        matches = manifest.find_procedures(target=target, action="manual test")
    dmcs = _unique_dmcs(proc.dmc for proc in matches)
    reason = f"target={target};action={action}" if dmcs else "no_graph_match"
    return GraphCandidateResult(dmcs=dmcs, reason=reason)


def _parse_procedure_title(title: str) -> tuple[str, str] | None:
    if " - " not in title:
        return None
    target_raw, action_raw = title.split(" - ", 1)
    target = _normalize_target(target_raw)
    action = _normalize_action(action_raw)
    if not target or not action:
        return None
    return target, action


def _parse_fault_title(title: str) -> str | None:
    if " - " not in title:
        return None
    target_raw, _fault_raw = title.split(" - ", 1)
    return _normalize_target(target_raw)


def _unique_dmcs(dmcs: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for dmc in dmcs:
        if dmc and dmc not in seen:
            result.append(dmc)
            seen.add(dmc)
    return result


def _is_fault_query(query: str) -> bool:
    q = _normalize_query(query)
    return any(term in q for term in ("fault", "고장", "결함", "켜지지", "작동하지", "동작하지"))


def _extract_target(query: str) -> str | None:
    q = _normalize_query(query)
    target_patterns: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("brake pad", ("브레이크 패드", "brake pad", "brake pads")),
        ("brake system", ("브레이크 시스템", "브레이크", "brake system", "brake", "brakes")),
        ("lights", ("조명", "light", "lights", "lighting")),
        ("chain", ("체인", "chain")),
        ("handlebar", ("핸들바", "핸들", "handlebar", "handle bar")),
        ("stem", ("스템", "stem")),
        ("tire", ("타이어", "tire", "tyre")),
    )
    for target, patterns in target_patterns:
        if any(_contains_phrase(q, pattern) for pattern in patterns):
            return target
    return None


def _extract_action(query: str) -> str | None:
    q = _normalize_query(query)
    action_patterns: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("remove", ("분리", "탈거", "제거", "remove", "removal")),
        ("install", ("장착", "설치", "install", "installation")),
        ("replace", ("교체", "replace", "replacement")),
        ("clean", ("청소", "세척", "clean", "cleaning")),
        ("oil", ("오일", "윤활", "기름", "oil", "lubricate", "lubrication")),
        ("check pressure", ("공기압", "압력", "pressure")),
        ("test", ("수동 점검", "수동 시험", "수동 테스트", "정상 동작", "동작 확인", "켜지지", "점검", "시험", "테스트", "test", "manual test")),
    )
    for action, patterns in action_patterns:
        if any(_contains_phrase(q, pattern) for pattern in patterns):
            return action
    if re.search(r"어떻게\s*확인", q):
        return "test"
    return None


def _normalize_target(value: str) -> str:
    text = _normalize_query(value)
    replacements = {
        "brake pads": "brake pad",
        "lights": "lights",
        "lighting system": "lights",
        "tire": "tire",
        "tyre": "tire",
    }
    text = re.sub(r"\bprocedures?\b", "", text).strip()
    return replacements.get(text, text)


def _normalize_action(value: str) -> str:
    text = _normalize_query(value)
    text = re.sub(r"\bprocedures?\b", "", text).strip()
    if text in {"manual test", "test", "testing"}:
        return "test"
    if text in {"check pressure", "pressure check"}:
        return "check pressure"
    if text in {"fill with air"}:
        return "fill air"
    if text.startswith("clean"):
        return "clean"
    if text.startswith("oil") or text.startswith("lubricat"):
        return "oil"
    if text.startswith("remove"):
        return "remove"
    if text.startswith("install"):
        return "install"
    if text.startswith("replace"):
        return "replace"
    if text.startswith("check") and "pressure" in text:
        return "check pressure"
    return text


def _normalize_token(value: str) -> str:
    return _normalize_query(value)


def _normalize_query(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase_norm = _normalize_query(phrase)
    if re.fullmatch(r"[a-z0-9 ]+", phrase_norm):
        return re.search(rf"(?<![a-z0-9]){re.escape(phrase_norm)}(?![a-z0-9])", text) is not None
    return phrase_norm in text
