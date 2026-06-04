"""Build and load the ontology manifest used by RAG v2."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from src.config import S1000D_GRAPH_MANIFEST_PATH
from src.rag.graph_retrieval import load_graph_manifest

from .schema import OntologyNode

DEFAULT_ONTOLOGY_MANIFEST_PATH = Path("chroma_db_full/s1000d_ontology_manifest.json")

DESCRIPTION_NODES: tuple[OntologyNode, ...] = (
    OntologyNode(
        dmc="BRAKE-AAA-DA1-00-00-00AA-041A-A",
        title="Brake system - Description",
        dm_type="descriptive",
        sns_code="DA1",
        target="brake system",
        aliases=("브레이크", "브레이크 시스템", "brake", "brake system"),
        metadata={"components": ["브레이크 레버", "브레이크 케이블", "브레이크 암", "브레이크 패드"]},
    ),
    OntologyNode(
        dmc="S1000DBIKE-AAA-DA1-00-00-00AA-041A-A",
        title="Brake system - Description",
        dm_type="descriptive",
        sns_code="DA1",
        target="brake system",
        aliases=("브레이크", "브레이크 시스템", "brake system"),
        metadata={"components": ["브레이크 레버", "브레이크 케이블", "브레이크 암", "브레이크 패드"]},
    ),
)

TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "brake system": ("브레이크 시스템", "브레이크", "brake system", "brake"),
    "brake pad": ("브레이크 패드", "브레이크패드", "brake pad", "brake pads"),
    "brake cable": ("브레이크 케이블", "브레이크케이블", "brake cable"),
    "front wheel": ("앞바퀴", "전륜", "프론트 휠", "front wheel"),
    "rear wheel": ("뒷바퀴", "후륜", "rear wheel"),
    "wheel": ("바퀴", "휠", "wheel", "wheels"),
    "tire": ("타이어", "tire", "tyre"),
    "chain": ("체인", "chain"),
    "handlebar": ("핸들바", "핸들 바", "handlebar"),
    "lights": ("조명 시스템", "조명", "라이트", "등화", "lights", "lighting"),
    "stem": ("스템", "stem"),
}

ACTION_ALIASES: dict[str, tuple[str, ...]] = {
    "describe": ("설명", "알려줘", "무엇", "정보", "describe"),
    "list_components": ("주요 구성품", "구성품", "부품", "components", "parts"),
    "clean": ("청소", "세척", "clean"),
    "install": ("설치", "장착", "조립", "install", "installation"),
    "remove": ("탈거", "제거", "분리", "remove", "removal"),
    "replace": ("교체", "바꾸", "replace", "replacement"),
    "oil": ("오일", "윤활", "기름", "바르는", "oil", "lubricate"),
    "test": ("점검", "검사", "시험", "테스트", "확인", "test", "check", "inspection"),
}


def build_ontology_manifest(graph_manifest_path: str | Path = S1000D_GRAPH_MANIFEST_PATH) -> list[OntologyNode]:
    nodes: dict[tuple[str, str | None, str | None], OntologyNode] = {}
    for node in DESCRIPTION_NODES:
        nodes[(node.dmc, node.target, node.action)] = node

    manifest = load_graph_manifest(graph_manifest_path)
    if manifest:
        for proc in manifest.procedures:
            target = _canonical_target(proc.target)
            action = _canonical_action(proc.action)
            aliases = (*TARGET_ALIASES.get(target, ()), *ACTION_ALIASES.get(action, ()))
            node = OntologyNode(
                dmc=proc.dmc,
                title=proc.title,
                dm_type=proc.dm_type,
                sns_code=proc.sns_code,
                target=target,
                action=action,
                aliases=tuple(dict.fromkeys(alias for alias in aliases if alias)),
            )
            nodes[(node.dmc, node.target, node.action)] = node
        for fault in manifest.faults:
            target = _canonical_target(fault.target)
            nodes[(fault.dmc, target, "fault")] = OntologyNode(
                dmc=fault.dmc,
                title=fault.title,
                dm_type=fault.dm_type,
                sns_code=fault.sns_code,
                target=target,
                action="fault",
                aliases=TARGET_ALIASES.get(target, ()),
            )
    return sorted(nodes.values(), key=lambda n: (n.dmc, n.target or "", n.action or ""))


def save_ontology_manifest(nodes: Iterable[OntologyNode], path: str | Path = DEFAULT_ONTOLOGY_MANIFEST_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(n) for n in nodes], ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_ontology_manifest(path: str | Path = DEFAULT_ONTOLOGY_MANIFEST_PATH) -> list[OntologyNode]:
    target = Path(path)
    if target.exists():
        data = json.loads(target.read_text(encoding="utf-8"))
        return [OntologyNode(**item) for item in data]
    return build_ontology_manifest()


def _canonical_target(value: str) -> str:
    v = " ".join(value.casefold().replace("pads", "pad").split())
    return {"brake pads": "brake pad", "lighting": "lights"}.get(v, v)


def _canonical_action(value: str) -> str:
    v = " ".join(value.casefold().split())
    if "clean" in v:
        return "clean"
    if "install" in v and "remove" not in v:
        return "install"
    if "remove" in v:
        return "remove"
    if "manual test" in v or v == "test":
        return "test"
    if "oil" in v:
        return "oil"
    if "replace" in v or "new item" in v:
        return "replace"
    return v
