"""Ontology-backed structured reference materials for RAG results.

This module is deliberately read-only against the existing ontology export.  If
that export is absent or malformed, callers receive an empty structure so the
normal text answer and chunk evidences continue to work.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.types.rag import Evidence, ReferenceMaterialItem, ReferenceMaterials

_DEFAULT_GRAPH_PATH = Path(__file__).resolve().parents[2] / "ontology" / "s1000d_ontology_graph.json"
_CATEGORY_BY_TYPE = {
    "DataModule": "data_modules",
    "Procedure": "procedures",
    "Fault": "faults",
    "Reference": "references",
    "Warning": "warnings",
    "Caution": "cautions",
    "Figure": "figures",
    "GraphicAsset": "graphic_assets",
    "Hotspot": "hotspots",
}
_PRIMARY_EDGE_PREDICATES = {
    "HAS_PROCEDURE",
    "HAS_FAULT_DOC",
    "HAS_WARNING",
    "HAS_CAUTION",
    "HAS_FIGURE",
    "REFERENCES",
}
_SECONDARY_EDGE_PREDICATES = {"USES_ASSET", "HAS_HOTSPOT", "REFERENCES"}
_DM_LIKE_TYPES = {"DataModule", "Reference", "Warning", "Caution", "Figure"}


def collect_reference_materials(
    evidences: Iterable[Evidence],
    graph_path: str | Path | None = None,
) -> ReferenceMaterials:
    """Collect deterministic ontology-backed evidence for retrieved DMCs.

    Missing/unreadable graph artifacts are a non-fatal empty result by product
    requirement.  Ordering follows retrieved DMC order, graph edge order, then a
    stable item key within each output category.
    """
    dmc_order = _unique_evidence_dmcs(evidences)
    if not dmc_order:
        return ReferenceMaterials()

    graph = _load_graph(graph_path or _DEFAULT_GRAPH_PATH)
    if graph is None:
        return ReferenceMaterials()

    nodes = {str(node.get("id")): node for node in graph.get("nodes", []) if node.get("id")}
    edges = [edge for edge in graph.get("edges", []) if edge.get("source") and edge.get("target")]
    dm_ids_by_norm = _index_data_module_ids(nodes)

    collected: dict[str, dict[str, ReferenceMaterialItem]] = defaultdict(dict)
    seen_edges: set[tuple[str, str, str]] = set()

    for source_rank, dmc in enumerate(dmc_order):
        source_dm_ids = dm_ids_by_norm.get(_normalize_dmc(dmc), {f"dm:{dmc}"})
        for dm_id in sorted(source_dm_ids):
            node = nodes.get(dm_id)
            if node:
                _add_item(collected, _item_from_node(node, "RETRIEVED", dmc, None, source_rank))

            first_hop_targets: list[str] = []
            for edge in edges:
                source = str(edge["source"])
                predicate = str(edge.get("predicate") or edge.get("relation") or edge.get("type") or "")
                target = str(edge["target"])
                if source == dm_id and predicate in _PRIMARY_EDGE_PREDICATES:
                    _collect_edge_target(collected, nodes, edge, dmc, source_rank)
                    first_hop_targets.append(target)
                    seen_edges.add((source, predicate, target))
                elif target == dm_id and predicate in {"GROUNDED_IN", "HAS_PROCEDURE", "HAS_FAULT_DOC"}:
                    _collect_edge_source(collected, nodes, edge, dmc, source_rank)
                    seen_edges.add((source, predicate, target))

            # Related material hanging from first-hop figures/references (assets,
            # hotspots, referenced DMs) is useful in the reference panel but does
            # not change retrieval.
            for edge in edges:
                source = str(edge["source"])
                predicate = str(edge.get("predicate") or edge.get("relation") or edge.get("type") or "")
                target = str(edge["target"])
                if source in first_hop_targets and predicate in _SECONDARY_EDGE_PREDICATES:
                    # Reference nodes already represent referenced documents in
                    # the references category; do not duplicate their target DMs
                    # as retrieved data modules.
                    target_node = nodes.get(target)
                    if predicate == "REFERENCES" and target_node and target_node.get("type") == "DataModule":
                        continue
                    if (source, predicate, target) not in seen_edges:
                        _collect_edge_target(collected, nodes, edge, dmc, source_rank)
                        seen_edges.add((source, predicate, target))

    return ReferenceMaterials(
        data_modules=_sorted_items(collected["data_modules"].values()),
        procedures=_sorted_items(collected["procedures"].values()),
        faults=_sorted_items(collected["faults"].values()),
        references=_sorted_items(collected["references"].values()),
        warnings=_sorted_items(collected["warnings"].values()),
        cautions=_sorted_items(collected["cautions"].values()),
        figures=_sorted_items(collected["figures"].values()),
        graphic_assets=_sorted_items(collected["graphic_assets"].values()),
        hotspots=_sorted_items(collected["hotspots"].values()),
    )


def _load_graph(path: str | Path) -> Mapping[str, Any] | None:
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            graph = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        return None
    return graph


def _unique_evidence_dmcs(evidences: Iterable[Evidence]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for evidence in evidences:
        dmc = (evidence.dmc or "").strip()
        if dmc and dmc not in seen:
            seen.add(dmc)
            ordered.append(dmc)
    return ordered


def _index_data_module_ids(nodes: Mapping[str, Mapping[str, Any]]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for node_id, node in nodes.items():
        node_type = node.get("type")
        props = node.get("properties") or {}
        if node_type == "DataModule":
            dmc = props.get("dmc") or str(node_id).removeprefix("dm:")
            index[_normalize_dmc(str(dmc))].add(str(node_id))
    return index


def _collect_edge_target(
    collected: dict[str, dict[str, ReferenceMaterialItem]],
    nodes: Mapping[str, Mapping[str, Any]],
    edge: Mapping[str, Any],
    source_dmc: str,
    source_rank: int,
) -> None:
    node = nodes.get(str(edge["target"]))
    if node:
        _add_item(collected, _item_from_node(node, str(edge.get("predicate") or ""), source_dmc, str(edge["target"]), source_rank))


def _collect_edge_source(
    collected: dict[str, dict[str, ReferenceMaterialItem]],
    nodes: Mapping[str, Mapping[str, Any]],
    edge: Mapping[str, Any],
    source_dmc: str,
    source_rank: int,
) -> None:
    node = nodes.get(str(edge["source"]))
    if node:
        _add_item(collected, _item_from_node(node, str(edge.get("predicate") or ""), source_dmc, str(edge["target"]), source_rank))


def _add_item(collected: dict[str, dict[str, ReferenceMaterialItem]], item: ReferenceMaterialItem) -> None:
    category = _CATEGORY_BY_TYPE.get(item.type or "")
    if category:
        existing = collected[category].get(item.id)
        if existing is None or item.source_rank < existing.source_rank:
            collected[category][item.id] = item


def _item_from_node(
    node: Mapping[str, Any],
    relation: str,
    source_dmc: str,
    target_id: str | None,
    source_rank: int,
) -> ReferenceMaterialItem:
    props = node.get("properties") or {}
    node_id = str(node.get("id") or "")
    node_type = str(node.get("type") or "")
    dmc = _display_dmc(props.get("dmc") or (node_id.removeprefix("dm:") if node_id.startswith("dm:") else None))
    return ReferenceMaterialItem(
        id=node_id,
        label=str(node.get("label") or props.get("title") or dmc or node_id),
        title=str(node.get("label") or props.get("title") or "") or None,
        dmc=dmc,
        type=node_type,
        category=_CATEGORY_BY_TYPE.get(node_type),
        relation=relation or None,
        source_dmc=_display_dmc(source_dmc),
        target_dmc=_target_dmc(node, target_id),
        text=str(props.get("text") or "") or None,
        metadata={str(k): v for k, v in props.items() if v is not None},
        source_rank=source_rank,
        sort_key=f"{source_rank:04d}|{node_type}|{dmc or ''}|{node.get('label') or ''}|{node_id}",
    )


def _target_dmc(node: Mapping[str, Any], target_id: str | None) -> str | None:
    props = node.get("properties") or {}
    if props.get("dmc"):
        return _display_dmc(str(props["dmc"]))
    if target_id and target_id.startswith("dm:"):
        return _display_dmc(target_id.removeprefix("dm:"))
    return None


def _sorted_items(items: Iterable[ReferenceMaterialItem]) -> list[ReferenceMaterialItem]:
    return sorted(items, key=lambda item: (item.source_rank, item.sort_key, item.id))


def _normalize_dmc(dmc: str) -> str:
    """Normalize DMCs across XML-export short integer variants.

    The ontology export sometimes stores issue segments like ``00-00`` as
    ``0-0`` or ``10-00`` as ``1-0``.  Removing leading zeros from purely numeric
    hyphen-separated tokens gives a stable join key without changing display.
    """
    parts = []
    for part in dmc.strip().upper().split("-"):
        if re.fullmatch(r"\d+", part):
            parts.append(str(int(part)))
        else:
            parts.append(part)
    return "-".join(parts)


def _display_dmc(dmc: object | None) -> str | None:
    if dmc is None:
        return None
    return str(dmc)
