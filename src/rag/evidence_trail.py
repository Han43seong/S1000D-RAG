"""Ontology-backed structured reference materials for RAG results.

This module is deliberately read-only against the existing ontology export.  If
that export is absent or malformed, callers receive an empty structure so the
normal text answer and chunk evidences continue to work.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.types.rag import Evidence, ReferenceMaterialItem, ReferenceMaterials

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GRAPH_PATH = _REPO_ROOT / "ontology" / "s1000d_ontology_graph.json"
DEFAULT_GRAPHIC_ASSET_ROOT = _REPO_ROOT / "docs" / "S1000D Issue 6" / "Bike Data Set for Release number 6 R2"
_RENDERABLE_EXTENSIONS = ("png", "jpg", "jpeg", "svg", "gif", "webp")
_GRAPHIC_EXTENSION_ORDER = (*_RENDERABLE_EXTENSIONS, "cgm")
_ASSET_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
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
    asset_root: str | Path | None = None,
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
                _add_item(collected, _item_from_node(node, "RETRIEVED", dmc, None, source_rank, asset_root=asset_root))

            first_hop_targets: list[str] = []
            for edge in edges:
                source = str(edge["source"])
                predicate = str(edge.get("predicate") or edge.get("relation") or edge.get("type") or "")
                target = str(edge["target"])
                if source == dm_id and predicate in _PRIMARY_EDGE_PREDICATES:
                    _collect_edge_target(collected, nodes, edge, dmc, source_rank, asset_root=asset_root)
                    first_hop_targets.append(target)
                    seen_edges.add((source, predicate, target))
                elif target == dm_id and predicate in {"GROUNDED_IN", "HAS_PROCEDURE", "HAS_FAULT_DOC"}:
                    _collect_edge_source(collected, nodes, edge, dmc, source_rank, asset_root=asset_root)
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
                        _collect_edge_target(collected, nodes, edge, dmc, source_rank, asset_root=asset_root)
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



@dataclass(frozen=True)
class GraphicAssetPreview:
    preview_url: str | None = None
    original_url: str | None = None
    asset_format: str | None = None
    preview_available: bool = False
    preview_status: str | None = None


def resolve_graphic_asset_preview(
    metadata: Mapping[str, Any] | None,
    asset_root: str | Path | None = None,
) -> GraphicAssetPreview:
    """Resolve a GraphicAsset preview from trusted ICN-like metadata only.

    The resolver never consumes arbitrary filesystem paths. It extracts an ICN or
    asset id, searches below the known S1000D asset root, and returns only safe
    backend URLs using deterministic extension preference.
    """
    stem = _graphic_asset_stem(metadata or {})
    if not stem:
        return GraphicAssetPreview(preview_status="missing")

    root = Path(asset_root or DEFAULT_GRAPHIC_ASSET_ROOT).resolve()
    matches = _find_graphic_asset_files(root, stem)
    if not matches:
        return GraphicAssetPreview(preview_status="missing")

    for ext in _RENDERABLE_EXTENSIONS:
        path = matches.get(ext)
        if path is not None:
            url = _graphic_asset_url(stem, ext)
            return GraphicAssetPreview(
                preview_url=url,
                original_url=url,
                asset_format=ext,
                preview_available=True,
                preview_status="available",
            )

    if matches.get("cgm") is not None:
        return GraphicAssetPreview(
            original_url=_graphic_asset_url(stem, "cgm"),
            asset_format="cgm",
            preview_available=False,
            preview_status="unsupported_cgm",
        )

    return GraphicAssetPreview(preview_status="missing")


def resolve_graphic_asset_file(asset_name: str, asset_root: str | Path | None = None) -> Path | None:
    """Return a safe repo-local asset file for a backend URL filename."""
    stem, ext = _split_safe_asset_name(asset_name)
    if stem is None or ext is None:
        return None
    root = Path(asset_root or DEFAULT_GRAPHIC_ASSET_ROOT).resolve()
    path = _find_graphic_asset_files(root, stem).get(ext)
    if path is None:
        return None
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return None
    return path


def _graphic_asset_stem(metadata: Mapping[str, Any]) -> str | None:
    for key in ("icn", "asset_id", "asset_key", "id", "label", "title"):
        value = metadata.get(key)
        if value is None:
            continue
        raw = str(value).strip()
        if not raw:
            continue
        if raw.startswith("asset:"):
            raw = raw.removeprefix("asset:")
        raw = Path(raw).name
        if "." in raw:
            candidate_stem, candidate_ext = raw.rsplit(".", 1)
            if candidate_ext.lower() not in _GRAPHIC_EXTENSION_ORDER:
                continue
            raw = candidate_stem
        if _ASSET_STEM_RE.fullmatch(raw):
            return raw
    return None


def _find_graphic_asset_files(root: Path, stem: str) -> dict[str, Path]:
    if not root.exists() or not root.is_dir() or not _ASSET_STEM_RE.fullmatch(stem):
        return {}
    stem_upper = stem.upper()
    matches: dict[str, Path] = {}
    for child in root.rglob("*"):
        if not child.is_file() or child.stem.upper() != stem_upper:
            continue
        ext = child.suffix.lower().lstrip(".")
        if ext not in _GRAPHIC_EXTENSION_ORDER:
            continue
        try:
            child.resolve().relative_to(root)
        except ValueError:
            continue
        matches.setdefault(ext, child)
    return matches


def _split_safe_asset_name(asset_name: str) -> tuple[str | None, str | None]:
    name = Path(str(asset_name)).name
    if name != asset_name or "/" in name or "\\" in name or "." not in name:
        return None, None
    stem, ext = name.rsplit(".", 1)
    ext = ext.lower()
    if ext not in _GRAPHIC_EXTENSION_ORDER or not _ASSET_STEM_RE.fullmatch(stem):
        return None, None
    return stem, ext


def _graphic_asset_url(stem: str, ext: str) -> str:
    return f"/assets/graphic/{stem}.{ext.lower()}"


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
    asset_root: str | Path | None = None,
) -> None:
    node = nodes.get(str(edge["target"]))
    if node:
        _add_item(collected, _item_from_node(node, str(edge.get("predicate") or ""), source_dmc, str(edge["target"]), source_rank, asset_root=asset_root))


def _collect_edge_source(
    collected: dict[str, dict[str, ReferenceMaterialItem]],
    nodes: Mapping[str, Mapping[str, Any]],
    edge: Mapping[str, Any],
    source_dmc: str,
    source_rank: int,
    asset_root: str | Path | None = None,
) -> None:
    node = nodes.get(str(edge["source"]))
    if node:
        _add_item(collected, _item_from_node(node, str(edge.get("predicate") or ""), source_dmc, str(edge["target"]), source_rank, asset_root=asset_root))


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
    asset_root: str | Path | None = None,
) -> ReferenceMaterialItem:
    props = node.get("properties") or {}
    node_id = str(node.get("id") or "")
    node_type = str(node.get("type") or "")
    dmc = _display_dmc(props.get("dmc") or (node_id.removeprefix("dm:") if node_id.startswith("dm:") else None))
    preview = resolve_graphic_asset_preview(props | {"id": node_id}, asset_root=asset_root) if node_type == "GraphicAsset" else GraphicAssetPreview()
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
        preview_url=preview.preview_url,
        original_url=preview.original_url,
        asset_format=preview.asset_format,
        preview_available=preview.preview_available,
        preview_status=preview.preview_status,
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
