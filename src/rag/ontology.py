"""S1000D ontology-style node/edge graph exports.

This is a lightweight ontology layer for the S1000D-RAG demo corpus.  It keeps
runtime dependencies minimal while producing stable node/edge JSON-LD and
Turtle-like RDF exports that can later be imported into Neo4j, Jena/Fuseki, or
an OWL/RDF toolchain.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .graph_retrieval import FaultNode, GraphManifest, ProcedureNode

S1000D_NS = "https://example.org/s1000d#"


@dataclass(frozen=True)
class OntologyNode:
    id: str
    type: str
    label: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OntologyEdge:
    source: str
    predicate: str
    target: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class OntologyGraph:
    nodes: dict[str, OntologyNode] = field(default_factory=dict)
    edges: list[OntologyEdge] = field(default_factory=list)

    def add_node(self, node_id: str, node_type: str, label: str = "", **properties: Any) -> None:
        if not node_id:
            return
        existing = self.nodes.get(node_id)
        if existing:
            merged = {**existing.properties, **{k: v for k, v in properties.items() if v not in (None, "")}}
            label = label or existing.label
            self.nodes[node_id] = OntologyNode(node_id, existing.type, label, merged)
            return
        self.nodes[node_id] = OntologyNode(
            id=node_id,
            type=node_type,
            label=label,
            properties={k: v for k, v in properties.items() if v not in (None, "")},
        )

    def add_edge(self, source: str, predicate: str, target: str, **properties: Any) -> None:
        if not source or not target or not predicate:
            return
        edge = OntologyEdge(
            source=source,
            predicate=predicate,
            target=target,
            properties={k: v for k, v in properties.items() if v not in (None, "")},
        )
        if edge not in self.edges:
            self.edges.append(edge)

    def has_edge(self, source: str, predicate: str, target: str) -> bool:
        return any(edge.source == source and edge.predicate == predicate and edge.target == target for edge in self.edges)

    def node_count(self, node_type: str | None = None) -> int:
        if node_type is None:
            return len(self.nodes)
        return sum(1 for node in self.nodes.values() if node.type == node_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in sorted(self.nodes.values(), key=lambda item: item.id)],
            "edges": [asdict(edge) for edge in sorted(self.edges, key=_edge_sort_key)],
        }

    def node_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes.values():
            counts[node.type] = counts.get(node.type, 0) + 1
        return dict(sorted(counts.items()))

    def edge_predicate_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for edge in self.edges:
            counts[edge.predicate] = counts.get(edge.predicate, 0) + 1
        return dict(sorted(counts.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Manifest → ontology
# ─────────────────────────────────────────────────────────────────────────────


def build_ontology_from_manifest(manifest: GraphManifest) -> OntologyGraph:
    graph = OntologyGraph()
    for proc in manifest.procedures:
        _add_procedure_node(graph, proc)
    for fault in manifest.faults:
        _add_fault_node(graph, fault)
    return graph


def _add_procedure_node(graph: OntologyGraph, proc: ProcedureNode) -> None:
    dm_id = _dm_id(proc.dmc)
    component_id = _component_id(proc.target)
    procedure_id = _procedure_id(proc.target, proc.action)
    action_id = _action_id(proc.action)

    graph.add_node(dm_id, "DataModule", proc.title, dmc=proc.dmc, dm_type=proc.dm_type, sns_code=proc.sns_code)
    graph.add_node(component_id, "Component", proc.target, canonical_name=proc.target)
    graph.add_node(action_id, "Action", proc.action, canonical_name=proc.action)
    graph.add_node(procedure_id, "Procedure", proc.title, target=proc.target, action=proc.action)
    graph.add_edge(component_id, "HAS_PROCEDURE", procedure_id)
    graph.add_edge(procedure_id, "USES_ACTION", action_id)
    graph.add_edge(procedure_id, "GROUNDED_IN", dm_id)
    graph.add_edge(dm_id, "DESCRIBES", component_id)


def _add_fault_node(graph: OntologyGraph, fault: FaultNode) -> None:
    dm_id = _dm_id(fault.dmc)
    component_id = _component_id(fault.target)
    graph.add_node(dm_id, "DataModule", fault.title, dmc=fault.dmc, dm_type=fault.dm_type, sns_code=fault.sns_code)
    graph.add_node(component_id, "Component", fault.target, canonical_name=fault.target)
    graph.add_node(f"fault:{_slug(fault.target)}", "Fault", fault.title, target=fault.target)
    graph.add_edge(component_id, "HAS_FAULT_DOC", dm_id)
    graph.add_edge(dm_id, "DESCRIBES", component_id)


# ─────────────────────────────────────────────────────────────────────────────
# XML → ontology
# ─────────────────────────────────────────────────────────────────────────────


def build_ontology_from_xml_dir(xml_dir: str | Path) -> OntologyGraph:
    root = Path(xml_dir)
    graph = OntologyGraph()
    for xml_path in sorted(root.rglob("DMC-*.XML")):
        _add_xml_file(graph, xml_path)
    return graph


def _add_xml_file(graph: OntologyGraph, xml_path: Path) -> None:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return
    root = tree.getroot()
    dm_code = _first_dm_code(root)
    if dm_code is None:
        return
    dmc = _dmc_from_dmcode(dm_code)
    dm_id = _dm_id(dmc)
    title = _extract_dm_title(root) or dmc
    graph.add_node(dm_id, "DataModule", title, dmc=dmc, source_path=str(xml_path))

    seen_ref_dmcs: set[str] = set()
    for ref_code in _iter_dmref_codes(root):
        ref_dmc = _dmc_from_dmcode(ref_code)
        if ref_dmc in seen_ref_dmcs:
            continue
        seen_ref_dmcs.add(ref_dmc)
        ref_id = _dm_id(ref_dmc)
        reference_id = f"reference:{dmc}:{_slug(ref_dmc)}"
        graph.add_node(ref_id, "DataModule", ref_dmc, dmc=ref_dmc)
        graph.add_node(reference_id, "Reference", ref_dmc, dmc=ref_dmc, source_dmc=dmc)
        graph.add_edge(dm_id, "REFERENCES", ref_id)
        graph.add_edge(dm_id, "REFERENCES", reference_id)
        graph.add_edge(reference_id, "REFERENCES", ref_id)

    _add_safety_nodes(graph, root, dmc, dm_id)

    for figure in root.iter():
        if _local_name(figure.tag) != "figure":
            continue
        figure_id = figure.attrib.get("id") or figure.attrib.get("figureNumber") or f"figure-{len(graph.edges)+1}"
        ontology_figure_id = f"figure:{dmc}:{figure_id}"
        figure_title = _extract_child_text(figure, "title") or figure_id
        graph.add_node(ontology_figure_id, "Figure", figure_title, figure_id=figure_id, dmc=dmc)
        graph.add_edge(dm_id, "HAS_FIGURE", ontology_figure_id)
        for child in figure.iter():
            name = _local_name(child.tag)
            if name == "graphic":
                icn = child.attrib.get("infoEntityIdent") or child.attrib.get("id")
                if icn:
                    asset_id = f"asset:{icn}"
                    graph.add_node(asset_id, "GraphicAsset", icn, icn=icn)
                    graph.add_edge(ontology_figure_id, "USES_ASSET", asset_id)
            elif name == "hotspot":
                hotspot_id = child.attrib.get("id") or child.attrib.get("applicationStructureIdent")
                if hotspot_id:
                    node_id = f"hotspot:{dmc}:{hotspot_id}"
                    graph.add_node(node_id, "Hotspot", hotspot_id, hotspot_id=hotspot_id)
                    graph.add_edge(ontology_figure_id, "HAS_HOTSPOT", node_id)


def _first_dm_code(root: ET.Element) -> ET.Element | None:
    for element in root.iter():
        if _local_name(element.tag) == "dmCode":
            return element
    return None


def _iter_dmref_codes(root: ET.Element) -> Iterable[ET.Element]:
    for element in root.iter():
        if _local_name(element.tag) != "dmRef":
            continue
        for child in element.iter():
            if _local_name(child.tag) == "dmCode":
                yield child
                break



def _add_safety_nodes(graph: OntologyGraph, root: ET.Element, dmc: str, dm_id: str) -> None:
    counters = {"warning": 0, "caution": 0}
    for element in root.iter():
        name = _local_name(element.tag)
        if name not in counters:
            continue
        counters[name] += 1
        text = _extract_safety_text(element) or element.attrib.get("id") or f"{name}-{counters[name]}"
        node_type = "Warning" if name == "warning" else "Caution"
        predicate = "HAS_WARNING" if name == "warning" else "HAS_CAUTION"
        node_id = f"{name}:{dmc}:{counters[name]}:{_slug(text)[:48]}"
        properties: dict[str, Any] = {"dmc": dmc, "text": text}
        if name == "warning":
            properties["warning_type"] = element.attrib.get("warningType")
        else:
            properties["caution_type"] = element.attrib.get("cautionType")
        graph.add_node(node_id, node_type, text, **properties)
        graph.add_edge(dm_id, predicate, node_id)


def _extract_safety_text(element: ET.Element) -> str:
    parts: list[str] = []
    for child in element.iter():
        if _local_name(child.tag) == "warningAndCautionPara":
            text = " ".join(part.strip() for part in child.itertext() if part and part.strip())
            if text:
                parts.append(text)
    if not parts:
        text = " ".join(part.strip() for part in element.itertext() if part and part.strip())
        if text:
            parts.append(text)
    return " ".join(parts)


def _dmc_from_dmcode(element: ET.Element) -> str:
    attrs = element.attrib
    prefix = attrs.get("modelIdentCode", "")
    system_diff = attrs.get("systemDiffCode", "")
    system = attrs.get("systemCode", "")
    subsystem = attrs.get("subSystemCode", "")
    subsubsystem = attrs.get("subSubSystemCode", "")
    assy = attrs.get("assyCode", "")  # Present in XML but not represented as a separate segment in this app's DMC labels.
    disassy = f"{attrs.get('disassyCode', '')}{attrs.get('disassyCodeVariant', '')}"
    info = f"{attrs.get('infoCode', '')}{attrs.get('infoCodeVariant', '')}"
    item = attrs.get("itemLocationCode", "")
    parts = [prefix, system_diff, system, subsystem, subsubsystem, disassy or assy, info, item]
    return "-".join(part for part in parts if part)


def _extract_dm_title(root: ET.Element) -> str:
    tech = ""
    info = ""
    for element in root.iter():
        name = _local_name(element.tag)
        if name == "techName" and element.text:
            tech = element.text.strip()
        elif name == "infoName" and element.text:
            info = element.text.strip()
    return " - ".join(part for part in (tech, info) if part)


def _extract_child_text(parent: ET.Element, local_name: str) -> str:
    for child in parent:
        if _local_name(child.tag) == local_name and child.text:
            return child.text.strip()
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Exporters
# ─────────────────────────────────────────────────────────────────────────────


def ontology_to_jsonld(graph: OntologyGraph) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for node in sorted(graph.nodes.values(), key=lambda item: item.id):
        payload: dict[str, Any] = {
            "@id": node.id,
            "@type": f"s1000d:{node.type}",
            "rdfs:label": node.label or node.id,
        }
        payload.update({f"s1000d:{key}": value for key, value in node.properties.items()})
        items.append(payload)
    for edge in sorted(graph.edges, key=_edge_sort_key):
        payload = {
            "@id": f"edge:{_slug(edge.source)}:{edge.predicate}:{_slug(edge.target)}",
            "@type": "s1000d:Relation",
            "s1000d:source": {"@id": edge.source},
            "s1000d:predicate": edge.predicate,
            "s1000d:target": {"@id": edge.target},
        }
        payload.update({f"s1000d:{key}": value for key, value in edge.properties.items()})
        items.append(payload)
    return {
        "@context": {
            "s1000d": S1000D_NS,
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        },
        "@graph": items,
    }


def ontology_to_turtle(graph: OntologyGraph) -> str:
    lines = [
        "@prefix s1000d: <https://example.org/s1000d#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "",
    ]
    for node in sorted(graph.nodes.values(), key=lambda item: item.id):
        lines.append(f"{_ttl_id(node.id)} a s1000d:{node.type} ;")
        lines.append(f"  rdfs:label {_ttl_literal(node.label or node.id)} .")
        lines.append("")
    for edge in sorted(graph.edges, key=_edge_sort_key):
        lines.append(f"{_ttl_id(edge.source)} s1000d:{edge.predicate} {_ttl_id(edge.target)} .")
    return "\n".join(lines).strip() + "\n"


def write_ontology_exports(graph: OntologyGraph, output_dir: str | Path) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    jsonld_path = target / "s1000d_ontology.jsonld"
    ttl_path = target / "s1000d_ontology.ttl"
    graph_path = target / "s1000d_ontology_graph.json"
    jsonld_path.write_text(json.dumps(ontology_to_jsonld(graph), ensure_ascii=False, indent=2), encoding="utf-8")
    ttl_path.write_text(ontology_to_turtle(graph), encoding="utf-8")
    graph_path.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"jsonld": str(jsonld_path), "turtle": str(ttl_path), "graph": str(graph_path)}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _edge_sort_key(edge: OntologyEdge) -> tuple[str, str, str, str]:
    return (edge.predicate, edge.source, edge.target, json.dumps(edge.properties, sort_keys=True))


def _dm_id(dmc: str) -> str:
    return f"dm:{dmc}"


def _component_id(target: str) -> str:
    return f"component:{_slug(target)}"


def _procedure_id(target: str, action: str) -> str:
    return f"procedure:{_slug(target)}:{_slug(action)}"


def _action_id(action: str) -> str:
    return f"action:{_slug(action)}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9가-힣]+", "-", value.strip().casefold()).strip("-")
    return slug or "unknown"


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _ttl_id(identifier: str) -> str:
    if re.match(r"^[a-zA-Z][\w-]*:[^\s]+$", identifier):
        return identifier
    return f"<{identifier}>"


def _ttl_literal(value: str) -> str:
    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'
