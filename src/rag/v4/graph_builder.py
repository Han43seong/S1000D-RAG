"""Build a lightweight v4 graph context from ontology manifest nodes."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.rag.ontology import OntologyNode

from .graph_schema import GraphEdge, GraphNode, NodeType, RelationType


@dataclass(frozen=True)
class GraphContext:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def find_node(self, node_id: str) -> GraphNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def related_dmcs_for_target(self, target: str | None) -> tuple[str, ...]:
        if not target:
            return ()
        family = _family(target)
        dmcs: list[str] = []
        for node in self.nodes:
            if node.node_type == NodeType.DATA_MODULE and _family(str(node.metadata.get("target") or "")) == family:
                if node.dmc:
                    dmcs.append(node.dmc)
        return tuple(dict.fromkeys(dmcs))


def build_graph_context(manifest_nodes: list[OntologyNode]) -> GraphContext:
    graph_nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []

    for node in manifest_nodes:
        dmc_id = f"dmc:{node.dmc}"
        graph_nodes[dmc_id] = GraphNode(
            id=dmc_id,
            node_type=NodeType.DATA_MODULE,
            label=node.title,
            dmc=node.dmc,
            metadata={"dm_type": node.dm_type, "target": node.target, "action": node.action, **dict(node.metadata or {})},
        )

        if node.target:
            target_id = _target_node_id(node.target)
            target_type = NodeType.SYSTEM if "system" in node.target or "시스템" in node.target else NodeType.COMPONENT
            graph_nodes.setdefault(target_id, GraphNode(id=target_id, node_type=target_type, label=node.target))
            relation = RelationType.HAS_DESCRIPTION if node.dm_type == "descriptive" else RelationType.RELATED_TO
            edges.append(GraphEdge(source_id=target_id, relation=relation, target_id=dmc_id, source_dmc=node.dmc))

        if node.dm_type == "procedural" and node.target and node.action:
            procedure_id = f"procedure:{_slug(node.target)}:{_slug(node.action)}"
            graph_nodes.setdefault(procedure_id, GraphNode(id=procedure_id, node_type=NodeType.PROCEDURE, label=f"{node.target} {node.action}", dmc=node.dmc))
            target_id = _target_node_id(node.target)
            graph_nodes.setdefault(target_id, GraphNode(id=target_id, node_type=NodeType.COMPONENT, label=node.target))
            edges.append(GraphEdge(source_id=target_id, relation=RelationType.HAS_PROCEDURE, target_id=procedure_id, source_dmc=node.dmc))
            edges.append(GraphEdge(source_id=procedure_id, relation=RelationType.REFERENCES, target_id=dmc_id, source_dmc=node.dmc))

    return GraphContext(nodes=tuple(graph_nodes.values()), edges=tuple(edges))


def _target_node_id(target: str) -> str:
    family = _family(target)
    prefix = "system" if "system" in target or "시스템" in target else "component"
    return f"{prefix}:{_slug(family if family else target)}"


def _family(target: str | None) -> str | None:
    if not target:
        return None
    if "brake" in target:
        return "brake system"
    if target in {"wheel", "front wheel", "rear wheel", "tire"}:
        return "wheel"
    return target


def _slug(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "-", value.casefold()).strip("-")
