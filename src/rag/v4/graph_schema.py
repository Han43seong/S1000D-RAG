"""Graph schema primitives for v4 ontology-guided Graph RAG."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class NodeType(StrEnum):
    DATA_MODULE = "data_module"
    SYSTEM = "system"
    COMPONENT = "component"
    PROCEDURE = "procedure"
    ACTION = "action"
    FAULT = "fault"
    WARNING = "warning"
    CAUTION = "caution"
    NOTE = "note"
    FIGURE = "figure"
    GRAPHIC = "graphic"
    TOOL = "tool"
    SUPPLY = "supply"
    REFERENCE = "reference"
    APPLICABILITY = "applicability"
    SECURITY_CLASS = "security_class"


class RelationType(StrEnum):
    HAS_COMPONENT = "has_component"
    HAS_PROCEDURE = "has_procedure"
    HAS_DESCRIPTION = "has_description"
    HAS_FAULT = "has_fault"
    HAS_WARNING = "has_warning"
    HAS_CAUTION = "has_caution"
    HAS_FIGURE = "has_figure"
    USES_TOOL = "uses_tool"
    USES_SUPPLY = "uses_supply"
    REFERENCES = "references"
    APPLIES_TO = "applies_to"
    PART_OF = "part_of"
    OPERATES_BY = "operates_by"
    PRODUCES_EFFECT = "produces_effect"
    RELATED_TO = "related_to"


@dataclass(frozen=True)
class GraphNode:
    id: str
    node_type: NodeType
    label: str
    dmc: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    relation: RelationType
    target_id: str
    source_dmc: str | None = None
    structure_path: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
