"""SHACL-like ontology shape validation for the v4 RDF layer.

This is a lightweight local validator for the project ontology manifest.  It is
not a replacement for SHACL; it defines the same data-quality contract in Python
so CI and closed-network demos can validate ontology shape before a GraphDB or
pySHACL dependency is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from src.rag.ontology import OntologyNode

KNOWN_DM_TYPES = {"descriptive", "procedural", "fault"}


class ShapeSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ShapeIssue:
    code: str
    message: str
    dmc: str | None = None
    severity: ShapeSeverity = ShapeSeverity.ERROR


def validate_ontology_nodes(nodes: Iterable[OntologyNode]) -> tuple[ShapeIssue, ...]:
    materialized = tuple(nodes)
    issues: list[ShapeIssue] = []
    seen_dmcs: set[str] = set()
    duplicate_dmcs: set[str] = set()

    for node in materialized:
        dmc = (node.dmc or "").strip()
        title = (node.title or "").strip()
        dm_type = (node.dm_type or "").strip()
        target = (node.target or "").strip()
        action = (node.action or "").strip()

        if not dmc:
            issues.append(ShapeIssue(code="missing-dmc", message="Ontology node must have a DMC."))
        elif dmc in seen_dmcs and dmc not in duplicate_dmcs:
            issues.append(ShapeIssue(code="duplicate-dmc", message=f"Duplicate DMC: {dmc}", dmc=dmc))
            duplicate_dmcs.add(dmc)
        else:
            seen_dmcs.add(dmc)

        if not title:
            issues.append(ShapeIssue(code="missing-title", message="Ontology node must have a title.", dmc=dmc or None))

        if dm_type not in KNOWN_DM_TYPES:
            issues.append(
                ShapeIssue(
                    code="unknown-dm-type",
                    message=f"Unknown data module type: {dm_type or '<empty>'}",
                    dmc=dmc or None,
                )
            )
            continue

        if dm_type == "descriptive" and not target:
            issues.append(
                ShapeIssue(
                    code="descriptive-missing-target",
                    message="Descriptive data modules must describe a target system/component.",
                    dmc=dmc or None,
                )
            )

        if dm_type == "procedural":
            if not target:
                issues.append(
                    ShapeIssue(
                        code="procedural-missing-target",
                        message="Procedural data modules must have a target system/component.",
                        dmc=dmc or None,
                    )
                )
            if not action:
                issues.append(
                    ShapeIssue(
                        code="procedural-missing-action",
                        message="Procedural data modules must have an action.",
                        dmc=dmc or None,
                    )
                )

    return tuple(issues)
