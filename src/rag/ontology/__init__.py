"""Ontology-first RAG v2 package.

This package intentionally supersedes the older ``src.rag.ontology`` module for
runtime imports.  A small legacy re-export shim keeps old ontology export tests
working while v2 imports use explicit submodules.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .answer_composer import compose_answer
from .evidence_planner import plan_evidence, retrieve_evidence
from .manifest_builder import build_ontology_manifest, load_ontology_manifest, save_ontology_manifest
from .query_parser import parse_query
from .quality_gate import check_answer_quality, enforce_quality
from .resolver import resolve_ontology
from .schema import (
    AnswerMode,
    Audience,
    CandidateEvidence,
    DetailLevel,
    Intent,
    OntologyNode,
    ParsedQuery,
    QualityGateResult,
    ResolutionResult,
    SupportLevel,
)

_legacy_path = Path(__file__).resolve().parent.parent / "ontology.py"
if _legacy_path.exists():
    _spec = importlib.util.spec_from_file_location("src.rag._legacy_ontology_module", _legacy_path)
    if _spec and _spec.loader:
        _legacy = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _legacy
        _spec.loader.exec_module(_legacy)
        for _name in (
            "OntologyGraph", "OntologyEdge", "build_ontology_from_manifest", "build_ontology_from_xml_dir",
            "ontology_to_jsonld", "ontology_to_turtle", "save_ontology_exports",
        ):
            if hasattr(_legacy, _name):
                globals()[_name] = getattr(_legacy, _name)

__all__ = [
    "AnswerMode", "Audience", "CandidateEvidence", "DetailLevel", "Intent", "OntologyNode", "ParsedQuery", "QualityGateResult", "ResolutionResult", "SupportLevel",
    "build_ontology_manifest", "load_ontology_manifest", "save_ontology_manifest", "parse_query", "resolve_ontology",
    "plan_evidence", "retrieve_evidence", "compose_answer", "check_answer_quality", "enforce_quality",
]
