"""Resolve parsed queries against ontology nodes with support levels."""
from __future__ import annotations

import re

from .schema import CandidateEvidence, Intent, OntologyNode, ParsedQuery, ResolutionResult, SupportLevel

_DMC_RE = re.compile(r"\b[A-Z0-9]+-[A-Z0-9-]+\b", re.I)


def resolve_ontology(parsed: ParsedQuery, nodes: list[OntologyNode]) -> ResolutionResult:
    if parsed.intent == Intent.UNKNOWN:
        return ResolutionResult(parsed, SupportLevel.NONE, (), "unparsed_query")

    if parsed.intent in {Intent.DMC_LOOKUP, Intent.DOCUMENT_SUMMARY}:
        match = _DMC_RE.search(parsed.original)
        dmc = match.group(0).upper() if match else ""
        exact = [n for n in nodes if n.dmc.upper() == dmc]
        return _result(parsed, SupportLevel.EXACT if exact else SupportLevel.NONE, exact, "exact_dmc_lookup")

    if parsed.intent == Intent.LIST_COMPONENTS:
        exact = [n for n in nodes if n.target == parsed.target and n.dm_type == "descriptive"]
        return _result(parsed, SupportLevel.EXACT if exact else SupportLevel.NONE, exact, "component_list_description")

    if parsed.intent == Intent.DESCRIBE:
        exact = [n for n in nodes if n.target == parsed.target and n.dm_type in {"descriptive"}]
        if exact:
            return _result(parsed, SupportLevel.EXACT, exact, "exact_description")
        related = [n for n in nodes if n.target == parsed.target][:5]
        return _result(parsed, SupportLevel.RELATED if related else SupportLevel.NONE, related, "description_related")

    if parsed.intent == Intent.PROCEDURE:
        exact = [n for n in nodes if n.dm_type == "procedural" and n.target == parsed.target and n.action == parsed.action]
        if exact:
            return _result(parsed, SupportLevel.EXACT, exact, "exact_target_action")
        # Broad wheel replacement is supported by multiple narrower documents.
        if parsed.target == "wheel" and parsed.action == "replace":
            partial = [n for n in nodes if (n.target in {"tire", "front wheel", "rear wheel"} and n.action in {"replace", "remove", "install"})]
            return _result(parsed, SupportLevel.PARTIAL if partial else SupportLevel.NONE, partial, "wheel_replacement_decomposed")
        same_target = [n for n in nodes if n.dm_type == "procedural" and n.target == parsed.target]
        if same_target:
            return _result(parsed, SupportLevel.RELATED, same_target, "same_target_different_action")
        if parsed.target == "brake cable":
            related = [n for n in nodes if n.target in {"brake system", "brake pad", "front brake"}][:6]
            return _result(parsed, SupportLevel.RELATED if related else SupportLevel.NONE, related, "brake_related_no_cable_procedure")
        family = _family(parsed.target)
        related = [n for n in nodes if family and _family(n.target) == family][:6]
        return _result(parsed, SupportLevel.RELATED if related else SupportLevel.NONE, related, "family_related")

    return ResolutionResult(parsed, SupportLevel.NONE, (), "unsupported_intent")


def _result(parsed: ParsedQuery, support: SupportLevel, nodes: list[OntologyNode], reason: str) -> ResolutionResult:
    candidates = tuple(CandidateEvidence(node=n, support=support, reason=reason, score=max(0.1, 1.0 - i * 0.08)) for i, n in enumerate(nodes))
    return ResolutionResult(parsed, support, candidates, reason)


def _family(target: str | None) -> str | None:
    if not target:
        return None
    if "brake" in target:
        return "brake"
    if target in {"wheel", "front wheel", "rear wheel", "tire"}:
        return "wheel"
    return target
