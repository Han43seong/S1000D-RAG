"""RDF-backed ontology resolver for v4 Graph RAG.

The default implementation indexes RDF-compatible triples in memory so the
project remains runnable without a local GraphDB server.  The exported Turtle
can be loaded into RDFLib/GraphDB later; this store keeps the resolver contract
and tests stable while the optional backend is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.rag.ontology import Intent, OntologyNode, ParsedQuery

from .rdf_exporter import BASE_IRI, Triple, action_uri, dm_uri, entity_uri, ontology_nodes_to_triples


@dataclass(frozen=True)
class RdfResolution:
    primary_dmcs: tuple[str, ...]
    related_dmcs: tuple[str, ...]
    graph_paths: tuple[str, ...] = ()

    @property
    def all_dmcs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.primary_dmcs, *self.related_dmcs)))


class RdfOntologyStore:
    def __init__(self, triples: Iterable[Triple]):
        self.triples = tuple(dict.fromkeys(triples))
        self._dm_uri_to_dmc = self._build_dm_index()

    @classmethod
    def from_nodes(cls, nodes: Iterable[OntologyNode]) -> "RdfOntologyStore":
        return cls(ontology_nodes_to_triples(nodes))

    def find_descriptive_dmcs(self, target: str) -> tuple[str, ...]:
        target_ref = entity_uri(target)
        return self._dmcs_for(predicate="s1000d:describes", obj=target_ref)

    def find_procedure_dmcs(self, target: str, action: str | None = None) -> tuple[str, ...]:
        target_ref = entity_uri(target)
        target_dmcs = set(self._dmcs_for(predicate="s1000d:hasTarget", obj=target_ref))
        if action:
            action_ref = action_uri(action)
            action_dmcs = set(self._dmcs_for(predicate="s1000d:hasAction", obj=action_ref))
            target_dmcs &= action_dmcs
        return tuple(sorted(target_dmcs))

    def related_dmcs_for_target(self, target: str | None) -> tuple[str, ...]:
        if not target:
            return ()
        family_entities = self._family_entities(target)
        related: list[str] = []
        for entity in family_entities:
            related.extend(self._dmcs_for(predicate="s1000d:describes", obj=entity))
            related.extend(self._dmcs_for(predicate="s1000d:hasTarget", obj=entity))
        return tuple(dict.fromkeys(related))

    def resolve_query(self, parsed: ParsedQuery) -> RdfResolution:
        primary: tuple[str, ...] = ()
        if parsed.referenced_dmcs:
            primary = parsed.referenced_dmcs
        elif parsed.intent == Intent.PROCEDURE and parsed.target:
            primary = self.find_procedure_dmcs(parsed.target, parsed.action)
        elif parsed.intent in {Intent.DESCRIBE, Intent.LIST_COMPONENTS, Intent.DOCUMENT_SUMMARY} and parsed.target:
            primary = self.find_descriptive_dmcs(parsed.target)
        elif parsed.target:
            primary = self.related_dmcs_for_target(parsed.target)

        related = tuple(dmc for dmc in self.related_dmcs_for_target(parsed.target) if dmc not in primary)
        paths = self._graph_paths_for(parsed.target, primary, related)
        return RdfResolution(primary_dmcs=primary, related_dmcs=related, graph_paths=paths)

    def _dmcs_for(self, predicate: str, obj: str) -> tuple[str, ...]:
        dmcs: list[str] = []
        for subject, pred, object_ in self.triples:
            if pred == predicate and object_ == obj and subject in self._dm_uri_to_dmc:
                dmcs.append(self._dm_uri_to_dmc[subject])
        return tuple(dict.fromkeys(dmcs))

    def _build_dm_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for subject, pred, obj in self.triples:
            if pred == "s1000d:dmc" and subject.startswith(f"<{BASE_IRI}/dm/"):
                index[subject] = obj.strip('"')
        return index

    def _family_entities(self, target: str) -> tuple[str, ...]:
        requested = entity_uri(_family(target) or target)
        entities = {requested, entity_uri(target)}
        for source, pred, obj in self.triples:
            if pred == "s1000d:hasComponent" and source == requested:
                entities.add(obj)
        family = _family(target)
        if family:
            for source, pred, obj in self.triples:
                if pred in {"s1000d:describes", "s1000d:hasTarget"} and obj.startswith(f"<{BASE_IRI}/entity/"):
                    label = obj.removeprefix(f"<{BASE_IRI}/entity/").removesuffix(">").replace("-", " ")
                    if _family(label) == family:
                        entities.add(obj)
        return tuple(sorted(entities))

    def _graph_paths_for(self, target: str | None, primary: tuple[str, ...], related: tuple[str, ...]) -> tuple[str, ...]:
        if not target:
            return ()
        paths: list[str] = []
        target_ref = entity_uri(target)
        for dmc in (*primary, *related):
            dm = dm_uri(dmc)
            for subject, pred, obj in self.triples:
                if subject == dm and obj.startswith(f"<{BASE_IRI}/entity/"):
                    paths.append(f"{target_ref} <- {pred} <- {dm}")
        return tuple(dict.fromkeys(paths))


def _family(target: str | None) -> str | None:
    if not target:
        return None
    if "brake" in target:
        return "brake system"
    if target in {"wheel", "front wheel", "rear wheel", "tire"}:
        return "wheel"
    return target
