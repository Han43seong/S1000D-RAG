"""RDF/Turtle export for the v4 canonical ontology layer.

This module intentionally has no mandatory rdflib dependency.  It emits a
small RDF-compatible triple set and Turtle document that can be loaded into
RDFLib, Apache Jena Fuseki, Ontotext GraphDB, or another SPARQL-compatible
store once the optional backend is enabled.
"""
from __future__ import annotations

import re
from typing import Iterable
from urllib.parse import quote

from src.rag.ontology import OntologyNode

RDF_PREFIXES = """@prefix s1000d: <https://example.org/s1000d/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
"""

BASE_IRI = "https://example.org/s1000d"

Triple = tuple[str, str, str]


def ontology_nodes_to_triples(nodes: Iterable[OntologyNode]) -> tuple[Triple, ...]:
    triples: list[Triple] = [
        ("s1000d:DataModule", "rdf:type", "owl:Class"),
        ("s1000d:DescriptiveDataModule", "rdfs:subClassOf", "s1000d:DataModule"),
        ("s1000d:ProceduralDataModule", "rdfs:subClassOf", "s1000d:DataModule"),
        ("s1000d:FaultDataModule", "rdfs:subClassOf", "s1000d:DataModule"),
        ("s1000d:System", "rdf:type", "owl:Class"),
        ("s1000d:Component", "rdf:type", "owl:Class"),
        ("s1000d:Action", "rdf:type", "owl:Class"),
    ]
    for node in nodes:
        dm = dm_uri(node.dmc)
        triples.append((dm, "rdf:type", dm_class_uri(node.dm_type)))
        triples.append((dm, "s1000d:dmc", literal(node.dmc)))
        triples.append((dm, "rdfs:label", literal(node.title)))
        triples.append((dm, "s1000d:dmType", literal(node.dm_type)))
        if node.sns_code:
            triples.append((dm, "s1000d:snsCode", literal(node.sns_code)))
        if node.target:
            target = entity_uri(node.target)
            triples.append((target, "rdf:type", target_class_uri(node.target)))
            triples.append((target, "rdfs:label", literal(node.target)))
            relation = "s1000d:describes" if node.dm_type == "descriptive" else "s1000d:hasTarget"
            triples.append((dm, relation, target))
        if node.action:
            action = action_uri(node.action)
            triples.append((action, "rdf:type", "s1000d:Action"))
            triples.append((action, "rdfs:label", literal(node.action)))
            triples.append((dm, "s1000d:hasAction", action))
        for alias in node.aliases:
            triples.append((dm, "s1000d:alias", literal(alias)))
        for component in _components(node):
            if node.target:
                triples.append((entity_uri(node.target), "s1000d:hasComponent", entity_uri(component)))
                triples.append((entity_uri(component), "rdf:type", "s1000d:Component"))
                triples.append((entity_uri(component), "rdfs:label", literal(component)))
    return tuple(dict.fromkeys(triples))


def export_ontology_turtle(nodes: Iterable[OntologyNode]) -> str:
    lines = [RDF_PREFIXES.rstrip(), ""]
    for subject, predicate, obj in ontology_nodes_to_triples(nodes):
        lines.append(f"{subject} {predicate} {obj} .")
    lines.append("")
    return "\n".join(lines)


def export_ontology_jsonld(nodes: Iterable[OntologyNode]) -> dict:
    graph: list[dict] = [
        {"@id": "s1000d:DataModule", "@type": "owl:Class"},
        {"@id": "s1000d:DescriptiveDataModule", "rdfs:subClassOf": {"@id": "s1000d:DataModule"}},
        {"@id": "s1000d:ProceduralDataModule", "rdfs:subClassOf": {"@id": "s1000d:DataModule"}},
        {"@id": "s1000d:FaultDataModule", "rdfs:subClassOf": {"@id": "s1000d:DataModule"}},
        {"@id": "s1000d:System", "@type": "owl:Class"},
        {"@id": "s1000d:Component", "@type": "owl:Class"},
        {"@id": "s1000d:Action", "@type": "owl:Class"},
    ]
    for node in nodes:
        item: dict = {
            "@id": _strip_angle_iri(dm_uri(node.dmc)),
            "@type": dm_class_uri(node.dm_type),
            "s1000d:dmc": node.dmc,
            "rdfs:label": node.title,
            "s1000d:dmType": node.dm_type,
        }
        if node.sns_code:
            item["s1000d:snsCode"] = node.sns_code
        if node.target:
            relation = "s1000d:describes" if node.dm_type == "descriptive" else "s1000d:hasTarget"
            item[relation] = {"@id": _strip_angle_iri(entity_uri(node.target))}
        if node.action:
            item["s1000d:hasAction"] = {"@id": _strip_angle_iri(action_uri(node.action))}
        if node.aliases:
            item["s1000d:alias"] = list(node.aliases)
        components = _components(node)
        if components and node.target:
            item["s1000d:hasComponent"] = [{"@id": _strip_angle_iri(entity_uri(component))} for component in components]
        graph.append(item)
    return {
        "@context": {
            "s1000d": f"{BASE_IRI}/",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        },
        "@graph": graph,
    }


def dm_uri(dmc: str) -> str:
    return _iri("dm", _safe_identifier(dmc))


def entity_uri(value: str) -> str:
    return _iri("entity", _slug(value))


def action_uri(value: str) -> str:
    return _iri("action", _slug(value))


def dm_class_uri(dm_type: str) -> str:
    if dm_type == "procedural":
        return "s1000d:ProceduralDataModule"
    if dm_type == "fault":
        return "s1000d:FaultDataModule"
    if dm_type == "descriptive":
        return "s1000d:DescriptiveDataModule"
    return "s1000d:DataModule"


def target_class_uri(target: str) -> str:
    return "s1000d:System" if "system" in target or "시스템" in target else "s1000d:Component"


def literal(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _components(node: OntologyNode) -> tuple[str, ...]:
    raw = node.metadata.get("components") if node.metadata else None
    if isinstance(raw, list):
        return tuple(str(item) for item in raw if str(item).strip())
    return ()


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣\-]+", "-", value).strip("-")


def _iri(kind: str, value: str) -> str:
    encoded = quote(value, safe="-")
    return f"<{BASE_IRI}/{kind}/{encoded}>"


def _strip_angle_iri(value: str) -> str:
    return value[1:-1] if value.startswith("<") and value.endswith(">") else value


def _slug(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "-", value.casefold()).strip("-")
