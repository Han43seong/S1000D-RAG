from src.rag.ontology.schema import Intent, OntologyNode, ParsedQuery, SupportLevel


def test_schema_contract_values():
    node = OntologyNode(dmc="DMC-1", title="Title", dm_type="procedural", target="chain", action="oil")
    parsed = ParsedQuery(original="q", normalized="q", intent=Intent.PROCEDURE, target="chain", action="oil")
    assert node.target == "chain"
    assert parsed.intent == Intent.PROCEDURE
    assert SupportLevel.EXACT.value == "exact"
