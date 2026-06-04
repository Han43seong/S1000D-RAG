from src.rag.ontology import OntologyNode
from src.rag.v4.shape_validator import ShapeSeverity, validate_ontology_nodes


def test_shape_validator_accepts_well_formed_descriptive_and_procedural_nodes():
    issues = validate_ontology_nodes(
        [
            OntologyNode(
                dmc="BRAKE-DESC",
                title="Brake system - Description",
                dm_type="descriptive",
                target="brake system",
            ),
            OntologyNode(
                dmc="BRAKE-PAD-CLEAN",
                title="Brake pads - Clean",
                dm_type="procedural",
                target="brake pad",
                action="clean",
            ),
        ]
    )

    assert issues == ()


def test_shape_validator_requires_dmc_title_and_known_dm_type():
    issues = validate_ontology_nodes(
        [
            OntologyNode(dmc="", title="", dm_type="unknown"),
        ]
    )

    issue_codes = {issue.code for issue in issues}
    assert "missing-dmc" in issue_codes
    assert "missing-title" in issue_codes
    assert "unknown-dm-type" in issue_codes
    assert all(issue.severity == ShapeSeverity.ERROR for issue in issues)


def test_shape_validator_requires_target_for_descriptive_nodes():
    issues = validate_ontology_nodes(
        [OntologyNode(dmc="BRAKE-DESC", title="Brake system - Description", dm_type="descriptive")]
    )

    assert [(issue.code, issue.dmc) for issue in issues] == [("descriptive-missing-target", "BRAKE-DESC")]


def test_shape_validator_requires_target_and_action_for_procedural_nodes():
    issues = validate_ontology_nodes(
        [OntologyNode(dmc="BRAKE-PROC", title="Brake procedure", dm_type="procedural")]
    )

    issue_codes = {issue.code for issue in issues}
    assert issue_codes == {"procedural-missing-target", "procedural-missing-action"}


def test_shape_validator_detects_duplicate_dmc_values():
    issues = validate_ontology_nodes(
        [
            OntologyNode(dmc="BRAKE-DESC", title="Brake description", dm_type="descriptive", target="brake system"),
            OntologyNode(dmc="BRAKE-DESC", title="Brake duplicate", dm_type="descriptive", target="brake system"),
        ]
    )

    assert [(issue.code, issue.dmc) for issue in issues] == [("duplicate-dmc", "BRAKE-DESC")]
