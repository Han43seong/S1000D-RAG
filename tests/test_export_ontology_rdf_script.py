import json
import subprocess
import sys


def test_export_ontology_rdf_script_writes_jsonld(tmp_path):
    output = tmp_path / "s1000d.jsonld"

    result = subprocess.run(
        [sys.executable, "scripts/export_ontology_rdf.py", "--format", "jsonld", "--output", str(output)],
        cwd=".",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["@context"]["s1000d"] == "https://example.org/s1000d/"
    assert any(item.get("s1000d:dmc") == "BRAKE-AAA-DA1-00-00-00AA-041A-A" for item in payload["@graph"])
    assert "jsonld" in result.stdout
