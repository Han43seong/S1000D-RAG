from __future__ import annotations

import pytest

from src.parser.visual_refs import extract_visual_refs_from_xml
from src.types.visual import VisualArtifactKind


def test_extract_visual_refs_from_tiny_xml_fixture():
    xml = """
    <dmodule>
      <content>
        <description>
          <levelledPara id="lp-1">
            <figure id="fig-front-wheel">
              <title>Front wheel assembly</title>
              <graphic infoEntityIdent="ICN-S1000DBIKE-FRONT-WHEEL" />
            </figure>
            <table id="tab-torque">
              <title>Torque values</title>
              <tgroup><tbody><row><entry>Axle</entry><entry>40 Nm</entry></row></tbody></tgroup>
            </table>
          </levelledPara>
        </description>
      </content>
    </dmodule>
    """

    refs = extract_visual_refs_from_xml(xml, dmc="DMC-TEST", source_path="DMC-TEST.xml")

    assert [ref.kind for ref in refs] == [VisualArtifactKind.FIGURE, VisualArtifactKind.TABLE]
    assert refs[0].ref_id == "fig-front-wheel"
    assert refs[0].title == "Front wheel assembly"
    assert refs[0].info_entity_ident == "ICN-S1000DBIKE-FRONT-WHEEL"
    assert refs[0].stable_key == "DMC-TEST:figure:fig-front-wheel"
    assert refs[1].ref_id == "tab-torque"
    assert refs[1].title == "Torque values"


def test_extract_visual_refs_rejects_invalid_xml():
    with pytest.raises(ValueError, match="Invalid XML"):
        extract_visual_refs_from_xml("<dmodule><content>")


def test_extract_visual_refs_captures_standalone_graphic_without_double_counting_figures():
    xml = """
    <dmodule>
      <content>
        <figure id="fig-a"><graphic infoEntityIdent="ICN-FIG" /></figure>
        <graphic id="gra-a" infoEntityIdent="ICN-STANDALONE" />
      </content>
    </dmodule>
    """

    refs = extract_visual_refs_from_xml(xml, dmc="DMC-TEST")

    assert [ref.kind for ref in refs] == [VisualArtifactKind.FIGURE, VisualArtifactKind.GRAPHIC]
    assert refs[1].ref_id == "gra-a"
    assert refs[1].info_entity_ident == "ICN-STANDALONE"
