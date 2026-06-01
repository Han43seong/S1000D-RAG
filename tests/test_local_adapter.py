from __future__ import annotations

import asyncio

from src.csdb.local_adapter import LocalCsdbAdapter


def test_list_data_modules_includes_uppercase_xml_and_deduplicates(tmp_path):
    (tmp_path / "DMC-ABC-AAA-DA1-00-00-00AA-041A-A_001-00_EN-US.XML").write_text("<dm/>", encoding="utf-8")
    (tmp_path / "DMC-DEF-AAA-DA1-00-00-00AA-041A-A_001-00_EN-US.xml").write_text("<dm/>", encoding="utf-8")
    (tmp_path / "DMC-DEF-AAA-DA1-00-00-00AA-041A-A_001-00_EN-US.XML").write_text("<dm/>", encoding="utf-8")
    (tmp_path / "PMC-IGNORED.XML").write_text("<pm/>", encoding="utf-8")

    adapter = LocalCsdbAdapter(tmp_path)

    dmcs = asyncio.run(adapter.list_data_modules())

    assert dmcs == [
        "DMC-ABC-AAA-DA1-00-00-00AA-041A-A_001-00_EN-US",
        "DMC-DEF-AAA-DA1-00-00-00AA-041A-A_001-00_EN-US",
    ]


def test_get_data_module_xml_resolves_actual_case_for_full_stem_and_without_dmc_prefix(tmp_path):
    stem = "DMC-ABC-AAA-DA1-00-00-00AA-041A-A_001-00_EN-US"
    (tmp_path / f"{stem}.XML").write_text("<dm>uppercase</dm>", encoding="utf-8")

    adapter = LocalCsdbAdapter(tmp_path)

    assert asyncio.run(adapter.get_data_module_xml(stem)) == "<dm>uppercase</dm>"
    assert asyncio.run(adapter.get_data_module_xml(stem.removeprefix("DMC-"))) == "<dm>uppercase</dm>"
    assert asyncio.run(adapter.get_data_module_xml(stem.lower())) == "<dm>uppercase</dm>"
