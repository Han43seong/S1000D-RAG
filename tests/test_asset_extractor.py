from __future__ import annotations

import json
import subprocess
import sys

from src.media.asset_extractor import (
    build_visual_asset_manifest,
    extract_visual_assets,
    write_visual_asset_manifest,
)


def _write_dm(path, graphic_ident="ICN-UNIT-FOUND"):
    path.write_text(
        f"""
        <dmodule>
          <content>
            <description>
              <figure id="fig-found">
                <title>Found figure</title>
                <graphic infoEntityIdent="{graphic_ident}" />
              </figure>
              <table id="tab-only">
                <title>Metadata table</title>
                <tgroup><tbody><row><entry>A</entry></row></tbody></tgroup>
              </table>
            </description>
          </content>
        </dmodule>
        """,
        encoding="utf-8",
    )


def test_extract_visual_assets_marks_found_and_table_metadata_only(tmp_path):
    _write_dm(tmp_path / "DMC-UNIT.xml")
    (tmp_path / "ICN-UNIT-FOUND.PNG").write_text("not real image bytes", encoding="utf-8")

    result = extract_visual_assets(tmp_path)

    assert result.dm_count == 1
    assert result.visual_ref_count == 2
    assert result.found_asset_count == 1
    assert result.missing_asset_count == 0
    assert result.table_ref_count == 1

    figure = next(asset for asset in result.assets if asset.ref.ref_id == "fig-found")
    table = next(asset for asset in result.assets if asset.ref.ref_id == "tab-only")
    assert figure.status == "found"
    assert figure.asset_path == tmp_path / "ICN-UNIT-FOUND.PNG"
    assert table.status == "metadata_only"
    assert table.metadata_only is True
    assert table.asset_path is None


def test_extract_visual_assets_marks_missing_non_table_asset(tmp_path):
    _write_dm(tmp_path / "DMC-UNIT.xml", graphic_ident="ICN-UNIT-MISSING")

    result = extract_visual_assets(tmp_path)

    assert result.found_asset_count == 0
    assert result.missing_asset_count == 1
    missing = next(asset for asset in result.assets if asset.status == "missing")
    assert missing.ref.info_entity_ident == "ICN-UNIT-MISSING"


def test_build_and_write_visual_asset_manifest(tmp_path):
    _write_dm(tmp_path / "DMC-UNIT.xml")
    (tmp_path / "ICN-UNIT-FOUND.cGm").write_text("not real image bytes", encoding="utf-8")
    result = extract_visual_assets(tmp_path)

    manifest = build_visual_asset_manifest(
        result,
        created_at="2026-06-01T00:00:00+00:00",
        git_commit="abc1234",
    )
    path = write_visual_asset_manifest(manifest, tmp_path / "out" / "assets_manifest.json")

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["data_dir"] == str(tmp_path)
    assert written["created_at"] == "2026-06-01T00:00:00+00:00"
    assert written["dm_count"] == 1
    assert written["visual_ref_count"] == 2
    assert written["found_asset_count"] == 1
    assert written["missing_asset_count"] == 0
    assert written["table_ref_count"] == 1
    assert written["git_commit"] == "abc1234"
    assert written["assets"] == written["refs"]
    assert any(ref["asset_path"] == "ICN-UNIT-FOUND.cGm" for ref in written["refs"])


def test_ingest_extract_assets_only_cli_writes_manifest_without_models(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_dm(data_dir / "DMC-UNIT.xml")
    (data_dir / "ICN-UNIT-FOUND.svg").write_text("<svg />", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"

    result = subprocess.run(
        [
            sys.executable,
            "ingest.py",
            "--extract-assets-only",
            "--data-dir",
            str(data_dir),
            "--assets-manifest",
            str(manifest_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dm_count"] == 1
    assert manifest["visual_ref_count"] == 2
    assert manifest["found_asset_count"] == 1
    assert manifest["missing_asset_count"] == 0
    assert manifest["table_ref_count"] == 1
    assert "임베딩 모델 로딩" not in result.stdout
    assert "[assets] Manifest:" in result.stdout
