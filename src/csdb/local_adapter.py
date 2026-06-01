from __future__ import annotations

from pathlib import Path

from .adapter import CsdbAdapter, DmFilter


class LocalCsdbAdapter(CsdbAdapter):
    """로컬 파일시스템 기반 CSDB 어댑터."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"CSDB directory not found: {self.root_dir}")

    def _iter_dmc_xml_files(self) -> list[Path]:
        """Return DMC XML files, matching the .xml suffix case-insensitively."""
        matches: dict[str, Path] = {}
        for path in self.root_dir.iterdir():
            if not path.is_file():
                continue
            if not path.name.casefold().startswith("dmc-"):
                continue
            if path.suffix.casefold() != ".xml":
                continue
            # De-duplicate only case variants of the same stem.
            matches.setdefault(path.stem.casefold(), path)
        return sorted(matches.values(), key=lambda p: p.name.casefold())

    async def list_data_modules(self, filters: DmFilter | None = None) -> list[str]:
        """디렉터리에서 DMC-*.xml/.XML 파일을 스캔하여 DMC 목록 반환."""
        dmcs = [f.stem for f in self._iter_dmc_xml_files()]
        if filters and filters.model_ident_code:
            prefix = f"DMC-{filters.model_ident_code}-".casefold()
            dmcs = [d for d in dmcs if d.casefold().startswith(prefix)]
        return dmcs

    def _resolve_xml_path(self, dmc: str) -> Path:
        candidates = [dmc]
        if not dmc.casefold().startswith("dmc-"):
            candidates.append(f"DMC-{dmc}")
        candidate_keys = {candidate.casefold() for candidate in candidates}

        for path in self._iter_dmc_xml_files():
            if path.stem.casefold() in candidate_keys:
                return path

        raise FileNotFoundError(f"DM XML not found for DMC: {dmc}")

    async def get_data_module_xml(self, dmc: str) -> str:
        """DMC에 해당하는 XML 파일 읽기."""
        xml_path = self._resolve_xml_path(dmc)
        return xml_path.read_text(encoding="utf-8")

    def get_data_module_path(self, dmc: str) -> Path:
        """Return the local XML path for a DMC without reading its contents."""
        return self._resolve_xml_path(dmc)
