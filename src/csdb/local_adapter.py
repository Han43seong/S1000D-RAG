from __future__ import annotations

from pathlib import Path

from .adapter import CsdbAdapter, DmFilter


class LocalCsdbAdapter(CsdbAdapter):
    """로컬 파일시스템 기반 CSDB 어댑터."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"CSDB directory not found: {self.root_dir}")

    async def list_data_modules(self, filters: DmFilter | None = None) -> list[str]:
        """디렉터리에서 DMC-*.xml 파일을 스캔하여 DMC 목록 반환."""
        xml_files = sorted(self.root_dir.glob("DMC-*.xml"))
        dmcs = [f.stem for f in xml_files]
        # TODO: filters 적용 (XML 내부 파싱 필요)
        return dmcs

    async def get_data_module_xml(self, dmc: str) -> str:
        """DMC에 해당하는 XML 파일 읽기."""
        # DMC 자체가 파일명 stem인 경우
        xml_path = self.root_dir / f"{dmc}.xml"
        if not xml_path.exists():
            # DMC- 접두사 없이 시도
            xml_path = self.root_dir / f"DMC-{dmc}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"DM XML not found for DMC: {dmc}")
        return xml_path.read_text(encoding="utf-8")
