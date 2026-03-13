from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel

from src.types.dm import DmType


class DmFilter(BaseModel):
    dm_type: Optional[DmType] = None
    security: Optional[str] = None
    language: Optional[str] = None


class CsdbAdapter(ABC):
    """S1000D CSDB 데이터 소스 추상화."""

    @abstractmethod
    async def list_data_modules(self, filters: DmFilter | None = None) -> list[str]:
        """DMC 목록 반환."""
        ...

    @abstractmethod
    async def get_data_module_xml(self, dmc: str) -> str:
        """DM XML 원문 반환."""
        ...
