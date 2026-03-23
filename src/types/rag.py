from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from .dm import DmType


class Evidence(BaseModel):
    dmc: str
    chunk_id: str
    score: float
    dm_type: Optional[DmType] = None
    security: Optional[str] = None
    applicability: Optional[str] = None


class RagResult(BaseModel):
    answer: str
    evidences: list[Evidence]


class SessionMeta(BaseModel):
    user_id: Optional[str] = None
    role: Optional[str] = None
    locale: Optional[str] = None
    security_clearance: Optional[str] = None


class RerankOptions(BaseModel):
    enabled: bool = True
    top_k: int = 3
    model_path: str = "models/bge-reranker-v2-m3-ko/"


class RagOptions(BaseModel):
    top_k: int = 10
    relevance_threshold: float = 0.3
    rerank: RerankOptions = RerankOptions()
    max_context_chars: int = 10000
    rewrite_query: bool = False
    expand_query: bool = True
    sns_filter: bool = True
