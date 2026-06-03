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
    # Optional API/UI bridge fields.  Keeping these nullable preserves existing
    # text-only callers while allowing visual-caption evidence to be serialized
    # without a separate model/index dependency.
    chunk_index: Optional[str] = None
    id: Optional[str] = None
    final_score: Optional[float] = None
    rank: Optional[int] = None
    modality: Optional[str] = None
    content_role: Optional[str] = None
    asset_key: Optional[str] = None
    asset_path: Optional[str] = None
    caption_path: Optional[str] = None
    title: Optional[str] = None
    kind: Optional[str] = None
    ref_id: Optional[str] = None
    display_label: Optional[str] = None
    source_label: Optional[str] = None
    text: Optional[str] = None


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
