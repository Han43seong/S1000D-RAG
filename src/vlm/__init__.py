"""Dependency-light visual captioning scaffold for S1000D-RAG."""

from src.vlm.captioner import CaptionerUnavailableError, MockVisualCaptioner, create_captioner
from src.vlm.documents import caption_to_document, captions_to_documents
from src.vlm.prompts import build_technical_manual_caption_prompt
from src.vlm.types import VisualCaptionRecord

__all__ = [
    "CaptionerUnavailableError",
    "MockVisualCaptioner",
    "VisualCaptionRecord",
    "build_technical_manual_caption_prompt",
    "caption_to_document",
    "captions_to_documents",
    "create_captioner",
]
