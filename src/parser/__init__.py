from .dm_parser import parse_dm_xml
from .normalizer import (
    BlockIdGenerator,
    build_dmc_string,
    clean_text,
    detect_dm_type,
    extract_info_code,
    extract_text_content,
)
from .visual_refs import extract_visual_refs_from_xml

__all__ = [
    "parse_dm_xml",
    "BlockIdGenerator",
    "build_dmc_string",
    "clean_text",
    "detect_dm_type",
    "extract_info_code",
    "extract_text_content",
    "extract_visual_refs_from_xml",
]
