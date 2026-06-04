"""Project-level tracing switch.

LangSmith tracing is disabled by default to avoid quota usage during local
development and automated tests.  Re-enable intentionally with:

    S1000D_LANGSMITH_TRACING=true

This module also sets the standard LangSmith/LangChain env flags to false before
pipeline modules import LangSmith, so checked-in code does not consume traces
just because a local .env contains legacy tracing settings.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_ENABLE_VALUES = {"1", "true", "yes", "on"}
_TRACE_ENV_KEYS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")


def tracing_enabled() -> bool:
    return os.getenv("S1000D_LANGSMITH_TRACING", "false").strip().casefold() in _ENABLE_VALUES


if not tracing_enabled():
    for key in _TRACE_ENV_KEYS:
        os.environ[key] = "false"


def traceable(*args: Any, **kwargs: Any):
    """Return LangSmith's decorator only when project tracing is explicitly enabled."""
    if tracing_enabled():
        from langsmith import traceable as langsmith_traceable

        return langsmith_traceable(*args, **kwargs)

    def decorator(func: F) -> F:
        return func

    # Match decorator behavior for both @traceable and @traceable(...).
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return decorator
