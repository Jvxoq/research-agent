"""LangSmith tracing for a script that runs a graph itself.

LangGraph Server loads ``.env`` into the process environment when it starts,
because ``langgraph.json`` names it under ``"env"``. LangSmith reads its
variables from there, so a served run needs nothing from this module.

A standalone script, such as the eval runner, has no such step. ``Settings``
reads ``.env`` but never exports it. :func:`tracing` turns those settings into a
tracing context the script opens once, around everything it runs. It writes no
environment variable, so one run cannot change tracing for another.

Tracing is off unless ``LANGSMITH_TRACING`` asks for it *and*
``LANGSMITH_API_KEY`` is present. A missing key is reported and ignored. It
never fails a run.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from langsmith import Client, tracing_context
from loguru import logger

from core.config import Settings


@contextmanager
def tracing(settings: Settings) -> Iterator[bool]:
    """Turn LangSmith tracing on or off for everything run inside the block.

    Args:
        settings: Process settings holding the LangSmith variables.

    Yields:
        True if traces will be sent, False otherwise.
    """
    active = bool(settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY)
    if settings.LANGSMITH_TRACING and not active:
        logger.warning(
            "LANGSMITH_TRACING is on but LANGSMITH_API_KEY is not set - "
            "traces will not be sent"
        )

    client = (
        Client(api_url=settings.LANGSMITH_ENDPOINT, api_key=settings.LANGSMITH_API_KEY)
        if active
        else None
    )
    with tracing_context(
        enabled=active, project_name=settings.LANGSMITH_PROJECT, client=client
    ):
        yield active
