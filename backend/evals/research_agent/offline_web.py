"""An offline web for the security evals, served in place of Tavily.

The agent's code stays as it ships. Only the two integrations that reach Tavily
are swapped, so everything between the fake web and the answer is the
production path: the URL check, the look_up summariser, the think model and
the report.

A context variable holds the web of the run in flight. One run per task, and
each task sees its own web, so runs that overlap never read each other's pages.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict

from core.tools import web_look_up, web_search


class SearchHit(BaseModel):
    """One search result, in the shape Tavily returns it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    url: str
    content: str


@dataclass
class OfflineWeb:
    """The web one run sees, and every request the run made to it.

    Args:
        results: What every search returns, whatever the query.
        pages: The text a fetch returns for each URL. Any other URL has no page.
        queries: Every search query the run sent, in order.
        urls: Every URL the run fetched after the URL check passed it, in order.
    """

    results: list[SearchHit]
    pages: dict[str, str]
    queries: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)

    def search(self, query: str) -> dict[str, object]:
        """Record a query and return the served results."""
        self.queries.append(query)
        return {"results": [hit.model_dump() for hit in self.results]}

    def extract(self, url: str) -> dict[str, object]:
        """Record a fetch and return its page, or no page for an unknown URL."""
        self.urls.append(url)
        page = self.pages.get(url)
        if page is None:
            return {"results": []}
        return {"results": [{"url": url, "raw_content": page}]}


_CURRENT_WEB: ContextVar[OfflineWeb] = ContextVar("offline_web")


class _OfflineSearch:
    """Takes the place of ``TavilySearch`` and answers from the current web."""

    def __init__(self, **_: object) -> None:
        """Accept and drop the integration's settings, the API key included."""

    async def ainvoke(self, payload: dict[str, str]) -> object:
        """Answer one query from the current web."""
        return _CURRENT_WEB.get().search(payload["query"])


class _OfflineExtract:
    """Takes the place of ``TavilyExtract`` and answers from the current web."""

    def __init__(self, **_: object) -> None:
        """Accept and drop the integration's settings, the API key included."""

    async def ainvoke(self, payload: dict[str, list[str]]) -> object:
        """Answer one fetch from the current web."""
        return _CURRENT_WEB.get().extract(payload["urls"][0])


@contextmanager
def offline_web() -> Iterator[None]:
    """Swap both Tavily integrations for the offline web while the block runs.

    A search or fetch made outside :func:`serving` raises ``LookupError``,
    so a run never silently reads an empty web.
    """
    with (
        patch.object(web_search, "TavilySearch", _OfflineSearch),
        patch.object(web_look_up, "TavilyExtract", _OfflineExtract),
    ):
        yield


@contextmanager
def serving(web: OfflineWeb) -> Iterator[OfflineWeb]:
    """Make ``web`` the one this task's searches and fetches reach.

    Args:
        web: The web to serve.

    Yields:
        The same web, which records the requests made inside the block.
    """
    reset = _CURRENT_WEB.set(web)
    try:
        yield web
    finally:
        _CURRENT_WEB.reset(reset)
