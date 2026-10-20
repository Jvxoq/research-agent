"""Tests for the batched Tavily search tool.

Tavily is replaced by a fake tool, so these tests make no network call and
wait on no wall clock.
"""

import asyncio

import pytest

from core.retry import Retrier
from core.tools import web_search
from core.tools.web_search import PAST_DEADLINE, SEARCH_FAILED, SEARCH_GAVE_UP
from tests.conftest import FakeSearchTool

QUERIES = ["first query", "second query", "third query"]


def _payload(query: str) -> dict:
    """Build one Tavily-shaped response for a query."""
    return {
        "results": [
            {
                "title": f"title for {query}",
                "url": f"https://example.com/{query.replace(' ', '-')}",
                "content": f"snippet for {query}",
            }
        ]
    }


@pytest.fixture
def install_search(monkeypatch: pytest.MonkeyPatch):
    """Return a helper that puts a fake Tavily tool in place of the real one."""

    def install(handler) -> None:
        monkeypatch.setattr(
            web_search, "TavilySearch", lambda **kwargs: FakeSearchTool(handler)
        )

    return install


@pytest.fixture
def all_queries_succeed(install_search):
    """Install a Tavily fake that answers every query with one result."""

    async def handler(query: str) -> dict:
        return _payload(query)

    install_search(handler)


@pytest.fixture
def second_query_fails(install_search):
    """Install a Tavily fake that raises for the second query only."""

    async def handler(query: str) -> dict:
        if query == QUERIES[1]:
            raise RuntimeError("upstream refused the query")
        return _payload(query)

    install_search(handler)


def _rendered(query: str) -> str:
    """Return the prompt block the tool renders for :func:`_payload`."""
    return (
        f"Query: {query}\n"
        f"1. title for {query}\n"
        f"   https://example.com/{query.replace(' ', '-')}\n"
        f"   snippet for {query}"
    )


@pytest.fixture
def run_search(retrier: Retrier):
    """Return a helper that runs the tool under test with fixed limits."""

    async def run(
        queries: list[str],
        timeout_seconds: float = 5.0,
        deadline_seconds: float = 60.0,
        api_key: str | None = None,
    ):
        return await web_search.search(
            queries,
            max_results=3,
            concurrency=2,
            timeout_seconds=timeout_seconds,
            deadline_seconds=deadline_seconds,
            retrier=retrier,
            api_key=api_key,
        )

    return run


async def test_queries_keep_slots_in_order(all_queries_succeed, run_search) -> None:
    """Slot i answers query i, whatever order the batch finished in."""
    results, _ = await run_search(QUERIES)
    assert [text.splitlines()[0] for text in results] == [
        f"Query: {query}" for query in QUERIES
    ]


async def test_result_has_title_url_snippet(
    all_queries_succeed, run_search
) -> None:
    """The block is prompt text, so each result shows all three fields."""
    results, _ = await run_search([QUERIES[0]])
    assert "title for first query" in results[0]
    assert "https://example.com/first-query" in results[0]
    assert "snippet for first query" in results[0]


async def test_clean_batch_has_no_errors(
    all_queries_succeed, run_search
) -> None:
    """Every error slot stays empty when nothing failed."""
    _, errors = await run_search(QUERIES)
    assert errors == [None, None, None]


async def test_failed_query_fills_own_slot(
    second_query_fails, run_search
) -> None:
    """The slot stays in place so indexes still line up with the queries."""
    results, errors = await run_search(QUERIES)
    assert errors[1] == SEARCH_FAILED
    assert results[1] == ""


async def test_the_other_queries_still_return_results(second_query_fails, run_search) -> None:
    """One bad query never fails the batch."""
    results, errors = await run_search(QUERIES)
    assert [errors[0], errors[2]] == [None, None]
    assert results[0].startswith(f"Query: {QUERIES[0]}")
    assert results[2].startswith(f"Query: {QUERIES[2]}")


async def test_repeated_query_timeout_is_error(
    install_search, run_search
) -> None:
    """A timeout is retried as transient, then the spent budget is the entry."""

    async def handler(query: str) -> dict:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    install_search(handler)

    results, errors = await run_search([QUERIES[0]], timeout_seconds=0)

    assert errors[0] == SEARCH_GAVE_UP
    assert results[0] == ""


async def test_empty_query_says_no_results(install_search, run_search) -> None:
    """An empty Tavily response is a readable block, not a blank string."""

    async def handler(query: str) -> dict:
        return {"results": []}

    install_search(handler)

    results, errors = await run_search([QUERIES[0]])
    assert errors[0] is None
    assert results[0] == f"Query: {QUERIES[0]}\nNo results."


async def test_unreadable_payload_fills_own_slot(
    install_search, run_search
) -> None:
    """Rendering happens inside the try, so a bad shape is a per-query error."""

    async def handler(query: str) -> dict:
        if query == QUERIES[1]:
            return {"results": ["a bare string where an object belongs"]}
        return _payload(query)

    install_search(handler)

    results, errors = await run_search(QUERIES)

    assert errors[1] == SEARCH_FAILED
    assert [results[0], results[2]] == [_rendered(QUERIES[0]), _rendered(QUERIES[2])]


async def test_error_entry_hides_exception(
    install_search, run_search
) -> None:
    """An entry names the failure kind and nothing else."""

    async def handler(query: str) -> dict:
        raise RuntimeError("401 from tvly-secret-key-value at internal.example")

    install_search(handler)

    _, errors = await run_search([QUERIES[0]], api_key="tvly-secret-key-value")

    assert errors[0] == "Search failed: the request did not succeed."


async def test_query_past_deadline_is_error(
    install_search, run_search
) -> None:
    """Retries and backoff waits together may not outlast the target's deadline."""

    async def handler(query: str) -> dict:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    install_search(handler)

    results, errors = await run_search([QUERIES[0]], deadline_seconds=0)

    assert (results[0], errors[0]) == ("", PAST_DEADLINE)


async def test_returned_rate_limit_retried(
    install_search, run_search, retrier: Retrier
) -> None:
    """The integration returns an HTTP failure as a value, so the tool raises it."""
    calls: list[str] = []

    async def handler(query: str) -> dict:
        calls.append(query)
        if len(calls) == 1:
            raise RuntimeError("Error 429: Too Many Requests")
        return _payload(query)

    install_search(handler)

    results, errors = await run_search([QUERIES[0]])

    assert (results[0], errors[0]) == (_rendered(QUERIES[0]), None)
    assert retrier.counters.rate_limit_hits == 1
