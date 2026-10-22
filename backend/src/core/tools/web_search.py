"""Batched Tavily web search.

The only module in this repository that imports the Tavily search integration.
One call carries a batch of queries, matching the paper's search tool, and the
batch runs concurrently behind a semaphore.
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import ToolException
from langchain_tavily import TavilySearch
from loguru import logger

from core.retry import Retrier, RetriesExhausted
from core.tools.tavily_errors import raise_returned_error

# The closed set of error entries. An observation is state the caller reads back
# and the next prompt, so an entry names the failure kind and nothing else. The
# detail goes to the log.
SEARCH_FAILED = "Search failed: the request did not succeed."
SEARCH_GAVE_UP = "Search failed: gave up after its retries."
PAST_DEADLINE = "Failed: the query ran past its deadline."


def _format_results(query: str, payload: object) -> str:
    """Render one query's results as prompt text.

    Args:
        query: The query that produced them.
        payload: Whatever the Tavily integration returned.

    Returns:
        A numbered block of title, URL and snippet per result.
    """
    results = payload.get("results", []) if isinstance(payload, dict) else []
    lines = [f'Query: {query}']
    for index, result in enumerate(results, start=1):
        lines.append(
            f'{index}. {result.get("title", "(no title)")}\n'
            f'   {result.get("url", "")}\n'
            f'   {result.get("content", "").strip()}'
        )
    return "\n".join(lines)


async def search(
    queries: list[str],
    max_results: int,
    concurrency: int,
    timeout_seconds: float,
    deadline_seconds: float,
    retrier: Retrier,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[list[str], list[str | None]]:
    """Search the web for a batch of queries.

    Args:
        queries: The queries to run.
        max_results: Results Tavily returns per query.
        concurrency: Most queries in flight at once.
        timeout_seconds: Ceiling for one request.
        deadline_seconds: Ceiling for one query as a whole, retries and
            backoff waits included.
        retrier: Backoff policy and counters for rate limits and transient
            failures. A timeout is a transient failure.
        api_key: Tavily credential. None lets the integration read its own
            environment variable, which only a server-hosted run has.
        base_url: Tavily endpoint. None uses the integration's own default.

    Returns:
        Results and errors, one entry per query, in the order given. A query
        that failed has an empty result and a fixed message from the closed
        set above in the errors list.
    """
    tool = TavilySearch(
        max_results=max_results,
        include_images=False,
        **({"tavily_api_key": api_key} if api_key else {}),
        **({"api_base_url": base_url} if base_url else {}),
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def fetch(query: str) -> object:
        return raise_returned_error(
            await asyncio.wait_for(tool.ainvoke({"query": query}), timeout_seconds)
        )

    async def run_one(query: str) -> tuple[str, str | None]:
        async with semaphore:
            try:
                async with asyncio.timeout(deadline_seconds):
                    payload = await retrier.run(lambda: fetch(query), "search")
                return _format_results(query, payload), None
            except RetriesExhausted as error:
                logger.warning(f"search gave up: {error}")
                return "", SEARCH_GAVE_UP
            except TimeoutError:
                logger.warning(f"search ran past its {deadline_seconds}s deadline")
                return "", PAST_DEADLINE
            except ToolException:
                # The integration raises this for a query with no hits.
                return f"Query: {query}\nNo results.", None
            except Exception:  # noqa: BLE001 - a fixed message reaches the model
                logger.exception("search failed")
                return "", SEARCH_FAILED

    pairs = await asyncio.gather(*(run_one(query) for query in queries))
    return [text for text, _ in pairs], [error for _, error in pairs]
