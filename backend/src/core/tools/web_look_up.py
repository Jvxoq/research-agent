"""Read web pages, keeping only what serves a goal.

The paper does not truncate observations, it summarises them against a goal
(Appendix C.2). Truncation cuts by position, so it keeps the first part of a
page whether or not the answer is there. Goal-conditioned summarisation cuts by
relevance, so it keeps the answer and drops the navigation menu.

The summariser returns a :class:`PageSummary` through structured output. Each
summary is checked, and a summary that fails the check is retried for that URL
alone. The observation is rendered from the checked fields, so its shape never
depends on how the model chose to format its reply.

The only module in this repository that imports the Tavily extract integration.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilyExtract
from loguru import logger
from pydantic import BaseModel, Field

from core.llm import UNUSABLE_OUTPUT_ERRORS, StructuredOutputError, read_structured
from core.retry import Retrier, RetriesExhausted
from core.tools.tavily_errors import raise_returned_error
from core.tools.text import cap, redact
from core.tools.url_safety import UnsafeUrl, check_url

_Reply = TypeVar("_Reply")

SUMMARY_CORRECTION = (
    "\n\nYour last summary failed the check: {error}\n"
    "Answer again with the same fields. Give at least one finding when coverage "
    "is FULL or PARTIAL, and none when it is NONE."
)

UNKNOWN_DATE = "unknown"

# The closed set of error entries. An observation is state the caller reads back
# and the next prompt, so an entry names the failure kind and nothing else. The
# detail goes to the log.
REFUSED = "Refused: not a public http or https address."
FETCH_FAILED = "Fetch failed: the request did not succeed."
FETCH_GAVE_UP = "Fetch failed: gave up after its retries."
NO_READABLE_TEXT = "Fetch failed: the page returned no readable text."
SUMMARY_FAILED = "Summary failed: the model call did not succeed."
SUMMARY_GAVE_UP = "Summary failed: gave up after its retries."
SUMMARY_UNCHECKED = "Summary failed: the summary did not pass its check."
PAST_DEADLINE = "Failed: the page ran past its deadline."


class PageSummary(BaseModel):
    """What one page says about the goal."""

    findings: list[str] = Field(
        description=(
            "One entry per useful fact, with the figures, dates, names or quotes "
            "and the page detail that proves it. Empty when coverage is NONE."
        )
    )
    source_date: str = Field(
        description='The visible publish or update date, or "unknown".'
    )
    coverage: Literal["FULL", "PARTIAL", "NONE"] = Field(
        description="How much of the goal this page answers."
    )
    coverage_note: str = Field(
        description="One line on what is missing, or why the page does not match the goal."
    )


class MalformedSummary(ValueError):
    """Raised when a structured summary fails the check."""


@dataclass
class _FailedChecks:
    """Summaries of one page that failed the check, kept across a deadline."""

    count: int = 0


@dataclass(frozen=True)
class LookUpSettings:
    """Limits for one look-up batch."""

    goal: str
    max_page_chars: int
    max_summary_chars: int
    max_error_chars: int
    max_summary_retries: int
    concurrency: int
    timeout_seconds: float
    deadline_seconds: float
    api_key: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class LookUpBatch:
    """What one look-up batch produced.

    Args:
        results: One rendered summary per URL, empty where that URL failed.
        errors: One message per URL, None where that URL succeeded.
        malformed_summaries: Summaries that failed the check across the batch,
            including ones a later retry repaired.
    """

    results: list[str]
    errors: list[str | None]
    malformed_summaries: int


def _page_text(payload: object) -> str:
    """Pull the extracted text for one URL out of a Tavily response.

    Args:
        payload: Whatever the extract integration returned for one URL.

    Returns:
        The page's text, or an empty string when the response holds none.
        Tavily may report a redirected address, so the one result is read
        whatever URL it names.
    """
    if not isinstance(payload, dict):
        return ""
    results = payload.get("results") or []
    return (results[0].get("raw_content") or "") if results else ""


def _check_summary(result: object) -> PageSummary:
    """Verify one structured summariser result.

    Args:
        result: The raw-and-parsed mapping ``include_raw`` produces.

    Returns:
        The summary, with blank findings dropped and a blank date read as unknown.

    Raises:
        MalformedSummary: If the result fails the schema, or its coverage and
            its findings contradict each other.
    """
    try:
        summary = read_structured(result, PageSummary)
    except StructuredOutputError as error:
        raise MalformedSummary(str(error)) from error

    findings = [finding.strip() for finding in summary.findings if finding.strip()]
    if summary.coverage != "NONE" and not findings:
        raise MalformedSummary(f"Coverage is {summary.coverage} but no findings were given.")
    if summary.coverage == "NONE" and findings:
        raise MalformedSummary("Coverage is NONE but findings were given.")

    return summary.model_copy(
        update={
            "findings": findings,
            "source_date": summary.source_date.strip() or UNKNOWN_DATE,
            "coverage_note": summary.coverage_note.strip(),
        }
    )


def _render_summary(url: str, summary: PageSummary) -> str:
    """Render a checked summary as the observation text for one URL.

    Args:
        url: The page the summary describes.
        summary: A summary that passed the check.

    Returns:
        One of two fixed layouts, so the think model reads a known shape. A
        FULL or PARTIAL page gives the URL, its findings, its source date and
        its coverage line. A NONE page gives only the URL and its coverage line,
        because it has no findings and no date worth showing.
    """
    coverage = f"COVERAGE: {summary.coverage}"
    if summary.coverage_note:
        coverage += f". {summary.coverage_note}"
    if summary.coverage == "NONE":
        return f"{url}\n{coverage}"
    findings = "\n".join(f"- {finding}" for finding in summary.findings)
    return f"{url}\nFINDINGS:\n{findings}\nSOURCE_DATE: {summary.source_date}\n{coverage}"


async def look_up(
    urls: list[str],
    settings: LookUpSettings,
    summariser: BaseChatModel,
    prompt_template: str,
    retrier: Retrier,
    run_config: RunnableConfig,
) -> LookUpBatch:
    """Fetch a batch of pages and summarise each against the shared goal.

    Args:
        urls: Pages to read.
        settings: Limits and the goal for this batch.
        summariser: Model that reduces a page to what serves the goal.
        prompt_template: Template with ``goal``, ``url``, ``max_chars`` and
            ``content`` placeholders.
        retrier: Backoff policy and counters for rate limits and transient
            failures. A timeout is a transient failure.
        run_config: Config for every summariser call, carrying the run's cost
            tracker and its parent trace.

    Returns:
        Summaries and errors, one entry per URL, in the order given, and the
        number of summaries that failed the check. A URL that failed, including
        one whose summary failed every check or that ran past its deadline, has
        an empty summary and a fixed message from the closed set above in the
        errors list.
    """
    extractor = TavilyExtract(
        **({"tavily_api_key": settings.api_key} if settings.api_key else {}),
        **({"api_base_url": settings.base_url} if settings.base_url else {}),
    )
    # include_raw hands a parsing failure back as a value, so a schema message
    # quoting a word like "connection" never reads as a transient network error.
    structured = summariser.with_structured_output(
        PageSummary, method="json_schema", include_raw=True
    )
    semaphore = asyncio.Semaphore(settings.concurrency)

    async def with_timeout_and_retry(
        call: Callable[[], Awaitable[_Reply]], label: str
    ) -> _Reply:
        """Run one network call under the timeout and the retrier."""

        async def once() -> _Reply:
            return await asyncio.wait_for(call(), settings.timeout_seconds)

        return await retrier.run(once, label)

    async def summarise_once(prompt: str) -> PageSummary:
        """Make one summariser call and check what it returned."""
        try:
            result = await with_timeout_and_retry(
                lambda: structured.ainvoke(prompt, run_config), "look_up.summary"
            )
        except UNUSABLE_OUTPUT_ERRORS as error:
            raise MalformedSummary(
                f"The response stopped before the summary was complete: {error}"
            ) from error
        return _check_summary(result)

    async def read_page(url: str, checks: _FailedChecks) -> tuple[str, str | None]:
        """Check, fetch and summarise one page."""
        try:
            await asyncio.to_thread(check_url, url)
        except UnsafeUrl as error:
            logger.warning(f"refused url: {error}")
            return "", REFUSED

        async def fetch() -> object:
            return raise_returned_error(await extractor.ainvoke({"urls": [url]}))

        try:
            payload = await with_timeout_and_retry(fetch, "look_up.fetch")
            content = _page_text(payload)
        except RetriesExhausted as error:
            logger.warning(f"fetch gave up for {url}: {error}")
            return "", FETCH_GAVE_UP
        except Exception:  # noqa: BLE001 - a fixed message reaches the model
            logger.exception(f"fetch failed for {url}")
            return "", FETCH_FAILED

        if not content.strip():
            return "", NO_READABLE_TEXT

        prompt = prompt_template.format(
            goal=settings.goal,
            url=url,
            max_chars=settings.max_summary_chars,
            content=content[: settings.max_page_chars],
        )
        for _ in range(settings.max_summary_retries + 1):
            try:
                summary = await summarise_once(prompt)
            except RetriesExhausted as error:
                logger.warning(f"summary gave up for {url}: {error}")
                return "", SUMMARY_GAVE_UP
            except MalformedSummary as error:
                correction = cap(
                    redact(str(error), (settings.api_key,)), settings.max_error_chars
                )
                logger.warning(f"summary check failed for {url}: {correction}")
                checks.count += 1
                # The correction accumulates, so a second retry still carries the first.
                prompt += SUMMARY_CORRECTION.format(error=correction)
                continue
            except Exception:  # noqa: BLE001 - a fixed message reaches the model
                logger.exception(f"summary failed for {url}")
                return "", SUMMARY_FAILED
            return _render_summary(url, summary), None

        return "", SUMMARY_UNCHECKED

    async def run_one(url: str) -> tuple[str, str | None, int]:
        checks = _FailedChecks()
        async with semaphore:
            try:
                async with asyncio.timeout(settings.deadline_seconds):
                    text, error = await read_page(url, checks)
            except TimeoutError:
                logger.warning(
                    f"{url} ran past its {settings.deadline_seconds}s deadline"
                )
                return "", PAST_DEADLINE, checks.count
        return text, error, checks.count

    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(run_one(url)) for url in urls]

    outcomes = [task.result() for task in tasks]
    return LookUpBatch(
        results=[text for text, _, _ in outcomes],
        errors=[error for _, error, _ in outcomes],
        malformed_summaries=sum(count for _, _, count in outcomes),
    )
