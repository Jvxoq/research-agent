"""Tests for the goal-conditioned page look-up tool.

Tavily extract, the summariser model and DNS are all replaced by fakes, so
these tests make no network call and wait on no wall clock.
"""

import asyncio
import dataclasses
import threading

import pytest
from openai import ContentFilterFinishReasonError

from core.retry import Retrier
from core.tools import web_look_up
from core.tools.web_look_up import (
    FETCH_FAILED,
    FETCH_GAVE_UP,
    NO_READABLE_TEXT,
    PAST_DEADLINE,
    REFUSED,
    SUMMARY_FAILED,
    SUMMARY_UNCHECKED,
    LookUpSettings,
    PageSummary,
    look_up,
)
from tests.conftest import PUBLIC_ADDRESS, FakeChatModel, FakeExtractor, Raises

GOAL = "the year the company was founded"
PROMPT_TEMPLATE = (
    "Goal: {goal}\nURL: {url}\nLimit: {max_chars} chars\nPage:\n{content}"
)
URLS = [
    "https://example.com/one",
    "https://example.com/two",
    "https://example.com/three",
]

FOUND = PageSummary(
    findings=["Founded in 1893 [About page, first paragraph]"],
    source_date="2024-05-01",
    coverage="FULL",
    coverage_note="",
)


def _extract_payload(url: str) -> dict:
    """Build one Tavily-extract-shaped response for a URL."""
    return {"results": [{"url": url, "raw_content": f"page text of {url}"}]}


def _rendered_found(url: str) -> str:
    """Return the observation text the tool renders for :data:`FOUND`."""
    return (
        f"{url}\nFINDINGS:\n- Founded in 1893 [About page, first paragraph]\n"
        "SOURCE_DATE: 2024-05-01\nCOVERAGE: FULL"
    )


@pytest.fixture(autouse=True)
def public_dns(resolves) -> None:
    """Resolve every hostname to one public address, without touching DNS."""
    resolves(default=PUBLIC_ADDRESS)


@pytest.fixture
def install_extractor(monkeypatch: pytest.MonkeyPatch):
    """Return a helper that puts a fake extractor in place of the real one."""

    def install(handler) -> None:
        monkeypatch.setattr(
            web_look_up, "TavilyExtract", lambda **kwargs: FakeExtractor(handler)
        )

    return install


@pytest.fixture
def every_page_extracts(install_extractor) -> None:
    """Install an extractor that returns readable text for every URL."""

    async def handler(url: str) -> dict:
        return _extract_payload(url)

    install_extractor(handler)


@pytest.fixture
def second_url_fails(install_extractor) -> None:
    """Install an extractor that raises for the second URL only."""

    async def handler(url: str) -> dict:
        if url == URLS[1]:
            raise RuntimeError("upstream refused the page")
        return _extract_payload(url)

    install_extractor(handler)


@pytest.fixture
def summariser() -> FakeChatModel:
    """A summariser whose every summary passes the check."""
    return FakeChatModel(structured_replies=[FOUND])


@pytest.fixture
def settings() -> LookUpSettings:
    """Limits shared by the look-up tests."""
    return LookUpSettings(
        goal=GOAL,
        max_page_chars=2000,
        max_summary_chars=300,
        max_error_chars=500,
        max_summary_retries=1,
        concurrency=2,
        timeout_seconds=5.0,
        deadline_seconds=60.0,
    )


async def test_goal_and_page_reach_summariser(
    every_page_extracts, summariser: FakeChatModel, settings: LookUpSettings, retrier: Retrier
) -> None:
    """The summariser reduces the page, keeping what serves the goal."""
    await look_up([URLS[0]], settings, summariser, PROMPT_TEMPLATE, retrier, {})

    assert GOAL in summariser.prompts[0]
    assert f"page text of {URLS[0]}" in summariser.prompts[0]


async def test_page_cut_before_summarising(
    install_extractor, summariser: FakeChatModel, settings: LookUpSettings, retrier: Retrier
) -> None:
    """An uncapped page would blow the summariser's own context."""

    async def handler(url: str) -> dict:
        return {"results": [{"url": url, "raw_content": "Z" * 500}]}

    install_extractor(handler)

    await look_up(
        [URLS[0]], dataclasses.replace(settings, max_page_chars=50), summariser, PROMPT_TEMPLATE, retrier, {}
    )
    assert "Page:\n" + "Z" * 50 == summariser.prompts[0].split("Limit: 300 chars\n")[1]


async def test_urls_keep_slots_in_order(
    every_page_extracts, summariser: FakeChatModel, settings: LookUpSettings, retrier: Retrier
) -> None:
    """Slot i answers URL i, whatever order the fetches finished in."""
    batch = await look_up(URLS, settings, summariser, PROMPT_TEMPLATE, retrier, {})
    assert [text.splitlines()[0] for text in batch.results] == URLS


async def test_clean_batch_has_no_errors(
    every_page_extracts, summariser: FakeChatModel, settings: LookUpSettings, retrier: Retrier
) -> None:
    """Every error slot stays empty when nothing failed."""
    batch = await look_up(URLS, settings, summariser, PROMPT_TEMPLATE, retrier, {})
    assert batch.errors == [None, None, None]


async def test_failed_fetch_fills_own_slot(
    second_url_fails, summariser: FakeChatModel, settings: LookUpSettings, retrier: Retrier
) -> None:
    """The slot stays in place so indexes still line up with the URLs."""
    batch = await look_up(URLS, settings, summariser, PROMPT_TEMPLATE, retrier, {})
    assert batch.errors[1] == FETCH_FAILED
    assert batch.results[1] == ""


async def test_the_other_urls_still_return_summaries(
    second_url_fails, summariser: FakeChatModel, settings: LookUpSettings, retrier: Retrier
) -> None:
    """One bad page never fails the batch."""
    batch = await look_up(URLS, settings, summariser, PROMPT_TEMPLATE, retrier, {})
    assert [batch.errors[0], batch.errors[2]] == [None, None]
    assert [batch.results[0].splitlines()[0], batch.results[2].splitlines()[0]] == [
        URLS[0],
        URLS[2],
    ]


async def test_unsafe_url_never_fetched(
    every_page_extracts, summariser: FakeChatModel, settings: LookUpSettings, retrier: Retrier
) -> None:
    """The guard runs first, so the model is told and no prompt is built."""
    batch = await look_up(
        ["file:///etc/passwd"], settings, summariser, PROMPT_TEMPLATE, retrier, {}
    )

    assert batch.errors[0] == REFUSED
    assert summariser.prompts == []


async def test_unsafe_url_spares_safe_urls(
    every_page_extracts, summariser: FakeChatModel, settings: LookUpSettings, retrier: Retrier
) -> None:
    """One refused URL never costs the batch its good pages."""
    targets = [URLS[0], "ftp://example.com/data", URLS[1]]
    batch = await look_up(targets, settings, summariser, PROMPT_TEMPLATE, retrier, {})
    assert [batch.errors[0], batch.errors[2]] == [None, None]
    assert batch.errors[1] == REFUSED


async def test_empty_page_is_error(
    install_extractor, summariser: FakeChatModel, settings: LookUpSettings, retrier: Retrier
) -> None:
    """An empty page is a failed target, not an empty summary that looks fine."""

    async def handler(url: str) -> dict:
        return {"results": [{"url": url, "raw_content": "   "}]}

    install_extractor(handler)

    batch = await look_up([URLS[0]], settings, summariser, PROMPT_TEMPLATE, retrier, {})
    assert batch.errors[0] == NO_READABLE_TEXT


async def test_repeated_fetch_timeout_is_error(
    install_extractor, summariser: FakeChatModel, settings: LookUpSettings, retrier: Retrier
) -> None:
    """A timeout is retried as transient, then the spent budget is the entry."""

    async def handler(url: str) -> dict:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    install_extractor(handler)

    batch = await look_up(
        [URLS[0]], dataclasses.replace(settings, timeout_seconds=0), summariser, PROMPT_TEMPLATE, retrier, {}
    )
    assert batch.errors[0] == FETCH_GAVE_UP


async def test_failed_summariser_is_error(
    every_page_extracts, settings: LookUpSettings, retrier: Retrier
) -> None:
    """A model error is reported per target, like a fetch error."""
    model = FakeChatModel(structured_replies=[Raises(RuntimeError("model refused the prompt"))])

    batch = await look_up([URLS[0]], settings, model, PROMPT_TEMPLATE, retrier, {})

    assert batch.errors[0] == SUMMARY_FAILED


async def test_summary_uses_fixed_layout(
    every_page_extracts, settings: LookUpSettings, retrier: Retrier
) -> None:
    model = FakeChatModel(
        structured_replies=[
            PageSummary(
                findings=["  Founded in 1893  ", " ", "Moved in 1920"],
                source_date="",
                coverage="PARTIAL",
                coverage_note="No founder name.",
            )
        ]
    )

    batch = await look_up([URLS[0]], settings, model, PROMPT_TEMPLATE, retrier, {})

    assert batch.results[0] == (
        f"{URLS[0]}\nFINDINGS:\n- Founded in 1893\n- Moved in 1920\n"
        "SOURCE_DATE: unknown\nCOVERAGE: PARTIAL. No founder name."
    )


async def test_off_goal_page_renders_coverage_only(
    every_page_extracts, settings: LookUpSettings, retrier: Retrier
) -> None:
    model = FakeChatModel(
        structured_replies=[
            PageSummary(
                findings=[],
                source_date="unknown",
                coverage="NONE",
                coverage_note="The page is a product list.",
            )
        ]
    )

    batch = await look_up([URLS[0]], settings, model, PROMPT_TEMPLATE, retrier, {})

    assert batch.results[0] == f"{URLS[0]}\nCOVERAGE: NONE. The page is a product list."


async def test_mapping_validated_into_summary(
    every_page_extracts, settings: LookUpSettings, retrier: Retrier
) -> None:
    model = FakeChatModel(structured_replies=[FOUND.model_dump()])

    batch = await look_up([URLS[0]], settings, model, PROMPT_TEMPLATE, retrier, {})

    assert batch.errors == [None]


async def test_failed_check_retried_with_error(
    every_page_extracts, settings: LookUpSettings, retrier: Retrier
) -> None:
    model = FakeChatModel(structured_replies=[ValueError("Invalid JSON at line 1"), FOUND])

    await look_up([URLS[0]], settings, model, PROMPT_TEMPLATE, retrier, {})

    assert (
        "Your last summary failed the check: "
        "The response did not match the schema: Invalid JSON at line 1\n"
    ) in model.prompts[1]


async def test_passing_retry_fills_slot(
    every_page_extracts, settings: LookUpSettings, retrier: Retrier
) -> None:
    model = FakeChatModel(structured_replies=[ValueError("Invalid JSON"), FOUND])

    batch = await look_up([URLS[0]], settings, model, PROMPT_TEMPLATE, retrier, {})

    assert batch.errors == [None]


async def test_corrections_accumulate(
    every_page_extracts, settings: LookUpSettings, retrier: Retrier
) -> None:
    """Resetting the prompt each retry throws away what the model was told."""
    model = FakeChatModel(structured_replies=[ValueError("Invalid JSON")])

    await look_up(
        [URLS[0]], dataclasses.replace(settings, max_summary_retries=2), model, PROMPT_TEMPLATE, retrier, {}
    )

    assert model.prompts[2].count("Your last summary failed the check") == 2


async def test_exhausted_retries_become_error(
    every_page_extracts, settings: LookUpSettings, retrier: Retrier
) -> None:
    model = FakeChatModel(structured_replies=[ValueError("Invalid JSON")])

    batch = await look_up([URLS[0]], settings, model, PROMPT_TEMPLATE, retrier, {})

    assert batch.errors[0] == SUMMARY_UNCHECKED


async def test_full_coverage_needs_findings(
    every_page_extracts, settings: LookUpSettings, retrier: Retrier
) -> None:
    empty = FOUND.model_copy(update={"findings": ["  "]})
    model = FakeChatModel(structured_replies=[empty])

    await look_up([URLS[0]], settings, model, PROMPT_TEMPLATE, retrier, {})

    assert (
        "failed the check: Coverage is FULL but no findings were given.\n"
        in model.prompts[1]
    )


async def test_no_coverage_rejects_findings(
    every_page_extracts, settings: LookUpSettings, retrier: Retrier
) -> None:
    contradiction = FOUND.model_copy(update={"coverage": "NONE"})
    model = FakeChatModel(structured_replies=[contradiction])

    await look_up([URLS[0]], settings, model, PROMPT_TEMPLATE, retrier, {})

    assert "failed the check: Coverage is NONE but findings were given.\n" in model.prompts[1]


async def test_cut_off_summary_fails_check_only(
    every_page_extracts, settings: LookUpSettings, retrier: Retrier
) -> None:
    model = FakeChatModel(structured_replies=[Raises(ContentFilterFinishReasonError()), FOUND])

    batch = await look_up([URLS[0]], settings, model, PROMPT_TEMPLATE, retrier, {})

    assert batch.errors == [None]


async def test_failed_checks_counted(
    every_page_extracts, settings: LookUpSettings, retrier: Retrier
) -> None:
    """The count comes back with the batch, so no caller can lose it."""
    model = FakeChatModel(structured_replies=[ValueError("Invalid JSON"), FOUND])

    batch = await look_up([URLS[0]], settings, model, PROMPT_TEMPLATE, retrier, {})

    assert batch.malformed_summaries == 1


async def test_schema_error_not_backed_off(
    every_page_extracts, settings: LookUpSettings, retrier: Retrier
) -> None:
    """The error text is the model's, so a network word in it is not a network error."""
    model = FakeChatModel(structured_replies=[ValueError("lost connection to source"), FOUND])

    await look_up([URLS[0]], settings, model, PROMPT_TEMPLATE, retrier, {})

    assert retrier.counters.transient_retries == 0


async def test_unreadable_payload_fills_own_slot(
    install_extractor, summariser: FakeChatModel, settings: LookUpSettings, retrier: Retrier
) -> None:
    """Parsing happens inside the try, so a bad shape is a per-URL error."""

    async def handler(url: str) -> dict:
        if url == URLS[1]:
            return {"results": ["a bare string where an object belongs"]}
        return _extract_payload(url)

    install_extractor(handler)

    batch = await look_up(URLS, settings, summariser, PROMPT_TEMPLATE, retrier, {})

    assert batch.errors[1] == FETCH_FAILED
    assert [batch.results[0], batch.results[2]] == [
        _rendered_found(URLS[0]),
        _rendered_found(URLS[2]),
    ]


async def test_error_entry_hides_exception(
    install_extractor, summariser: FakeChatModel, settings: LookUpSettings, retrier: Retrier
) -> None:
    """An entry names the failure kind and nothing else."""

    async def handler(url: str) -> dict:
        raise RuntimeError("500 from internal.example with body " + "X" * 5000)

    install_extractor(handler)

    batch = await look_up([URLS[0]], settings, summariser, PROMPT_TEMPLATE, retrier, {})

    assert batch.errors[0] == "Fetch failed: the request did not succeed."


async def test_a_refused_url_names_no_host(
    resolves, every_page_extracts, summariser: FakeChatModel,
    settings: LookUpSettings, retrier: Retrier
) -> None:
    resolves({"internal.example.com": "10.0.0.5"})

    batch = await look_up(
        ["https://internal.example.com/secrets"], settings, summariser, PROMPT_TEMPLATE, retrier, {}
    )

    assert batch.errors[0] == "Refused: not a public http or https address."


async def test_page_past_deadline_is_error(
    install_extractor, summariser: FakeChatModel, settings: LookUpSettings, retrier: Retrier
) -> None:
    """The fetch, every summary try and every wait share one deadline."""

    async def handler(url: str) -> dict:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    install_extractor(handler)

    batch = await look_up(
        [URLS[0]], dataclasses.replace(settings, deadline_seconds=0), summariser, PROMPT_TEMPLATE, retrier, {}
    )

    assert (batch.results[0], batch.errors[0]) == ("", PAST_DEADLINE)


async def test_url_check_runs_off_loop(
    monkeypatch, every_page_extracts, summariser: FakeChatModel,
    settings: LookUpSettings, retrier: Retrier
) -> None:
    """check_url resolves DNS synchronously, which would stall every worker."""
    threads: list[str] = []

    def record(url: str) -> str:
        threads.append(threading.current_thread().name)
        return url

    monkeypatch.setattr(web_look_up, "check_url", record)

    await look_up([URLS[0]], settings, summariser, PROMPT_TEMPLATE, retrier, {})

    assert threads and "MainThread" not in threads


async def test_returned_rate_limit_retried(
    install_extractor, summariser: FakeChatModel, settings: LookUpSettings, retrier: Retrier
) -> None:
    """The integration returns an HTTP failure as a value, so the tool raises it."""
    calls: list[str] = []

    async def handler(url: str) -> dict:
        calls.append(url)
        if len(calls) == 1:
            raise RuntimeError("Error 429: Too Many Requests")
        return _extract_payload(url)

    install_extractor(handler)

    batch = await look_up([URLS[0]], settings, summariser, PROMPT_TEMPLATE, retrier, {})

    assert (batch.results[0], batch.errors[0]) == (_rendered_found(URLS[0]), None)
    assert retrier.counters.rate_limit_hits == 1
