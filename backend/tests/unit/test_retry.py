"""The two transport retry families, their ceilings and their counters.

No test here waits on the wall clock. ``call_with_retries`` takes its sleep and
its jitter source as arguments, so every wait is recorded rather than slept.
"""

from __future__ import annotations

from typing import Any

import httpx2
import pytest

from core.retry import (
    BackoffPolicy,
    RetriesExhausted,
    RetryCounters,
    RetryTotals,
    backoff_wait,
    call_with_retries,
    counters_from_config,
    is_rate_limit,
    is_transient,
    retry_after_seconds,
)


class RateLimited(Exception):
    """A provider rate-limit error, carrying a status code like a real one."""

    status_code = 429


class ServerError(Exception):
    """A provider server error."""

    status_code = 503


class Unrecoverable(Exception):
    """An error no amount of waiting will fix."""


@pytest.fixture
def policy() -> BackoffPolicy:
    """A policy with round numbers, so an expected wait is easy to read."""
    return BackoffPolicy(
        base_seconds=1.0,
        max_seconds=8.0,
        jitter=0.0,
        max_wait_seconds=30.0,
        max_transient_retries=2,
    )


@pytest.fixture
def counters() -> RetryCounters:
    """Fresh counters for one run."""
    return RetryCounters()


@pytest.fixture
def waits() -> list[float]:
    """Every wait the retrier asked for, in order."""
    return []


@pytest.fixture
def sleep(waits: list[float]):
    """A sleep that records its argument instead of waiting."""

    async def record(seconds: float) -> None:
        waits.append(seconds)

    return record


def failing(errors: list[BaseException], result: Any = "ok"):
    """Return an operation that raises each error in turn, then succeeds."""
    queue = list(errors)

    async def operation() -> Any:
        if queue:
            raise queue.pop(0)
        return result

    return operation


def test_status_429_is_rate_limit() -> None:
    assert is_rate_limit(RateLimited("slow down"))


def test_message_429_is_not_rate_limit() -> None:
    """A message can quote a model's output. Only the status code counts."""
    assert not is_rate_limit(ValueError("the schema allows 4290 tokens"))


def test_connection_message_not_transient() -> None:
    assert not is_transient(ValueError("field 'connection' is missing"))


TRANSPORT_FAILURES = [
    httpx2.ConnectError("refused"),
    httpx2.ReadTimeout("timed out"),
]


@pytest.mark.parametrize(
    "cause", TRANSPORT_FAILURES, ids=["connect", "read-timeout"]
)
def test_transport_cause_is_transient(
    cause: Exception,
) -> None:
    """The OpenAI SDK raises its own error from an httpx2 transport one."""
    try:
        try:
            raise cause
        except Exception as raised:
            raise RuntimeError("Request timed out.") from raised
    except RuntimeError as error:
        wrapped = error

    assert is_transient(wrapped)


def test_a_server_error_is_transient() -> None:
    assert is_transient(ServerError("bad gateway"))


def test_a_timeout_is_transient() -> None:
    assert is_transient(TimeoutError("took too long"))


def test_an_unrecoverable_error_is_neither_family() -> None:
    error = Unrecoverable("bad request")

    assert not is_rate_limit(error)
    assert not is_transient(error)


def test_retry_after_read_from_header() -> None:
    class WithHeaders(Exception):
        response = type("Response", (), {"headers": {"retry-after": "7"}})()

    assert retry_after_seconds(WithHeaders()) == 7.0


def test_no_retry_after_hint_returns_none() -> None:
    assert retry_after_seconds(Unrecoverable("nothing here")) is None


def test_backoff_doubles_to_cap(policy: BackoffPolicy) -> None:
    waits = [backoff_wait(attempt, policy, lambda: 0.0) for attempt in range(5)]

    assert waits == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_long_streak_stays_at_cap(policy: BackoffPolicy) -> None:
    """2.0 ** 1024 overflows a float. The cap must hold long before that."""
    assert backoff_wait(5000, policy, lambda: 0.0) == 8.0


def test_jitter_adds_fraction() -> None:
    policy = BackoffPolicy(
        base_seconds=10.0,
        max_seconds=60.0,
        jitter=0.5,
        max_wait_seconds=300.0,
        max_transient_retries=2,
    )

    assert backoff_wait(0, policy, lambda: 1.0) == 15.0


async def test_a_successful_call_never_waits(
    policy: BackoffPolicy, counters: RetryCounters, waits: list[float], sleep
) -> None:
    result = await call_with_retries(failing([]), policy, counters, sleep=sleep)

    assert result == "ok"
    assert waits == []


async def test_rate_limit_backs_off_then_succeeds(
    policy: BackoffPolicy, counters: RetryCounters, waits: list[float], sleep
) -> None:
    operation = failing([RateLimited("429"), RateLimited("429")])

    result = await call_with_retries(
        operation, policy, counters, sleep=sleep, roll=lambda: 0.0
    )

    assert result == "ok"
    assert waits == [1.0, 2.0]


async def test_rate_limit_honours_retry_after(
    policy: BackoffPolicy, counters: RetryCounters, waits: list[float], sleep
) -> None:
    error = RateLimited("429")
    error.retry_after = 5

    await call_with_retries(
        failing([error]), policy, counters, sleep=sleep, roll=lambda: 0.0
    )

    assert waits == [5.0]


async def test_zero_retry_after_still_backs_off(
    policy: BackoffPolicy, counters: RetryCounters, waits: list[float], sleep
) -> None:
    """A zero hint would otherwise retry at once, and the wait would never grow."""
    error = RateLimited("429")
    error.retry_after = 0

    await call_with_retries(
        failing([error]), policy, counters, sleep=sleep, roll=lambda: 0.0
    )

    assert waits == [1.0]


async def test_endless_zero_wait_gives_up(
    counters: RetryCounters, sleep
) -> None:
    """The retry loop must end even when every Retry-After says 0."""
    policy = BackoffPolicy(
        base_seconds=1.0,
        max_seconds=8.0,
        jitter=0.0,
        max_wait_seconds=30.0,
        max_transient_retries=2,
    )
    errors = []
    for _ in range(1000):
        error = RateLimited("429")
        error.retry_after = 0
        errors.append(error)

    with pytest.raises(RetriesExhausted):
        await call_with_retries(
            failing(errors), policy, counters, sleep=sleep, roll=lambda: 0.0
        )


async def test_rate_limit_hits_counted(
    policy: BackoffPolicy, counters: RetryCounters, sleep
) -> None:
    operation = failing([RateLimited("429"), RateLimited("429")])

    await call_with_retries(
        operation, policy, counters, sleep=sleep, roll=lambda: 0.0
    )

    assert counters.to_dict()["rate_limit_hits"] == 2
    assert counters.to_dict()["rate_limit_wait_seconds"] == 3.0


async def test_rate_limit_skips_transient_counter(
    policy: BackoffPolicy, counters: RetryCounters, sleep
) -> None:
    """Mixing the two budgets hides which problem the run actually had."""
    rate_limits = failing([RateLimited("429"), RateLimited("429")])

    await call_with_retries(rate_limits, policy, counters, sleep=sleep, roll=lambda: 0.0)

    assert counters.transient_retries == 0


async def test_transient_skips_rate_limit_counter(
    policy: BackoffPolicy, counters: RetryCounters, sleep
) -> None:
    await call_with_retries(
        failing([ServerError("503")]), policy, counters, sleep=sleep, roll=lambda: 0.0
    )

    assert counters.rate_limit_hits == 0


async def test_transient_gives_up_at_ceiling(
    policy: BackoffPolicy, counters: RetryCounters, sleep
) -> None:
    operation = failing([ServerError("503")] * 5)

    with pytest.raises(RetriesExhausted):
        await call_with_retries(
            operation, policy, counters, sleep=sleep, roll=lambda: 0.0
        )

    assert counters.transient_retries == policy.max_transient_retries


async def test_rate_limit_stops_at_wait_ceiling(
    counters: RetryCounters, waits: list[float], sleep
) -> None:
    policy = BackoffPolicy(
        base_seconds=1.0,
        max_seconds=8.0,
        jitter=0.0,
        max_wait_seconds=3.0,
        max_transient_retries=2,
    )
    operation = failing([RateLimited("429")] * 50)

    with pytest.raises(RetriesExhausted):
        await call_with_retries(
            operation, policy, counters, sleep=sleep, roll=lambda: 0.0
        )

    assert waits == [1.0, 2.0]


async def test_fatal_error_raised_without_wait(
    policy: BackoffPolicy, counters: RetryCounters, waits: list[float], sleep
) -> None:
    with pytest.raises(Unrecoverable):
        await call_with_retries(
            failing([Unrecoverable("bad request")]), policy, counters, sleep=sleep
        )

    assert waits == []


def test_counters_from_configurable() -> None:
    mine = RetryCounters(rate_limit_hits=4)

    found = counters_from_config({"configurable": {"retry_counters": mine}}, None)

    assert found is mine


def test_missing_counters_get_fresh_set() -> None:
    assert counters_from_config(None, None).rate_limit_hits == 0


def test_served_run_continues_state_totals() -> None:
    """A request body cannot carry counters, so state is the only carrier."""
    spent: RetryTotals = {
        "rate_limit_hits": 3,
        "rate_limit_wait_seconds": 4.5,
        "transient_retries": 1,
        "transient_wait_seconds": 2.0,
        "malformed_summaries": 2,
    }

    found = counters_from_config({"configurable": {"thread_id": "t"}}, spent)

    assert found.to_dict() == spent
