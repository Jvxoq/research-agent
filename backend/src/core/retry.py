"""Retry policy for the two transport failure families.

There are three failure families, and they never share a counter. Two of them
are transport problems and live here:

1. A rate limit, HTTP 429. Backs off with jitter, waits at least as long as
   ``Retry-After`` asks, and gives up once its cumulative wait crosses the
   ceiling.
2. A transient network or server error: a timeout, a dropped connection or an
   HTTP 5xx. Backs off the same way, bounded by its own attempt count and its
   own cumulative wait.

An error is classified by its type and its status code, never by its message.
A message can quote a model's output, and a stray "429" or "connection" in it
would otherwise send a bad request into a backoff loop.

The third family, a malformed model response, is not a transport problem. It is
retried where the response is checked, without backoff, because waiting does not
improve a model's output. Nothing here retries it. ``RetryCounters`` only carries
the count of failed summary checks, because the look-up tool has no other way to
hand a number back to the run.

Each family keeps its own attempt count and its own wait total, so a run that
spent its budget waiting on rate limits is never mistaken for one that spent it
on broken connections.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx2
from langchain_core.runnables import RunnableConfig
from loguru import logger
from typing_extensions import TypedDict

# The key under which a caller puts its counters in the run's ``configurable``
# mapping. The eval runner creates one per question and reads the totals back
# out when the run ends.
COUNTERS_KEY = "retry_counters"

# Transport failures with no status code. The OpenAI SDK raises its own
# connection and timeout errors from an httpx2 one, so the cause chain is read.
_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    httpx2.TransportError,
)


class RetriesExhausted(RuntimeError):
    """Raised when a retry family spends its whole budget without succeeding."""


@dataclass(frozen=True)
class BackoffPolicy:
    """Backoff shape and the ceiling for each transport family.

    Args:
        base_seconds: The first wait. It doubles on each further attempt. Must be
            positive, so every retry adds to the cumulative wait.
        max_seconds: Cap on any single wait.
        jitter: Random fraction added to a wait, so retries desynchronise.
        max_wait_seconds: Cumulative wait ceiling, applied to each family
            separately.
        max_transient_retries: Attempt ceiling for the transient family. The
            rate-limit family is bounded by cumulative wait alone, because a
            provider may legitimately ask for many short waits in a row.
    """

    base_seconds: float
    max_seconds: float
    jitter: float
    max_wait_seconds: float
    max_transient_retries: int


class RetryTotals(TypedDict):
    """What a run waited for so far, as it is carried between nodes."""

    rate_limit_hits: int
    rate_limit_wait_seconds: float
    transient_retries: int
    transient_wait_seconds: float
    malformed_summaries: int


@dataclass
class RetryCounters:
    """What one run waited for, counted per family.

    A slow run and an expensive run are different results, and ``Y`` means
    nothing without these numbers beside it.
    """

    rate_limit_hits: int = 0
    rate_limit_wait_seconds: float = 0.0
    transient_retries: int = 0
    transient_wait_seconds: float = 0.0
    malformed_summaries: int = 0

    def to_dict(self) -> RetryTotals:
        """Return a JSON-serialisable snapshot of the totals."""
        return {
            "rate_limit_hits": self.rate_limit_hits,
            "rate_limit_wait_seconds": round(self.rate_limit_wait_seconds, 3),
            "transient_retries": self.transient_retries,
            "transient_wait_seconds": round(self.transient_wait_seconds, 3),
            "malformed_summaries": self.malformed_summaries,
        }


def counters_from_config(
    config: RunnableConfig | None, spent: RetryTotals | None
) -> RetryCounters:
    """Return the caller's counters, or a fresh set that continues the run's totals.

    An in-process caller, such as the eval runner, creates one set per run and
    hands it in through ``configurable``. A request body cannot carry that
    object, so a served run carries its totals in graph state instead, and each
    node continues from ``spent``.

    Args:
        config: The run's configuration, if any.
        spent: The totals the run's earlier nodes wrote to state, if any.

    Returns:
        The caller's counters, or a new set starting from ``spent``.
    """
    configurable = config.get("configurable") if config else None
    counters = (configurable or {}).get(COUNTERS_KEY)
    if isinstance(counters, RetryCounters):
        return counters
    return RetryCounters(**spent) if spent else RetryCounters()


def _status_code(error: BaseException) -> int | None:
    """Read an HTTP status code off a provider error, if it carries one."""
    for attribute in ("status_code", "http_status", "code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _causes(error: BaseException) -> list[BaseException]:
    """Return the error and every exception it was raised ``from``, in order.

    Only an explicit cause counts. An error raised while some other exception
    was being handled carries that one as context, and a timeout handled
    upstream must not make an unrelated error look transient.
    """
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__
    return chain


def is_rate_limit(error: BaseException) -> bool:
    """Report whether an error is a provider rate limit.

    Args:
        error: The exception a call raised.

    Returns:
        True when the error carries HTTP status 429.
    """
    return _status_code(error) == 429


def is_transient(error: BaseException) -> bool:
    """Report whether an error is a transient network or server failure.

    Args:
        error: The exception a call raised.

    Returns:
        True for an HTTP 5xx, or when the error or anything it was raised from
        is a timeout or a transport failure.
    """
    status = _status_code(error)
    if status is not None and 500 <= status < 600:
        return True
    return any(isinstance(cause, _TRANSIENT_TYPES) for cause in _causes(error))


def retry_after_seconds(error: BaseException) -> float | None:
    """Read the provider's own ``Retry-After`` hint, when it sent one.

    Args:
        error: The exception a call raised.

    Returns:
        The number of seconds the provider asked for, or None.
    """
    for value in (
        getattr(error, "retry_after", None),
        (getattr(getattr(error, "response", None), "headers", None) or {}).get(
            "retry-after"
        ),
    ):
        try:
            if value is not None:
                return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return None


def backoff_wait(attempt: int, policy: BackoffPolicy, roll: Callable[[], float]) -> float:
    """Return the wait for one attempt, exponential and jittered.

    Args:
        attempt: Zero-based attempt number for this family.
        policy: Backoff shape.
        roll: Source of the jitter fraction, injected so a test is not random.

    Returns:
        Seconds to wait before the next attempt.
    """
    # Past this many doublings the wait is capped anyway. Stopping there keeps
    # a long rate-limit streak from overflowing the float.
    doublings_to_cap = math.log2(policy.max_seconds / policy.base_seconds)
    if attempt >= doublings_to_cap:
        wait = policy.max_seconds
    else:
        wait = math.ldexp(policy.base_seconds, attempt)
    return wait * (1.0 + policy.jitter * roll())


async def call_with_retries[T](
    operation: Callable[[], Awaitable[T]],
    policy: BackoffPolicy,
    counters: RetryCounters,
    *,
    label: str = "call",
    sleep: Callable[[float], Awaitable[None]] | None = None,
    roll: Callable[[], float] | None = None,
) -> T:
    """Run one async operation, retrying the two transport families.

    Each family keeps its own attempt count and its own cumulative wait, so
    neither can exhaust the other's budget. Every other exception is re-raised
    untouched, which is how a malformed decision reaches its own handler.

    Args:
        operation: The call to make. Takes no arguments so a caller can close
            over whatever it needs.
        policy: Backoff shape and both ceilings.
        counters: The run's counters. Mutated in place.
        label: Name used in log lines, so a slow tool is identifiable.
        sleep: Injected so a test never waits on the wall clock. Defaults to
            ``asyncio.sleep``, looked up when the call runs rather than when
            this function was defined, so a test may replace it.
        roll: Injected so a test is not random. Defaults to ``random.random``.

    Returns:
        Whatever the operation returned.

    Raises:
        RetriesExhausted: When a family crosses its ceiling.
        BaseException: Whatever the operation raised, when it is neither a rate
            limit nor a transient failure.
    """
    pause = sleep or asyncio.sleep
    dice = roll or random.random
    rate_limit_attempts = 0
    rate_limit_wait = 0.0
    transient_attempts = 0
    transient_wait = 0.0

    while True:
        try:
            return await operation()
        except Exception as error:
            if is_rate_limit(error):
                # The provider's hint is a floor, never the whole wait. A
                # Retry-After of 0 would otherwise retry at once, forever,
                # because the cumulative wait would never grow.
                wait = max(
                    retry_after_seconds(error) or 0.0,
                    backoff_wait(rate_limit_attempts, policy, dice),
                )
                counters.rate_limit_hits += 1

                if rate_limit_wait + wait > policy.max_wait_seconds:
                    logger.warning(
                        f"{label}: rate limited again after "
                        f"{rate_limit_wait:.1f}s of waiting, giving up"
                    )
                    raise RetriesExhausted(
                        f"{label} stayed rate limited for "
                        f"{rate_limit_wait:.1f}s"
                    ) from error

                rate_limit_attempts += 1
                rate_limit_wait += wait
                counters.rate_limit_wait_seconds += wait
                logger.warning(
                    f"{label}: rate limited, waiting {wait:.1f}s "
                    f"(attempt {rate_limit_attempts})"
                )
                await pause(wait)
                continue

            if is_transient(error):
                wait = backoff_wait(transient_attempts, policy, dice)
                if (
                    transient_attempts >= policy.max_transient_retries
                    or transient_wait + wait > policy.max_wait_seconds
                ):
                    logger.warning(
                        f"{label}: transient failure after "
                        f"{transient_attempts} retries, giving up"
                    )
                    raise RetriesExhausted(
                        f"{label} failed {transient_attempts + 1} times: "
                        f"{type(error).__name__}"
                    ) from error

                transient_attempts += 1
                transient_wait += wait
                counters.transient_retries += 1
                counters.transient_wait_seconds += wait
                logger.warning(
                    f"{label}: {type(error).__name__}, retrying in {wait:.1f}s "
                    f"(attempt {transient_attempts})"
                )
                await pause(wait)
                continue

            raise


@dataclass(frozen=True)
class Retrier:
    """A policy and a counter set, carried together into a tool.

    Tools take limits as arguments and never read configuration themselves, so
    this is how a node hands its retry budget down to one.
    """

    policy: BackoffPolicy
    counters: RetryCounters = field(default_factory=RetryCounters)

    async def run[T](self, operation: Callable[[], Awaitable[T]], label: str) -> T:
        """Run one operation under this policy.

        Args:
            operation: The call to make.
            label: Name used in log lines.

        Returns:
            Whatever the operation returned.
        """
        return await call_with_retries(
            operation, self.policy, self.counters, label=label
        )
