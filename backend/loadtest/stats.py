"""Turn one load level's runs into the speed and reliability SLO numbers.

Pure functions, so the arithmetic is tested without a server.

``SPEED_SLOS`` repeats the targets in ``SLO.md``. Each entry keeps the target
text exactly as ``SLO.md`` writes it, and a test checks the two still agree, so
a target changed in one place and not the other fails the suite.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from loadtest.scenario import RunOutcome

# The users level at which SLO.md's throughput key asks every speed SLO to hold.
THROUGHPUT_USERS = 50


@dataclass(frozen=True)
class RunRecord:
    """One run, as the load client saw it.

    Args:
        graph_id: The assistant the run used.
        users: How many users ran at once in this level.
        ttft_seconds: Seconds to the first streamed update, or None when the
            run produced none.
        latency_seconds: Seconds from the request to the end of the stream.
        timed_out: The run hit the whole-run timeout.
        error: The capped error text when the run failed, else None.
        outcome: What the final state says, or None when there was none.
    """

    graph_id: str
    users: int
    ttft_seconds: float | None
    latency_seconds: float
    timed_out: bool
    error: str | None
    outcome: RunOutcome | None

    @property
    def succeeded(self) -> bool:
        """The run returned an answer with no error and no timeout."""
        return (
            not self.timed_out
            and self.error is None
            and self.outcome is not None
            and self.outcome.answered
        )


@dataclass(frozen=True)
class LevelReport:
    """The SLO numbers for one graph at one users level.

    A field is None when no run produced the value it needs.
    """

    graph_id: str
    users: int
    runs: int
    wall_seconds: float
    throughput_per_minute: float
    ttft_p95: float | None
    latency_p50: float | None
    latency_p95: float | None
    latency_p99: float | None
    rounds_per_task_p95: float | None
    run_success_rate: float
    timeout_rate: float
    retries_rate_limit: float | None
    retries_transient: float | None
    retries_bad_response: float | None
    rate_limit_wait_p95: float | None


@dataclass(frozen=True)
class SpeedSlo:
    """One speed and reliability target.

    Args:
        key: The metric name in ``SLO.md`` and on :class:`LevelReport`.
        target: The target text exactly as ``SLO.md`` writes it.
        passes: True when a value meets the target.
    """

    key: str
    target: str
    passes: Callable[[float], bool]


SPEED_SLOS: tuple[SpeedSlo, ...] = (
    SpeedSlo("ttft_p95", "< 2 s", lambda v: v < 2),
    SpeedSlo("latency_p50", "< 90 s", lambda v: v < 90),
    SpeedSlo("latency_p95", "< 3 min", lambda v: v < 180),
    SpeedSlo("latency_p99", "< 5 min", lambda v: v < 300),
    SpeedSlo("rounds_per_task_p95", "<= 8", lambda v: v <= 8),
    SpeedSlo("run_success_rate", ">= 99% of runs", lambda v: v >= 0.99),
    SpeedSlo("timeout_rate", "< 0.5% of runs", lambda v: v < 0.005),
    SpeedSlo("retries_rate_limit", "<= 1 per task on average", lambda v: v <= 1),
    SpeedSlo("retries_transient", "<= 1 per task on average", lambda v: v <= 1),
    SpeedSlo("retries_bad_response", "<= 1 per task on average", lambda v: v <= 1),
    SpeedSlo("rate_limit_wait_p95", "< 10 s per task", lambda v: v < 10),
)


@dataclass(frozen=True)
class SloCheck:
    """One target checked against one level.

    Args:
        key: The metric name.
        value: The measured value, or None when nothing measured it.
        target: The target text from ``SLO.md``.
        passed: None when there is no value to check.
    """

    key: str
    value: float | None
    target: str
    passed: bool | None


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Return the nearest-rank percentile, or None for no values.

    Nearest rank returns a value that was really measured. With 20 values, p99
    is the slowest one, which is what a reader should be told it is.

    Args:
        values: The measurements.
        pct: The percentile, from 0 to 100.

    Returns:
        The smallest value with at least ``pct`` percent of values at or below it.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarise(
    graph_id: str, users: int, runs: Sequence[RunRecord], wall_seconds: float
) -> LevelReport:
    """Compute one level's SLO numbers from its runs.

    Latency covers every run, failed ones included, because a run that errors
    after four minutes still kept a user waiting four minutes. Rounds and
    retries come only from runs whose final state was read.

    Args:
        graph_id: The assistant the level ran.
        users: How many users ran at once.
        runs: Every run of the level.
        wall_seconds: Time from the first request to the last run's end.

    Returns:
        The level's report.

    Raises:
        ValueError: When ``runs`` is empty, since no rate means anything then.
    """
    if not runs:
        raise ValueError(f"{graph_id} at {users} users finished no runs")
    outcomes = [run.outcome for run in runs if run.outcome is not None]
    latencies = [run.latency_seconds for run in runs]
    finished = sum(1 for run in runs if not run.timed_out)
    return LevelReport(
        graph_id=graph_id,
        users=users,
        runs=len(runs),
        wall_seconds=wall_seconds,
        throughput_per_minute=finished / wall_seconds * 60 if wall_seconds > 0 else 0.0,
        ttft_p95=percentile([r.ttft_seconds for r in runs if r.ttft_seconds is not None], 95),
        latency_p50=percentile(latencies, 50),
        latency_p95=percentile(latencies, 95),
        latency_p99=percentile(latencies, 99),
        rounds_per_task_p95=percentile([float(o.rounds) for o in outcomes], 95),
        run_success_rate=sum(1 for run in runs if run.succeeded) / len(runs),
        timeout_rate=sum(1 for run in runs if run.timed_out) / len(runs),
        retries_rate_limit=_mean([float(o.rate_limit_retries) for o in outcomes]),
        retries_transient=_mean([float(o.transient_retries) for o in outcomes]),
        retries_bad_response=_mean([float(o.bad_response_retries) for o in outcomes]),
        rate_limit_wait_p95=percentile([o.rate_limit_wait_seconds for o in outcomes], 95),
    )


def check(report: LevelReport) -> list[SloCheck]:
    """Check every speed SLO against one level's report."""
    checks: list[SloCheck] = []
    for slo in SPEED_SLOS:
        value: float | None = getattr(report, slo.key)
        checks.append(
            SloCheck(
                key=slo.key,
                value=value,
                target=slo.target,
                passed=None if value is None else slo.passes(value),
            )
        )
    return checks


def throughput_holds(reports: Sequence[LevelReport]) -> bool | None:
    """Report whether every speed SLO holds at the throughput users level.

    Returns:
        None when no level ran at :data:`THROUGHPUT_USERS` users. False when any
        target fails or has no value there, because an unmeasured target has
        not been shown to hold.
    """
    at_level = [report for report in reports if report.users == THROUGHPUT_USERS]
    if not at_level:
        return None
    return all(c.passed is True for report in at_level for c in check(report))
