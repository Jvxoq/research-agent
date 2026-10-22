"""The load test's SLO arithmetic, and its agreement with SLO.md."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from loadtest.scenario import RunOutcome
from loadtest.stats import (
    SPEED_SLOS,
    THROUGHPUT_USERS,
    LevelReport,
    RunRecord,
    SloCheck,
    check,
    percentile,
    summarise,
    throughput_holds,
)

SLO_FILE = Path(__file__).resolve().parents[4] / "SLO.md"

CLEAN = RunOutcome(
    answered=True,
    rounds=3,
    rate_limit_retries=0,
    rate_limit_wait_seconds=0.0,
    transient_retries=0,
    bad_response_retries=0,
)


def _run(
    latency: float = 5.0,
    ttft: float | None = 1.0,
    timed_out: bool = False,
    error: str | None = None,
    outcome: RunOutcome | None = CLEAN,
) -> RunRecord:
    return RunRecord(
        graph_id="g",
        users=10,
        ttft_seconds=ttft,
        latency_seconds=latency,
        timed_out=timed_out,
        error=error,
        outcome=outcome,
    )


def _report(users: int = THROUGHPUT_USERS, **values: float | None) -> LevelReport:
    """A level where every SLO passes, with any value overridden."""
    passing = summarise("g", users, [_run()], wall_seconds=60.0)
    return replace(passing, **values)


def _speed_section() -> dict[str, str]:
    """Parse the Speed and reliability keys out of SLO.md's YAML block."""
    block = re.search(r"```yaml\n(.*?)```", SLO_FILE.read_text(), re.DOTALL)
    assert block is not None, "SLO.md has no yaml block"
    section = re.search(r"# Speed and reliability\n(.*?)(?=\n#|\Z)", block.group(1), re.DOTALL)
    assert section is not None, "SLO.md has no Speed and reliability section"
    parsed = yaml.safe_load(section.group(1))
    assert isinstance(parsed, dict) and parsed
    return {str(key): str(value) for key, value in parsed.items()}


def test_targets_match_slo_md() -> None:
    """A target changed in SLO.md and not here would grade runs against a stale bar."""
    expected = {key: text for key, text in _speed_section().items() if key != "throughput"}

    assert {slo.key: slo.target for slo in SPEED_SLOS} == expected


@pytest.mark.parametrize(
    "pct,expected", [(50, 10.0), (95, 19.0), (99, 20.0), (100, 20.0)], ids=str
)
def test_percentile_is_nearest_rank(pct: float, expected: float) -> None:
    """Nearest rank reports a value that was really measured."""
    assert percentile([float(v) for v in range(20, 0, -1)], pct) == expected


def test_percentile_of_nothing_is_none() -> None:
    assert percentile([], 95) is None


def test_summary_counts_every_run() -> None:
    """Failed and timed-out runs still kept a user waiting, so latency counts them."""
    retried = replace(
        CLEAN,
        rounds=8,
        rate_limit_retries=2,
        rate_limit_wait_seconds=4.0,
        transient_retries=3,
        bad_response_retries=1,
    )
    runs = [
        _run(latency=4.0, ttft=0.5),
        _run(latency=6.0, ttft=1.5, outcome=retried),
        _run(latency=300.0, ttft=None, timed_out=True, outcome=None),
        _run(latency=9.0, ttft=2.0, error="boom", outcome=None),
    ]

    report = summarise("g", 10, runs, wall_seconds=30.0)

    assert report == LevelReport(
        graph_id="g",
        users=10,
        runs=4,
        wall_seconds=30.0,
        throughput_per_minute=6.0,
        ttft_p95=2.0,
        latency_p50=6.0,
        latency_p95=300.0,
        latency_p99=300.0,
        rounds_per_task_p95=8.0,
        run_success_rate=0.5,
        timeout_rate=0.25,
        retries_rate_limit=1.0,
        retries_transient=1.5,
        retries_bad_response=0.5,
        rate_limit_wait_p95=4.0,
    )


def test_unanswered_run_is_not_success() -> None:
    """A run that stopped on max rounds returned no answer, so it did not succeed."""
    stopped = replace(CLEAN, answered=False)

    report = summarise("g", 10, [_run(), _run(outcome=stopped)], wall_seconds=10.0)

    assert report.run_success_rate == 0.5


def test_no_outcomes_leaves_retries_unmeasured() -> None:
    """With no final state read, a retry average of 0 would claim a clean run."""
    report = summarise("g", 10, [_run(error="boom", outcome=None)], wall_seconds=10.0)

    assert (report.retries_transient, report.rounds_per_task_p95) == (None, None)


def test_summary_refuses_no_runs() -> None:
    with pytest.raises(ValueError, match="g at 10 users finished no runs"):
        summarise("g", 10, [], wall_seconds=1.0)


@pytest.mark.parametrize(
    "key,passing,failing",
    [
        ("ttft_p95", 1.99, 2.0),
        ("latency_p50", 89.9, 90.0),
        ("latency_p95", 179.9, 180.0),
        ("latency_p99", 299.9, 300.0),
        ("rounds_per_task_p95", 8.0, 9.0),
        ("run_success_rate", 0.99, 0.989),
        ("timeout_rate", 0.0049, 0.005),
        ("retries_rate_limit", 1.0, 1.01),
        ("retries_transient", 1.0, 1.01),
        ("retries_bad_response", 1.0, 1.01),
        ("rate_limit_wait_p95", 9.99, 10.0),
    ],
)
def test_slo_boundary(key: str, passing: float, failing: float) -> None:
    """Each target is checked at its edge, so a < written as <= fails here."""
    verdicts = [
        next(c.passed for c in check(_report(**{key: value})) if c.key == key)
        for value in (passing, failing)
    ]

    assert verdicts == [True, False]


def test_unmeasured_value_has_no_verdict() -> None:
    checks = check(_report(ttft_p95=None))

    assert checks[0] == SloCheck(key="ttft_p95", value=None, target="< 2 s", passed=None)


def test_throughput_holds_when_all_pass() -> None:
    assert throughput_holds([_report(users=10), _report()]) is True


def test_throughput_fails_on_one_slo() -> None:
    assert throughput_holds([_report(latency_p95=181.0)]) is False


def test_throughput_fails_on_unmeasured() -> None:
    """An SLO nobody measured at 50 users has not been shown to hold."""
    assert throughput_holds([_report(ttft_p95=None)]) is False


def test_throughput_unknown_without_level() -> None:
    assert throughput_holds([_report(users=10)]) is None
