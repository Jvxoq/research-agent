"""The load runner: one run, one level, its wiring and its exit code.

The LangGraph client is a fake, so no test reaches a server.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
import yaml
from langgraph_sdk import get_client as real_get_client
from langgraph_sdk.schema import StreamPart

from core.auth import authenticate
from core.tools.text import CUT_MARK
from evals.research_agent.load import SCENARIO
from loadtest import run
from loadtest.run import (
    COMPOSE,
    MANIFEST,
    SetupError,
    Wiring,
    exit_code,
    one_run,
    parse_args,
    read_graphs,
    read_wiring,
    run_level,
    token_lifetime,
)
from loadtest.scenario import RunOutcome
from loadtest.stats import RunRecord, summarise
from tests.conftest import PeakGauge, hang, logged, off_loop, settle

FINAL_STATE = {
    "question": "q",
    "round": 3,
    "answer": "an answer",
    "stop_reason": "answer",
    "retries": {
        "rate_limit_hits": 2,
        "rate_limit_wait_seconds": 1.5,
        "transient_retries": 1,
        "transient_wait_seconds": 0.5,
        "malformed_summaries": 3,
    },
}


@dataclass
class FakeClock:
    """A clock the fake stream moves forward as each event arrives."""

    now: float = 100.0

    def __call__(self) -> float:
        return self.now


@dataclass
class Step:
    """One thing the fake stream does: wait until ``at`` seconds, then yield ``part``."""

    at: float
    part: StreamPart


def _part(event: str, data: object) -> StreamPart:
    return StreamPart(event=event, data=data, id=None)


ANSWERED = [
    Step(0.1, _part("metadata", {"run_id": "run-1"})),
    Step(0.2, _part("values", {"question": "q"})),
    Step(1.5, _part("updates", {"think": {"round": 1}})),
    Step(3.0, _part("values", FINAL_STATE)),
]


@dataclass
class FakeRuns:
    """Stands in for ``client.runs``: streams scripted steps, then ends, hangs or raises."""

    clock: FakeClock
    steps: list[Step]
    then: str = "end"
    gauge: PeakGauge | None = None
    streams: int = 0
    cancels: list[tuple[str, str]] = field(default_factory=list)
    cancel_fails: bool = False

    async def stream(self, thread_id: str, assistant_id: str, **kwargs: Any) -> AsyncIterator[StreamPart]:
        self.streams += 1
        if self.gauge is not None:
            await self.gauge.hold()
        for step in self.steps:
            self.clock.now = 100.0 + step.at
            yield step.part
        if self.then == "hang":
            await hang()
        if self.then == "raise":
            raise httpx.ConnectError("connection refused")
        self.clock.now = 104.0

    async def cancel(self, thread_id: str, run_id: str) -> None:
        if self.cancel_fails:
            raise httpx.ConnectError("server gone")
        self.cancels.append((thread_id, run_id))


@dataclass
class FakeThreads:
    created: int = 0

    async def create(self) -> dict[str, str]:
        self.created += 1
        return {"thread_id": f"thread-{self.created}"}


@dataclass
class FakeClient:
    runs: FakeRuns
    threads: FakeThreads = field(default_factory=FakeThreads)
    headers: dict[str, str] = field(default_factory=dict)
    closed: bool = False

    async def aclose(self) -> None:
        self.closed = True


def _args(*extra: str) -> Any:
    return parse_args(["--max-error-chars", "60", *extra])


async def _one(runs: FakeRuns, *extra: str) -> RunRecord:
    return await one_run(
        FakeClient(runs), "research_agent", 5, {"question": "q"}, SCENARIO, _args(*extra), runs.clock
    )


def _wiring() -> Wiring:
    return read_wiring(COMPOSE)


async def test_answered_run_record() -> None:
    """TTFT counts to the first node update, not the stream's opening input echo."""
    clock = FakeClock()

    record = await _one(FakeRuns(clock, ANSWERED))

    assert record == RunRecord(
        graph_id="research_agent",
        users=5,
        ttft_seconds=1.5,
        latency_seconds=4.0,
        timed_out=False,
        error=None,
        outcome=RunOutcome(
            answered=True,
            rounds=3,
            rate_limit_retries=2,
            rate_limit_wait_seconds=1.5,
            transient_retries=1,
            bad_response_retries=3,
        ),
    )


async def test_error_event_fails_run() -> None:
    steps = [*ANSWERED[:3], Step(2.0, _part("error", {"message": "boom"}))]

    record = await _one(FakeRuns(FakeClock(), steps))

    assert (record.error, record.outcome, record.succeeded) == ('{"message": "boom"}', None, False)


async def test_huge_error_is_capped() -> None:
    """A server error can quote the whole request. It must not flood the report."""
    steps = [Step(1.0, _part("error", {"message": "x" * 10_000}))]

    record = await _one(FakeRuns(FakeClock(), steps))

    assert record.error is not None
    assert (len(record.error), record.error.endswith(CUT_MARK)) == (60, True)


async def test_transport_failure_fails_run(log_records) -> None:
    """One broken connection fills its own run's slot and is logged with its trace."""
    record = await _one(FakeRuns(FakeClock(), ANSWERED[:3], then="raise"))

    assert record.error == "ConnectError: connection refused"
    assert logged(log_records) == [("ERROR", "research_agent run failed", True)]


async def test_hung_run_times_out_and_cancels() -> None:
    """A run past the ceiling is cancelled, so it stops holding a server worker."""
    runs = FakeRuns(FakeClock(), ANSWERED[:3], then="hang")

    record = await _one(runs, "--run-timeout-seconds", "0.05")

    assert (record.timed_out, record.error, runs.cancels) == (True, None, [("thread-1", "run-1")])


async def test_hang_before_run_id_skips_cancel() -> None:
    runs = FakeRuns(FakeClock(), [], then="hang")

    record = await _one(runs, "--run-timeout-seconds", "0.05")

    assert (record.timed_out, runs.cancels) == (True, [])


async def test_failed_cancel_keeps_timeout(log_records) -> None:
    runs = FakeRuns(FakeClock(), ANSWERED[:3], then="hang", cancel_fails=True)

    record = await _one(runs, "--run-timeout-seconds", "0.05")

    assert record.timed_out is True
    assert logged(log_records) == [("ERROR", "could not cancel run run-1", True)]


async def test_no_final_state_fails_run() -> None:
    record = await _one(FakeRuns(FakeClock(), [ANSWERED[0], ANSWERED[2]]))

    assert record.error == "the stream ended with no final state"


async def test_wrong_final_state_fails_run() -> None:
    steps = [*ANSWERED[:3], Step(3.0, _part("values", {"round": "three"}))]

    record = await _one(FakeRuns(FakeClock(), steps))

    assert record.error is not None
    assert record.error.startswith("final state has the wrong shape:")


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch) -> list[FakeClient]:
    """Replace the SDK's client factory, recording every client a level builds."""
    built: list[FakeClient] = []
    gauge = PeakGauge()
    clock = FakeClock()

    def connect(**kwargs: Any) -> FakeClient:
        # Build the real client too: its headers are what would leave the process.
        real = real_get_client(**kwargs)
        sent = dict(real.http.client.headers)
        client = FakeClient(FakeRuns(clock, ANSWERED, gauge=gauge), headers=sent)
        built.append(client)
        return client

    monkeypatch.setattr(run, "get_client", connect)
    return built


async def _level(clients: list[FakeClient], users: int, runs: int) -> tuple[int, int, int]:
    """Run a level, hold its runs open until all start, then let them finish.

    Returns:
        Runs in flight while held, the peak, and the runs the level reported.
    """
    level = asyncio.create_task(
        run_level("research_agent", users, SCENARIO, _wiring(), _args("--runs-per-level", str(runs)))
    )
    await settle()
    gauge = clients[0].runs.gauge
    assert gauge is not None
    held = gauge.in_flight
    gauge.gate.set()
    report = await level
    return held, gauge.peak, report.runs


async def test_level_runs_exactly_users_at_once(clients: list[FakeClient]) -> None:
    """A level of 3 users must hold 3 runs open, not 1 and not all 8."""
    assert await _level(clients, users=3, runs=8) == (3, 3, 8)


async def test_level_closes_every_client(clients: list[FakeClient]) -> None:
    await _level(clients, users=3, runs=4)

    assert [client.closed for client in clients] == [True, True, True]


async def test_each_user_passes_real_auth(clients: list[FakeClient], set_env) -> None:
    """Every user's token must pass the server's own JWT check as a separate tenant."""
    wiring = _wiring()
    set_env(
        JWT_SECRET=wiring.jwt_secret,
        JWT_ALGORITHM=wiring.jwt_algorithm,
        JWT_ISSUER=wiring.jwt_issuer,
        JWT_AUDIENCE=wiring.jwt_audience,
    )
    await _level(clients, users=3, runs=3)

    users = [await authenticate(client.headers) for client in clients]

    assert sorted(user["identity"] for user in users) == [
        "loadtest-user-0",
        "loadtest-user-1",
        "loadtest-user-2",
    ]


async def test_level_sends_no_langsmith_key(clients: list[FakeClient], set_env) -> None:
    """The SDK reads LANGSMITH_API_KEY from this machine unless told not to."""
    set_env(LANGSMITH_API_KEY="lsv2-this-machine-only")

    await _level(clients, users=2, runs=2)

    assert [client.headers.get("x-api-key") for client in clients] == [None, None]


def test_token_outlives_worst_level() -> None:
    """One user running every run to the timeout, then the cancel wait, still holds a valid token."""
    assert token_lifetime(300.0, 200) == timedelta(seconds=310 * 200)


def test_user_token_carries_wiring_claims() -> None:
    wiring = _wiring()

    claims = jwt.decode(
        run.user_token(wiring, 7, timedelta(hours=1)),
        wiring.jwt_secret,
        algorithms=[wiring.jwt_algorithm],
        audience=wiring.jwt_audience,
    )

    assert (claims["sub"], claims["iss"], claims["exp"] - claims["iat"]) == (
        "loadtest-user-7",
        wiring.jwt_issuer,
        3600,
    )


def test_wiring_reads_compose() -> None:
    """The runner and the server read the fake port and the JWT key from one file."""
    environment = yaml.safe_load(COMPOSE.read_text())["services"]["langgraph-api"]["environment"]

    assert _wiring() == Wiring(
        fake_port=8790,
        jwt_secret=environment["JWT_SECRET"],
        jwt_algorithm="HS256",
        jwt_issuer="loadtest",
        jwt_audience="loadtest",
    )


def _compose_with(tmp_path: Path, **environment: str) -> Path:
    document = yaml.safe_load(COMPOSE.read_text())
    document["services"]["langgraph-api"]["environment"].update(environment)
    path = tmp_path / "compose.yml"
    path.write_text(yaml.safe_dump(document))
    return path


@pytest.mark.parametrize(
    "key", ["BASE_URL", "THINK_BASE_URL", "LOOK_UP_BASE_URL", "TAVILY_BASE_URL"]
)
def test_wiring_refuses_port_drift(tmp_path: Path, key: str) -> None:
    """A URL on another port would send that traffic somewhere the fakes are not."""
    compose = _compose_with(tmp_path, **{key: "http://host.docker.internal:9999/v1"})

    with pytest.raises(SetupError, match="must name the same explicit port"):
        read_wiring(compose)


def test_wiring_refuses_missing_port(tmp_path: Path) -> None:
    compose = _compose_with(
        tmp_path,
        BASE_URL="http://host.docker.internal/v1",
        THINK_BASE_URL="http://host.docker.internal/v1",
        LOOK_UP_BASE_URL="http://host.docker.internal/v1",
        TAVILY_BASE_URL="http://host.docker.internal",
    )

    with pytest.raises(SetupError, match="must name the same explicit port"):
        read_wiring(compose)


@pytest.mark.parametrize(
    "document",
    [{"services": {}}, {"services": {"langgraph-api": {"environment": {"BASE_URL": "x"}}}}, None],
    ids=["no-service", "missing-keys", "empty-file"],
)
def test_wiring_refuses_bad_compose(tmp_path: Path, document: object) -> None:
    compose = tmp_path / "compose.yml"
    compose.write_text(yaml.safe_dump(document))

    with pytest.raises(SetupError, match="has no usable langgraph-api environment"):
        read_wiring(compose)


def test_graphs_come_from_manifest() -> None:
    assert read_graphs(MANIFEST) == list(json.loads(MANIFEST.read_text())["graphs"])


def test_manifest_without_graphs_refused(tmp_path: Path) -> None:
    manifest = tmp_path / "langgraph.json"
    manifest.write_text('{"graphs": {}}')

    with pytest.raises(SetupError, match="lists no graphs"):
        read_graphs(manifest)


@pytest.mark.parametrize(
    "flag,value,message",
    [
        ("--runs-per-level", "0", "must be at least 1, got 0"),
        ("--users", "10,0", "every users level must be a positive integer"),
        ("--users", "ten", "bad users levels 'ten'"),
    ],
    ids=["zero-runs", "zero-users", "not-a-number"],
)
def test_bad_counts_refused(capsys, flag: str, value: str, message: str) -> None:
    """A level with no runs would crash the summary after the server did the work."""
    with pytest.raises(SystemExit):
        parse_args([flag, value])

    assert message in capsys.readouterr().err


def _passing(users: int) -> Any:
    record = RunRecord("g", users, 1.0, 5.0, False, None, RunOutcome(True, 3, 0, 0.0, 0, 0))
    return summarise("g", users, [record], wall_seconds=60.0)


def test_exit_zero_when_all_hold() -> None:
    assert exit_code([_passing(10), _passing(50)]) == 0


def test_exit_two_without_throughput_level() -> None:
    """Passing at 10 users says nothing about 50, so the run has not shown the SLO."""
    assert exit_code([_passing(10)]) == 2


def test_exit_two_on_any_failure() -> None:
    assert exit_code([replace(_passing(10), ttft_p95=3.0), _passing(50)]) == 2


async def test_unknown_graph_refused(log_records) -> None:
    code = await run.main(["--graph", "no_such_agent"])

    assert code == 1
    assert logged(log_records) == [
        ("ERROR", "refusing to start: not in langgraph.json: ['no_such_agent']", False)
    ]


async def test_startup_reads_off_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file read on the event loop stalls it. The runner reads its files in a thread."""
    monkeypatch.setattr(run, "read_graphs", off_loop(["research_agent"]))
    monkeypatch.setattr(run, "read_wiring", off_loop(_wiring()))

    assert await run.main(["--graph", "no_such_agent"]) == 1
