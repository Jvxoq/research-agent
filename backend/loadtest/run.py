"""Load test every agent the LangGraph Server serves, and check the speed SLOs.

Start the server under test first, wired to the fakes::

    uv run langgraph up -d loadtest/docker-compose.yml --wait

Then run::

    uv run python -m loadtest.run

For each graph in ``langgraph.json`` the runner starts the fake model and fake
Tavily for that graph's scenario, then runs each users level in turn. A level
of ``N`` users is ``N`` clients, each with its own signed JWT, each starting
its next run as soon as its last one ends, until the level's runs are done.
Every run goes through ``/threads`` and ``/runs/stream``, the same path a real
caller takes.

The report prints every speed and reliability SLO per level and writes the
numbers to ``--out`` as JSON.

Exit codes: 0 when every SLO holds at every level, the 50-user throughput
level included. 2 when one fails, was not measured, or that level did not run.
1 when the setup is broken.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
import jwt
import yaml
from langgraph_sdk import get_client
from langgraph_sdk.client import LangGraphClient
from langgraph_sdk.schema import Input
from loguru import logger
from pydantic import BaseModel, ConfigDict, ValidationError

from core.tools.text import cap
from loadtest.fakes import HEALTH_PATH
from loadtest.scenario import MissingScenario, Scenario, load_scenario
from loadtest.stats import (
    THROUGHPUT_USERS,
    LevelReport,
    RunRecord,
    check,
    summarise,
    throughput_holds,
)

BACKEND = Path(__file__).resolve().parent.parent
MANIFEST = BACKEND / "langgraph.json"
COMPOSE = Path(__file__).resolve().parent / "docker-compose.yml"
API_SERVICE = "langgraph-api"
CANCEL_TIMEOUT_SECONDS = 10.0


class SetupError(RuntimeError):
    """Raised when the load test cannot start: a bad file or a missing server."""


class _ApiEnvironment(BaseModel):
    """The compose values the runner shares with the server under test."""

    model_config = ConfigDict(extra="ignore")

    BASE_URL: str
    THINK_BASE_URL: str
    LOOK_UP_BASE_URL: str
    TAVILY_BASE_URL: str
    JWT_SECRET: str
    JWT_ALGORITHM: str
    JWT_ISSUER: str
    JWT_AUDIENCE: str


class _ApiService(BaseModel):
    model_config = ConfigDict(extra="ignore")

    environment: _ApiEnvironment


class _Compose(BaseModel):
    model_config = ConfigDict(extra="ignore")

    services: dict[str, object]


@dataclass(frozen=True)
class Wiring:
    """How the runner reaches the fakes and signs its users' tokens."""

    fake_port: int
    jwt_secret: str
    jwt_algorithm: str
    jwt_issuer: str
    jwt_audience: str


def read_graphs(manifest: Path) -> list[str]:
    """Return the graph ids ``langgraph.json`` serves, in file order.

    Raises:
        SetupError: When the manifest has no ``graphs`` mapping.
    """
    graphs = json.loads(manifest.read_text()).get("graphs")
    if not isinstance(graphs, dict) or not graphs:
        raise SetupError(f"{manifest} lists no graphs")
    return [str(name) for name in graphs]


def read_wiring(compose: Path) -> Wiring:
    """Read the fake port and the JWT settings from the compose override.

    Every model URL and the Tavily URL must share one port, because the runner
    starts one fake process per graph and it serves them all.

    Raises:
        SetupError: When a value is missing, or the URLs disagree on the port.
    """
    try:
        services = _Compose.model_validate(yaml.safe_load(compose.read_text())).services
        environment = _ApiService.model_validate(services[API_SERVICE]).environment
    except (KeyError, ValidationError) as error:
        raise SetupError(f"{compose} has no usable {API_SERVICE} environment: {error}") from error
    urls = (
        environment.BASE_URL,
        environment.THINK_BASE_URL,
        environment.LOOK_UP_BASE_URL,
        environment.TAVILY_BASE_URL,
    )
    port = urlsplit(urls[0]).port
    if port is None or any(urlsplit(url).port != port for url in urls):
        raise SetupError(f"{compose}: every model and Tavily URL must name the same explicit port")
    return Wiring(
        fake_port=port,
        jwt_secret=environment.JWT_SECRET,
        jwt_algorithm=environment.JWT_ALGORITHM,
        jwt_issuer=environment.JWT_ISSUER,
        jwt_audience=environment.JWT_AUDIENCE,
    )


def user_token(wiring: Wiring, user: int, lifetime: timedelta) -> str:
    """Sign a token for one simulated user, as the web application would."""
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "sub": f"loadtest-user-{user}",
            "iss": wiring.jwt_issuer,
            "aud": wiring.jwt_audience,
            "iat": now,
            "exp": now + lifetime,
        },
        wiring.jwt_secret,
        algorithm=wiring.jwt_algorithm,
    )


def parse_levels(text: str) -> list[int]:
    """Parse ``"10,50,100"`` into users levels.

    Raises:
        argparse.ArgumentTypeError: When a level is not a positive integer.
    """
    try:
        levels = [int(part) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"bad users levels {text!r}") from error
    if not levels or any(level < 1 for level in levels):
        raise argparse.ArgumentTypeError("every users level must be a positive integer")
    return levels


def _positive_int(text: str) -> int:
    """Parse a count that must be at least 1."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="http://127.0.0.1:8123", help="Server under test.")
    parser.add_argument("--users", type=parse_levels, default=[10, THROUGHPUT_USERS, 100])
    parser.add_argument(
        "--runs-per-level",
        type=_positive_int,
        default=200,
        help="Runs per users level. Fewer than 100 cannot show a p99.",
    )
    parser.add_argument(
        "--run-timeout-seconds",
        type=float,
        default=300.0,
        help="Whole-run ceiling. A run past it counts as timed out and is cancelled.",
    )
    parser.add_argument("--graph", action="append", help="Test only this graph. Repeatable.")
    parser.add_argument("--fake-host", default="127.0.0.1")
    parser.add_argument("--model-delay-seconds", type=float, default=1.0)
    parser.add_argument("--search-delay-seconds", type=float, default=0.3)
    parser.add_argument("--extract-delay-seconds", type=float, default=0.5)
    parser.add_argument("--jitter", type=float, default=0.25)
    parser.add_argument("--max-error-chars", type=int, default=500)
    parser.add_argument("--out", type=Path, default=None, help="JSON report path.")
    return parser.parse_args(argv)


async def one_run(
    client: LangGraphClient,
    graph_id: str,
    users: int,
    payload: Input,
    scenario: Scenario,
    args: argparse.Namespace,
    clock: Callable[[], float] = time.monotonic,
) -> RunRecord:
    """Run one request through ``/threads`` and ``/runs/stream``.

    Time to first update counts from the thread request to the first
    ``updates`` event, which is the first node finishing. The ``values`` event
    the stream opens with only echoes the input, so it does not count.

    Args:
        client: This user's client.
        graph_id: The assistant to run.
        users: The level's users count, recorded on the result.
        payload: The run input.
        scenario: Reads the final state.
        args: Timeout and error cap.
        clock: Seconds from any fixed point. Injected so a test reads exact times.

    Returns:
        The run's record, whether it answered, failed or timed out.
    """
    started = clock()
    ttft: float | None = None
    final: object = None
    thread_id: str | None = None
    run_id: str | None = None
    error: str | None = None
    timed_out = False
    try:
        async with asyncio.timeout(args.run_timeout_seconds):
            thread = await client.threads.create()
            thread_id = thread["thread_id"]
            async for part in client.runs.stream(
                thread_id,
                graph_id,
                input=payload,
                stream_mode=["updates", "values"],
            ):
                if part.event == "metadata" and isinstance(part.data, dict):
                    run_id = part.data.get("run_id")
                elif part.event == "updates" and ttft is None:
                    ttft = clock() - started
                elif part.event == "values":
                    final = part.data
                elif part.event == "error":
                    error = cap(json.dumps(part.data, default=str), args.max_error_chars)
    except TimeoutError:
        timed_out = True
        await cancel_run(client, thread_id, run_id)
    except Exception as failure:  # noqa: BLE001 - one failed run fills its own slot
        logger.exception(f"{graph_id} run failed")
        error = cap(f"{type(failure).__name__}: {failure}", args.max_error_chars)
    latency = clock() - started

    outcome = None
    if not timed_out and error is None:
        if isinstance(final, dict):
            try:
                outcome = scenario.outcome(final)
            except ValueError as bad_state:
                error = cap(f"final state has the wrong shape: {bad_state}", args.max_error_chars)
        else:
            error = "the stream ended with no final state"
    return RunRecord(
        graph_id=graph_id,
        users=users,
        ttft_seconds=ttft,
        latency_seconds=latency,
        timed_out=timed_out,
        error=error,
        outcome=outcome,
    )


async def cancel_run(client: LangGraphClient, thread_id: str | None, run_id: str | None) -> None:
    """Cancel a timed-out run, so it stops holding a server worker."""
    if thread_id is None or run_id is None:
        return
    try:
        async with asyncio.timeout(CANCEL_TIMEOUT_SECONDS):
            await client.runs.cancel(thread_id, run_id)
    except Exception:  # noqa: BLE001 - the timeout is already recorded
        logger.exception(f"could not cancel run {run_id}")


def token_lifetime(run_timeout_seconds: float, runs_per_level: int) -> timedelta:
    """Return how long a user's token must stay valid for one level.

    The longest a level can last is one user running every run, each one
    timing out and then waiting the full cancel timeout. A token signed when
    the level starts outlives that, so no run fails on an expired token.
    """
    return timedelta(seconds=(run_timeout_seconds + CANCEL_TIMEOUT_SECONDS) * runs_per_level)


async def run_level(
    graph_id: str,
    users: int,
    scenario: Scenario,
    wiring: Wiring,
    args: argparse.Namespace,
) -> LevelReport:
    """Run one users level for one graph and summarise it.

    ``users`` workers share one counter, so at most ``users`` runs are in
    flight and the level stops at ``--runs-per-level`` runs.
    """
    runs: list[RunRecord] = []
    next_index = 0
    lifetime = token_lifetime(args.run_timeout_seconds, args.runs_per_level)

    async def worker(user: int) -> None:
        nonlocal next_index
        client = get_client(
            url=args.url,
            # None keeps the SDK from sending a LangSmith key from this
            # machine's environment to the server.
            api_key=None,
            headers={"Authorization": f"Bearer {user_token(wiring, user, lifetime)}"},
            timeout=args.run_timeout_seconds,
        )
        try:
            while next_index < args.runs_per_level:
                index = next_index
                next_index += 1
                payload = scenario.inputs[index % len(scenario.inputs)]
                runs.append(await one_run(client, graph_id, users, payload, scenario, args))
        finally:
            await client.aclose()

    logger.info(f"{graph_id}: {users} users, {args.runs_per_level} runs")
    started = time.monotonic()
    await asyncio.gather(*(worker(user) for user in range(users)))
    report = summarise(graph_id, users, runs, time.monotonic() - started)
    failed = [run.error for run in runs if run.error is not None]
    if failed:
        logger.warning(f"{graph_id} at {users} users: {len(failed)} failed, first: {failed[0]}")
    return report


async def wait_until_up(url: str, seconds: float) -> None:
    """Poll ``url`` until it answers 200.

    Raises:
        SetupError: When it never does within ``seconds``.
    """
    deadline = time.monotonic() + seconds
    timeout = aiohttp.ClientTimeout(total=2)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return
            except (aiohttp.ClientError, TimeoutError):
                pass
            await asyncio.sleep(0.5)
    raise SetupError(f"{url} did not answer within {seconds:.0f}s")


async def start_fakes(graph_id: str, wiring: Wiring, args: argparse.Namespace) -> asyncio.subprocess.Process:
    """Start the fakes for one graph in their own process and wait for them.

    A separate process keeps the fakes' work off the load client's event loop,
    so a busy fake never shows up as server latency.
    """
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "loadtest.fakes",
        "--graph",
        graph_id,
        "--host",
        args.fake_host,
        "--port",
        str(wiring.fake_port),
        "--model-delay-seconds",
        str(args.model_delay_seconds),
        "--search-delay-seconds",
        str(args.search_delay_seconds),
        "--extract-delay-seconds",
        str(args.extract_delay_seconds),
        "--jitter",
        str(args.jitter),
        cwd=BACKEND,
    )
    try:
        await wait_until_up(f"http://{args.fake_host}:{wiring.fake_port}{HEALTH_PATH}", 30)
    except SetupError:
        await stop_fakes(process)
        raise
    return process


async def stop_fakes(process: asyncio.subprocess.Process) -> None:
    """Stop a fakes process, killing it when it does not stop in time."""
    if process.returncode is not None:
        return
    process.terminate()
    try:
        async with asyncio.timeout(10):
            await process.wait()
    except TimeoutError:
        process.kill()
        await process.wait()


def log_report(report: LevelReport) -> None:
    """Log one level's SLO table."""
    logger.info(
        f"{report.graph_id} at {report.users} users: {report.runs} runs in "
        f"{report.wall_seconds:.1f}s, {report.throughput_per_minute:.1f} runs/min"
    )
    for result in check(report):
        value = "not measured" if result.value is None else f"{result.value:.3f}"
        verdict = {True: "pass", False: "FAIL", None: "no data"}[result.passed]
        logger.info(f"  {result.key:<22} {value:>14}  target {result.target:<26} {verdict}")


async def main(argv: list[str] | None = None) -> int:
    """Load test every graph, log the SLO tables and write the JSON report."""
    args = parse_args(argv)
    try:
        graphs = await asyncio.to_thread(read_graphs, MANIFEST)
        wiring = await asyncio.to_thread(read_wiring, COMPOSE)
        scenarios = {graph: load_scenario(graph) for graph in graphs}
        if args.graph:
            unknown = sorted(set(args.graph) - set(graphs))
            if unknown:
                raise SetupError(f"not in langgraph.json: {unknown}")
            graphs = [graph for graph in graphs if graph in args.graph]
        await wait_until_up(f"{args.url}/ok", 10)
    except (SetupError, MissingScenario) as problem:
        logger.error(f"refusing to start: {problem}")
        return 1

    reports: list[LevelReport] = []
    for graph in graphs:
        try:
            fakes = await start_fakes(graph, wiring, args)
        except SetupError as problem:
            logger.error(f"refusing to start {graph}: {problem}")
            return 1
        try:
            for users in args.users:
                report = await run_level(graph, users, scenarios[graph], wiring, args)
                log_report(report)
                reports.append(report)
        finally:
            await stop_fakes(fakes)

    holds = throughput_holds(reports)
    logger.info(
        f"throughput at {THROUGHPUT_USERS} users: "
        + {True: "every speed SLO holds", False: "FAIL", None: "level not run"}[holds]
    )
    out = args.out or BACKEND / "results" / "loadtest" / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    document = {
        "levels": [
            {**asdict(report), "checks": [asdict(c) for c in check(report)]} for report in reports
        ],
        "throughput_holds": holds,
        "settings": {
            "runs_per_level": args.runs_per_level,
            "run_timeout_seconds": args.run_timeout_seconds,
            "model_delay_seconds": args.model_delay_seconds,
            "search_delay_seconds": args.search_delay_seconds,
            "extract_delay_seconds": args.extract_delay_seconds,
            "jitter": args.jitter,
        },
    }
    await asyncio.to_thread(_write_json, out, document)
    logger.info(f"report in {out}")
    return exit_code(reports)


def exit_code(reports: Sequence[LevelReport]) -> int:
    """Return 0 when every SLO holds at every level, the throughput level included, else 2."""
    every_check = [c.passed for report in reports for c in check(report)]
    holds = throughput_holds(reports) is True
    return 0 if holds and all(passed is True for passed in every_check) else 2


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
