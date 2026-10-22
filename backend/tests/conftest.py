"""Shared fixtures and the one model fake every suite uses.

Every test runs against a known environment. The developer's own shell, and
any `.env` file, must not change a result.

Every fake lives here rather than beside one agent's tests. Two fakes for
one thing drift, and a test that passes against a stale fake proves nothing.
"""

from __future__ import annotations

import asyncio
import os
import re
import socket
import uuid
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn

import pytest
from langchain_core.callbacks import AsyncCallbackManager
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import ToolException
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_tavily import TavilyExtract, TavilySearch
from langsmith.schemas import TracerSession
from langsmith.types.run_get_url_response import RunGetURLResponse
from loguru import logger

from agents.research_agent.schemas import Decision, SearchAction
from core.config import Settings, get_settings
from core.llm import reset_rate_limiter
from core.retry import COUNTERS_KEY, BackoffPolicy, Retrier
from core.tools import web_look_up, web_search
from core.tools.web_look_up import LookUpBatch
from loadtest.scenario import ModelRequest, Scenario
from tests.structure.conftest import configuration_classes

if TYPE_CHECKING:
    from loguru import Record

# Every field any agent can configure. A variable left in the developer's shell
# would otherwise win over a test's own configuration, silently.
CONFIGURABLE_NAMES = {
    name
    for _, configuration in configuration_classes()
    for name in configuration.model_fields
}


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any test that opens a real connection or resolves a real name.

    Each attempt raises, and the test also fails at teardown, because code
    under test with a broad ``except`` can swallow the raise.
    """
    attempts: list[str] = []

    def refuse(*args: object, **kwargs: object) -> NoReturn:
        attempts.append(repr(args))
        raise AssertionError(f"a test reached the real network: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield
    assert attempts == [], f"a test reached the real network: {attempts}"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Clear agent settings from the environment and drop cached settings."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in list(os.environ):
        if name.lower() in CONFIGURABLE_NAMES:
            monkeypatch.delenv(name, raising=False)
    for name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "JWT_SECRET",
        "JWT_ALGORITHM",
        "JWT_ISSUER",
        "JWT_AUDIENCE",
        "ENVIRONMENT",
        "MODEL_PRICES_USD_PER_MTOK",
        "TAVILY_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-used")
    get_settings.cache_clear()
    reset_rate_limiter()
    yield
    get_settings.cache_clear()
    reset_rate_limiter()


@dataclass(frozen=True)
class Raises:
    """A scripted structured reply that makes the call itself raise.

    Any other exception in the script is a parsing failure, which the real
    integration hands back as a value when ``include_raw`` is set.
    """

    error: BaseException


@dataclass
class FakeStructuredModel:
    """Stands in for ``model.with_structured_output(..., include_raw=True)``.

    Every caller asks for the raw output, so the fake returns only that shape.
    """

    model: FakeChatModel

    async def ainvoke(self, prompt: str, config: Any = None) -> Any:
        """Record the call and return the next scripted structured reply."""
        self.model.prompts.append(prompt)
        index = min(self.model.structured_calls, len(self.model.structured_replies) - 1)
        self.model.structured_calls += 1
        reply = self.model.structured_replies[index]

        if isinstance(reply, Raises):
            raise reply.error
        await self.model.report_usage(config)
        if isinstance(reply, BaseException):
            return {"raw": AIMessage(content=""), "parsed": None, "parsing_error": reply}
        return {"raw": AIMessage(content=""), "parsed": reply, "parsing_error": None}


@dataclass
class FakeChatModel:
    """A chat model that replays scripted responses, so no test reaches a provider.

    Args:
        replies: Raw responses returned in order by the plain path. The last
            one repeats once the script runs out, so a test never has to count
            retries twice.
        structured_replies: What the structured path returns, in order, with
            the same repeat rule. An exception is a parsing failure, and a
            :class:`Raises` makes the call itself raise.
        usage: Input and output tokens each answered call reports to the
            callbacks in its config, the way a real provider response does.
            None reports nothing.
        model_name: The model the reported usage is priced as.
    """

    replies: list[str | list[dict[str, str]]] = field(
        default_factory=lambda: ["final answer"]
    )
    structured_replies: list[Any] = field(default_factory=lambda: [None])
    usage: tuple[int, int] | None = None
    model_name: str = "zai-org/GLM-5.3-Flash"
    prompts: list[str] = field(default_factory=list)
    # The schema each structured call sent to the provider, in order.
    schemas: list[Any] = field(default_factory=list)
    plain_calls: int = 0
    structured_calls: int = 0

    async def report_usage(self, config: Any) -> None:
        """Start and end one model run on the config's callbacks, with usage.

        The run carries the config's metadata, so a handler sees the graph
        step a call belongs to, as it does with a real chat model.
        """
        if self.usage is None or not config:
            return
        manager = AsyncCallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            inheritable_metadata=config.get("metadata"),
        )
        runs = await manager.on_chat_model_start({}, [[]], name="chat_model")
        result = LLMResult(
            generations=[[]],
            llm_output={
                "model_name": self.model_name,
                "token_usage": {
                    "prompt_tokens": self.usage[0],
                    "completion_tokens": self.usage[1],
                },
            },
        )
        for run in runs:
            await run.on_llm_end(result)

    def with_structured_output(
        self, schema: Any, method: str = "json_schema", include_raw: bool = True
    ) -> FakeStructuredModel:
        """Record the schema and return the structured path."""
        self.schemas.append(schema)
        return FakeStructuredModel(model=self)

    async def ainvoke(self, prompt: str, config: Any = None) -> AIMessage:
        """Return the next scripted reply."""
        index = min(self.plain_calls, len(self.replies) - 1)
        self.plain_calls += 1
        self.prompts.append(prompt)
        await self.report_usage(config)
        return AIMessage(content=self.replies[index])


@pytest.fixture
def set_env(monkeypatch):
    """Return a helper that sets environment variables and drops cached settings."""

    def apply(**values: str) -> None:
        for name, value in values.items():
            monkeypatch.setenv(name, value)
        get_settings.cache_clear()

    return apply


@pytest.fixture
def chat_model() -> type[FakeChatModel]:
    """Return the fake chat model class, so a test can script its replies."""
    return FakeChatModel


@dataclass
class FakeSearch:
    """Stands in for ``core.tools.web_search.search``, so no test reaches Tavily.

    Args:
        batches: Results and errors returned in order, one pair per call. The
            last pair repeats once the script runs out.
        error: Raised by every call instead of returning, when set.
    """

    batches: list[tuple[list[str], list[str | None]]] = field(
        default_factory=lambda: [(["a result"], [None])]
    )
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, **kwargs: Any) -> tuple[list[str], list[str | None]]:
        """Record the call and return a copy of the next scripted batch."""
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        results, errors = self.batches[min(len(self.calls), len(self.batches)) - 1]
        # The node caps results in place, so a shared list would leak between calls.
        return list(results), list(errors)


@pytest.fixture
def no_backoff_wait(monkeypatch) -> None:
    """Replace the retrier's sleep, so no test waits on the wall clock."""

    async def skip(seconds: float) -> None:
        """Return at once instead of sleeping."""

    monkeypatch.setattr("core.retry.asyncio.sleep", skip)


@pytest.fixture
def retrier(no_backoff_wait) -> Retrier:
    """A retrier whose waits take 0 seconds and whose counters a test can read.

    The waits are real numbers, so the cumulative wait ceiling still ends a
    persistent rate limit. Only the sleep is skipped.
    """
    return Retrier(
        policy=BackoffPolicy(
            base_seconds=1.0,
            max_seconds=1.0,
            jitter=0.0,
            max_wait_seconds=10.0,
            max_transient_retries=2,
        )
    )


def decision(
    report: str, answer: str | None = None, queries: list[str] | None = None
) -> Decision:
    """Build one structured decision.

    Args:
        report: The rewritten report.
        answer: The final answer, when the model is answering.
        queries: A search batch, when the model is acting.

    Returns:
        The decision the structured path hands back.
    """
    action = SearchAction(queries=queries) if queries is not None else None
    return Decision(report=report, answer=answer, action=action)


TRACE_PROJECT_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")


def trace_page(project_id: object, trace_id: object, run_id: object) -> str:
    """Return the page LangSmith names for one run of one trace."""
    return f"https://smith.test/p/{project_id}/trace/{trace_id}/run/{run_id}"


@dataclass
class FakeRunsResource:
    """Stands in for ``Client.runs``, the async API that names a run's page."""

    error: Exception | None = None
    lookups: list[dict[str, object]] = field(default_factory=list)

    async def get_url(
        self, run_id: str, *, project_id: str, trace_id: str, timeout: float
    ) -> RunGetURLResponse:
        """Record the lookup, then answer with the page or raise the set error."""
        self.lookups.append({"run_id": run_id, "trace_id": trace_id, "timeout": timeout})
        if self.error is not None:
            raise self.error
        return RunGetURLResponse(url=trace_page(project_id, trace_id, run_id))


@dataclass
class FakeLangSmithClient:
    """A LangSmith client that records how it was built and every run it is sent."""

    api_url: str | None = None
    api_key: str | None = None
    created: list[dict[str, Any]] = field(default_factory=list)
    builds: list[dict[str, Any]] = field(default_factory=list)
    project_reads: list[str] = field(default_factory=list)
    project_error: Exception | None = None
    runs: FakeRunsResource = field(default_factory=FakeRunsResource)

    def build(self, **settings: Any) -> FakeLangSmithClient:
        """Record one construction, the way the runner builds its own client."""
        self.builds.append(settings)
        return self

    def create_run(self, **run: Any) -> None:
        """Record a run the tracer posts, the way LangSmith would store it."""
        self.created.append(run)

    def update_run(self, run_id: Any, **run: Any) -> None:
        """Accept the tracer's end-of-run patch. Its metadata repeats the post."""

    def read_project(self, *, project_name: str) -> TracerSession:
        """Answer a blocking project read, which must run off the event loop."""
        off_loop(None)()
        self.project_reads.append(project_name)
        if self.project_error is not None:
            raise self.project_error
        return TracerSession(
            id=TRACE_PROJECT_ID, tenant_id=uuid.uuid4(), reference_dataset_id=None
        )


@pytest.fixture
def langsmith_client(monkeypatch):
    """Replace the LangSmith client, whose constructor calls the network."""
    monkeypatch.setattr("core.tracing.Client", FakeLangSmithClient)


@pytest.fixture
def langsmith_fake(monkeypatch) -> FakeLangSmithClient:
    """Replace every LangSmith client the tracer and the eval runner build."""
    client = FakeLangSmithClient()
    monkeypatch.setattr("core.tracing.Client", lambda **kwargs: client)
    monkeypatch.setattr("evals.research_agent.run.Client", client.build)
    return client


@dataclass(frozen=True)
class TracedRun:
    """One run as LangSmith files it: its name and the thread keys it carries."""

    name: str
    question_id: str | None
    thread_id: str | None


@pytest.fixture
def traced_runs(langsmith_fake, set_env) -> Callable[[], list[TracedRun]]:
    """Turn tracing on the way the operator does, and record every posted run.

    Returns:
        A callable that waits for the tracer and returns the runs posted so far.
    """
    set_env(LANGSMITH_TRACING="true", LANGSMITH_API_KEY="ls-test-key-not-used")

    def read() -> list[TracedRun]:
        wait_for_all_tracers()
        found = []
        for run in langsmith_fake.created:
            metadata = (run.get("extra") or {}).get("metadata") or {}
            found.append(
                TracedRun(
                    name=run["name"],
                    question_id=metadata.get("question_id"),
                    thread_id=metadata.get("thread_id"),
                )
            )
        return found

    return read


PUBLIC_ADDRESS = "93.184.216.34"


@dataclass
class FakeResolver:
    """Stands in for ``socket.getaddrinfo``, so no test touches DNS.

    Args:
        addresses: Each hostname mapped to the address, or addresses, it
            resolves to.
        default: The address any unmapped hostname resolves to. None makes an
            unmapped hostname fail to resolve.
    """

    addresses: dict[str, str | list[str]] = field(default_factory=dict)
    default: str | None = None

    def __call__(self, host: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Answer the way getaddrinfo does, from the scripted addresses."""
        mapped = self.addresses.get(host, self.default)
        if mapped is None:
            raise socket.gaierror(f"unmapped host {host!r}")
        found = [mapped] if isinstance(mapped, str) else mapped
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0)) for address in found]


@pytest.fixture
def resolves(monkeypatch):
    """Return a helper that pins hostnames to addresses, with no DNS."""

    def install(
        addresses: dict[str, str | list[str]] | None = None, default: str | None = None
    ) -> None:
        monkeypatch.setattr(
            socket, "getaddrinfo", FakeResolver(addresses or {}, default)
        )

    return install


async def _as_tavily_answers(reply: Awaitable[object]) -> object:
    """Answer the way ``langchain_tavily`` does around its HTTP call.

    A failure is returned as ``{"error": exception}``, never raised, and an
    answer with no results raises ``ToolException``.
    """
    try:
        payload = await reply
    except Exception as error:  # noqa: BLE001 - the integration returns it
        return {"error": error}
    if isinstance(payload, dict) and not payload.get("results"):
        raise ToolException("No results found.")
    return payload


@dataclass
class FakeExtractor:
    """Stands in for ``TavilyExtract``, answering from a caller-supplied handler."""

    handler: Callable[[str], Awaitable[object]]

    async def ainvoke(self, payload: dict[str, list[str]]) -> object:
        """Answer for the one URL the tool sends per call."""
        return await _as_tavily_answers(self.handler(payload["urls"][0]))


@dataclass
class FakeSearchTool:
    """Stands in for ``TavilySearch``, answering from a caller-supplied handler."""

    handler: Callable[[str], Awaitable[object]]

    async def ainvoke(self, payload: dict[str, str]) -> object:
        """Answer for the one query the tool sends per call."""
        return await _as_tavily_answers(self.handler(payload["query"]))


@dataclass
class FakeLookUp:
    """Stands in for ``core.tools.web_look_up.look_up``, so no test fetches a page.

    Args:
        results: The summaries every call returns.
        errors: The error slots every call returns.
        malformed_summaries: The failed checks every call reports.
    """

    results: list[str]
    errors: list[str | None]
    malformed_summaries: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __call__(self, **kwargs: Any) -> LookUpBatch:
        """Record the call and return a fresh copy of the scripted batch."""
        self.calls.append(kwargs)
        return LookUpBatch(
            results=list(self.results),
            errors=list(self.errors),
            malformed_summaries=self.malformed_summaries,
        )


class FakeGraph:
    """A compiled graph whose answer for each question is scripted."""

    def __init__(self, answers: dict[str, str], raises: set[str] | None = None) -> None:
        self.answers = answers
        self.raises = raises or set()
        self.configs: list[dict[str, Any]] = []

    async def ainvoke(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        """Return a final state, or raise for a question scripted to fail."""
        question = state["question"]
        self.configs.append(config or {})
        if question in self.raises:
            raise RuntimeError("provider is down")
        return {
            "answer": self.answers[question],
            "report": "a report",
            "round": 2,
            "stop_reason": "answer",
            "usage": {"input_tokens": 1000, "output_tokens": 200},
        }


class CountingGraph(FakeGraph):
    """A graph that records a wait in each retry family, the way the agent's retrier does."""

    async def ainvoke(self, state: dict[str, Any], config: Any = None) -> dict[str, Any]:
        """Charge the run's counters, then answer normally."""
        counters = config["configurable"][COUNTERS_KEY]
        counters.rate_limit_hits += 2
        counters.rate_limit_wait_seconds += 1.5
        counters.transient_retries += 1
        counters.transient_wait_seconds += 0.5
        return await super().ainvoke(state, config)


class FakeUser:
    """The authenticated user an auth handler reads its identity from."""

    identity = "user-42"


class FakeAuthContext:
    """The auth context LangGraph Server hands a resource handler."""

    user = FakeUser()


def _grader_section(prompt: str, label: str) -> str:
    """Return one bracketed section of the grader prompt."""
    match = re.search(rf"\[{label}\]: (.*?)\n\n", prompt, re.DOTALL)
    return match.group(1) if match else ""


class ContainmentJudge:
    """A judge that says yes when the gold answer appears in the response.

    A stand-in for a real judge that is right on obvious cases, which is all
    the known-answer set contains.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, messages, **kwargs):
        """Return a judge reply for the question in the prompt."""
        prompt = messages[0].content
        self.prompts.append(prompt)
        gold = _grader_section(prompt, "correct_answer").strip()
        response = _grader_section(prompt, "response")
        correct = bool(gold) and gold.lower() in response.lower()
        return AIMessage(
            content=f"extracted_final_answer: {gold if correct else 'None'}\n"
            "reasoning: compared the two answers\n"
            f"correct: {'yes' if correct else 'no'}\n"
            "confidence: 90"
        )


def traced_judge() -> RunnableLambda[Any, AIMessage]:
    """Return the containment judge as a runnable, so LangSmith traces its calls."""
    return RunnableLambda(ContainmentJudge().ainvoke, name="judge")


class AlwaysWrongJudge:
    """A judge that calls everything incorrect, including the known answers."""

    async def ainvoke(self, messages, **kwargs):
        """Return a reply that always says no."""
        return AIMessage(content="extracted_final_answer: None\nreasoning: no\ncorrect: no")


class BlockJudge:
    """A judge that answers in content blocks, the way a reasoning model does."""

    async def ainvoke(self, messages, **kwargs):
        """Return a yes split across a reasoning block and a text block."""
        return AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "correct: no, still thinking"},
                {"type": "text", "text": "extracted_final_answer: Paris\ncorrect: yes"},
            ]
        )


class BrokenJudge:
    """A judge whose provider is down, so every call raises."""

    async def ainvoke(self, messages, **kwargs):
        """Raise the way a failing provider does."""
        raise RuntimeError("provider is down")


class GoldenJudge(ContainmentJudge):
    """A judge that passes the known-answer check and scripts its golden checklist.

    The runner checks the grader first with plain calls, then judges each golden
    report through the structured path, so one fake has to serve both.
    """

    def __init__(self, structured: FakeChatModel) -> None:
        super().__init__()
        self.structured = structured

    def with_structured_output(
        self, schema: Any, method: str = "json_schema", include_raw: bool = True
    ) -> RunnableLambda[str, Any]:
        """Return the scripted structured path as a runnable, so LangSmith traces it."""
        scripted = self.structured.with_structured_output(schema, method, include_raw)
        return RunnableLambda(scripted.ainvoke, name="golden_judge")


@dataclass
class PeakGauge:
    """Hold every call open until the test opens the gate, and count them."""

    in_flight: int = 0
    peak: int = 0
    gate: asyncio.Event = field(default_factory=asyncio.Event)

    async def hold(self) -> None:
        """Count this call in, wait for the gate, then count it out."""
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await self.gate.wait()
        finally:
            self.in_flight -= 1


async def settle(rounds: int = 100) -> None:
    """Run every ready task until it blocks. Fakes do no I/O, so this ends."""
    for _ in range(rounds):
        await asyncio.sleep(0)


def off_loop(answer: object) -> Callable[..., object]:
    """Return a fake sync function that fails on the event loop thread."""

    def fake(*args: object, **kwargs: object) -> object:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return answer
        raise AssertionError("a blocking call ran on the event loop thread")

    return fake


async def hang(*args: object, **kwargs: object) -> NoReturn:
    """Never answer, like a dependency that stopped responding."""
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


@pytest.fixture
def log_records() -> Iterator[list[Record]]:
    """Collect every loguru record the test produces."""
    records: list[Record] = []
    sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
    yield records
    logger.remove(sink)


def logged(records: list[Record], min_level: int = 30) -> list[tuple[str, str, bool]]:
    """Return level name, message and whether an exception is attached."""
    return [
        (record["level"].name, record["message"], record["exception"] is not None)
        for record in records
        if record["level"].no >= min_level
    ]


@pytest.fixture
def tavily_endpoints(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Build the real Tavily client each tool asks for, and record its endpoint.

    The real client is the last object before the network, so the endpoint it
    holds is where its request would go. None means Tavily's own default. A
    fake answers in its place, so nothing leaves the process.
    """
    endpoints: list[str | None] = []

    async def answer_search(query: str) -> dict[str, object]:
        return {"results": [{"title": "t", "url": "https://example.com/a", "content": "c"}]}

    async def answer_extract(url: str) -> dict[str, object]:
        return {"results": [{"url": url, "raw_content": "page text"}]}

    def search_tool(**kwargs: Any) -> FakeSearchTool:
        endpoints.append(TavilySearch(**kwargs).api_wrapper.api_base_url)
        return FakeSearchTool(answer_search)

    def extract_tool(**kwargs: Any) -> FakeExtractor:
        endpoints.append(TavilyExtract(**kwargs).apiwrapper.api_base_url)
        return FakeExtractor(answer_extract)

    monkeypatch.setattr(web_search, "TavilySearch", search_tool)
    monkeypatch.setattr(web_look_up, "TavilyExtract", extract_tool)
    return endpoints


@dataclass
class ScenarioStructured:
    """The structured path of :class:`ScenarioChatModel`, for one schema."""

    model: ScenarioChatModel
    schema: Any

    async def ainvoke(self, prompt: str, config: Any = None) -> dict[str, Any]:
        """Answer from the scenario and parse it the way the provider's schema would."""
        content = self.model.answer(self.schema.__name__, prompt)
        return {
            "raw": AIMessage(content=content),
            "parsed": self.schema.model_validate_json(content),
            "parsing_error": None,
        }


@dataclass
class ScenarioChatModel:
    """A chat model that answers the way the load test's fake provider does.

    It hands the real prompt and schema name to a load scenario's ``reply``,
    so a test proves the scenario can drive the real graph.
    """

    scenario: Scenario
    schemas: list[str | None] = field(default_factory=list)

    def answer(self, schema_name: str | None, prompt: str) -> str:
        """Record the schema and return the scenario's reply."""
        self.schemas.append(schema_name)
        return self.scenario.reply(ModelRequest("fake", schema_name, prompt))

    def with_structured_output(
        self, schema: Any, method: str = "json_schema", include_raw: bool = True
    ) -> ScenarioStructured:
        """Return the structured path for one schema."""
        return ScenarioStructured(model=self, schema=schema)

    async def ainvoke(self, prompt: str, config: Any = None) -> AIMessage:
        """Answer a plain-text call."""
        return AIMessage(content=self.answer(None, prompt))
