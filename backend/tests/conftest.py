"""Shared fixtures and the one model fake every suite uses.

Every test runs against a known environment. The developer's own shell, and
any `.env` file, must not change a result.

Every fake lives here rather than beside one agent's tests. Two fakes for
one thing drift, and a test that passes against a stale fake proves nothing.
"""

from __future__ import annotations

import os
import re
import socket
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, NoReturn

import pytest
from langchain_core.callbacks import AsyncCallbackManager
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult
from langchain_core.tools import ToolException

from agents.research_agent.schemas import Decision, SearchAction
from core.config import Settings, get_settings
from core.llm import reset_rate_limiter
from core.retry import COUNTERS_KEY, BackoffPolicy, Retrier
from core.tools.web_look_up import LookUpBatch
from tests.structure.conftest import configuration_classes

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
        runs = await manager.on_chat_model_start({}, [[]])
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


@dataclass
class FakeLangSmithClient:
    """A LangSmith client that records how it was built and sends nothing."""

    api_url: str | None = None
    api_key: str | None = None


@pytest.fixture
def langsmith_client(monkeypatch):
    """Replace the LangSmith client, whose constructor calls the network."""
    monkeypatch.setattr("core.tracing.Client", FakeLangSmithClient)


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
    ) -> FakeStructuredModel:
        """Return the scripted structured path."""
        return self.structured.with_structured_output(schema, method, include_raw)
