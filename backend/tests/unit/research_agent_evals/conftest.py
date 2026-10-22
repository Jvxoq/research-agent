"""Fixtures for the research agent eval tests.

Nothing here touches the network, a real model or the wall clock. The graph,
the judge model and the dataset loader are all replaced.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from typing import Any

import pytest
from loguru import logger

from evals.research_agent.dataset import Row
from tests.conftest import ContainmentJudge


def _collect(level: str) -> Iterator[list[str]]:
    """Collect every loguru message at or above a level while the test runs."""
    messages: list[str] = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]), level=level
    )
    yield messages
    logger.remove(sink)


@pytest.fixture
def error_log() -> Iterator[list[str]]:
    """Collect every error message loguru records during the test."""
    yield from _collect("ERROR")


@pytest.fixture
def info_log() -> Iterator[list[str]]:
    """Collect every message loguru records at INFO and above during the test."""
    yield from _collect("INFO")


@pytest.fixture
def rows() -> list[Row]:
    """Return three benchmark rows."""
    return [
        Row(question_id="q1", question="Capital of France?", answer="Paris"),
        Row(question_id="q2", question="Who wrote Frankenstein?", answer="Mary Shelley"),
        Row(question_id="q3", question="Apollo 11 landing year?", answer="1969"),
    ]


@pytest.fixture
def judge() -> ContainmentJudge:
    """Return a judge that grades the known-answer set exactly right."""
    return ContainmentJudge()


# What a scripted agent does for one question, returning its final state.
Script = Callable[[str], Awaitable[dict[str, object]]]


class ScriptedAgent:
    """Stands in for the compiled graph, acting out one script per question."""

    def __init__(self, script: Script) -> None:
        self.questions: list[str] = []
        self.script = script

    async def ainvoke(self, state: dict[str, Any], config: Any = None) -> dict[str, object]:
        """Record the question, run the script for it and return its final state."""
        self.questions.append(state["question"])
        return await self.script(state["question"])


@pytest.fixture
def scripted_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str, Script], ScriptedAgent]:
    """Return a helper that replaces one eval module's graph with a scripted agent."""

    def install(module: str, script: Script) -> ScriptedAgent:
        agent = ScriptedAgent(script)
        monkeypatch.setattr(f"{module}.graph", agent)
        return agent

    return install
