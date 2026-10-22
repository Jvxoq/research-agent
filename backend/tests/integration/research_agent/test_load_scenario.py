"""The load test's research scenario, played through the real compiled graph.

The fake provider answers from the prompt's Last Tool Call slot. If the prompt
changes shape, the load test would send every run down a broken path, so this
runs the real graph with the scenario answering, and the fakes' own search and
extract replies standing in for Tavily.
"""

from __future__ import annotations

import pytest

from agents.research_agent.graph import graph
from core.tools import web_look_up, web_search
from evals.research_agent.load import QUESTIONS, SCENARIO
from loadtest.fakes import extract_results, search_results
from loadtest.scenario import RunOutcome
from tests.conftest import (
    PUBLIC_ADDRESS,
    FakeExtractor,
    FakeSearchTool,
    ScenarioChatModel,
)


@pytest.fixture
def scenario_model(monkeypatch, resolves, set_env) -> ScenarioChatModel:
    """Answer every model call from the scenario, and every Tavily call from the fakes."""
    model = ScenarioChatModel(scenario=SCENARIO)
    set_env(TAVILY_API_KEY="tvly-test")
    resolves(default=PUBLIC_ADDRESS)

    async def search(query: str) -> object:
        return search_results(query)

    async def extract(url: str) -> object:
        return extract_results([url])

    monkeypatch.setattr("agents.research_agent.nodes.build_chat_model", lambda spec: model)
    monkeypatch.setattr(web_search, "TavilySearch", lambda **kwargs: FakeSearchTool(search))
    monkeypatch.setattr(web_look_up, "TavilyExtract", lambda **kwargs: FakeExtractor(extract))
    return model


async def test_scenario_answers_in_three_rounds(scenario_model: ScenarioChatModel) -> None:
    """Every load test run must answer, in the same three rounds, or latency means nothing."""
    final = await graph.ainvoke(
        {"question": QUESTIONS[0]}, {"configurable": {"thread_id": "load-scenario"}}
    )

    assert SCENARIO.outcome(final) == RunOutcome(
        answered=True,
        rounds=3,
        rate_limit_retries=0,
        rate_limit_wait_seconds=0.0,
        transient_retries=0,
        bad_response_retries=0,
    )


async def test_scenario_takes_fixed_path(scenario_model: ScenarioChatModel) -> None:
    """Search, read two pages, answer: five model calls in a fixed order."""
    await graph.ainvoke(
        {"question": QUESTIONS[0]}, {"configurable": {"thread_id": "load-scenario-path"}}
    )

    assert scenario_model.schemas == [
        "Decision",
        "Decision",
        "PageSummary",
        "PageSummary",
        "Decision",
    ]
