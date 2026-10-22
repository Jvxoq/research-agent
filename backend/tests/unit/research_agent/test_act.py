"""``act`` runs the one action the model chose, and returns one observation.

The tools themselves are covered by their own tests. Here they are replaced, so
each test checks only what the node does with what they return.
"""

from __future__ import annotations

from typing import Any

import pytest

from agents.research_agent.nodes import act
from core.retry import COUNTERS_KEY, RetryCounters
from core.tools.web_look_up import PageSummary
from tests.conftest import PUBLIC_ADDRESS, FakeChatModel, FakeLookUp, FakeSearch


@pytest.fixture
def patch_tools(monkeypatch):
    """Install fake tools, returning whatever the test asks them to return.

    Returns:
        A helper that installs the fakes and hands back both of them.
    """

    def install(
        results: list[str] | None = None,
        errors: list[str | None] | None = None,
        malformed_summaries: int = 0,
    ) -> tuple[FakeSearch, FakeLookUp]:
        pair = (results or ["a result"], errors or [None])
        search = FakeSearch(batches=[pair])
        look_up = FakeLookUp(*pair, malformed_summaries=malformed_summaries)
        monkeypatch.setattr("agents.research_agent.nodes.search", search)
        monkeypatch.setattr("agents.research_agent.nodes.look_up", look_up)
        monkeypatch.setattr(
            "agents.research_agent.nodes.build_chat_model", lambda spec: object()
        )
        return search, look_up

    return install


@pytest.fixture
def search_state() -> dict[str, Any]:
    """A state with a pending search action."""
    return {
        "question": "q",
        "round": 1,
        "action": {"name": "search", "queries": ["alpha", "beta"]},
    }


@pytest.fixture
def look_up_state() -> dict[str, Any]:
    """A state with a pending look-up action."""
    return {
        "question": "q",
        "round": 1,
        "action": {
            "name": "look_up",
            "urls": ["https://a.test", "https://b.test"],
            "goal": "the founding year",
        },
    }


async def test_search_batch_becomes_observation(
    patch_tools, config, search_state
) -> None:
    search, _ = patch_tools(results=["alpha result", "beta result"], errors=[None, None])

    update = await act(search_state, config)

    assert search.calls[0]["queries"] == ["alpha", "beta"]
    assert update["observation"]["results"] == ["alpha result", "beta result"]
    assert update["observation"]["call"] == search_state["action"]


async def test_look_up_sends_goal(
    patch_tools, config, look_up_state
) -> None:
    _, look_up = patch_tools()

    await act(look_up_state, config)

    assert look_up.calls[0]["settings"].goal == "the founding year"


async def test_failed_target_keeps_error_slot(
    patch_tools, config, search_state
) -> None:
    patch_tools(results=["alpha result", ""], errors=[None, "Search failed: Boom"])

    update = await act(search_state, config)

    assert update["observation"]["errors"] == [None, "Search failed: Boom"]
    assert update["observation"]["results"][0] == "alpha result"


async def test_cap_cuts_long_results_only(
    patch_tools, config, search_state
) -> None:
    """The cap applies per batch entry, not to the batch as a whole."""
    patch_tools(results=["x" * 50, "short"], errors=[None, None])
    config["configurable"]["max_tool_chars"] = 20

    update = await act(search_state, config)

    assert update["observation"]["capped"] is True
    assert update["observation"]["results"][0] == "x" * 14 + "\n[cut]"
    assert update["observation"]["results"][1] == "short"


async def test_only_full_history_records_history(
    patch_tools, config, set_env, search_state
) -> None:
    """The arm is the whole ablation, so both sides are one behaviour."""
    patch_tools()

    report_arm = await act(search_state, config)
    set_env(CARRY_FULL_HISTORY="true")
    history_arm = await act(search_state, config)

    assert "history" not in report_arm
    assert history_arm["history"] == [history_arm["observation"]]


async def test_summary_failures_reach_counters(
    patch_tools, config, look_up_state
) -> None:
    patch_tools(malformed_summaries=2)
    counters = RetryCounters()
    config["configurable"][COUNTERS_KEY] = counters

    await act(look_up_state, config)

    assert counters.malformed_summaries == 2


async def test_act_writes_back_usage(
    patch_tools, config, search_state
) -> None:
    """A search makes no model call, so the totals leave act as they arrived."""
    patch_tools()
    spent = {"input_tokens": 100, "output_tokens": 20}

    update = await act({**search_state, "usage": spent}, config)

    assert update["usage"] == spent


async def test_summary_failures_reach_state_totals(
    patch_tools, config, look_up_state
) -> None:
    """A served run has no caller counters, so state carries the count."""
    patch_tools(malformed_summaries=2)
    spent = {
        "rate_limit_hits": 0,
        "rate_limit_wait_seconds": 0.0,
        "transient_retries": 0,
        "transient_wait_seconds": 0.0,
        "malformed_summaries": 1,
    }

    update = await act({**look_up_state, "retries": spent}, config)

    assert update["retries"]["malformed_summaries"] == 3


@pytest.fixture
def real_tools(monkeypatch, resolves, set_env) -> None:
    """Run the real tools on fake Tavily clients, with a key from the environment."""
    set_env(TAVILY_API_KEY="tvly-test")
    resolves(default=PUBLIC_ADDRESS)
    summary = PageSummary(
        findings=["founded 1893"], source_date="unknown", coverage="FULL", coverage_note=""
    )
    monkeypatch.setattr(
        "agents.research_agent.nodes.build_chat_model",
        lambda spec: FakeChatModel(structured_replies=[summary]),
    )


@pytest.mark.parametrize("state_name", ["search_state", "look_up_state"])
async def test_tools_call_env_endpoint(
    request, real_tools, tavily_endpoints, set_env, config, state_name
) -> None:
    """A served run reads the Tavily endpoint from the environment, as the load test sets it."""
    set_env(TAVILY_BASE_URL="http://fakes.test:8790")

    await act(request.getfixturevalue(state_name), config)

    assert set(tavily_endpoints) == {"http://fakes.test:8790"}


@pytest.mark.parametrize("state_name", ["search_state", "look_up_state"])
async def test_request_cannot_set_endpoint(
    request, real_tools, tavily_endpoints, config, state_name
) -> None:
    """Whoever names the endpoint receives the Tavily key, so a caller never names it."""
    config["configurable"].update(
        {"tavily_base_url": "http://attacker.test", "TAVILY_BASE_URL": "http://attacker.test"}
    )

    await act(request.getfixturevalue(state_name), config)

    assert set(tavily_endpoints) == {None}
