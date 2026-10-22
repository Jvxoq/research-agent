"""The compiled research agent graph, run end to end with the network mocked.

The last test here is the one that proves the thesis. Round 3 sees round 2's
observation and does not see round 1's. If that ever stops being true, the
agent has quietly become a growing-context loop.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agents.research_agent.graph import build_graph, graph
from agents.research_agent.nodes import FINISHED_THREAD, MISSING_THREAD
from agents.research_agent.schemas import (
    Answered,
    Decision,
    FinishedThread,
    MissingThread,
)
from core.config import get_settings
from core.tracing import tracing
from tests.conftest import FakeChatModel, FakeSearch, decision

QUESTION = "In what year was the observatory founded?"

# Every run LangSmith receives for three rounds: the graph, its nodes and its router.
TRACED_NAMES = ("research-agent", "think", "act", "route_after_think")


@pytest.fixture
def three_round_script() -> list[Decision]:
    """Two research rounds, then an answer."""
    return [
        decision("round one report", queries=["first query"]),
        decision("round two report", queries=["second query"]),
        decision("round three report", answer="1893"),
    ]


@pytest.fixture
def to_the_limit() -> list[Decision | Answered]:
    """Two research rounds, then the forced answer of a run whose limit is 2."""
    return [
        decision("round one report", queries=["first query"]),
        decision("round two report", queries=["second query"]),
        Answered(report="final report", answer="1893, unconfirmed"),
    ]


@pytest.fixture
def run_graph(monkeypatch):
    """Return a helper that runs the graph against a scripted model."""

    def prepare(decisions: list[Decision | Answered]) -> FakeChatModel:
        model = FakeChatModel(structured_replies=decisions)
        search = FakeSearch(batches=[(["FINDING 1"], [None]), (["FINDING 2"], [None])])

        monkeypatch.setattr(
            "agents.research_agent.nodes.build_chat_model", lambda spec: model
        )
        monkeypatch.setattr("agents.research_agent.nodes.search", search)
        return model

    return prepare


@pytest.fixture
def config() -> dict[str, Any]:
    """A run configuration with the date frozen and the loop bounded."""
    return {
        "configurable": {
            "current_date_override": "2026-01-02",
            "max_rounds": 6,
            "thread_id": "thread-integration",
        }
    }


async def test_the_loop_researches_twice_then_answers(
    run_graph, config, three_round_script
) -> None:
    run_graph(three_round_script)

    final = await graph.ainvoke({"question": QUESTION}, config)

    assert final["answer"] == "1893"
    assert final["stop_reason"] == "answer"


async def test_round_record_not_in_output(
    run_graph, config, three_round_script
) -> None:
    run_graph(three_round_script)

    final = await graph.ainvoke({"question": QUESTION}, config)

    assert "per_round" not in final


async def test_output_holds_token_totals(
    run_graph, config, three_round_script
) -> None:
    model = run_graph(three_round_script)
    model.usage = (1000, 200)

    final = await graph.ainvoke({"question": QUESTION}, config)

    assert final["usage"] == {"input_tokens": 3000, "output_tokens": 600}


async def test_round_three_sees_last_observation_only(
    run_graph, config, three_round_script
) -> None:
    """This is the thesis. The workspace is rebuilt, never appended to."""
    model = run_graph(three_round_script)

    await graph.ainvoke({"question": QUESTION}, config)

    third_prompt = model.prompts[2]
    assert "round two report" in third_prompt
    assert "FINDING 2" in third_prompt
    assert "FINDING 1" not in third_prompt


async def test_the_report_arm_keeps_no_history(
    run_graph, config, three_round_script
) -> None:
    run_graph(three_round_script)

    final = await graph.ainvoke({"question": QUESTION}, config)

    assert final.get("history", []) == []


async def test_full_history_keeps_first_observation(
    run_graph, config, set_env, three_round_script
) -> None:
    model = run_graph(three_round_script)
    set_env(CARRY_FULL_HISTORY="true")

    await graph.ainvoke({"question": QUESTION}, config)

    third_prompt = model.prompts[2]
    assert "FINDING 1" in third_prompt
    assert "FINDING 2" in third_prompt


async def test_round_limit_ends_with_answer(
    run_graph, config, to_the_limit
) -> None:
    run_graph(to_the_limit)
    config["configurable"]["max_rounds"] = 2

    final = await graph.ainvoke({"question": QUESTION}, config)

    assert final["answer"] == "1893, unconfirmed"


async def test_round_limit_stops_max_rounds(
    run_graph, config, to_the_limit
) -> None:
    run_graph(to_the_limit)
    config["configurable"]["max_rounds"] = 2

    final = await graph.ainvoke({"question": QUESTION}, config)

    assert final["stop_reason"] == "max_rounds"


async def test_forced_answer_sees_last_observation(
    run_graph, config, to_the_limit
) -> None:
    """The last search a run paid for must reach the answer it gives."""
    model = run_graph(to_the_limit)
    config["configurable"]["max_rounds"] = 2

    await graph.ainvoke({"question": QUESTION}, config)

    assert "FINDING 2" in model.prompts[-1]


async def test_spend_never_stops_a_served_run(run_graph, config, to_the_limit) -> None:
    """Only the round limit ends a run that keeps searching."""
    model = run_graph(to_the_limit)
    # Two million input tokens per decision, and still no stop.
    model.usage = (2_000_000, 0)
    config["configurable"]["max_rounds"] = 2

    final = await graph.ainvoke({"question": QUESTION}, config)

    assert final["stop_reason"] == "max_rounds"


async def test_finished_thread_refuses_question(
    run_graph, config, three_round_script
) -> None:
    """A reused thread would carry the old report, round and usage into it."""
    model = run_graph(three_round_script)
    saved = build_graph().compile(checkpointer=InMemorySaver())
    thread = {**config, "configurable": {**config["configurable"], "thread_id": "t1"}}
    await saved.ainvoke({"question": QUESTION}, thread)
    calls_after_first_run = len(model.prompts)

    with pytest.raises(FinishedThread, match=FINISHED_THREAD):
        await saved.ainvoke({"question": "Who founded it?"}, thread)

    assert len(model.prompts) == calls_after_first_run


async def test_graph_refuses_without_thread(run_graph, config, three_round_script) -> None:
    run_graph(three_round_script)
    del config["configurable"]["thread_id"]

    with pytest.raises(MissingThread, match=f"^{re.escape(MISSING_THREAD)}"):
        await graph.ainvoke({"question": "When was it founded?"}, config)


async def test_every_traced_run_carries_thread(
    traced_runs, run_graph, config, three_round_script
) -> None:
    """LangSmith files a run under a thread only when the run itself names it."""
    run_graph(three_round_script)

    with tracing(get_settings()):
        await graph.ainvoke({"question": "When was it founded?"}, config)

    assert {run.name: run.thread_id for run in traced_runs()} == {
        name: "thread-integration" for name in TRACED_NAMES
    }

async def test_input_round_cannot_extend_limit(
    run_graph, config, to_the_limit
) -> None:
    """The input is the question alone, so a caller cannot skip max_rounds."""
    run_graph(to_the_limit)
    config["configurable"]["max_rounds"] = 2

    final = await graph.ainvoke({"question": QUESTION, "round": -100}, config)

    assert (final["stop_reason"], final["round"]) == ("max_rounds", 3)

