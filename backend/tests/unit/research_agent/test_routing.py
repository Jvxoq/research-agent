"""``route_after_think`` sends the run to the tools, or ends it."""

from __future__ import annotations

from agents.research_agent.nodes import route_after_think

PENDING_ACTION = {"name": "search", "queries": ["alpha"]}


def test_a_pending_action_goes_to_act() -> None:
    assert route_after_think({"action": PENDING_ACTION}) == "act"


def test_an_answer_ends_the_run() -> None:
    state = {"answer": "1893", "stop_reason": "answer", "action": None}

    assert route_after_think(state) == "__end__"


def test_stop_reason_beats_pending_action() -> None:
    state = {"stop_reason": "malformed", "action": PENDING_ACTION}

    assert route_after_think(state) == "__end__"


def test_no_action_ends_run() -> None:
    assert route_after_think({"action": None}) == "__end__"
