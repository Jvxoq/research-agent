"""The research agent's load scenario: its fake replies and how it reads a final state."""

from __future__ import annotations

import pytest

from agents.research_agent.prompts import EMPTY_SLOT
from agents.research_agent.schemas import Answered, Decision, LookUpAction, SearchAction
from core.tools.web_look_up import PageSummary
from evals.research_agent.load import (
    ANSWER,
    PAGES,
    QUESTIONS,
    REPORT,
    SCENARIO,
    outcome,
    reply,
)
from loadtest.scenario import ModelRequest, RunOutcome


def _round(slot: str) -> ModelRequest:
    """A Decision request whose Last Tool Call slot holds ``slot``."""
    prompt = f"Question\n- Last Tool Call:\n<tool_call>\n{slot}\n</tool_call>\n- Last Tool Response:"
    return ModelRequest(model="m", schema_name="Decision", prompt=prompt)


def test_first_round_searches() -> None:
    decision = Decision.model_validate_json(reply(_round(EMPTY_SLOT)))

    assert decision == Decision(
        report=REPORT,
        answer=None,
        action=SearchAction(queries=["load test query one", "load test query two"]),
    )


def test_after_search_reads_pages() -> None:
    decision = Decision.model_validate_json(reply(_round('{"name": "search", "queries": ["a"]}')))

    assert decision.action == LookUpAction(urls=list(PAGES), goal="Find the fixture fact.")


def test_after_look_up_answers() -> None:
    slot = '{"name": "look_up", "urls": ["u"], "goal": "g"}'

    assert Decision.model_validate_json(reply(_round(slot))) == Decision(
        report=REPORT, answer=ANSWER, action=None
    )


def test_last_slot_wins() -> None:
    """The instructions above the input can show a tool call too. The input slot is last."""
    request = _round('{"name": "search", "queries": ["a"]}')
    shown_first = ModelRequest(
        model="m",
        schema_name="Decision",
        prompt=f"<tool_call>\n{EMPTY_SLOT}\n</tool_call>\n{request.prompt}",
    )

    assert Decision.model_validate_json(reply(shown_first)).action == LookUpAction(
        urls=list(PAGES), goal="Find the fixture fact."
    )


def test_forced_answer() -> None:
    request = ModelRequest(model="m", schema_name="Answered", prompt="p")

    assert Answered.model_validate_json(reply(request)) == Answered(report=REPORT, answer=ANSWER)


def test_page_summary_passes_check() -> None:
    """A FULL summary with findings is the shape the look-up check accepts."""
    request = ModelRequest(model="m", schema_name="PageSummary", prompt="p")

    summary = PageSummary.model_validate_json(reply(request))

    assert (summary.coverage, summary.findings) == (
        "FULL",
        ["The fixture page states the fact the question needs."],
    )


def test_plain_call_answers() -> None:
    assert reply(ModelRequest(model="m", schema_name=None, prompt="p")) == ANSWER


@pytest.mark.parametrize(
    "request_,message",
    [
        (ModelRequest(model="m", schema_name="Verdict", prompt="p"), "never asks for schema 'Verdict'"),
        (ModelRequest(model="m", schema_name="Decision", prompt="no slot"), "has no Last Tool Call slot"),
        (_round('{"queries": ["a"]}'), "names no tool"),
        (_round("[1, 2]"), "names no tool"),
        (_round("{not json"), "Expecting property name"),
    ],
    ids=["unknown-schema", "no-slot", "no-name", "not-object", "bad-json"],
)
def test_unreadable_request_refused(request_: ModelRequest, message: str) -> None:
    """The fake turns this into a 400, so a broken prompt fails its run visibly."""
    with pytest.raises(ValueError, match=message):
        reply(request_)


def test_outcome_reads_every_field() -> None:
    state = {
        "round": 4,
        "answer": "x",
        "stop_reason": "answer",
        "retries": {
            "rate_limit_hits": 2,
            "rate_limit_wait_seconds": 1.5,
            "transient_retries": 3,
            "transient_wait_seconds": 0.5,
            "malformed_summaries": 1,
        },
    }

    assert outcome(state) == RunOutcome(
        answered=True,
        rounds=4,
        rate_limit_retries=2,
        rate_limit_wait_seconds=1.5,
        transient_retries=3,
        bad_response_retries=1,
    )


@pytest.mark.parametrize(
    "state",
    [
        {"round": 8, "answer": "partial", "stop_reason": "max_rounds"},
        {"round": 3, "answer": "", "stop_reason": "answer"},
        {"round": 3, "answer": None, "stop_reason": "answer"},
    ],
    ids=["max-rounds", "blank-answer", "no-answer"],
)
def test_outcome_not_answered(state: dict[str, object]) -> None:
    assert outcome(state).answered is False


def test_outcome_without_retries_counts_none() -> None:
    assert outcome({"round": 2, "answer": "x", "stop_reason": "answer"}) == RunOutcome(
        answered=True,
        rounds=2,
        rate_limit_retries=0,
        rate_limit_wait_seconds=0.0,
        transient_retries=0,
        bad_response_retries=0,
    )


def test_outcome_refuses_wrong_shape() -> None:
    with pytest.raises(ValueError, match="round"):
        outcome({"round": "three"})


def test_inputs_are_questions() -> None:
    """Each run sends a question, the only field the research agent's input takes."""
    assert list(SCENARIO.inputs) == [{"question": question} for question in QUESTIONS]
