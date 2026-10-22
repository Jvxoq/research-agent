"""The research agent's part of the server load test.

The fake model plays a short, fixed research loop: search, then read two
pages, then answer. Every run takes three rounds and five model calls, so a
change in latency comes from the server, not from the model choosing a longer
path.

The loop is read from the prompt's Last Tool Call slot, the same slot the real
model reads, so the fake needs no memory between requests.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from agents.research_agent.prompts import EMPTY_SLOT
from core.retry import RetryTotals
from loadtest.scenario import ModelRequest, RunOutcome, Scenario

QUESTIONS: tuple[str, ...] = (
    "Which river runs through the capital of the country that hosted the 1992 Summer Olympics?",
    "How many years passed between the first ascent of Everest and the first Moon landing?",
    "What is the population of the largest city in the state where the Wright brothers first flew?",
    "Which element was discovered in the same year the Eiffel Tower opened?",
    "Who directed the film that won Best Picture the year the Berlin Wall fell?",
)
PAGES: tuple[str, ...] = (
    "https://example.com/loadtest/1",
    "https://example.com/loadtest/2",
)
REPORT = (
    "## Confirmed Facts\n- Fixture fact from a load test page.\n"
    "## Gaps\n- None.\n"
    "## History\n- Load test round.\n"
    "## Plan\n- Answer."
)
ANSWER = "Load test answer."
# The Last Tool Call slot is the last <tool_call> block in the round prompt.
_TOOL_CALL = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)


def _last_tool(prompt: str) -> str | None:
    """Name the tool the last round called, or None in round one.

    Raises:
        ValueError: When the prompt has no Last Tool Call slot, or the slot
            holds something other than a tool call.
    """
    slots = _TOOL_CALL.findall(prompt)
    if not slots:
        raise ValueError("the round prompt has no Last Tool Call slot")
    text = slots[-1]
    if text == EMPTY_SLOT:
        return None
    call: object = json.loads(text)
    name = call.get("name") if isinstance(call, dict) else None
    if not isinstance(name, str):
        raise ValueError("the Last Tool Call slot names no tool")
    return name


def _decision(prompt: str) -> dict[str, object]:
    """Choose this round's step from the step the last round took."""
    last = _last_tool(prompt)
    if last is None:
        return {
            "report": REPORT,
            "answer": None,
            "action": {"name": "search", "queries": ["load test query one", "load test query two"]},
        }
    if last == "search":
        return {
            "report": REPORT,
            "answer": None,
            "action": {"name": "look_up", "urls": list(PAGES), "goal": "Find the fixture fact."},
        }
    return {"report": REPORT, "answer": ANSWER, "action": None}


def reply(request: ModelRequest) -> str:
    """Answer one of the research agent's model calls.

    Args:
        request: The schema the call asked for and its prompt.

    Returns:
        The JSON for a ``Decision``, ``Answered`` or ``PageSummary``, or the
        plain answer for a call with no schema.

    Raises:
        ValueError: For a schema this agent never asks for, or a round prompt
            the loop cannot read.
    """
    if request.schema_name == "Decision":
        return json.dumps(_decision(request.prompt))
    if request.schema_name == "Answered":
        return json.dumps({"report": REPORT, "answer": ANSWER})
    if request.schema_name == "PageSummary":
        return json.dumps(
            {
                "findings": ["The fixture page states the fact the question needs."],
                "source_date": "unknown",
                "coverage": "FULL",
                "coverage_note": "",
            }
        )
    if request.schema_name is None:
        return ANSWER
    raise ValueError(f"the research agent never asks for schema {request.schema_name!r}")


class _FinalState(TypedDict, total=False):
    """The output fields the load test reads."""

    round: int
    answer: str | None
    stop_reason: str | None
    retries: RetryTotals


_FINAL_STATE = TypeAdapter(_FinalState)


def outcome(state: Mapping[str, object]) -> RunOutcome:
    """Read a finished research run's final state.

    ``bad_response_retries`` counts page summaries that failed their check. A
    retried round decision is counted per round, and the per-round record stays
    out of a served run's output, so it is not part of this number.

    Raises:
        ValueError: When the state does not match the research agent's output.
    """
    try:
        final = _FINAL_STATE.validate_python(dict(state))
    except ValidationError as error:
        raise ValueError(str(error)) from error
    retries = final.get("retries")
    return RunOutcome(
        answered=final.get("stop_reason") == "answer" and bool(final.get("answer")),
        rounds=final.get("round", 0),
        rate_limit_retries=retries["rate_limit_hits"] if retries else 0,
        rate_limit_wait_seconds=retries["rate_limit_wait_seconds"] if retries else 0.0,
        transient_retries=retries["transient_retries"] if retries else 0,
        bad_response_retries=retries["malformed_summaries"] if retries else 0,
    )


SCENARIO = Scenario(
    inputs=tuple({"question": question} for question in QUESTIONS),
    reply=reply,
    outcome=outcome,
)
