"""Graph state for the research agent.

The workspace is rebuilt every round, never appended to. ``report``, ``action``
and ``observation`` are replaced whole, so the state stays the same size however
many rounds a run takes. ``history`` is the one exception, and it exists only to
serve the full-history ablation baseline.

``per_round`` is private: it holds the latest round's record and never reaches
the run's output. The eval runner reads each record as the node returns it.
"""

from __future__ import annotations

import operator
from typing import Literal

from typing_extensions import Annotated, TypedDict

from core.cost import UsageTotals
from core.retry import RetryTotals

# Fields holding the rebuilt workspace. tests/structure/test_state_reducers.py
# reads this name and fails the build if any of them accumulates.
ROUND_LOOP_FIELDS = ("report", "action", "observation")

# Why a run ended. Only "answer" means the model finished its research. "error"
# means the decision call spent its transport retries.
StopReason = Literal["answer", "max_rounds", "malformed", "error"]


class SearchCall(TypedDict):
    """A search the model chose, in the flat shape of its action."""

    name: Literal["search"]
    queries: list[str]


class LookUpCall(TypedDict):
    """A look-up the model chose, in the flat shape of its action."""

    name: Literal["look_up"]
    urls: list[str]
    goal: str


ToolCall = SearchCall | LookUpCall


class Observation(TypedDict):
    """What one tool call returned, one entry per target in its batch."""

    call: ToolCall
    results: list[str]
    errors: list[str | None]
    capped: bool


class RoundRecord(TypedDict):
    """One round's accounting. The eval runner builds its charts from these."""

    round: int
    prompt_chars: int
    report_chars: int
    observation_chars: int
    action_name: str | None
    malformed_retries: int


class ResearchAgentInput(TypedDict):
    """What a caller may send to start a run.

    The question alone. Every other field is the loop's own bookkeeping, and a
    caller who could set ``round`` or ``stop_reason`` would skip ``max_rounds``
    or reopen a finished thread.
    """

    question: str


class ResearchAgentOutput(TypedDict, total=False):
    """What a finished run returns to its caller."""

    current_date: str
    question: str
    report: str
    action: ToolCall | None
    observation: Observation | None
    round: int
    answer: str | None
    stop_reason: StopReason | None
    # Replaced whole by each node, which reads the totals so far and writes the
    # new ones. The run totals span nodes only because of this field.
    usage: UsageTotals
    # Replaced whole, like usage. A served run has no caller-held counters, so
    # its rate-limit and transient retry totals span nodes only through here.
    retries: RetryTotals
    # Ablation baseline only. Stays empty unless carry_full_history is set.
    history: Annotated[list[Observation], operator.add]


class ResearchAgentState(ResearchAgentOutput, total=False):
    """State carried through the research agent loop."""

    # Left out of the output, so a served run's result holds no per-round log.
    per_round: RoundRecord


class ResearchAgentUpdate(TypedDict, total=False):
    """What one node writes back. Every node is annotated with this type.

    Every key must also exist on :class:`ResearchAgentState`. Annotating a node
    ``-> dict`` lets a mistyped key reach the reducer and vanish.
    """

    current_date: str
    report: str
    action: ToolCall | None
    observation: Observation
    round: int
    answer: str | None
    stop_reason: StopReason
    per_round: RoundRecord
    usage: UsageTotals
    retries: RetryTotals
    history: list[Observation]
