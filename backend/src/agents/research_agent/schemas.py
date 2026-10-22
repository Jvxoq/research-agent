"""The decision the model returns each round.

``think`` asks the model for a :class:`Decision` through structured output, so
the provider enforces the shape and no text is parsed here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MalformedDecision(ValueError):
    """Raised when a structured response fails the decision check."""


class FinishedThread(ValueError):
    """Raised when a run starts on a thread whose research already ended."""


class SearchAction(BaseModel):
    """Search the web for one batch of queries."""

    name: Literal["search"] = "search"
    queries: list[str] = Field(min_length=1)


class LookUpAction(BaseModel):
    """Read one batch of pages, keeping only what serves ``goal``."""

    name: Literal["look_up"] = "look_up"
    urls: list[str] = Field(min_length=1)
    goal: str = Field(min_length=1)


Action = SearchAction | LookUpAction


class Decision(BaseModel):
    """One round's output: the rewritten report, and what to do next.

    Exactly one of ``answer`` and ``action`` is set. ``answer`` ends the run.
    """

    report: str = Field(
        description="The rewritten report, with this round's findings folded in."
    )
    answer: str | None = Field(
        default=None,
        description="The final answer, or null to keep researching.",
    )
    action: Action | None = Field(
        default=None,
        description="The one tool call for this round, or null when answering.",
    )


class Answered(BaseModel):
    """A decision that passed the check and ends the run with an answer.

    Also the schema of the last round's call, which has no ``action`` field so
    the model cannot pick a tool there.
    """

    report: str
    answer: str


class Acting(BaseModel):
    """A decision that passed the check and runs one tool call."""

    report: str
    action: Action


def trim_action(
    action: Action,
    max_queries_per_search: int,
    max_urls_per_look_up: int,
) -> tuple[Action, list[str]]:
    """Cut an over-long batch down to its ceiling.

    A model asking for more targets than the configuration allows is a cost
    problem, not a correctness one, so the batch is trimmed rather than failed.

    Args:
        action: The action the model chose.
        max_queries_per_search: Batch ceiling for a search action.
        max_urls_per_look_up: Batch ceiling for a look-up action.

    Returns:
        The action, trimmed if needed, and notes describing any trim.
    """
    notes: list[str] = []
    if isinstance(action, SearchAction) and len(action.queries) > max_queries_per_search:
        notes.append(
            f"Trimmed {len(action.queries)} queries to {max_queries_per_search}."
        )
        action = action.model_copy(
            update={"queries": action.queries[:max_queries_per_search]}
        )
    elif isinstance(action, LookUpAction) and len(action.urls) > max_urls_per_look_up:
        notes.append(f"Trimmed {len(action.urls)} urls to {max_urls_per_look_up}.")
        action = action.model_copy(update={"urls": action.urls[:max_urls_per_look_up]})
    return action, notes
