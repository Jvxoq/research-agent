"""What one agent hands the load test.

The load runner is generic. It knows the server, not any agent. Each agent in
``langgraph.json`` supplies a :class:`Scenario` in ``evals/<agent>/load.py``:
the inputs to send, how the fake model answers that agent's prompts, and how to
read the agent's final state. A manifest graph with no scenario stops the run.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType

from langgraph_sdk.schema import Input


@dataclass(frozen=True)
class ModelRequest:
    """One chat completion request, as the fake model sees it.

    Args:
        model: The model name the request asked for.
        schema_name: The ``json_schema`` name of a structured-output request,
            which is the Pydantic class name. None for a plain-text request.
        prompt: The text of the last user message.
    """

    model: str
    schema_name: str | None
    prompt: str


@dataclass(frozen=True)
class RunOutcome:
    """What one finished run's final state says, read by its agent's scenario.

    Args:
        answered: The run ended with an answer, not a stop or an error.
        rounds: Think-act rounds the run used.
        rate_limit_retries: Retries after a provider rate limit.
        rate_limit_wait_seconds: Seconds spent waiting on rate limits.
        transient_retries: Retries after a network or server error.
        bad_response_retries: Retries after a reply that failed its schema.
    """

    answered: bool
    rounds: int
    rate_limit_retries: int
    rate_limit_wait_seconds: float
    transient_retries: int
    bad_response_retries: int


@dataclass(frozen=True)
class Scenario:
    """One agent's part of the load test.

    Args:
        inputs: Run inputs, used in turn. Run ``n`` sends ``inputs[n % len]``.
        reply: The fake model's message content for one request to this agent.
        outcome: Reads a finished run's final state. Raises ``ValueError`` when
            the state does not have the shape the agent promises.
    """

    inputs: Sequence[Input]
    reply: Callable[[ModelRequest], str]
    outcome: Callable[[Mapping[str, object]], RunOutcome]

    def __post_init__(self) -> None:
        """Refuse a scenario with nothing to send."""
        if not self.inputs:
            raise ValueError("a load scenario needs at least one input")


class MissingScenario(LookupError):
    """Raised when a graph in the manifest has no load scenario."""


def load_scenario(graph_id: str) -> Scenario:
    """Import the scenario for one manifest graph.

    Args:
        graph_id: The graph's name in ``langgraph.json``, which is also its
            agent package name.

    Returns:
        The module-level ``SCENARIO`` of ``evals/<graph_id>/load.py``.

    Raises:
        MissingScenario: When the module or its ``SCENARIO`` is absent.
    """
    name = f"evals.{graph_id}.load"
    try:
        module: ModuleType = importlib.import_module(name)
    except ModuleNotFoundError as error:
        # A broken import inside the scenario is a bug, not a missing scenario.
        if error.name not in {name, f"evals.{graph_id}"}:
            raise
        raise MissingScenario(
            f"{graph_id} has no evals/{graph_id}/load.py, so it cannot be load tested"
        ) from error
    scenario = getattr(module, "SCENARIO", None)
    if not isinstance(scenario, Scenario):
        raise MissingScenario(f"evals/{graph_id}/load.py defines no Scenario named SCENARIO")
    return scenario
