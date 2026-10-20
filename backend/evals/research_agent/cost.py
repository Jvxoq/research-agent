"""Dollar accounting for eval runs.

The served graph counts tokens only. The eval runner attaches a
:class:`RunAccountant` to each run it starts. The accountant sees every model
call, prices it, and joins it to the round that made it, so a result line holds
per-round tokens and dollars without the graph storing either.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from loguru import logger
from pydantic import TypeAdapter, ValidationError

from agents.research_agent.state import RoundRecord
from core.config import get_settings
from core.cost import token_counts

# Published list prices in USD per million tokens, as (input, output).
# MODEL_PRICES_USD_PER_MTOK in the environment adds to this table or overrides
# an entry. A model with no price is counted in tokens at $0, and each of its
# calls logs a warning naming it.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # Together
    "deepseek-ai/DeepSeek-V4.1-Flash": (0.30, 1.20),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (0.14, 0.28),
    "zai-org/GLM-5.3-Flash": (0.15, 0.50),
}

# The state key the think node writes its round record to.
ROUND_RECORD_KEY = "per_round"

_ROUND_RECORD = TypeAdapter(RoundRecord)


class RoundLine(RoundRecord):
    """One round of a result line: the node's record plus what its calls cost."""

    prompt_tokens: int
    completion_tokens: int
    usd: float


def price_for(model: str) -> tuple[float, float] | None:
    """Return (input, output) price per million tokens, or None if unknown."""
    configured = get_settings().MODEL_PRICES_USD_PER_MTOK or {}
    return configured.get(model) or PRICES_USD_PER_MTOK.get(model)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of one call, or 0.0 when the model has no price."""
    price = price_for(model)
    if price is None:
        return 0.0
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


@dataclass
class _Call:
    """One finished model call and the graph step that made it."""

    step: int
    input_tokens: int
    output_tokens: int
    usd: float


class RunAccountant(AsyncCallbackHandler):
    """Count tokens and dollars for one graph run, per round and in total.

    LangGraph tags every run inside a node with ``langgraph_step``. A round
    starts at a ``think`` step, whose output carries the round record, and
    every call up to the next ``think`` step belongs to that round.
    """

    def __init__(self) -> None:
        """Start with no calls and no rounds."""
        self._steps: dict[UUID, int] = {}
        self._calls: list[_Call] = []
        self._records: dict[int, RoundRecord] = {}
        self._unpriced: set[str] = set()

    def _remember_step(self, run_id: UUID, metadata: dict[str, Any] | None) -> None:
        """Note which graph step a starting run belongs to."""
        step = (metadata or {}).get("langgraph_step")
        if isinstance(step, int):
            self._steps[run_id] = step

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> None:
        """Note the step of a node run, so its output can be placed."""
        self._remember_step(run_id, metadata)

    async def on_chain_end(
        self, outputs: dict[str, Any], *, run_id: UUID, **kwargs: object
    ) -> None:
        """Keep the round record a node run wrote, keyed by its step."""
        step = self._steps.pop(run_id, None)
        if step is None or not isinstance(outputs, dict):
            return
        if ROUND_RECORD_KEY not in outputs:
            return
        try:
            self._records[step] = _ROUND_RECORD.validate_python(
                outputs[ROUND_RECORD_KEY]
            )
        except ValidationError:
            logger.exception(f"step {step} wrote an unreadable round record")

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> None:
        """Note the step of a chat model call."""
        self._remember_step(run_id, metadata)

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> None:
        """Note the step of a completion model call."""
        self._remember_step(run_id, metadata)

    async def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, **kwargs: object
    ) -> None:
        """Price one finished call and keep it with its step."""
        output = response.llm_output or {}
        model = str(output.get("model_name") or output.get("model") or "unknown")
        input_tokens, output_tokens = token_counts(response)
        if price_for(model) is None and model not in self._unpriced:
            self._unpriced.add(model)
            logger.warning(
                f"no price for model {model!r}; counted at $0. "
                "Add it to MODEL_PRICES_USD_PER_MTOK."
            )
        self._calls.append(
            _Call(
                # A call LangGraph did not tag counts in the totals only.
                step=self._steps.pop(run_id, -1),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                usd=estimate_cost(model, input_tokens, output_tokens),
            )
        )

    @property
    def tokens(self) -> int:
        """Input plus output tokens over every call."""
        return sum(call.input_tokens + call.output_tokens for call in self._calls)

    @property
    def usd(self) -> float:
        """Dollars over every call, rounded to a millionth."""
        return round(sum(call.usd for call in self._calls), 6)

    def rounds(self) -> list[RoundLine]:
        """Return one line per round, in round order, with its calls' spend.

        Returns:
            Each round record joined with the tokens and dollars of the calls
            made from its ``think`` step up to the next one.
        """
        starts = sorted(self._records)
        lines = [
            RoundLine(
                **self._records[step], prompt_tokens=0, completion_tokens=0, usd=0.0
            )
            for step in starts
        ]
        for call in self._calls:
            index = bisect_right(starts, call.step) - 1
            if index < 0:
                continue
            lines[index]["prompt_tokens"] += call.input_tokens
            lines[index]["completion_tokens"] += call.output_tokens
            lines[index]["usd"] += call.usd
        for line in lines:
            line["usd"] = round(line["usd"], 6)
        return lines
