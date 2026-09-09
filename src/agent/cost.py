"""Token accounting and hard spend limits.

Two jobs, and the second is the important one:

1. Count tokens and dollars per node, so a run reports what it cost.
2. Stop the run the moment it crosses a ceiling you set.

The ceiling is enforced inside the model callback, so it fires between calls
rather than after the damage is done. A run that trips it raises
:class:`BudgetExceeded`, which the benchmark runner records as a normal
failed result rather than a crash.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from loguru import logger

# Published list prices in USD per million tokens, as (input, output).
# Unknown models are counted in tokens but priced at zero — see
# ``Usage.has_unpriced_models``, which the runner surfaces so a zero cost is
# never mistaken for a free run.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Groq (free tier bills nothing but still consumes quota)
    "openai/gpt-oss-120b": (0.15, 0.75),
    "openai/gpt-oss-20b": (0.10, 0.50),
    "qwen/qwen3.8-27b": (0.20, 0.60),
    # Together
    "deepseek-ai/DeepSeek-V3": (1.25, 1.25),
}


class BudgetExceeded(RuntimeError):
    """Raised when a run crosses its token or cost ceiling."""


@dataclass
class Usage:
    """Accumulated token and cost totals for one run."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    per_model: dict[str, dict[str, float]] = field(default_factory=dict)
    unpriced_models: set[str] = field(default_factory=set)

    @property
    def total_tokens(self) -> int:
        """Return input plus output tokens."""
        return self.input_tokens + self.output_tokens

    @property
    def has_unpriced_models(self) -> bool:
        """Report whether any model was counted without a known price."""
        return bool(self.unpriced_models)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "per_model": self.per_model,
            "unpriced_models": sorted(self.unpriced_models),
        }


def price_for(model: str) -> tuple[float, float] | None:
    """Return (input, output) price per million tokens, or None if unknown."""
    return PRICES_USD_PER_MTOK.get(model)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of one call, or 0.0 when the model has no price."""
    price = price_for(model)
    if price is None:
        return 0.0
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def _extract_usage(response: LLMResult) -> tuple[int, int]:
    """Pull (input, output) token counts out of a provider response.

    Providers report usage in more than one place depending on the client, so
    both known shapes are handled. Returns zeros when neither is present.
    """
    output = response.llm_output or {}
    for key in ("token_usage", "usage"):
        usage = output.get(key)
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            completion = (
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            if prompt or completion:
                return int(prompt), int(completion)

    for generations in response.generations:
        for generation in generations:
            message = getattr(generation, "message", None)
            meta = getattr(message, "usage_metadata", None)
            if meta:
                return int(meta.get("input_tokens", 0)), int(
                    meta.get("output_tokens", 0)
                )

    return 0, 0


class CostTracker(BaseCallbackHandler):
    """Accumulate usage across a run and enforce a hard ceiling.

    Args:
        max_tokens: Stop the run once total tokens exceed this. None disables.
        max_cost_usd: Stop the run once cost exceeds this. None disables.
    """

    # LangChain swallows exceptions raised inside callbacks unless a handler
    # opts in. Without this the ceiling silently does nothing, which is worse
    # than having no ceiling at all.
    raise_error = True

    def __init__(
        self,
        max_tokens: int | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        """Initialise the tracker with optional ceilings."""
        self.usage = Usage()
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self._lock = threading.Lock()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Record one call's usage and enforce the ceiling."""
        model = (response.llm_output or {}).get("model_name") or (
            response.llm_output or {}
        ).get("model", "unknown")
        input_tokens, output_tokens = _extract_usage(response)
        cost = estimate_cost(model, input_tokens, output_tokens)

        with self._lock:
            self.usage.calls += 1
            self.usage.input_tokens += input_tokens
            self.usage.output_tokens += output_tokens
            self.usage.cost_usd += cost

            entry = self.usage.per_model.setdefault(
                model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0}
            )
            entry["calls"] += 1
            entry["input_tokens"] += input_tokens
            entry["output_tokens"] += output_tokens
            entry["cost_usd"] = round(entry["cost_usd"] + cost, 6)

            if price_for(model) is None and (input_tokens or output_tokens):
                self.usage.unpriced_models.add(model)

            totals = (self.usage.total_tokens, self.usage.cost_usd)

        logger.info(
            f"usage: call={self.usage.calls} +{input_tokens}in/+{output_tokens}out "
            f"total={totals[0]} tokens ${totals[1]:.4f}"
        )
        self._enforce(*totals)

    def _enforce(self, total_tokens: int, cost_usd: float) -> None:
        """Raise BudgetExceeded when a configured ceiling is crossed."""
        if self.max_tokens is not None and total_tokens > self.max_tokens:
            raise BudgetExceeded(
                f"token budget exceeded: {total_tokens} > {self.max_tokens}"
            )
        if self.max_cost_usd is not None and cost_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"cost budget exceeded: ${cost_usd:.4f} > ${self.max_cost_usd:.4f}"
            )
