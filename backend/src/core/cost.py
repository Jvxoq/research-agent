"""Token accounting.

Counts input and output tokens for every model call. A served run carries its
totals in graph state, so each node starts a tracker from what the run already
used. Dollars are not counted here: the eval runner prices the calls it
observes, so the served graph holds no price table.
"""

from __future__ import annotations

import threading

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from loguru import logger
from typing_extensions import TypedDict


class UsageTotals(TypedDict):
    """The tokens a run has used so far, as they are carried between nodes."""

    input_tokens: int
    output_tokens: int


def token_counts(response: LLMResult) -> tuple[int, int]:
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


class TokenTracker(BaseCallbackHandler):
    """Accumulate input and output tokens across a run.

    Args:
        spent: What the run used before this tracker, so the totals hold
            across nodes. None starts from zero.
    """

    def __init__(self, spent: UsageTotals | None = None) -> None:
        """Initialise the tracker from the run's prior usage."""
        self._input_tokens = spent["input_tokens"] if spent else 0
        self._output_tokens = spent["output_tokens"] if spent else 0
        self._lock = threading.Lock()

    def on_llm_end(self, response: LLMResult, **kwargs: object) -> None:
        """Record one call's tokens."""
        input_tokens, output_tokens = token_counts(response)
        with self._lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            totals = self.totals()
        logger.info(
            f"usage: +{input_tokens}in/+{output_tokens}out "
            f"total={totals['input_tokens']}in/{totals['output_tokens']}out"
        )

    def totals(self) -> UsageTotals:
        """Return the running totals."""
        return {
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
        }
