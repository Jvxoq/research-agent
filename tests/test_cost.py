"""Tests for token accounting and hard spend limits."""

import pytest
from langchain_core.outputs import LLMResult

from agent.configuration import Configuration
from agent.cost import (
    BudgetExceeded,
    CostTracker,
    estimate_cost,
    price_for,
)


def _result(model: str, prompt: int, completion: int) -> LLMResult:
    """Build a minimal LLMResult carrying the usage a provider would return."""
    return LLMResult(
        generations=[[]],
        llm_output={
            "model_name": model,
            "token_usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
            },
        },
    )


# ----------------------------------------------------------------------
# Pricing
# ----------------------------------------------------------------------


def test_known_model_has_a_price():
    assert price_for("claude-opus-5") == (5.00, 25.00)


def test_unknown_model_has_no_price():
    assert price_for("some/unreleased-model") is None


def test_cost_is_computed_per_million_tokens():
    # 1M in + 1M out on Opus 5 = $5 + $25
    assert estimate_cost("claude-opus-5", 1_000_000, 1_000_000) == pytest.approx(30.0)


def test_unknown_model_costs_zero_but_is_flagged():
    tracker = CostTracker()
    tracker.on_llm_end(_result("mystery/model", 100, 100))
    assert tracker.usage.cost_usd == 0.0
    assert tracker.usage.total_tokens == 200
    assert tracker.usage.has_unpriced_models is True


# ----------------------------------------------------------------------
# Accounting
# ----------------------------------------------------------------------


def test_usage_accumulates_across_calls():
    tracker = CostTracker()
    tracker.on_llm_end(_result("claude-opus-5", 1000, 500))
    tracker.on_llm_end(_result("claude-opus-5", 2000, 100))
    assert tracker.usage.calls == 2
    assert tracker.usage.input_tokens == 3000
    assert tracker.usage.output_tokens == 600
    assert tracker.usage.total_tokens == 3600


def test_usage_is_broken_down_per_model():
    tracker = CostTracker()
    tracker.on_llm_end(_result("claude-opus-5", 1000, 0))
    tracker.on_llm_end(_result("claude-haiku-4-5", 500, 0))
    assert set(tracker.usage.per_model) == {"claude-opus-5", "claude-haiku-4-5"}
    assert tracker.usage.per_model["claude-opus-5"]["input_tokens"] == 1000


def test_usage_serialises():
    tracker = CostTracker()
    tracker.on_llm_end(_result("claude-opus-5", 1000, 500))
    data = tracker.usage.to_dict()
    assert data["calls"] == 1
    assert data["total_tokens"] == 1500
    assert data["cost_usd"] > 0


def test_missing_usage_does_not_crash():
    """A provider that omits usage must not break the run."""
    tracker = CostTracker()
    tracker.on_llm_end(LLMResult(generations=[[]], llm_output={"model_name": "x"}))
    assert tracker.usage.calls == 1
    assert tracker.usage.total_tokens == 0


# ----------------------------------------------------------------------
# Hard ceilings — the part that protects the wallet
# ----------------------------------------------------------------------


def test_token_ceiling_stops_the_run():
    tracker = CostTracker(max_tokens=1000)
    tracker.on_llm_end(_result("claude-opus-5", 400, 400))  # 800, under
    with pytest.raises(BudgetExceeded, match="token budget"):
        tracker.on_llm_end(_result("claude-opus-5", 400, 0))  # 1200, over


def test_cost_ceiling_stops_the_run():
    tracker = CostTracker(max_cost_usd=0.01)
    with pytest.raises(BudgetExceeded, match="cost budget"):
        tracker.on_llm_end(_result("claude-opus-5", 1_000_000, 0))  # $5


def test_no_ceiling_never_raises():
    tracker = CostTracker()
    for _ in range(5):
        tracker.on_llm_end(_result("claude-opus-5", 1_000_000, 1_000_000))
    assert tracker.usage.cost_usd > 100


def test_usage_is_recorded_even_when_the_ceiling_trips():
    """The run stops, but you still learn what it spent getting there."""
    tracker = CostTracker(max_tokens=100)
    with pytest.raises(BudgetExceeded):
        tracker.on_llm_end(_result("claude-opus-5", 500, 0))
    assert tracker.usage.total_tokens == 500
    assert tracker.usage.calls == 1


def test_ceilings_are_configurable_by_env(monkeypatch):
    monkeypatch.setenv("MAX_RUN_TOKENS", "50000")
    monkeypatch.setenv("MAX_RUN_COST_USD", "0.25")
    config = Configuration.from_runnable_config()
    assert config.max_run_tokens == 50000
    assert config.max_run_cost_usd == 0.25


def test_ceilings_are_off_by_default():
    config = Configuration()
    assert config.max_run_tokens is None
    assert config.max_run_cost_usd is None


def test_tracker_opts_into_exception_propagation():
    """LangChain swallows callback exceptions unless raise_error is set.

    Without this the ceiling silently does nothing — a spend limit that does
    not stop spending. Regression guard.
    """
    assert CostTracker.raise_error is True
