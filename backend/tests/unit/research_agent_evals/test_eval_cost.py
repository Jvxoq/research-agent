"""The eval accountant: prices, and joining each call to its round."""

from uuid import uuid4

import pytest
from langchain_core.outputs import LLMResult

from core.config import get_settings
from evals.research_agent.cost import RunAccountant, estimate_cost, price_for

RECORD = {
    "round": 0,
    "prompt_chars": 100,
    "report_chars": 10,
    "observation_chars": 0,
    "action_name": "search",
    "malformed_retries": 0,
}


def _result(model: str, prompt: int, completion: int) -> LLMResult:
    """Build a minimal LLMResult carrying the usage a provider would return."""
    return LLMResult(
        generations=[[]],
        llm_output={
            "model_name": model,
            "token_usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        },
    )


async def _node(accountant: RunAccountant, step: int, outputs: dict) -> None:
    """Run one graph node through the accountant's callbacks."""
    run_id = uuid4()
    await accountant.on_chain_start({}, {}, run_id=run_id, metadata={"langgraph_step": step})
    await accountant.on_chain_end(outputs, run_id=run_id)


async def _call(
    accountant: RunAccountant, step: int | None, model: str, prompt: int, completion: int
) -> None:
    """Run one model call through the accountant's callbacks."""
    run_id = uuid4()
    metadata = {} if step is None else {"langgraph_step": step}
    await accountant.on_chat_model_start({}, [[]], run_id=run_id, metadata=metadata)
    await accountant.on_llm_end(_result(model, prompt, completion), run_id=run_id)


def test_known_model_has_a_price():
    assert price_for("zai-org/GLM-5.3-Flash") == (0.15, 0.50)


def test_unknown_model_has_no_price():
    assert price_for("some/unreleased-model") is None


def test_cost_is_computed_per_million_tokens():
    # 1M in + 1M out on GLM-5.3-Flash = $0.15 + $0.50
    assert estimate_cost("zai-org/GLM-5.3-Flash", 1_000_000, 1_000_000) == pytest.approx(0.65)


def test_env_price_adds_model(monkeypatch):
    monkeypatch.setenv("MODEL_PRICES_USD_PER_MTOK", '{"mystery/model": [0.5, 2.0]}')
    get_settings.cache_clear()

    assert price_for("mystery/model") == (0.5, 2.0)


def test_env_price_overrides_builtin(monkeypatch):
    monkeypatch.setenv("MODEL_PRICES_USD_PER_MTOK", '{"zai-org/GLM-5.3-Flash": [1.0, 1.0]}')
    get_settings.cache_clear()

    assert price_for("zai-org/GLM-5.3-Flash") == (1.0, 1.0)


async def test_act_call_counts_to_prior_round():
    """act runs the action think chose, so its calls belong to that round."""
    accountant = RunAccountant()
    await _call(accountant, 1, "zai-org/GLM-5.3-Flash", 1000, 0)
    await _node(accountant, 1, {"per_round": RECORD})
    await _call(accountant, 2, "zai-org/GLM-5.3-Flash", 500, 0)
    await _node(accountant, 2, {"observation": {}})
    await _call(accountant, 3, "zai-org/GLM-5.3-Flash", 2000, 0)
    await _node(accountant, 3, {"per_round": {**RECORD, "round": 1}})

    assert [line["prompt_tokens"] for line in accountant.rounds()] == [1500, 2000]


async def test_unpriced_model_costs_zero():
    accountant = RunAccountant()
    await _call(accountant, 1, "mystery/model", 100, 100)

    assert (accountant.tokens, accountant.usd) == (200, 0.0)


async def test_stepless_call_counts_in_totals():
    accountant = RunAccountant()
    await _node(accountant, 1, {"per_round": RECORD})
    await _call(accountant, None, "zai-org/GLM-5.3-Flash", 1000, 0)

    assert (accountant.tokens, accountant.rounds()[0]["prompt_tokens"]) == (1000, 0)


async def test_totals_price_each_model():
    accountant = RunAccountant()
    await _call(accountant, 1, "deepseek-ai/DeepSeek-V4.1-Flash", 1000, 0)
    await _call(accountant, 2, "zai-org/GLM-5.3-Flash", 1000, 0)

    # $0.30 and $0.15 per million input tokens.
    assert accountant.usd == 0.00045
