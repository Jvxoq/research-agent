"""Where a configuration value comes from, and which source wins.

Order, highest first: the run's `configurable` mapping, then the process
environment, then the field default. Getting this backwards means a caller
cannot change anything per request.
"""

import pytest
from pydantic import ValidationError

from agents.research_agent.configuration import DEFAULT_THINK_MODEL, Configuration


def test_env_used_when_alone(set_env):
    set_env(THINK_MODEL="from-env")

    config = Configuration.from_runnable_config()

    assert config.think_model == "from-env"


def test_request_used_when_env_silent():
    config = Configuration.from_runnable_config(
        {"configurable": {"answer_style": "report"}}
    )

    assert config.answer_style == "report"


def test_configurable_wins_over_the_environment(set_env):
    set_env(ANSWER_STYLE="short")

    config = Configuration.from_runnable_config(
        {"configurable": {"answer_style": "report"}}
    )

    assert config.answer_style == "report"


def test_default_used_when_unset():
    config = Configuration.from_runnable_config()

    assert config.think_model == DEFAULT_THINK_MODEL


def test_env_number_coerced(set_env):
    set_env(MAX_ROUNDS="5")

    config = Configuration.from_runnable_config()

    assert config.max_rounds == 5


def test_request_number_coerced():
    config = Configuration.from_runnable_config(
        {"configurable": {"max_rounds": "7"}}
    )

    assert config.max_rounds == 7


def test_server_keys_do_not_block_fields():
    """LangGraph Server puts thread_id and similar keys in the same mapping."""
    config = Configuration.from_runnable_config(
        {"configurable": {"thread_id": "abc", "not_a_field": "value", "max_rounds": 3}}
    )

    assert config.max_rounds == 3


def test_a_missing_config_argument_is_accepted():
    assert Configuration.from_runnable_config(None).think_model == DEFAULT_THINK_MODEL


def test_env_role_model_reaches_spec(set_env):
    set_env(THINK_MODEL="env-think-model")

    spec = Configuration.from_runnable_config().model_spec_for("think")

    assert spec.model == "env-think-model"


def test_role_url_falls_back_to_shared(set_env):
    set_env(BASE_URL="https://gateway.example/v1")

    config = Configuration.from_runnable_config()

    assert config.model_spec_for("think").base_url == "https://gateway.example/v1"


def test_role_url_overrides_shared(set_env):
    set_env(BASE_URL="https://shared.example/v1", THINK_BASE_URL="https://answer.example/v1")

    config = Configuration.from_runnable_config()

    assert config.model_spec_for("think").base_url == "https://answer.example/v1"
    assert config.model_spec_for("look_up").base_url == "https://shared.example/v1"


def test_request_cannot_redirect_endpoint(set_env):
    set_env(BASE_URL="https://gateway.example/v1")

    config = Configuration.from_runnable_config(
        {"configurable": {"base_url": "https://attacker.example/v1"}}
    )

    assert config.model_spec_for("think").base_url == "https://gateway.example/v1"


def test_request_cannot_redirect_role_endpoint(set_env):
    set_env(BASE_URL="https://gateway.example/v1")

    config = Configuration.from_runnable_config(
        {"configurable": {"think_base_url": "https://attacker.example/v1"}}
    )

    assert config.model_spec_for("think").base_url == "https://gateway.example/v1"


@pytest.mark.parametrize("role", ["think", "look_up"])
def test_request_cannot_choose_model(set_env, role):
    """The model decides what every call costs on the operator's key."""
    set_env(**{f"{role.upper()}_MODEL": "operator-model"})

    config = Configuration.from_runnable_config(
        {"configurable": {f"{role}_model": "expensive-model"}}
    )

    assert config.model_for(role) == "operator-model"


def test_request_cannot_enable_full_history():
    """That arm grows the prompt every round, so only the operator runs it."""
    config = Configuration.from_runnable_config(
        {"configurable": {"carry_full_history": True}}
    )

    assert config.carry_full_history is False


def test_env_enables_full_history(set_env):
    set_env(CARRY_FULL_HISTORY="true")

    config = Configuration.from_runnable_config()

    assert config.arm == "full_history"


def test_model_key_from_env(set_env):
    set_env(LLM_API_KEY="server-side-key")

    spec = Configuration.from_runnable_config().model_spec_for("think")

    assert spec.api_key == "server-side-key"


def test_request_cannot_raise_env_ceiling(set_env):
    set_env(MAX_ROUNDS="4")

    config = Configuration.from_runnable_config({"configurable": {"max_rounds": 400}})

    assert config.max_rounds == 4


def test_request_cannot_raise_default_ceiling():
    default = Configuration.model_fields["max_urls_per_look_up"].default

    config = Configuration.from_runnable_config(
        {"configurable": {"max_urls_per_look_up": default + 50}}
    )

    assert config.max_urls_per_look_up == default


def test_the_request_may_lower_a_ceiling(set_env):
    set_env(MAX_ROUNDS="12")

    config = Configuration.from_runnable_config({"configurable": {"max_rounds": 3}})

    assert config.max_rounds == 3


def test_zero_limit_fails_validation():
    """A zero concurrency would build a Semaphore(0) that never lets a target run."""
    with pytest.raises(ValidationError, match="tool_concurrency"):
        Configuration.from_runnable_config({"configurable": {"tool_concurrency": 0}})


def test_request_cannot_raise_wait_ceiling():
    default = Configuration.model_fields["rate_limit_max_wait_seconds"].default

    config = Configuration.from_runnable_config(
        {"configurable": {"rate_limit_max_wait_seconds": 100_000.0}}
    )

    assert config.rate_limit_max_wait_seconds == default


def test_request_cannot_set_request_rate(set_env):
    """Every run in the process shares one rate limiter."""
    set_env(REQUESTS_PER_SECOND="0.5")

    config = Configuration.from_runnable_config(
        {"configurable": {"requests_per_second": 50.0}}
    )

    assert config.requests_per_second == 0.5


def test_request_cannot_raise_token_ceiling():
    """With no MAX_TOKENS in the environment, the field default is the ceiling."""
    default = Configuration.model_fields["max_tokens"].default

    config = Configuration.from_runnable_config(
        {"configurable": {"max_tokens": 10_000_000}}
    )

    assert config.max_tokens == default


@pytest.mark.parametrize(
    "field", ["backoff_base_seconds", "backoff_max_seconds", "backoff_jitter"]
)
def test_request_cannot_set_backoff(field):
    """A zero backoff would retry a rate-limited provider with no wait at all."""
    default = Configuration.model_fields[field].default

    config = Configuration.from_runnable_config({"configurable": {field: 0.0}})

    assert getattr(config, field) == default


def test_text_limit_above_ceiling_capped(set_env):
    """JSON callers send numbers as strings, and a string must not skip the cap."""
    set_env(MAX_ROUNDS="4")

    config = Configuration.from_runnable_config({"configurable": {"max_rounds": "400"}})

    assert config.max_rounds == 4


def test_non_numeric_limit_uses_ceiling(set_env):
    set_env(MAX_ROUNDS="4")

    config = Configuration.from_runnable_config({"configurable": {"max_rounds": "many"}})

    assert config.max_rounds == 4


def test_list_limit_uses_ceiling(set_env):
    set_env(MAX_ROUNDS="4")

    config = Configuration.from_runnable_config({"configurable": {"max_rounds": [400]}})

    assert config.max_rounds == 4


def test_normal_tier_uses_regular_settings():
    config = Configuration.from_runnable_config(
        {"configurable": {"research_depth": "normal"}}
    )

    assert (config.think_reasoning_effort, config.max_urls_per_look_up) == ("medium", 2)


def test_deep_tier_uses_deep_settings():
    config = Configuration.from_runnable_config(
        {"configurable": {"research_depth": "deep"}}
    )

    assert (config.think_reasoning_effort, config.max_urls_per_look_up) == ("high", 5)


def test_env_sets_deep_tier(set_env):
    set_env(DEEP_THINK_REASONING_EFFORT="max", DEEP_MAX_URLS_PER_LOOK_UP="4")

    config = Configuration.from_runnable_config(
        {"configurable": {"research_depth": "deep"}}
    )

    assert (config.think_reasoning_effort, config.max_urls_per_look_up) == ("max", 4)


def test_deep_tier_keeps_lower_batch():
    config = Configuration.from_runnable_config(
        {"configurable": {"research_depth": "deep", "max_urls_per_look_up": 1}}
    )

    assert config.max_urls_per_look_up == 1


def test_deep_tier_allows_more_urls():
    """The deep batch size is the ceiling, so a request between the two stands."""
    config = Configuration.from_runnable_config(
        {"configurable": {"research_depth": "deep", "max_urls_per_look_up": 4}}
    )

    assert config.max_urls_per_look_up == 4


def test_deep_tier_caps_batch():
    config = Configuration.from_runnable_config(
        {"configurable": {"research_depth": "deep", "max_urls_per_look_up": 50}}
    )

    assert config.max_urls_per_look_up == 5


def test_request_cannot_set_effort():
    """Otherwise a normal run could ask for deep spending."""
    config = Configuration.from_runnable_config(
        {"configurable": {"think_reasoning_effort": "xhigh"}}
    )

    assert config.reasoning_effort_for("think") == "medium"


def test_request_cannot_change_deep_tier():
    config = Configuration.from_runnable_config(
        {
            "configurable": {
                "research_depth": "deep",
                "deep_max_urls_per_look_up": 50,
                "deep_think_reasoning_effort": "max",
            }
        }
    )

    assert (config.think_reasoning_effort, config.max_urls_per_look_up) == ("high", 5)


def test_an_unknown_depth_is_refused():
    with pytest.raises(ValidationError):
        Configuration.from_runnable_config({"configurable": {"research_depth": "extreme"}})

