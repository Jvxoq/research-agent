"""The chat model factory and the per-role settings that feed it.

Covers per-role model resolution, the one OpenAI-compatible client every role
builds, the endpoint and credential it receives, the shared rate limiter, the
output-token and reasoning-effort settings, and the structured output reader.
"""

import json

import httpx
import pytest
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agents.research_agent.configuration import ROLES, Configuration
from core.config import get_settings
from core.cost import TokenTracker
from core.llm import (
    ModelSpec,
    StructuredOutputError,
    build_chat_model,
    get_rate_limiter,
    read_structured,
)


def test_roles_have_own_model_and_temperature():
    config = Configuration(
        think_model="m-think",
        look_up_model="m-look-up",
        think_temperature=0.1,
        look_up_temperature=0.2,
    )
    assert [config.model_for(n) for n in ROLES] == ["m-think", "m-look-up"]
    assert [config.temperature_for(n) for n in ROLES] == [0.1, 0.2]


@pytest.mark.parametrize("role", ROLES)
def test_role_model_set_by_env(role, monkeypatch):
    monkeypatch.setenv(f"{role.upper()}_MODEL", "some-model")

    config = Configuration.from_runnable_config()

    assert config.model_for(role) == "some-model"


def test_roles_build_client_offline():
    """One client, and only one."""
    config = Configuration.from_runnable_config()

    clients = {type(build_chat_model(config.model_spec_for(role))) for role in ROLES}

    assert clients == {ChatOpenAI}


def test_no_pace_no_rate_limiter():
    assert get_rate_limiter(None) is None


def test_rate_limiter_is_shared_across_roles():
    """Parallel branches must contend for one budget, not one each."""
    first = get_rate_limiter(0.5)
    second = get_rate_limiter(0.5)
    assert first is second


def test_rate_limiter_rebuilt_on_pace_change():
    first = get_rate_limiter(0.5)
    second = get_rate_limiter(2.0)
    assert first is not second


def test_max_tokens_is_configurable_by_env(monkeypatch):
    monkeypatch.setenv("MAX_TOKENS", "900")
    assert Configuration.from_runnable_config().max_tokens == 900


def test_max_tokens_reaches_the_model():
    """Providers reject oversized requests outright, so this must be applied."""
    config = Configuration(max_tokens=900)
    assert build_chat_model(config.model_spec_for("think")).max_tokens == 900


def test_think_defaults_to_medium_effort():
    spec = Configuration.from_runnable_config().model_spec_for("think")
    assert build_chat_model(spec).reasoning_effort == "medium"


def test_no_effort_keeps_provider_default():
    spec = Configuration.from_runnable_config().model_spec_for("look_up")
    assert build_chat_model(spec).reasoning_effort is None


def test_reasoning_effort_is_configurable_by_env(monkeypatch):
    monkeypatch.setenv("THINK_REASONING_EFFORT", "max")
    spec = Configuration.from_runnable_config().model_spec_for("think")
    assert build_chat_model(spec).reasoning_effort == "max"


def test_gateway_key_from_env(monkeypatch):
    """LLM_API_KEY is provider-agnostic, so switching gateways moves no name."""
    monkeypatch.setenv("LLM_API_KEY", "from-env")

    config = Configuration.from_runnable_config()

    model = build_chat_model(config.model_spec_for("think"))
    assert model.openai_api_key.get_secret_value() == "from-env"


def test_request_cannot_swap_gateway_key(monkeypatch):
    """A caller-supplied key would travel in the body and land in the trace."""
    monkeypatch.setenv("LLM_API_KEY", "from-env")

    config = Configuration.from_runnable_config(
        {"configurable": {"llm_api_key": "from-run"}}
    )

    model = build_chat_model(config.model_spec_for("think"))
    assert model.openai_api_key.get_secret_value() == "from-env"


ZEN_URL = "https://opencode.ai/zen/v1"


def test_no_endpoint_leaves_the_sdk_default():
    """An unset BASE_URL must not invent a URL the OpenAI SDK would reject."""
    model = build_chat_model(Configuration().model_spec_for("think"))

    assert model.openai_api_base is None


def test_base_url_applies_to_every_role():
    config = Configuration(base_url=ZEN_URL)
    for role in ROLES:
        assert build_chat_model(config.model_spec_for(role)).openai_api_base == ZEN_URL


def test_role_url_overrides_global():
    """One role may run against a different gateway than the rest."""
    config = Configuration(
        base_url=ZEN_URL,
        think_base_url="https://other.gateway/v1",
    )
    assert config.base_url_for("look_up") == ZEN_URL
    assert config.base_url_for("think") == "https://other.gateway/v1"
    assert build_chat_model(config.model_spec_for("think")).openai_api_base == (
        "https://other.gateway/v1"
    )


def test_the_endpoint_is_configurable_by_env(monkeypatch):
    """Switching provider is an environment change, never a code change."""
    monkeypatch.setenv("BASE_URL", ZEN_URL)

    config = Configuration.from_runnable_config()

    assert build_chat_model(config.model_spec_for("think")).openai_api_base == ZEN_URL


class _Pick(BaseModel):
    """A one-field schema for the structured reader."""

    choice: str


def test_mapping_validated_into_schema():
    """A provider may return the mapping instead of the model."""
    result = {"raw": None, "parsed": {"choice": "a"}, "parsing_error": None}

    assert read_structured(result, _Pick) == _Pick(choice="a")


def test_parsing_error_raised_with_message():
    result = {"raw": None, "parsed": None, "parsing_error": ValueError("Invalid JSON")}

    with pytest.raises(StructuredOutputError) as caught:
        read_structured(result, _Pick)

    assert str(caught.value) == "The response did not match the schema: Invalid JSON"


def test_empty_parse_refused():
    result = {"raw": None, "parsed": None, "parsing_error": None}

    with pytest.raises(StructuredOutputError) as caught:
        read_structured(result, _Pick)

    assert str(caught.value) == "The response held no structured output."


def test_invalid_mapping_refused():
    result = {"raw": None, "parsed": {"other": "a"}, "parsing_error": None}

    with pytest.raises(StructuredOutputError):
        read_structured(result, _Pick)


def test_non_raw_result_refused():
    with pytest.raises(StructuredOutputError) as caught:
        read_structured(_Pick(choice="a"), _Pick)

    assert str(caught.value) == "Expected the raw and parsed output, got _Pick."


def test_the_model_timeout_reaches_the_client(monkeypatch):
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "42")
    get_settings.cache_clear()
    spec = Configuration.from_runnable_config().model_spec_for("think")

    assert build_chat_model(spec).request_timeout == 42.0


def test_the_client_retries_nothing_itself():
    """Retrier counts every retry. An SDK retry would happen out of its sight."""
    spec = Configuration.from_runnable_config().model_spec_for("think")

    assert build_chat_model(spec).max_retries == 0


def _stream_reply(request: httpx.Request) -> httpx.Response:
    """Stream one reply, with a usage chunk only when the request asked for it."""
    body = json.loads(request.content)
    chunks: list[dict[str, object]] = [
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
            ],
        }
    ]
    if (body.get("stream_options") or {}).get("include_usage"):
        chunks.append(
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            }
        )
    text = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})


async def test_streamed_reply_reports_tokens():
    """A caller streaming messages makes the model stream, and tokens still count."""
    model = build_chat_model(
        ModelSpec(
            model="m",
            temperature=0.0,
            timeout_seconds=5.0,
            base_url="https://api.together.ai/v1",
            api_key="test-key-not-used",
        )
    )
    model.root_async_client._client = httpx.AsyncClient(transport=httpx.MockTransport(_stream_reply))
    tracker = TokenTracker()

    async for _ in model.astream("prompt", {"callbacks": [tracker]}):
        pass

    assert tracker.totals() == {"input_tokens": 100, "output_tokens": 10}
