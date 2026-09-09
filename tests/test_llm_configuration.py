"""Tests enforcing provider-agnostic, per-node LLM configuration.

These are guard-rail tests: they fail if the agent regresses to importing a
provider SDK directly, hard-coding a model, or losing a node's ability to be
configured independently.
"""

import ast
import os
import pathlib

import pytest

from agent.configuration import SUPPORTED_PROVIDERS, Configuration
from agent.llm import NODES, build_chat_model, get_rate_limiter, reset_rate_limiter

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "agent"
GRAPH = SRC / "graph.py"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Clear agent env vars so tests are not affected by the developer's shell."""
    for name in list(os.environ):
        if name.lower() in Configuration.model_fields:
            monkeypatch.delenv(name, raising=False)
    reset_rate_limiter()
    yield
    reset_rate_limiter()


# ----------------------------------------------------------------------
# The graph must not depend on any single provider
# ----------------------------------------------------------------------


def test_graph_does_not_import_a_provider_sdk():
    """graph.py must build models through the factory, not a provider SDK."""
    tree = ast.parse(GRAPH.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)

    provider_packages = {
        "langchain_groq",
        "langchain_anthropic",
        "langchain_openai",
        "groq",
        "anthropic",
        "openai",
    }
    leaked = imported & provider_packages
    assert not leaked, f"graph.py imports provider SDKs directly: {sorted(leaked)}"


def test_graph_builds_every_model_through_the_factory():
    """Each of the four nodes must call build_chat_model exactly once."""
    source = GRAPH.read_text()
    assert source.count("build_chat_model(") == len(NODES), (
        "expected one build_chat_model call per node; "
        "a node is constructing its model some other way"
    )
    for node in NODES:
        assert f'build_chat_model("{node}"' in source, f"node {node} not wired"


# ----------------------------------------------------------------------
# Per-node provider and model resolution
# ----------------------------------------------------------------------


def test_default_provider_applies_to_every_node():
    config = Configuration(llm_provider="anthropic")
    for node in NODES:
        assert config.provider_for(node) == "anthropic"


def test_node_provider_overrides_the_default():
    config = Configuration(llm_provider="groq", reflection_provider="anthropic")
    assert config.provider_for("reflection") == "anthropic"
    assert config.provider_for("answer") == "groq"


def test_every_node_has_its_own_model_and_temperature():
    config = Configuration(
        query_generator_model="m-query",
        web_research_model="m-research",
        reflection_model="m-reflect",
        answer_model="m-answer",
        query_generator_temperature=0.1,
        web_research_temperature=0.2,
        reflection_temperature=0.3,
        answer_temperature=0.4,
    )
    assert [config.model_for(n) for n in NODES] == [
        "m-query",
        "m-research",
        "m-reflect",
        "m-answer",
    ]
    assert [config.temperature_for(n) for n in NODES] == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.parametrize("node", NODES)
def test_each_node_is_independently_configurable_by_env(node, monkeypatch):
    monkeypatch.setenv(f"{node.upper()}_PROVIDER", "anthropic")
    monkeypatch.setenv(f"{node.upper()}_MODEL", "some-model")
    config = Configuration.from_runnable_config()
    assert config.provider_for(node) == "anthropic"
    assert config.model_for(node) == "some-model"
    for other in NODES:
        if other != node:
            assert config.provider_for(other) == config.llm_provider


def test_env_overrides_runnable_config(monkeypatch):
    monkeypatch.setenv("ANSWER_MODEL", "from-env")
    config = Configuration.from_runnable_config(
        {"configurable": {"answer_model": "from-config"}}
    )
    assert config.model_for("answer") == "from-env"


# ----------------------------------------------------------------------
# Factory behaviour
# ----------------------------------------------------------------------


def test_unknown_node_is_rejected():
    with pytest.raises(ValueError, match="Unknown node"):
        build_chat_model("not_a_node", Configuration())


def test_unsupported_provider_is_rejected():
    config = Configuration(llm_provider="not_a_provider")
    with pytest.raises(ValueError, match="Unsupported provider"):
        build_chat_model("answer", config)


def test_groq_is_supported_and_builds():
    """The default provider must construct without network access."""
    os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
    model = build_chat_model("answer", Configuration(llm_provider="groq"))
    assert model.__class__.__name__ == "ChatGroq"


def test_supported_providers_are_declared():
    for provider in ("groq", "anthropic", "openai"):
        assert provider in SUPPORTED_PROVIDERS


# ----------------------------------------------------------------------
# Shared rate limiter
# ----------------------------------------------------------------------


def test_rate_limiter_is_disabled_by_default():
    assert get_rate_limiter(None) is None
    assert Configuration().requests_per_second is None


def test_rate_limiter_is_shared_across_nodes():
    """Parallel branches must contend for one budget, not one each."""
    first = get_rate_limiter(0.5)
    second = get_rate_limiter(0.5)
    assert first is second


def test_rate_limiter_is_rebuilt_when_the_pace_changes():
    first = get_rate_limiter(0.5)
    second = get_rate_limiter(2.0)
    assert first is not second


# ----------------------------------------------------------------------
# Output-token ceiling
# ----------------------------------------------------------------------


def test_max_tokens_is_unset_by_default():
    assert Configuration().max_tokens is None


def test_max_tokens_is_configurable_by_env(monkeypatch):
    monkeypatch.setenv("MAX_TOKENS", "900")
    assert Configuration.from_runnable_config().max_tokens == 900


def test_max_tokens_reaches_the_model():
    """Providers reject oversized requests outright, so this must be applied."""
    os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
    model = build_chat_model("answer", Configuration(max_tokens=900))
    assert model.max_tokens == 900


# ----------------------------------------------------------------------
# Together / DeepSeek path
# ----------------------------------------------------------------------


def test_together_is_a_supported_provider():
    assert "together" in SUPPORTED_PROVIDERS


def test_together_builds_a_chat_model():
    """The provider that will actually run the benchmark must construct."""
    os.environ.setdefault("TOGETHER_API_KEY", "test-key-not-used")
    config = Configuration(
        llm_provider="together", answer_model="deepseek-ai/DeepSeek-V3"
    )
    model = build_chat_model("answer", config)
    assert model.__class__.__name__ == "ChatTogether"


def test_switching_all_nodes_to_together_is_one_setting(monkeypatch):
    os.environ.setdefault("TOGETHER_API_KEY", "test-key-not-used")
    monkeypatch.setenv("LLM_PROVIDER", "together")
    monkeypatch.setenv("QUERY_GENERATOR_MODEL", "deepseek-ai/DeepSeek-V3")
    monkeypatch.setenv("WEB_RESEARCH_MODEL", "deepseek-ai/DeepSeek-V3")
    monkeypatch.setenv("REFLECTION_MODEL", "deepseek-ai/DeepSeek-V3")
    monkeypatch.setenv("ANSWER_MODEL", "deepseek-ai/DeepSeek-V3")
    config = Configuration.from_runnable_config()
    for node in NODES:
        assert config.provider_for(node) == "together"
        assert build_chat_model(node, config).__class__.__name__ == "ChatTogether"


# ----------------------------------------------------------------------
# OpenAI-compatible endpoints (OpenCode Zen, OpenRouter, vLLM, ...)
# ----------------------------------------------------------------------

ZEN_URL = "https://opencode.ai/zen/v1"


def test_openai_compatible_is_a_supported_provider():
    assert "openai_compatible" in SUPPORTED_PROVIDERS


def test_openai_compatible_builds_an_openai_client():
    """The wire format is OpenAI's, so the client must be ChatOpenAI."""
    os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
    config = Configuration(llm_provider="openai_compatible", api_key="zen-key")
    model = build_chat_model("answer", config)
    assert model.__class__.__name__ == "ChatOpenAI"


def test_base_url_reaches_the_client():
    os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
    config = Configuration(
        llm_provider="openai_compatible", base_url=ZEN_URL, api_key="zen-key"
    )
    assert build_chat_model("answer", config).openai_api_base == ZEN_URL


def test_base_url_applies_to_every_node():
    os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
    config = Configuration(
        llm_provider="openai_compatible", base_url=ZEN_URL, api_key="zen-key"
    )
    for node in NODES:
        assert build_chat_model(node, config).openai_api_base == ZEN_URL


def test_node_base_url_overrides_the_global_one():
    """One node may run against a different gateway than the rest."""
    os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
    config = Configuration(
        llm_provider="openai_compatible",
        base_url=ZEN_URL,
        answer_base_url="https://other.gateway/v1",
        api_key="k",
    )
    assert config.base_url_for("reflection") == ZEN_URL
    assert config.base_url_for("answer") == "https://other.gateway/v1"
    assert build_chat_model("answer", config).openai_api_base == (
        "https://other.gateway/v1"
    )


def test_endpoint_is_configurable_by_env(monkeypatch):
    os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("BASE_URL", ZEN_URL)
    monkeypatch.setenv("API_KEY", "zen-key")
    monkeypatch.setenv("ANSWER_MODEL", "zen/some-model")
    config = Configuration.from_runnable_config()
    assert config.llm_provider == "openai_compatible"
    assert build_chat_model("answer", config).openai_api_base == ZEN_URL


def test_base_url_is_unset_by_default():
    assert Configuration().base_url is None
    assert Configuration().base_url_for("answer") is None


def test_sdk_providers_ignore_base_url():
    """A base_url must not leak into providers with their own SDK."""
    os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
    config = Configuration(llm_provider="groq", base_url=ZEN_URL)
    model = build_chat_model("answer", config)
    assert model.__class__.__name__ == "ChatGroq"
    assert ZEN_URL not in str(getattr(model, "groq_api_base", "") or "")
