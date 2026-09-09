"""Tests enforcing LangSmith tracing configuration."""

import os

import pytest

from agent.configuration import Configuration
from agent.tracing import (
    API_KEY_ENV_VAR,
    ENDPOINT_ENV_VAR,
    PROJECT_ENV_VAR,
    TRACING_ENV_VAR,
    configure_tracing,
    tracing_enabled,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (TRACING_ENV_VAR, API_KEY_ENV_VAR, PROJECT_ENV_VAR, ENDPOINT_ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.lower() in Configuration.model_fields:
            monkeypatch.delenv(name, raising=False)
    yield


def test_tracing_is_off_by_default():
    config = Configuration()
    assert config.langsmith_tracing is False
    assert configure_tracing(config) is False
    assert os.environ[TRACING_ENV_VAR] == "false"


def test_tracing_without_an_api_key_does_not_activate():
    """A missing key must degrade quietly, never fail the run."""
    config = Configuration(langsmith_tracing=True)
    assert configure_tracing(config) is False
    assert os.environ[TRACING_ENV_VAR] == "false"


def test_tracing_activates_with_a_key(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "ls-test-key")
    config = Configuration(langsmith_tracing=True, langsmith_project="proj-x")
    assert configure_tracing(config) is True
    assert os.environ[TRACING_ENV_VAR] == "true"
    assert os.environ[PROJECT_ENV_VAR] == "proj-x"
    assert os.environ[ENDPOINT_ENV_VAR] == "https://api.smith.langchain.com"


def test_tracing_enabled_reports_effective_state(monkeypatch):
    assert tracing_enabled(Configuration(langsmith_tracing=True)) is False
    monkeypatch.setenv(API_KEY_ENV_VAR, "ls-test-key")
    assert tracing_enabled(Configuration(langsmith_tracing=True)) is True
    assert tracing_enabled(Configuration(langsmith_tracing=False)) is False


def test_tracing_is_configurable_by_env(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "from-env")
    config = Configuration.from_runnable_config()
    assert config.langsmith_tracing is True
    assert config.langsmith_project == "from-env"


def test_custom_endpoint_is_applied(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV_VAR, "ls-test-key")
    config = Configuration(
        langsmith_tracing=True, langsmith_endpoint="https://eu.smith.langchain.com"
    )
    assert configure_tracing(config) is True
    assert os.environ[ENDPOINT_ENV_VAR] == "https://eu.smith.langchain.com"


# ----------------------------------------------------------------------
# Run identification — a trace without these is not comparable
# ----------------------------------------------------------------------


def test_metadata_records_every_node_model():
    from agent.llm import NODES
    from agent.tracing import run_metadata

    config = Configuration(
        llm_provider="together",
        answer_model="deepseek-ai/DeepSeek-V3",
        reflection_provider="groq",
    )
    meta = run_metadata(config, query_id="769")
    assert meta["query_id"] == "769"
    assert meta["llm_provider"] == "together"
    assert meta["answer.model"] == "deepseek-ai/DeepSeek-V3"
    assert meta["reflection.provider"] == "groq"
    for node in NODES:
        assert f"{node}.model" in meta
        assert f"{node}.provider" in meta


def test_tags_identify_provider_and_models():
    from agent.tracing import run_tags

    config = Configuration(
        llm_provider="together", answer_model="deepseek-ai/DeepSeek-V3"
    )
    tags = run_tags(config, "browsecomp-plus")
    assert "provider:together" in tags
    assert "model:deepseek-ai/DeepSeek-V3" in tags
    assert "browsecomp-plus" in tags
