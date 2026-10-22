"""LangSmith tracing comes from settings and never touches the environment."""

import os

import pytest
from langsmith.run_helpers import get_tracing_context
from langsmith.utils import tracing_is_enabled

from agents.research_agent.configuration import (
    DEFAULT_LOOK_UP_MODEL,
    Configuration,
)
from core.config import Settings
from core.tracing import tracing

pytestmark = pytest.mark.usefixtures("langsmith_client")


@pytest.fixture
def requested_with_key() -> Settings:
    """Settings that ask for tracing and hold a key."""
    return Settings(LANGSMITH_TRACING=True, LANGSMITH_API_KEY="ls-test-key")


def test_tracing_off_by_default():
    with tracing(Settings()):
        assert tracing_is_enabled() is False


def test_tracing_off_without_key():
    """A missing key must degrade quietly, never fail the run."""
    with tracing(Settings(LANGSMITH_TRACING=True)):
        assert tracing_is_enabled() is False


def test_tracing_on_with_key(requested_with_key: Settings):
    with tracing(requested_with_key):
        assert tracing_is_enabled() is True


def test_block_reports_tracing_on(requested_with_key: Settings):
    with tracing(requested_with_key) as active:
        assert active is True


def test_shell_variable_cannot_enable_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    with tracing(Settings(LANGSMITH_TRACING=False, LANGSMITH_API_KEY="ls-test-key")):
        assert tracing_is_enabled() is False


def test_traces_go_to_the_configured_project():
    settings = Settings(
        LANGSMITH_TRACING=True,
        LANGSMITH_API_KEY="ls-test-key",
        LANGSMITH_PROJECT="proj-x",
    )

    with tracing(settings):
        assert get_tracing_context()["project_name"] == "proj-x"


def test_traces_go_to_the_configured_endpoint():
    settings = Settings(
        LANGSMITH_TRACING=True,
        LANGSMITH_API_KEY="ls-test-key",
        LANGSMITH_ENDPOINT="https://eu.smith.langchain.com",
    )

    with tracing(settings):
        assert get_tracing_context()["client"].api_url == (
            "https://eu.smith.langchain.com"
        )


def test_opening_tracing_writes_no_environment_variable(
    requested_with_key: Settings,
):
    """The environment belongs to the whole process, not to one run."""
    before = dict(os.environ)

    with tracing(requested_with_key):
        assert dict(os.environ) == before


@pytest.fixture
def mixed_config() -> Configuration:
    """A run whose two roles use different endpoints and models."""
    return Configuration(
        base_url="https://api.together.ai/v1",
        think_model="zai-org/GLM-5.3-Flash",
        look_up_base_url="https://api.groq.com/openai/v1",
    )


def test_metadata_names_role_endpoints_models(
    mixed_config: Configuration,
):
    """A trace that does not say what ran is not comparable to another run."""
    expected = {
        "query_id": "769",
        "think.base_url": "https://api.together.ai/v1",
        "think.model": "zai-org/GLM-5.3-Flash",
        "look_up.base_url": "https://api.groq.com/openai/v1",
        "look_up.model": DEFAULT_LOOK_UP_MODEL,
    }

    meta = mixed_config.run_metadata(query_id="769")

    assert {key: meta[key] for key in expected} == expected


def test_tags_name_arm_models_extras(
    mixed_config: Configuration,
):
    tags = mixed_config.run_tags("browsecomp")

    assert tags == [
        "arm:report",
        f"model:{DEFAULT_LOOK_UP_MODEL}",
        "model:zai-org/GLM-5.3-Flash",
        "browsecomp",
    ]
