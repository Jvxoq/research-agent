"""LangSmith tracing setup.

LangChain reads tracing settings from environment variables at call time, so
enabling tracing means exporting the right variables before the graph runs.
This module does that from :class:`~agent.configuration.Configuration` so the
setting lives with the rest of the agent's configuration rather than only in
the shell.

Tracing is off unless both ``LANGSMITH_TRACING`` is truthy and
``LANGSMITH_API_KEY`` is present. A missing key is reported once and ignored —
it never fails a run.
"""

from __future__ import annotations

import os

from loguru import logger

from agent.configuration import Configuration

TRACING_ENV_VAR = "LANGSMITH_TRACING"
API_KEY_ENV_VAR = "LANGSMITH_API_KEY"
PROJECT_ENV_VAR = "LANGSMITH_PROJECT"
ENDPOINT_ENV_VAR = "LANGSMITH_ENDPOINT"


def tracing_enabled(config: Configuration) -> bool:
    """Report whether tracing will actually send anything.

    Args:
        config: Resolved agent configuration.

    Returns:
        True when tracing is requested *and* an API key is available.
    """
    return bool(config.langsmith_tracing and os.environ.get(API_KEY_ENV_VAR))


def configure_tracing(config: Configuration) -> bool:
    """Apply LangSmith settings to the environment.

    Args:
        config: Resolved agent configuration.

    Returns:
        True if tracing is active, False otherwise.
    """
    if not config.langsmith_tracing:
        os.environ[TRACING_ENV_VAR] = "false"
        return False

    if not os.environ.get(API_KEY_ENV_VAR):
        logger.warning(
            f"langsmith_tracing is on but {API_KEY_ENV_VAR} is not set — "
            "traces will not be sent"
        )
        os.environ[TRACING_ENV_VAR] = "false"
        return False

    os.environ[TRACING_ENV_VAR] = "true"
    os.environ[PROJECT_ENV_VAR] = config.langsmith_project
    os.environ[ENDPOINT_ENV_VAR] = config.langsmith_endpoint

    logger.info(f"langsmith tracing on — project={config.langsmith_project}")
    return True


def run_metadata(config: Configuration, **extra: object) -> dict:
    """Describe a run so its LangSmith trace is identifiable.

    Without this a trace shows timings but not which models produced them,
    which makes comparing two configurations impossible.

    Args:
        config: Resolved agent configuration.
        **extra: Additional fields to attach (e.g. ``query_id``).

    Returns:
        A metadata mapping for ``RunnableConfig``.
    """
    from agent.llm import NODES

    metadata: dict = {
        "llm_provider": config.llm_provider,
        "max_search_results": config.max_search_results,
        "max_research_loops": config.max_research_loops,
        "requests_per_second": config.requests_per_second,
        "max_tokens": config.max_tokens,
    }
    for node in NODES:
        metadata[f"{node}.provider"] = config.provider_for(node)
        metadata[f"{node}.model"] = config.model_for(node)
    metadata.update(extra)
    return metadata


def run_tags(config: Configuration, *extra: str) -> list[str]:
    """Return LangSmith tags for filtering runs in the UI.

    Args:
        config: Resolved agent configuration.
        *extra: Additional tags.

    Returns:
        Tags identifying the provider and the distinct models in play.
    """
    from agent.llm import NODES

    models = sorted({config.model_for(node) for node in NODES})
    return [f"provider:{config.llm_provider}", *(f"model:{m}" for m in models), *extra]
