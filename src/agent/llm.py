"""Provider-agnostic chat model construction.

Nodes never import a provider SDK. They ask this module for the model that
their node is configured to use, and the provider is resolved at runtime from
:class:`~agent.configuration.Configuration`.

Swapping every node from Groq to Anthropic is one environment variable::

    LLM_PROVIDER=anthropic ANSWER_MODEL=claude-opus-5 ...

A single process-wide rate limiter is shared by every model instance, so
pacing holds across parallel branches rather than per-node.
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter
from loguru import logger

from agent.configuration import (
    OPENAI_COMPATIBLE_PROVIDERS,
    SUPPORTED_PROVIDERS,
    Configuration,
)

# Nodes that own an independently configurable model.
NODES = ("query_generator", "web_research", "reflection", "answer")

_rate_limiter: BaseRateLimiter | None = None
_rate_limiter_rps: float | None = None


def get_rate_limiter(requests_per_second: float | None) -> BaseRateLimiter | None:
    """Return the shared rate limiter for the configured pace.

    One limiter is shared by every model in the process so that parallel graph
    branches contend for the same budget. Returns ``None`` when pacing is
    disabled.

    Args:
        requests_per_second: Target pace, or ``None`` to disable pacing.

    Returns:
        The shared limiter, or ``None``.
    """
    global _rate_limiter, _rate_limiter_rps

    if not requests_per_second:
        return None

    if _rate_limiter is None or _rate_limiter_rps != requests_per_second:
        _rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=0.5,
            max_bucket_size=1,
        )
        _rate_limiter_rps = requests_per_second
        logger.info(f"rate limiter set to {requests_per_second} req/s")

    return _rate_limiter


def reset_rate_limiter() -> None:
    """Drop the shared limiter. Intended for tests."""
    global _rate_limiter, _rate_limiter_rps
    _rate_limiter = None
    _rate_limiter_rps = None


def build_chat_model(node: str, config: Configuration) -> BaseChatModel:
    """Build the chat model a node is configured to use.

    Args:
        node: One of :data:`NODES`.
        config: Resolved agent configuration.

    Returns:
        A chat model for the node's configured provider and model.

    Raises:
        ValueError: If the node is unknown or the provider is unsupported.
    """
    if node not in NODES:
        raise ValueError(f"Unknown node {node!r}. Expected one of {NODES}.")

    provider = config.provider_for(node)
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider {provider!r} for node {node!r}. "
            f"Expected one of {SUPPORTED_PROVIDERS}."
        )

    model = config.model_for(node)
    temperature = config.temperature_for(node)

    logger.info(f"node={node} provider={provider} model={model} temp={temperature}")

    kwargs: dict = {
        "model": model,
        "model_provider": provider,
        "temperature": temperature,
        "max_retries": config.max_retries,
    }

    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        # "openai_compatible" is our own name for "any gateway speaking the
        # OpenAI wire format"; the client that talks to it is the OpenAI one.
        kwargs["model_provider"] = "openai"

        base_url = config.base_url_for(node)
        if base_url:
            kwargs["base_url"] = base_url
        if config.api_key:
            kwargs["api_key"] = config.api_key

        logger.info(f"node={node} endpoint={base_url or 'provider default'}")

    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens

    limiter = get_rate_limiter(config.requests_per_second)
    if limiter is not None:
        kwargs["rate_limiter"] = limiter

    return init_chat_model(**kwargs)
