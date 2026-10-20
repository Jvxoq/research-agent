"""Chat model construction through the one OpenAI-compatible client.

Nodes never import a provider SDK. They ask this module for a model, described
by a :class:`ModelSpec` their agent's configuration produced. Every model goes
through the OpenAI-compatible client, so the endpoint is the provider: Together
AI, Groq or any other gateway is chosen by ``BASE_URL`` alone. Switching
provider is an environment change, never a code change.

A single process-wide rate limiter is shared by every model instance, so
pacing holds across parallel branches rather than per-node.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from loguru import logger
from openai import ContentFilterFinishReasonError, LengthFinishReasonError
from pydantic import BaseModel, SecretStr, ValidationError

# Raised by the OpenAI client's structured output call when the reply stopped
# before the schema was complete. Nothing is wrong with the connection, so these
# are not transport errors. Agents may not import a provider SDK, so they read
# the tuple from here.
UNUSABLE_OUTPUT_ERRORS: tuple[type[Exception], ...] = (
    LengthFinishReasonError,
    ContentFilterFinishReasonError,
)


class StructuredOutputError(ValueError):
    """Raised when a structured output call returned nothing the schema accepts."""


def read_structured[Schema: BaseModel](result: object, schema: type[Schema]) -> Schema:
    """Read one ``with_structured_output(..., include_raw=True)`` result.

    ``include_raw`` hands a parsing failure back as a value instead of raising
    it inside the call. A caller that retries the call on transport errors
    therefore never mistakes the model's own text for a network failure.

    Args:
        result: The raw-and-parsed mapping the structured call returned.
        schema: The model the parsed output must validate as.

    Returns:
        The parsed output as an instance of ``schema``.

    Raises:
        StructuredOutputError: If the result is not the mapping, holds a parsing
            error, holds no parsed output, or fails to validate as ``schema``.
    """
    if not isinstance(result, dict):
        raise StructuredOutputError(
            f"Expected the raw and parsed output, got {type(result).__name__}."
        )
    parsing_error = result.get("parsing_error")
    if parsing_error is not None:
        raise StructuredOutputError(
            f"The response did not match the schema: {parsing_error}"
        )
    parsed = result.get("parsed")
    if parsed is None:
        raise StructuredOutputError("The response held no structured output.")

    # A provider may hand back the mapping instead of the model.
    if isinstance(parsed, schema):
        return parsed
    try:
        return schema.model_validate(parsed)
    except ValidationError as error:
        raise StructuredOutputError(
            f"The response did not match the schema: {error}"
        ) from error

_rate_limiter: BaseRateLimiter | None = None
_rate_limiter_rps: float | None = None


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to build one chat model."""

    model: str
    temperature: float
    timeout_seconds: float
    base_url: str | None = None
    api_key: str | None = None
    max_tokens: int | None = None
    requests_per_second: float | None = None
    reasoning_effort: str | None = None


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


def build_chat_model(spec: ModelSpec) -> BaseChatModel:
    """Build a chat model from a spec.

    Args:
        spec: Model, endpoint and call settings for one role.

    Returns:
        An OpenAI-compatible chat model. With no ``base_url`` it calls the
        OpenAI SDK's own default endpoint.
    """
    logger.info(
        f"model={spec.model} temp={spec.temperature} "
        f"endpoint={spec.base_url or 'sdk default'}"
    )
    return ChatOpenAI(
        model=spec.model,
        temperature=spec.temperature,
        timeout=spec.timeout_seconds,
        # Retrier retries and counts every retry. SDK retries would hide them.
        max_retries=0,
        # A caller streaming messages makes the client stream too. The client
        # asks for usage in a stream only on OpenAI's own endpoint, so every
        # other provider would report 0 tokens without this.
        stream_usage=True,
        base_url=spec.base_url,
        api_key=SecretStr(spec.api_key) if spec.api_key else None,
        max_completion_tokens=spec.max_tokens,
        # None leaves the provider's own default.
        reasoning_effort=spec.reasoning_effort,
        rate_limiter=get_rate_limiter(spec.requests_per_second),
    )
