"""Process-wide settings.

This is the only module in the repository that reads environment variables. Every other module receives what it needs as an argument, or asks
:func:`get_settings` for it.

Almost every field defaults to ``None`` on purpose. ``None`` means "this
variable is not set in the environment", which lets an agent's per-run
configuration tell an unset variable apart from a real value and fall back to
its own field default.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-sourced settings shared by every agent."""

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", case_sensitive=False
    )

    # Anything but "development" refuses an unauthenticated request. Local
    # `langgraph dev` sets ENVIRONMENT=development in .env.
    ENVIRONMENT: str = "production"

    # Search. The endpoint is set here only, because whoever names it receives
    # the key. A load test points it at a local fake.
    TAVILY_API_KEY: str | None = None
    TAVILY_BASE_URL: str | None = None

    # Default endpoint. It chooses the provider.
    BASE_URL: str | None = None
    LLM_API_KEY: str | None = None

    # Throughput and ceilings
    MAX_RETRIES: int | None = None
    MODEL_TIMEOUT_SECONDS: float | None = None
    MAX_TOKENS: int | None = None
    REQUESTS_PER_SECOND: float | None = None
    # USD per million tokens as {"model": [input, output]}, in JSON. Merged over
    # the built-in table, so a new model is priced without a code change.
    MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] | None = None

    # Roles
    THINK_MODEL: str | None = None
    THINK_BASE_URL: str | None = None
    THINK_TEMPERATURE: float | None = None
    THINK_REASONING_EFFORT: str | None = None

    LOOK_UP_MODEL: str | None = None
    LOOK_UP_BASE_URL: str | None = None
    LOOK_UP_TEMPERATURE: float | None = None
    LOOK_UP_REASONING_EFFORT: str | None = None

    # The round loop
    MAX_ROUNDS: int | None = None
    MAX_DECISION_RETRIES: int | None = None
    ANSWER_STYLE: Literal["short", "report"] | None = None
    CURRENT_DATE_OVERRIDE: str | None = None

    # Tools
    MAX_QUERIES_PER_SEARCH: int | None = None
    MAX_URLS_PER_LOOK_UP: int | None = None
    LOOK_UP_MAX_PAGE_CHARS: int | None = None
    LOOK_UP_SUMMARY_MAX_CHARS: int | None = None
    MAX_SUMMARY_RETRIES: int | None = None
    TOOL_TIMEOUT_SECONDS: float | None = None
    TARGET_DEADLINE_SECONDS: float | None = None
    TOOL_CONCURRENCY: int | None = None
    MAX_TOOL_CHARS: int | None = None
    MAX_SEARCH_RESULTS: int | None = None
    MAX_ERROR_CHARS: int | None = None

    # Prompt limits
    MAX_REPORT_CHARS: int | None = None
    WARN_PROMPT_CHARS: int | None = None

    # Backoff
    RATE_LIMIT_MAX_WAIT_SECONDS: float | None = None
    BACKOFF_BASE_SECONDS: float | None = None
    BACKOFF_MAX_SECONDS: float | None = None
    BACKOFF_JITTER: float | None = None

    # Research depth. The caller names the tier. The environment sets what
    # the deep tier spends.
    RESEARCH_DEPTH: Literal["normal", "deep"] | None = None
    DEEP_THINK_REASONING_EFFORT: str | None = None
    DEEP_MAX_URLS_PER_LOOK_UP: int | None = None

    # Ablation
    CARRY_FULL_HISTORY: bool | None = None

    # Observability
    LANGSMITH_TRACING: bool | None = None
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str | None = None
    LANGSMITH_ENDPOINT: str | None = None

    # Server auth. The web application signs its users' JWTs with this
    # secret, so this server verifies them with the same one.
    JWT_SECRET: str | None = None
    JWT_ALGORITHM: str = "HS256"
    JWT_ISSUER: str | None = None
    JWT_AUDIENCE: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings."""
    return Settings()

