"""Per-run configuration for the research agent.

Two roles call a model: ``think`` makes the round decision, ``look_up``
summarises a fetched page against a goal. Each role has its own model,
endpoint, temperature and reasoning effort. An unset endpoint falls back to the
run-wide ``base_url``. The endpoint is the provider: every role goes through the
one OpenAI-compatible client.

Three sources feed a field, highest priority first: the ``configurable`` mapping
on the run's ``RunnableConfig``, then process settings from
:func:`core.config.get_settings`, then the field default declared here.

Defaults are sized for a free Tavily tier.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from core.config import get_settings
from core.llm import ModelSpec
from core.retry import BackoffPolicy

Role = Literal["think", "look_up"]

# Roles that own an independently configurable model.
ROLES: tuple[Role, ...] = ("think", "look_up")

DEFAULT_LOOK_UP_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_THINK_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"

# Fields a caller may never set. The ``configurable`` mapping arrives in the
# request body, so in production it is written by someone who is not the
# operator. An endpoint decides where this server sends a credential it holds:
# a model endpoint carries LLM_API_KEY and the run's prompts. The request rate
# paces the one rate limiter every run in the process shares. The backoff shape
# decides how hard a rate-limited provider is hit again, and a smaller value
# there is not cheaper, it is faster. A model, a reasoning effort and the deep
# tier's values decide what a run spends, and a caller picks a tier by name only,
# so a normal run cannot ask for deep spending. The full history arm grows the
# prompt every round, so only the operator runs the ablation. The prompt warning
# threshold is the operator's log setting. These come from the environment alone,
# whatever the request asks for.
SERVER_ONLY_FIELDS = frozenset(
    {
        "base_url",
        *(f"{role}_model" for role in ROLES),
        *(f"{role}_base_url" for role in ROLES),
        *(f"{role}_reasoning_effort" for role in ROLES),
        "carry_full_history",
        "warn_prompt_chars",
        "deep_think_reasoning_effort",
        "deep_max_urls_per_look_up",
        "requests_per_second",
        "backoff_base_seconds",
        "backoff_max_seconds",
        "backoff_jitter",
    }
)

# Fields where the environment sets a ceiling rather than a default. A caller
# may lower one to run something cheaper. Raising one spends the operator's
# provider and Tavily credit, so a higher value is refused and logged.
CEILING_FIELDS = frozenset(
    {
        "max_rounds",
        "max_decision_retries",
        "max_tokens",
        "max_queries_per_search",
        "max_urls_per_look_up",
        "max_search_results",
        "tool_concurrency",
        "tool_timeout_seconds",
        "target_deadline_seconds",
        "look_up_max_page_chars",
        "max_summary_retries",
        "max_tool_chars",
        "max_error_chars",
        "max_report_chars",
        "max_retries",
        "model_timeout_seconds",
        "rate_limit_max_wait_seconds",
        "look_up_summary_max_chars",
    }
)


def _within_ceiling(
    name: str, requested: object, ceiling: float | None
) -> object:
    """Return the caller's value when it does not exceed the operator's.

    Args:
        name: The field being resolved, used in the log line.
        requested: What the request asked for.
        ceiling: What the environment allows, or None when it set none.

    Returns:
        The requested value, or the ceiling when the request asked for more.
    """
    if ceiling is None:
        return requested
    # The request body carries JSON, so a limit can arrive as a string.
    # Comparing that against a number raises rather than capping.
    asked: float | None = None
    if isinstance(requested, (int, float, str)):
        try:
            asked = int(requested) if isinstance(ceiling, int) else float(requested)
        except ValueError:
            asked = None
    if asked is None:
        logger.warning(
            f"{name}: request sent {type(requested).__name__}, using the ceiling"
        )
        return ceiling
    if asked > ceiling:
        logger.warning(f"{name}: request asked for {asked}, capped at {ceiling}")
        return ceiling
    return asked


class Configuration(BaseModel):
    """The configuration for one research agent run."""

    # A misspelt name in code fails at construction. A request's unknown keys
    # never get here: from_runnable_config reads declared names only, because
    # the server puts its own keys, such as thread_id, in the same mapping.
    model_config = ConfigDict(extra="forbid")

    base_url: str | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "OpenAI-compatible endpoint every role calls. It chooses the "
                "provider. Any role may override."
            )
        },
    )

    think_model: str = Field(
        default=DEFAULT_THINK_MODEL,
        json_schema_extra={"description": "Model that writes the report and picks the action."},
    )
    think_base_url: str | None = Field(
        default=None,
        json_schema_extra={"description": "Endpoint override for the think role."},
    )
    think_temperature: float = Field(
        default=1.0,
        json_schema_extra={
            "description": (
                "Sampling temperature for the round decision. DeepSeek advises 1.0 "
                "in thinking mode, because a lower value can cut the reasoning short."
            )
        },
    )
    think_reasoning_effort: str | None = Field(
        default="medium",
        json_schema_extra={
            "description": "How long the think model reasons first. None keeps the provider default."
        },
    )

    look_up_model: str = Field(
        default=DEFAULT_LOOK_UP_MODEL,
        json_schema_extra={
            "description": (
                "Model that reduces a fetched page to what serves the goal. "
                "Use a cheap fast model; it runs once per page visited."
            )
        },
    )
    look_up_base_url: str | None = Field(
        default=None,
        json_schema_extra={"description": "Endpoint override for the look_up role."},
    )
    look_up_temperature: float = Field(
        default=0.0,
        json_schema_extra={"description": "Sampling temperature for the summariser."},
    )
    look_up_reasoning_effort: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "How long the summariser reasons first. None keeps the provider default."
        },
    )

    max_rounds: int = Field(
        default=12,
        ge=1,
        json_schema_extra={"description": "Hard cap on research rounds in one run."},
    )
    max_decision_retries: int = Field(
        default=2,
        ge=0,
        json_schema_extra={
            "description": (
                "Retries when a structured response fails the decision check. "
                "Never backed off: waiting does not improve the output."
            )
        },
    )
    answer_style: Literal["short", "report"] = Field(
        default="short",
        json_schema_extra={
            "description": "'short' for benchmarks, 'report' for a rich product answer."
        },
    )
    current_date_override: str | None = Field(
        default=None,
        json_schema_extra={"description": "Freeze the date the agent reasons against."},
    )

    max_queries_per_search: int = Field(
        default=3,
        ge=1,
        json_schema_extra={"description": "Batch ceiling for one search action."},
    )
    max_urls_per_look_up: int = Field(
        default=2,
        ge=1,
        json_schema_extra={
            "description": (
                "Batch ceiling for one look_up action. Each URL spends one "
                "Tavily extraction and one summariser call."
            )
        },
    )
    max_search_results: int = Field(
        default=5,
        ge=1,
        json_schema_extra={"description": "Results returned per search query."},
    )
    look_up_max_page_chars: int = Field(
        default=60000,
        ge=1,
        json_schema_extra={"description": "Page text sent to the summariser."},
    )
    look_up_summary_max_chars: int = Field(
        default=4000,
        ge=1,
        json_schema_extra={"description": "Summary length the prompt asks for."},
    )
    max_summary_retries: int = Field(
        default=1,
        ge=0,
        json_schema_extra={
            "description": (
                "Retries per URL when a structured summary fails its check. "
                "Never backed off, and each one is a summariser call."
            )
        },
    )
    tool_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        json_schema_extra={"description": "Ceiling on each call a tool makes, in seconds."},
    )
    target_deadline_seconds: float = Field(
        default=300.0,
        gt=0,
        json_schema_extra={
            "description": (
                "Ceiling on one batch target as a whole, in seconds: its fetch, "
                "every summary attempt and every backoff wait together."
            )
        },
    )
    tool_concurrency: int = Field(
        default=5,
        ge=1,
        json_schema_extra={"description": "Semaphore over targets inside one batch."},
    )
    max_error_chars: int = Field(
        default=500,
        ge=1,
        json_schema_extra={
            "description": (
                "Ceiling on one error message. An error is an observation like "
                "any other, and a provider exception can carry a whole body."
            )
        },
    )
    max_tool_chars: int = Field(
        default=6000,
        ge=1,
        json_schema_extra={
            "description": (
                "Safety net per batch entry. A goal-focused summary should never "
                "reach it, so every trigger is logged as a warning."
            )
        },
    )

    max_report_chars: int = Field(
        default=8000,
        ge=1,
        json_schema_extra={
            "description": (
                "Limit the prompt states. Never cut in code: truncating the "
                "report mid-sentence destroys the agent's memory."
            )
        },
    )
    warn_prompt_chars: int | None = Field(
        default=None,
        ge=1,
        json_schema_extra={"description": "Warn when a built workspace exceeds this."},
    )

    max_retries: int = Field(
        default=2,
        ge=0,
        json_schema_extra={
            "description": (
                "Retries for one call after a transient network or server error. "
                "The retrier counts them. The provider SDK retries nothing itself."
            )
        },
    )
    model_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        json_schema_extra={"description": "Ceiling on one model request, in seconds."},
    )
    max_tokens: int = Field(
        default=32_000,
        ge=1,
        json_schema_extra={
            "description": (
                "Cap on generated tokens per call, reasoning included. A caller "
                "may only lower it."
            )
        },
    )
    requests_per_second: float | None = Field(
        default=None,
        gt=0,
        json_schema_extra={
            "description": (
                "Client-side pacing shared by every role. This paces requests, "
                "not tokens, so set it from the worst round, not the average."
            )
        },
    )
    rate_limit_max_wait_seconds: float = Field(
        default=300.0,
        ge=0,
        json_schema_extra={"description": "Total backoff for one call before giving up."},
    )
    backoff_base_seconds: float = Field(
        default=1.0,
        gt=0,
        json_schema_extra={"description": "First wait; doubles each attempt."},
    )
    backoff_max_seconds: float = Field(
        default=60.0,
        gt=0,
        json_schema_extra={"description": "Cap on a single wait."},
    )
    backoff_jitter: float = Field(
        default=0.25,
        ge=0,
        json_schema_extra={"description": "Random fraction added, so retries desynchronise."},
    )

    research_depth: Literal["normal", "deep"] = Field(
        default="normal",
        json_schema_extra={
            "description": (
                "'normal' runs the regular fields. 'deep' replaces the think "
                "reasoning effort and the look_up batch size with the operator's "
                "deep values."
            )
        },
    )
    deep_think_reasoning_effort: str | None = Field(
        default="high",
        json_schema_extra={"description": "Think reasoning effort in the deep tier."},
    )
    deep_max_urls_per_look_up: int = Field(
        default=5,
        ge=1,
        json_schema_extra={
            "description": (
                "Batch ceiling for one look_up action in the deep tier. Each URL "
                "spends one Tavily extraction and one summariser call."
            )
        },
    )

    carry_full_history: bool = Field(
        default=False,
        json_schema_extra={
            "description": (
                "Ablation baseline. True replaces the evolving report with the "
                "full history of every action and observation."
            )
        },
    )

    @property
    def arm(self) -> Literal["report", "full_history"]:
        """Name the ablation arm this configuration runs."""
        return "full_history" if self.carry_full_history else "report"

    def model_for(self, role: Role) -> str:
        """Return the model identifier a role should use."""
        return self.think_model if role == "think" else self.look_up_model

    def base_url_for(self, role: Role) -> str | None:
        """Return the endpoint a role should call, or None for the default."""
        role_url = self.think_base_url if role == "think" else self.look_up_base_url
        return role_url or self.base_url

    def temperature_for(self, role: Role) -> float:
        """Return the sampling temperature a role should use."""
        return self.think_temperature if role == "think" else self.look_up_temperature

    def reasoning_effort_for(self, role: Role) -> str | None:
        """Return the reasoning effort a role should ask for, or None for the default."""
        if role == "think":
            return self.think_reasoning_effort
        return self.look_up_reasoning_effort

    def model_spec_for(self, role: Role) -> ModelSpec:
        """Return the chat model spec for a role.

        The credential is read from the environment here, not carried on this
        model. A secret on a per-run field would travel in the request body and
        land in the checkpoint and the trace. Every endpoint is server-only, so
        the key only ever reaches a host the operator named.

        Args:
            role: One of :data:`ROLES`.

        Returns:
            A spec :func:`core.llm.build_chat_model` can build.
        """
        return ModelSpec(
            model=self.model_for(role),
            temperature=self.temperature_for(role),
            timeout_seconds=self.model_timeout_seconds,
            base_url=self.base_url_for(role),
            api_key=get_settings().LLM_API_KEY,
            max_tokens=self.max_tokens,
            requests_per_second=self.requests_per_second,
            reasoning_effort=self.reasoning_effort_for(role),
        )

    def backoff_policy(self) -> BackoffPolicy:
        """Return the backoff shape and ceilings for the two transport families.

        Returns:
            A policy :func:`core.retry.call_with_retries` can apply.
        """
        return BackoffPolicy(
            base_seconds=self.backoff_base_seconds,
            max_seconds=self.backoff_max_seconds,
            jitter=self.backoff_jitter,
            max_wait_seconds=self.rate_limit_max_wait_seconds,
            max_transient_retries=self.max_retries,
        )

    def run_metadata(self, **extra: object) -> dict[str, object]:
        """Describe a run so its trace is identifiable.

        Args:
            **extra: Additional fields to attach, for example ``question_id``.

        Returns:
            A metadata mapping for ``RunnableConfig``.
        """
        metadata: dict[str, object] = {
            "max_rounds": self.max_rounds,
            "max_queries_per_search": self.max_queries_per_search,
            "max_urls_per_look_up": self.max_urls_per_look_up,
            "answer_style": self.answer_style,
            "research_depth": self.research_depth,
            "carry_full_history": self.carry_full_history,
        }
        for role in ROLES:
            metadata[f"{role}.base_url"] = self.base_url_for(role)
            metadata[f"{role}.model"] = self.model_for(role)
            metadata[f"{role}.reasoning_effort"] = self.reasoning_effort_for(role)
        metadata.update(extra)
        return metadata

    def run_tags(self, *extra: str) -> list[str]:
        """Return tags for filtering runs in the tracing UI."""
        models = sorted({self.model_for(role) for role in ROLES})
        return [
            f"arm:{self.arm}",
            *(f"model:{m}" for m in models),
            *extra,
        ]

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> Configuration:
        """Build the configuration for one run.

        Precedence, highest first: the request's ``configurable`` mapping, then
        process settings, then the field default declared here.

        Two exceptions, both because ``configurable`` is caller-supplied.
        A name in :data:`SERVER_ONLY_FIELDS` ignores the request entirely.
        A name in :data:`CEILING_FIELDS` takes the caller's value only when it
        does not exceed the operator's value.

        A ``deep`` run first replaces the think reasoning effort and the
        look_up batch size with the deep fields, which only the environment
        sets. The deep batch size is then the ceiling a caller may lower.

        Args:
            config: The run's configuration, if any.

        Returns:
            A resolved Configuration.
        """
        configurable: dict[str, Any] = (
            config["configurable"] if config and "configurable" in config else {}
        )
        settings = get_settings()

        operator: dict[str, Any] = {}
        for name, field in cls.model_fields.items():
            from_env = getattr(settings, name.upper(), None)
            operator[name] = field.default if from_env is None else from_env

        depth = configurable.get("research_depth") or operator["research_depth"]
        if depth == "deep":
            operator["think_reasoning_effort"] = operator["deep_think_reasoning_effort"]
            operator["max_urls_per_look_up"] = operator["deep_max_urls_per_look_up"]

        values: dict[str, Any] = {}
        for name, operator_value in operator.items():
            requested = None if name in SERVER_ONLY_FIELDS else configurable.get(name)
            if requested is not None and name in CEILING_FIELDS:
                requested = _within_ceiling(name, requested, operator_value)
            values[name] = operator_value if requested is None else requested

        return cls(**values)
