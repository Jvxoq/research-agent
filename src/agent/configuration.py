"""Configuration for the research agent.

Every node's LLM is configured independently: provider, model, temperature and
retry count. A single ``LLM_PROVIDER`` sets the default for all four nodes; each
node can then override it.

All fields are settable through environment variables named after the field in
upper case (``WEB_RESEARCH_MODEL``, ``REFLECTION_PROVIDER``, ...) or through the
``configurable`` key of a ``RunnableConfig``.
"""

import os
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Provider identifiers accepted by ``langchain.chat_models.init_chat_model``.
# Extend this list as provider packages are added to the project dependencies.
SUPPORTED_PROVIDERS = (
    "openai_compatible",
    "groq",
    "anthropic",
    "openai",
    "google_genai",
    "ollama",
    "together",
    "fireworks",
    "mistralai",
)

# Providers served by an OpenAI-compatible client, so a custom ``base_url``
# points them at any gateway speaking that wire format (OpenCode Zen,
# OpenRouter, vLLM, LM Studio, a local proxy).
OPENAI_COMPATIBLE_PROVIDERS = frozenset({"openai", "openai_compatible"})

DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "openai/gpt-oss-120b"


class Configuration(BaseModel):
    """The configuration for the agent."""

    # ------------------------------------------------------------------
    # Provider defaults
    # ------------------------------------------------------------------

    llm_provider: str = Field(
        default=DEFAULT_PROVIDER,
        json_schema_extra={
            "description": "Default provider for every node that does not name its own."
        },
    )

    base_url: str | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Endpoint for OpenAI-compatible providers, e.g. "
                "https://opencode.ai/zen/v1. Ignored by providers with their "
                "own SDK. Any node may override it."
            )
        },
    )

    api_key: str | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Explicit credential for OpenAI-compatible providers. Leave "
                "unset to fall back to the provider SDK's own environment "
                "variable (OPENAI_API_KEY, TOGETHER_API_KEY, ...)."
            )
        },
    )

    # ------------------------------------------------------------------
    # Per-node model configuration
    # ------------------------------------------------------------------

    query_generator_provider: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Provider for query generation. Falls back to llm_provider."
        },
    )
    query_generator_model: str = Field(
        default=DEFAULT_MODEL,
        json_schema_extra={
            "description": "Model used to generate search queries."
        },
    )
    query_generator_base_url: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Endpoint override for this node. Falls back to base_url."
        },
    )
    query_generator_temperature: float = Field(
        default=0.4,
        json_schema_extra={"description": "Sampling temperature for query generation."},
    )

    web_research_provider: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Provider for web research. Falls back to llm_provider."
        },
    )
    web_research_model: str = Field(
        default=DEFAULT_MODEL,
        json_schema_extra={
            "description": "Model used for the web research tool-calling agent."
        },
    )
    web_research_base_url: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Endpoint override for this node. Falls back to base_url."
        },
    )
    web_research_temperature: float = Field(
        default=0.0,
        json_schema_extra={"description": "Sampling temperature for web research."},
    )

    reflection_provider: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Provider for reflection. Falls back to llm_provider."
        },
    )
    reflection_model: str = Field(
        default=DEFAULT_MODEL,
        json_schema_extra={
            "description": "Model used to evaluate results and find knowledge gaps."
        },
    )
    reflection_base_url: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Endpoint override for this node. Falls back to base_url."
        },
    )
    reflection_temperature: float = Field(
        default=1.0,
        json_schema_extra={"description": "Sampling temperature for reflection."},
    )

    answer_provider: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Provider for the final answer. Falls back to llm_provider."
        },
    )
    answer_model: str = Field(
        default=DEFAULT_MODEL,
        json_schema_extra={
            "description": "Model used to synthesise the final answer."
        },
    )
    answer_base_url: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Endpoint override for this node. Falls back to base_url."
        },
    )
    answer_temperature: float = Field(
        default=0.1,
        json_schema_extra={"description": "Sampling temperature for the final answer."},
    )

    # ------------------------------------------------------------------
    # Research loop
    # ------------------------------------------------------------------

    number_of_initial_queries: int = Field(
        default=3,
        json_schema_extra={
            "description": "The number of initial search queries to generate."
        },
    )

    max_research_loops: int = Field(
        default=1,
        json_schema_extra={
            "description": "The maximum number of research loops to perform."
        },
    )

    max_search_results: int = Field(
        default=1,
        json_schema_extra={
            "description": "The maximum number of search results to return per query."
        },
    )

    # ------------------------------------------------------------------
    # Throughput and resilience
    # ------------------------------------------------------------------

    max_retries: int = Field(
        default=2,
        json_schema_extra={
            "description": "Provider-level retries for a single model call."
        },
    )

    max_tokens: int | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Cap on generated tokens per call. Providers enforce an "
                "output-tokens-per-minute ceiling that rejects oversized "
                "requests outright, so this is a correctness setting, not "
                "only a cost one."
            )
        },
    )

    max_run_tokens: int | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Hard ceiling on total tokens for one run. The run stops the "
                "moment it is crossed. None disables the ceiling."
            )
        },
    )

    max_run_cost_usd: float | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Hard ceiling on estimated USD cost for one run. The run stops "
                "the moment it is crossed. None disables the ceiling."
            )
        },
    )

    requests_per_second: float | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Client-side pacing shared by every node. Set this below the "
                "provider's tokens-per-minute ceiling to avoid 429s; leave unset "
                "to disable pacing."
            )
        },
    )

    # ------------------------------------------------------------------
    # Observability (LangSmith)
    # ------------------------------------------------------------------

    langsmith_tracing: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Send traces to LangSmith. Requires LANGSMITH_API_KEY."
        },
    )

    langsmith_project: str = Field(
        default="research-agent",
        json_schema_extra={"description": "LangSmith project that receives traces."},
    )

    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        json_schema_extra={"description": "LangSmith API endpoint."},
    )

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def provider_for(self, node: str) -> str:
        """Return the provider a node should use.

        Args:
            node: One of ``query_generator``, ``web_research``, ``reflection``,
                ``answer``.

        Returns:
            The node's own provider if set, otherwise ``llm_provider``.
        """
        override = getattr(self, f"{node}_provider", None)
        return override or self.llm_provider

    def model_for(self, node: str) -> str:
        """Return the model identifier a node should use."""
        return getattr(self, f"{node}_model")

    def base_url_for(self, node: str) -> str | None:
        """Return the endpoint a node should call, or None for the default."""
        override = getattr(self, f"{node}_base_url", None)
        return override or self.base_url

    def temperature_for(self, node: str) -> float:
        """Return the sampling temperature a node should use."""
        return getattr(self, f"{node}_temperature")

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig.

        Environment variables take precedence over the ``configurable`` mapping,
        which in turn takes precedence over the field defaults.
        """
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
