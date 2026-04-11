import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the agent."""

    query_generator_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        json_schema_extra={
            "description": "The name of the language model to use for the agent's query generation."
        },
    )

    web_research_model: str = Field(
        default="llama-3.3-70b-versatile",
        json_extra_schema={
            "description": "The model used for Web Reasearch Tool Calling"
        }
    )
    reflection_model: str = Field(
        default="llama-3.3-70b-versatile",
        json_schema_extra={
            "description": "The name of the language model to use for the agent's reflection."
        },
    )

    answer_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        json_schema_extra={
            "description": "The name of the language model to use for the agent's answer."
        },
    )

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

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}
        
        return cls(**values)
