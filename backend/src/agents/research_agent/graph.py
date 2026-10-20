"""Research agent graph wiring."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.research_agent.configuration import Configuration
from agents.research_agent.nodes import act, route_after_think, think
from agents.research_agent.state import (
    ResearchAgentInput,
    ResearchAgentOutput,
    ResearchAgentState,
)

ResearchAgentBuilder = StateGraph[
    ResearchAgentState, None, ResearchAgentInput, ResearchAgentOutput
]


def build_graph() -> ResearchAgentBuilder:
    """Assemble the research agent graph without compiling it.

    Returns:
        The uncompiled builder, so a caller may attach a checkpointer.
    """
    # Nodes read Configuration from the run's ``configurable`` mapping, so the
    # schema the API advertises is the config schema. A context schema would
    # describe a runtime context nothing reads.
    builder: ResearchAgentBuilder = StateGraph(
        ResearchAgentState,
        input_schema=ResearchAgentInput,
        output_schema=ResearchAgentOutput,
        # LangGraph 1.x types config_schema only as a deprecated keyword.
        config_schema=Configuration,  # type: ignore[call-arg]
    )

    builder.add_node("think", think)
    builder.add_node("act", act)

    builder.add_edge(START, "think")
    builder.add_conditional_edges(
        "think", route_after_think, {"act": "act", "__end__": END}
    )
    builder.add_edge("act", "think")

    return builder


graph = build_graph().compile(name="research-agent")
