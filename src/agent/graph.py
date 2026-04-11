import os

from loguru import logger
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from agent.tools_and_schemas import SearchQueryList, Reflection, ResearchResponse
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)

from agent.utils import get_research_topic

load_dotenv()

if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY is not set")


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """Generate search queries based on the user's research topic.

    Uses a structured LLM to generate a list of diverse, optimized search queries
    that cover different aspects of the research topic. The number of queries
    is determined by the configuration.

    Args:
        state: Current graph state containing messages with the user's question.
        config: Runnable configuration with model and query count settings.

    Returns:
        Dictionary with 'search_query' key containing list of generated queries.
    """
    logger.info("Node: generate_query")
    configurable = Configuration.from_runnable_config(config)
    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # Init groq
    llm = ChatGroq(
        model=configurable.query_generator_model,
        temperature=0.4,
        max_retries=2,
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    logger.info(f"generate_query result: {result}")
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """Distribute search queries to parallel web research nodes.

    A conditional edge function that transforms a list of search queries into
    parallel web_research node calls. Each query spawns its own web_research
    node with a unique ID.

    Args:
        state: Current state containing generated search queries.

    Returns:
        List of Send objects to invoke web_research nodes for each query.
    """
    logger.info("Node: continue_to_web_research")
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """Perform web research for a single search query using Tavily search.

    Executes a web search using the Tavily API combined with an LLM to gather
    information on the research topic. Extracts structured results including
    search summaries and source URLs.

    Args:
        state: Current state with search_query string and unique id.
        config: Runnable configuration with search result limits.

    Returns:
        Dictionary with search_query, web_research_result, and sources_gathered.
    """
    logger.info("Node: web_research")
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    tavily_search = TavilySearch(
        max_results=configurable.max_search_results,
        include_images=False,
    )

    llm = ChatGroq(
        model=configurable.web_research_model,
        temperature=0,
        max_retries=1,
    )

    agent = create_react_agent(
        model=llm,
        tools=[tavily_search],
        prompt=formatted_prompt,
    )

    # Invoke the agent with the search query
    response = agent.invoke({"messages": [HumanMessage(content=state["search_query"])]})

    # Structured Output Implementation
    structured_llm = llm.with_structured_output(ResearchResponse)
    structured_response = structured_llm.invoke(response["messages"][-1].content)

    # Extract the search results from the agent's response
    search_results = structured_response.search_result
    sources_gathered = structured_response.urls
    logger.info(f"web_research result - sources: {sources_gathered}")
    return {
        "search_query": [state["search_query"]],
        "web_research_result": [search_results],
        "sources_gathered": sources_gathered,
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """Evaluate research results and identify knowledge gaps.

    Analyzes the collected web research summaries to determine if sufficient
    information has been gathered. If not, identifies knowledge gaps and
    generates follow-up queries for additional research.

    Args:
        state: Current state with web_research_result and message history.
        config: Runnable configuration with reflection model settings.

    Returns:
        Dictionary with is_sufficient, knowledge_gap, follow_up_queries,
        research_loop_count, and number_of_ran_queries.
    """
    logger.info("Node: reflection")
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reflection_model = state.get("reflection_model", configurable.reflection_model)
    logger.info(f"reflection using model: {reflection_model}")
    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = ChatGroq(
        model=reflection_model,
        temperature=1.0,
        max_retries=2,
    )

    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)
    logger.info(
        f"reflection result - is_sufficient: {result.is_sufficient}, knowledge_gap: {result.knowledge_gap}, follow_up_queries: {result.follow_up_queries}"
    )
    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """Determine next step in the research loop.

    Routing function that decides whether to continue gathering more
    information or to finalize the answer based on reflection results
    and configured maximum research loops.

    Args:
        state: Current state with is_sufficient flag and research_loop_count.
        config: Runnable configuration with max_research_loops setting.

    Returns:
        "finalize_answer" to end the research, or list of Send objects
        to spawn additional web_research nodes for follow-up queries.
    """
    logger.info("Node: evaluate_research")
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """Generate the final answer based on all collected research.

    Uses an LLM to synthesize all web research results into a comprehensive
    answer to the user's original question. Formats the output with source
    citations.

    Args:
        state: Current state with web_research_result and original messages.
        config: Runnable configuration with answer model settings.

    Returns:
        Dictionary with messages containing the final AI response.
    """
    logger.info("Node: finalize_answer")
    configurable = Configuration.from_runnable_config(config)
    answer_model = state.get("answer_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # Init Reasoning Model
    llm = ChatGroq(
        model=answer_model,
        temperature=0.1,
        max_retries=2,
    )
    result = llm.invoke(formatted_prompt)
    logger.info(f"finalize_answer result: {result.content[:100]}...")
    return {
        "messages": [AIMessage(content=result.content)],
    }


# Research Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the starting edge
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="research-agent")
