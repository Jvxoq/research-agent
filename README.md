# LangGraph Research Agent

An autonomous research agent that uses LLM-powered graph orchestration to conduct multi-step web research, evaluate findings, and generate comprehensive reports.

## Tech Stack

- **Python 3.11+** with type hints
- **LangGraph** - Graph-based agent orchestration with checkpointing
- **LangChain** - LLM abstraction and structured output
- **Groq** - Free Fast LLM inference (Llama)
- **Tavily** - Web search API
- **Loguru** - Structured logging

## Architecture

```
User Query → Generate Queries → Parallel Web Research → Reflection → Evaluate
                                                                         ↓
                                                              Final Answer (END)
```

**Graph Nodes:**
- `generate_query` - Creates diverse search queries using structured LLM output
- `web_research` - Executes Tavily searches via ReAct agent
- `reflection` - Evaluates results, identifies knowledge gaps
- `evaluate_research` - Routes to continue research loops or finalize
- `finalize_answer` - Synthesizes findings into final report

**State Management:** TypedDict-based state with `Annotated` for reducer functions.

## Quick Start

```bash
# Navigate to backend
cd backend

# Install dependencies (creates .venv)
uv sync

# Activate venv
source .venv/bin/activate

# Run the agent
python -m demo.main "What are the latest advancements in AI agents?"
```

## Configuration

Set API keys in `.env`:
```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

Override defaults via environment variables or `Configuration` class in `src/agent/configuration.py`.

## Key Design Decisions

- **Parallel execution** - Multiple search queries run concurrently via `Send` nodes
- **Structured output** - Pydantic models enforce consistent schema between nodes
- **Configurable loops** - Control research depth via `max_research_loops`
- **Single responsibility nodes** - Each node focuses on one task for maintainability