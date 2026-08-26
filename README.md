# Research Agent — LangGraph

> Multi-step web research that actually finishes: generate queries → parallel research → reflect → synthesize.

An autonomous agent that takes a question, breaks it into diverse search queries, runs them in parallel via Tavily, checks what is missing, loops if needed, and writes a cited report.

**Stack — agent + backend ready:** `LangGraph` · `LangChain` · `Groq (Llama)` · `Tavily` · `Pydantic structured output` · `FastAPI-ready` (expose `src/agent/graph.py` as an endpoint, see below)

## Architecture

```
User Query → Generate Queries → Parallel Web Research → Reflection → Evaluate
                                                                         ↓
                                                              Final Answer
```

**Graph nodes (`src/agent/graph.py`):**

- `generate_query` — LLM creates diverse search queries (structured output, Pydantic)
- `web_research` — ReAct agent runs Tavily searches in parallel via `Send`
- `reflection` — LLM evaluates results, finds knowledge gaps
- `evaluate_research` — routes to another research loop or to `finalize_answer`
- `finalize_answer` — synthesizes findings into a final report with sources

**State:** `TypedDict` with `Annotated` reducers. Configurable `max_research_loops`.

## Quick Start

```bash
uv sync
cp .env.example .env   # set GROQ_API_KEY, TAVILY_API_KEY
uv run python -m demo.main "What are the latest advancements in AI agents?"
```

Override defaults via env or `src/agent/configuration.py`:

```bash
GROQ_API_KEY=... TAVILY_API_KEY=... uv run python -m demo.main "Your question"
```

## Use as a backend service

The graph is backend-ready. Wrap it in FastAPI in ~15 lines:

```python
from fastapi import FastAPI
from src.agent.graph import graph

app = FastAPI()

@app.post("/research")
async def research(query: str):
    result = await graph.ainvoke({"query": query})
    return {"report": result["final_answer"], "sources": result["sources"]}
```

Then `uv run fastapi dev app:main` — same pattern as `alignai` and `talk-to-web`.

## Configuration

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

`src/agent/configuration.py` holds defaults (`max_research_loops`, model, etc.) — tweak via env.

## Design Decisions

- **Parallel execution** — `Send` nodes run multiple Tavily queries concurrently, not sequentially
- **Structured output** — Pydantic models enforce schema between nodes, so a bad LLM shape fails fast
- **Configurable loops** — `max_research_loops` caps cost/latency
- **Single-responsibility nodes** — each node does one thing, easy to test in `tests/`

## Related

- **[alignai](https://github.com/Jvxoq/alignai)** — same LangGraph + FastAPI pattern, but RAG over the EU AI Act + Postgres + Qdrant + SSE
- **[talk-to-web](https://github.com/Jvxoq/talk-to-web)** — RAG chat with onion architecture (`domain → application → adapters → api`), enforced by `lint-imports`
