# Research Agent — LangGraph

> Multi-step web research that actually finishes: generate queries → parallel research → reflect → synthesize. Provider-agnostic, cost-metered, and benchmarked.

An autonomous agent that takes a question, breaks it into diverse search queries, runs them via Tavily, checks what is missing, loops if needed, and writes a cited report.

**Stack:** `LangGraph` · `LangChain` · `Tavily` · `Pydantic structured output` · `LangSmith` — with any OpenAI-compatible provider behind a configuration switch.

## Architecture

```
User Query → Generate Queries → Web Research → Reflection → Evaluate
                                                                ↓
                                                         Final Answer
```

**Graph nodes (`src/agent/graph.py`):**

- `generate_query` — LLM creates diverse search queries (structured output, Pydantic)
- `web_research` — ReAct agent runs Tavily searches, fanned out via `Send`
- `reflection` — LLM evaluates results, finds knowledge gaps
- `evaluate_research` — routes to another research loop or to `finalize_answer`
- `finalize_answer` — synthesizes findings into a final report with sources

**Supporting modules:**

| Module | Responsibility |
| --- | --- |
| `src/agent/llm.py` | Builds every node's model. The graph imports no provider SDK. |
| `src/agent/cost.py` | Counts tokens, prices them, and halts a run at a ceiling. |
| `src/agent/tracing.py` | LangSmith setup plus the metadata that makes traces comparable. |
| `src/agent/benchmark.py` | BrowseComp-Plus loading, decryption and grading. |
| `scripts/run_benchmark.py` | Runs questions and writes one result row each. |

## Quick Start

```bash
uv sync
cp .env.example .env      # fill in a provider key and TAVILY_API_KEY
uv run python -m demo.main "What are the latest advancements in AI agents?"
```

## Choosing a provider

No node names a provider in code. Two ways to point the agent somewhere:

**Any OpenAI-compatible endpoint** — set a base URL and key:

```env
LLM_PROVIDER=openai_compatible
BASE_URL=https://api.together.xyz/v1
API_KEY=your-key
ANSWER_MODEL=deepseek-ai/DeepSeek-V3
```

Works for Together, OpenRouter, vLLM, LM Studio, or a local proxy.

**A provider with its own SDK** — `groq`, `anthropic`, `together`, `openai`, `google_genai`, `ollama`, `fireworks`, `mistralai`. Credentials come from that SDK's own variable and `BASE_URL` is ignored:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key
ANSWER_MODEL=claude-opus-5
```

**Per node.** Provider, model, temperature, output cap and endpoint are independent for each of `query_generator`, `web_research`, `reflection`, `answer`:

```env
LLM_PROVIDER=groq                  # default for all four
ANSWER_PROVIDER=anthropic          # except this one
ANSWER_MODEL=claude-opus-5
ANSWER_BASE_URL=https://…/v1       # and it may use a different endpoint
```

Two tests enforce this: one parses `graph.py` and fails if a provider SDK is imported, another fails unless all four nodes build their model through the factory.

## Cost control

Token usage is read from the provider's own response, priced per model, and recorded on every result — including failed ones.

```env
MAX_RUN_TOKENS=40000       # halt the run past this many tokens
MAX_RUN_COST_USD=0.05      # halt the run past this many dollars
```

The ceiling is enforced inside the model callback, so it fires *between* calls rather than after the money is gone. Crossing it raises `BudgetExceeded`, which the runner records as a normal failed result.

Models missing from the price table are still counted in tokens, priced at zero, and flagged — so a zero cost is never mistaken for a free run.

> **Note:** LangChain swallows exceptions raised inside callbacks unless a handler sets `raise_error = True`. Without it the ceiling runs, decides you are over budget, and the agent keeps spending. `CostTracker` sets the flag and a regression test asserts it.

## Rate limits and resilience

Providers refuse in more ways than one, and only some are retryable:

| Limit | Fixed by |
| --- | --- |
| Tokens per minute | `REQUESTS_PER_SECOND` pacing |
| Tokens per day | Nothing — wait, or upgrade |
| Output tokens per minute | `MAX_TOKENS` — the request is rejected before it runs, so retrying cannot help |
| Requests per minute | Pacing |

```env
REQUESTS_PER_SECOND=0.07   # one shared budget across all parallel branches
MAX_RETRIES=4
MAX_TOKENS=800
```

The rate limiter is process-wide, so parallel branches contend for one budget rather than getting one each.

## Observability

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT=research-agent
```

Every run is named (`bcp-<query_id>`), tagged (`provider:…`, `model:…`), and stamped with each node's provider and model — without that, a trace shows timings but not which configuration produced them, and two runs cannot be compared.

Tracing without a key logs a warning and continues. It never fails a run.

## Benchmarking

The agent is evaluated against [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus), whose 830 questions ship encrypted so they do not leak into training corpora.

```bash
uv run python scripts/run_benchmark.py --limit 1
uv run python scripts/run_benchmark.py --offset 0 --limit 30 --out results/run1.jsonl
```

Each question produces one JSON line: correctness, wall clock, query and source counts, full token usage, the resolved configuration, and — when it fails — the error, preserved rather than swallowed.

## Configuration reference

Every field is settable by environment variable (upper case) or through a `RunnableConfig`. Environment wins.

| Setting | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `groq` | Default provider for all nodes |
| `BASE_URL` / `API_KEY` | unset | Endpoint and key for OpenAI-compatible providers |
| `<NODE>_PROVIDER` / `_MODEL` / `_TEMPERATURE` / `_BASE_URL` | — | Per-node overrides |
| `MAX_TOKENS` | unset | Output cap per call |
| `MAX_RETRIES` | `2` | Provider-level retries |
| `REQUESTS_PER_SECOND` | unset | Client-side pacing |
| `MAX_RUN_TOKENS` / `MAX_RUN_COST_USD` | unset | Hard run ceilings |
| `NUMBER_OF_INITIAL_QUERIES` | `3` | Queries generated up front |
| `MAX_RESEARCH_LOOPS` | `1` | Reflection loops |
| `MAX_SEARCH_RESULTS` | `1` | Tavily results per query |
| `LANGSMITH_TRACING` / `_API_KEY` / `_PROJECT` / `_ENDPOINT` | off | Tracing |

## Tests

```bash
uv run pytest tests/ -q      # 73 tests
uv run ruff check src/ tests/ scripts/
```

The suite is mostly guard rails rather than behaviour checks — it fails if the graph regains a provider dependency, if a node stops being independently configurable, if the benchmark's decryption breaks, or if the spend ceiling stops propagating.

## Design Decisions

- **Provider is configuration, not code** — swapping models is an environment change, enforced by tests rather than convention
- **Cost is measured, not estimated** — token counts come from the provider's own response
- **Ceilings halt between calls** — a limit checked after the run is an invoice, not a limit
- **Errors are recorded, never swallowed** — a failed question still produces a result row with the reason
- **One shared rate limiter** — parallel branches contend for one budget
- **Structured output** — Pydantic models enforce the schema between nodes, so a bad LLM shape fails fast

## Related

- **[alignai](https://github.com/Jvxoq/alignai)** — same LangGraph + FastAPI pattern, RAG over the EU AI Act + Postgres + Qdrant + SSE
- **[talk-to-web](https://github.com/Jvxoq/talk-to-web)** — RAG chat with onion architecture (`domain → application → adapters → api`), enforced by `lint-imports`
