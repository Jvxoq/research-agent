r"""Run the research agent against BrowseComp-Plus questions.

Every model is chosen by configuration, so switching provider is an
environment change rather than a code change::

    LLM_PROVIDER=together \\
    QUERY_GENERATOR_MODEL=deepseek-ai/DeepSeek-V3 \\
    WEB_RESEARCH_MODEL=deepseek-ai/DeepSeek-V3 \\
    REFLECTION_MODEL=deepseek-ai/DeepSeek-V3 \\
    ANSWER_MODEL=deepseek-ai/DeepSeek-V3 \\
    LANGSMITH_TRACING=true \\
    uv run python scripts/run_benchmark.py --limit 1

Results are appended as JSON lines so a run can be resumed or extended
without losing what has already completed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.benchmark import (  # noqa: E402
    RunResult,
    fetch_questions,
    grade,
    load_question,
)
from agent.configuration import Configuration  # noqa: E402
from agent.cost import CostTracker  # noqa: E402
from agent.graph import graph  # noqa: E402
from agent.tracing import (  # noqa: E402
    configure_tracing,
    run_metadata,
    run_tags,
    tracing_enabled,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset", type=int, default=0, help="Row offset.")
    parser.add_argument("--limit", type=int, default=1, help="Questions to run.")
    parser.add_argument(
        "--question-file", default=None, help="Run one saved question instead."
    )
    parser.add_argument(
        "--out", default="results/benchmark.jsonl", help="Results file (JSON lines)."
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Parallel branches. Keep at 1 on rate-limited tiers.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write one JSON line per question."""
    load_dotenv()
    args = parse_args(argv)

    config = Configuration.from_runnable_config()
    traced = configure_tracing(config)
    logger.info(
        f"provider={config.llm_provider} tracing={'on' if traced else 'off'} "
        f"project={config.langsmith_project}"
    )
    if config.langsmith_tracing and not tracing_enabled(config):
        logger.warning("tracing requested but inactive — traces will not be recorded")

    questions = (
        [load_question(args.question_file)]
        if args.question_file
        else fetch_questions(offset=args.offset, length=args.limit)
    )
    logger.info(f"loaded {len(questions)} question(s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    correct_count = 0
    with out_path.open("a", encoding="utf-8") as sink:
        for question in questions:
            logger.info(f"query_id={question.query_id}")
            tracker = CostTracker(
                max_tokens=config.max_run_tokens,
                max_cost_usd=config.max_run_cost_usd,
            )
            started = time.time()
            try:
                state = graph.invoke(
                    {
                        "messages": [HumanMessage(content=question.query)],
                        "initial_search_query_count": config.number_of_initial_queries,
                        "max_research_loops": config.max_research_loops,
                    },
                    config={
                        "max_concurrency": args.max_concurrency,
                        "run_name": f"bcp-{question.query_id}",
                        "metadata": run_metadata(config, query_id=question.query_id),
                        "tags": run_tags(config, "browsecomp-plus"),
                        "callbacks": [tracker],
                    },
                )
                answer = state["messages"][-1].content
                result = RunResult(
                    query_id=question.query_id,
                    correct=grade(answer, question.answer),
                    wall_clock_seconds=round(time.time() - started, 2),
                    n_queries=len(state.get("search_query", [])),
                    n_sources=len(state.get("sources_gathered", [])),
                    agent_answer=answer,
                    gold_answer=question.answer,
                    config=run_metadata(config),
                    usage=tracker.usage.to_dict(),
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                logger.error(f"query_id={question.query_id} failed: {exc}")
                result = RunResult(
                    query_id=question.query_id,
                    correct=False,
                    wall_clock_seconds=round(time.time() - started, 2),
                    gold_answer=question.answer,
                    error=f"{type(exc).__name__}: {exc}",
                    config=run_metadata(config),
                    usage=tracker.usage.to_dict(),
                )

            correct_count += int(result.correct)
            sink.write(result.to_json() + "\n")
            sink.flush()
            usage = tracker.usage
            logger.info(
                f"query_id={result.query_id} correct={result.correct} "
                f"{result.wall_clock_seconds}s | {usage.calls} calls "
                f"{usage.total_tokens} tokens ${usage.cost_usd:.4f}"
            )
            if usage.has_unpriced_models:
                logger.warning(
                    "no published price for "
                    f"{sorted(usage.unpriced_models)} — cost is a lower bound"
                )

    logger.info(f"{correct_count}/{len(questions)} correct — results in {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
