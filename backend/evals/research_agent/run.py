r"""Run the research agent against a benchmark subset.

Every model is chosen by configuration, and the endpoint is the provider, so
switching provider is an environment change rather than a code change::

    BASE_URL=https://api.together.ai/v1 \\
    THINK_MODEL=deepseek-ai/DeepSeek-V3 \\
    LOOK_UP_MODEL=deepseek-ai/DeepSeek-V3 \\
    uv run python -m evals.research_agent.run \\
        --benchmark frames --subset-size 10 --seed 7

The ablation arm is an environment setting too. ``CARRY_FULL_HISTORY=true``
runs the ``full_history`` arm, and every result line records the arm that ran.

Results are appended as JSON lines. A second pass over the same output file
skips questions that already finished and retries the ones that errored, so a
killed run loses nothing.

The run aborts before it starts on a broken setup: a missing key, or a grader
that disagrees with its own known-answer set. Burning an hour to discover the
grader was wrong is the failure this prevents.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.context import collect_runs
from langchain_core.tracers.schemas import Run
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import NotRequired, TypedDict

from agents.research_agent.configuration import ROLES, Configuration
from agents.research_agent.graph import graph
from agents.research_agent.state import ResearchAgentOutput, StopReason
from core.config import get_settings
from core.llm import ModelSpec, build_chat_model
from core.retry import COUNTERS_KEY, Retrier, RetryCounters
from core.tracing import tracing
from evals.research_agent.cost import RoundLine, RunAccountant, price_for
from evals.research_agent.dataset import (
    BENCHMARKS,
    GOLDEN,
    GoldenRow,
    Row,
    load_golden,
    load_subset,
)
from evals.research_agent.judge import (
    GoldenVerdict,
    JudgeFailure,
    Verdict,
    grade,
    judge_report,
    known_answer_check,
)

# Stop reasons that mean the question is finished. Anything else, in practice
# "error", is retried on the next pass.
TERMINAL_STOP_REASONS: frozenset[StopReason] = frozenset(
    {"answer", "max_rounds", "malformed"}
)

class ResultLine(TypedDict):
    """One JSON line of the results file, as spec.md section 8.4 names it."""

    question_id: str
    benchmark: str
    arm: str
    correct: bool
    judge_error: str | None
    rounds: int
    tokens: int
    usd: float
    seconds: float
    stop_reason: StopReason | None
    malformed_count: int
    rate_limit_hits: int
    rate_limit_wait_seconds: float
    # Optional so a line written before these fields existed still reads as
    # finished, and a resumed pass does not pay for that question again.
    transient_retries: NotRequired[int]
    transient_wait_seconds: NotRequired[float]
    # Kept so a claim in the answer can be checked against the sourced facts
    # in the report, and a failed tool target counted against all of them.
    answer: NotRequired[str | None]
    report: NotRequired[str | None]
    tool_targets: NotRequired[int]
    tool_target_errors: NotRequired[int]
    # Written on golden set lines only, which a benchmark grader never fills.
    missing_facts: NotRequired[list[str]]
    claims_made: NotRequired[list[str]]
    malformed_summaries: int
    reasoning_effort: str | None
    answer_style: str
    subset_size: int
    seed: int
    model: str
    trace_url: str | None
    per_round: list[RoundLine]


_RESULT_LINE = TypeAdapter(ResultLine)

BASELINE_PATH = Path(__file__).with_name("baseline.json")
# spec.md section 8.6: three more failures out of 30 is a regression.
GATE_DROP_POINTS = 10


class Baseline(BaseModel):
    """The accepted golden run that ``--gate`` compares against, spec.md section 8.6."""

    model_config = ConfigDict(extra="forbid")

    passed: int = Field(ge=0)
    total: int = Field(ge=1)
    think_model: str
    look_up_model: str
    judge_model: str
    date: str


def gate_failure(accuracy: Accuracy, total: int, baseline: Baseline) -> str | None:
    """Say why a golden pass fails the release gate, or None when it passes.

    Fractions keep the comparison exact. 22 of 30 against a baseline of 25 of
    30 is exactly 10 points down, and a float could put it on either side.

    Args:
        accuracy: The score of the pass.
        total: How many questions the golden set holds.
        baseline: The accepted run.

    Returns:
        The reason the gate fails, or None.
    """
    if total != baseline.total:
        return (
            f"the golden set holds {total} questions but the baseline was "
            f"taken on {baseline.total}. Accept a new baseline first."
        )
    if accuracy.graded != total:
        return (
            f"{total - accuracy.graded} of {total} questions have no grade, "
            "so the pass rate cannot be compared"
        )
    rate = Fraction(accuracy.correct, total)
    floor = Fraction(baseline.passed, baseline.total) - Fraction(GATE_DROP_POINTS, 100)
    if rate < floor:
        return (
            f"pass rate {float(rate):.1%} is more than {GATE_DROP_POINTS} points "
            f"below the baseline {baseline.passed}/{baseline.total}"
        )
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=(*BENCHMARKS, GOLDEN),
        default="browsecomp",
        help="Which dataset. The golden set always runs whole.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=10,
        help="Questions to run. Ignored by the golden set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed fixing the subset.")
    parser.add_argument(
        "--offset", type=int, default=0, help="Window start in the shuffled order."
    )
    parser.add_argument(
        "--out",
        default="results/research_agent.jsonl",
        help="Results file (JSON lines). Reread on start to resume.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Questions in flight at once. Keep at 1 or 2 on a free tier.",
    )
    parser.add_argument(
        "--grader-provider",
        default=None,
        help=(
            "Endpoint of the judge's OpenAI-compatible provider. It receives "
            "LLM_API_KEY. Defaults to the think role's endpoint."
        ),
    )
    parser.add_argument(
        "--grader-model", default=None, help="Judge model identifier."
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=None,
        help=(
            "Stop starting questions once every results file beside --out has "
            "spent this much. Unset means no stop."
        ),
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Golden set only. Exit 1 when the pass rate falls below baseline.json.",
    )
    return parser.parse_args(argv)


def latest_results(path: Path) -> dict[str, ResultLine]:
    """Return the last recorded result line for each question id.

    A malformed line is ignored rather than fatal, because a killed process can
    leave a half-written last line behind. A line that does not match
    :class:`ResultLine` is ignored the same way, so that question runs again. A later pass appends a newer line for
    a question it ran again, so the last line wins.

    Args:
        path: The results file.

    Returns:
        Each question id mapped to its newest result record.
    """
    if not path.exists():
        return {}

    latest: dict[str, ResultLine] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = _RESULT_LINE.validate_json(line)
        except ValidationError:
            logger.warning("skipping an unreadable result line")
            continue
        latest[record["question_id"]] = record
    return latest


def spent_usd(directory: Path) -> float:
    """Return what every result line in a directory's JSON lines files spent.

    Every line counts, a retried question's earlier attempts included, because
    each attempt was paid for.

    Args:
        directory: The folder holding the results files.

    Returns:
        The summed ``usd`` field, in dollars.
    """
    total = 0.0
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record: object = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"skipping an unreadable line in {path.name}")
                continue
            usd = record.get("usd") if isinstance(record, dict) else None
            if isinstance(usd, int | float):
                total += usd
    return total


@dataclass
class Budget:
    """What the metric runs may spend, and what they have spent so far.

    The check runs before a question starts, so questions already in flight
    can each overshoot the limit by what one question costs.
    """

    limit: float | None
    spent: float

    def exhausted(self) -> bool:
        """Report whether no further question may start."""
        return self.limit is not None and self.spent >= self.limit

    def charge(self, usd: float) -> None:
        """Add what one finished question spent."""
        self.spent += usd


def finished_question_ids(results: dict[str, ResultLine]) -> set[str]:
    """Return the question ids whose newest result has a terminal stop reason.

    A line whose judge call failed is not finished, so the next pass runs and
    grades that question again.

    Args:
        results: The newest result record per question id.

    Returns:
        The question ids that must not be run again.
    """
    return {
        question_id
        for question_id, record in results.items()
        if record["stop_reason"] in TERMINAL_STOP_REASONS
        and record["judge_error"] is None
    }


def append_line(path: Path, record: ResultLine) -> None:
    """Append one result record to the results file as a JSON line.

    Args:
        path: The results file.
        record: The result record to write.
    """
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(record, ensure_ascii=False) + "\n")


def _trace_url(runs: list[Run]) -> str | None:
    """Return the LangSmith URL of the collected root run, when there is one.

    Args:
        runs: Runs collected around one ``ainvoke``.

    Returns:
        The trace URL, or None when tracing recorded none.
    """
    return getattr(runs[0], "url", None) if runs else None


def build_result(
    row: Row | GoldenRow,
    args: argparse.Namespace,
    config: Configuration,
    state: ResearchAgentOutput,
    verdict: Verdict | GoldenVerdict | JudgeFailure | None,
    seconds: float,
    trace_url: str | None,
    counters: RetryCounters,
    accountant: RunAccountant,
) -> ResultLine:
    """Assemble the JSON line for one question.

    Args:
        row: The benchmark or golden question.
        args: Parsed command line arguments.
        config: The run configuration.
        state: The agent's final state.
        verdict: What the grader or the golden judge decided, the judge's
            failure, or None when the run itself failed and nothing was graded.
        seconds: Wall clock for this question.
        trace_url: The run's trace, when tracing recorded one.
        counters: What this question waited for, per retry family.
        accountant: The tokens, dollars and round records of the run's calls.

    Returns:
        The result record, holding the fields spec.md section 8.4 names plus
        ``judge_error``, the answer, the final report and the run's tool target
        and tool error counts. A golden verdict adds the facts the report
        missed and the wrong claims it made.
    """
    per_round = accountant.rounds()
    if isinstance(verdict, Verdict):
        correct = verdict.correct
    elif isinstance(verdict, GoldenVerdict):
        correct = verdict.passed
    else:
        correct = False
    result: ResultLine = {
        "question_id": row.question_id,
        "benchmark": args.benchmark,
        "arm": config.arm,
        "correct": correct,
        "judge_error": verdict.error if isinstance(verdict, JudgeFailure) else None,
        "rounds": state.get("round", 0),
        "tokens": accountant.tokens,
        "usd": accountant.usd,
        "seconds": round(seconds, 2),
        "stop_reason": state.get("stop_reason"),
        "malformed_count": sum(r["malformed_retries"] for r in per_round),
        "rate_limit_hits": counters.rate_limit_hits,
        "rate_limit_wait_seconds": round(counters.rate_limit_wait_seconds, 3),
        "transient_retries": counters.transient_retries,
        "transient_wait_seconds": round(counters.transient_wait_seconds, 3),
        "answer": state.get("answer"),
        "report": state.get("report"),
        "tool_targets": sum(r.get("tool_targets", 0) for r in per_round),
        "tool_target_errors": sum(r.get("tool_target_errors", 0) for r in per_round),
        "malformed_summaries": counters.malformed_summaries,
        "reasoning_effort": config.reasoning_effort_for("think"),
        "answer_style": config.answer_style,
        "subset_size": args.subset_size,
        "seed": args.seed,
        "model": config.model_for("think"),
        "trace_url": trace_url,
        "per_round": per_round,
    }
    if isinstance(verdict, GoldenVerdict):
        result["missing_facts"] = verdict.missing_facts
        result["claims_made"] = verdict.claims_made
    return result


def failed_result(
    row: Row | GoldenRow, args: argparse.Namespace, config: Configuration,
    seconds: float, counters: RetryCounters, accountant: RunAccountant,
) -> ResultLine:
    """Assemble the JSON line for a question whose run raised.

    Args:
        row: The benchmark or golden question.
        args: Parsed command line arguments.
        config: The run configuration.
        seconds: Wall clock before the failure.
        counters: What the failed attempt waited for, per retry family.
        accountant: What the failed attempt spent before it raised.

    Returns:
        A result record with ``stop_reason`` set to ``"error"``, so the next
        pass retries it.
    """
    return build_result(
        row=row,
        args=args,
        config=config,
        state={"stop_reason": "error"},
        verdict=None,
        seconds=seconds,
        trace_url=None,
        counters=counters,
        accountant=accountant,
    )


async def run_question(
    row: Row | GoldenRow,
    args: argparse.Namespace,
    config: Configuration,
    judge: BaseChatModel,
) -> ResultLine:
    """Run and grade one question, recording a failure rather than raising.

    A benchmark answer goes to the grader with the gold answer. A golden report
    goes to the golden judge with the line's checklist.

    Args:
        row: The benchmark or golden question.
        args: Parsed command line arguments.
        config: The run configuration.
        judge: The chat model acting as judge.

    Returns:
        One result record, whether the run succeeded or failed.
    """
    counters = RetryCounters()
    accountant = RunAccountant()
    # A new thread per attempt, so a retried question lands in its own
    # LangSmith thread, apart from the attempt that failed.
    thread_id = str(uuid.uuid4())
    started = time.monotonic()
    try:
        with collect_runs() as collector:
            # The graph counts tokens only. The accountant prices every call
            # and joins it to its round, so no dollars live in the graph.
            final = await graph.ainvoke(
                {"question": row.question},
                config={
                    "run_name": f"{args.benchmark}-{row.question_id}",
                    "metadata": config.run_metadata(question_id=row.question_id),
                    "tags": config.run_tags(args.benchmark),
                    "callbacks": [accountant],
                    "configurable": {COUNTERS_KEY: counters, "thread_id": thread_id},
                    "recursion_limit": config.max_rounds * 2 + 10,
                },
            )
    except Exception:
        logger.exception(f"{row.question_id} failed")
        return failed_result(
            row, args, config, time.monotonic() - started, counters, accountant
        )

    # ainvoke types its return as a plain dict. The graph's output schema is
    # ResearchAgentOutput, so this names the shape rather than changing it.
    state = cast(ResearchAgentOutput, final)

    retrier = Retrier(policy=config.backoff_policy(), counters=counters)
    judge_config = RunnableConfig(
        metadata={"thread_id": thread_id, "question_id": row.question_id}
    )
    verdict: Verdict | GoldenVerdict | JudgeFailure
    if isinstance(row, GoldenRow):
        verdict = await judge_report(
            judge=judge,
            question=row.question,
            report=state.get("answer") or "",
            required_facts=row.required_facts,
            wrong_claims=row.wrong_claims,
            retrier=retrier,
            max_error_chars=config.max_error_chars,
            config=judge_config,
        )
    else:
        verdict = await grade(
            judge=judge,
            question=row.question,
            response=state.get("answer") or "",
            correct_answer=row.answer,
            retrier=retrier,
            max_error_chars=config.max_error_chars,
            config=judge_config,
        )
    return build_result(
        row=row,
        args=args,
        config=config,
        state=state,
        verdict=verdict,
        seconds=time.monotonic() - started,
        trace_url=_trace_url(collector.traced_runs),
        counters=counters,
        accountant=accountant,
    )


async def _run_and_write(
    row: Row | GoldenRow,
    args: argparse.Namespace,
    config: Configuration,
    judge: BaseChatModel,
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock,
    out_path: Path,
    budget: Budget,
) -> ResultLine | None:
    """Run one question behind the concurrency bound and append its line.

    Args:
        row: The benchmark or golden question.
        args: Parsed command line arguments.
        config: The run configuration.
        judge: The chat model acting as judge.
        semaphore: Bounds how many questions are in flight.
        lock: Serialises writes to the results file.
        out_path: The results file.
        budget: What the runs may still spend. Charged with this question's cost.

    Returns:
        The result record that was written, or None when the budget was spent
        before this question could start.
    """
    async with semaphore:
        if budget.exhausted():
            return None
        result = await run_question(row, args, config, judge)
        # Charged before the slot is released, so the next question to take
        # it already sees this one's spend.
        budget.charge(result["usd"])

    async with lock:
        await asyncio.to_thread(append_line, out_path, result)

    logger.info(
        f"{result['question_id']} correct={result['correct']} "
        f"rounds={result['rounds']} {result['seconds']}s "
        f"tokens={result['tokens']} usd={result['usd']} "
        f"stop={result['stop_reason']} judge_error={result['judge_error']}"
    )
    return result


def judge_spec(args: argparse.Namespace, config: Configuration) -> ModelSpec:
    """Describe the judge model for :func:`core.llm.build_chat_model`.

    The judge falls back to the think role's endpoint and model when no grader
    is named, so a run always has a judge.

    Args:
        args: Parsed command line arguments.
        config: The run configuration.

    Returns:
        The judge's model spec.
    """
    return ModelSpec(
        model=args.grader_model or config.model_for("think"),
        temperature=0.0,
        timeout_seconds=config.model_timeout_seconds,
        base_url=args.grader_provider or config.base_url_for("think"),
        api_key=get_settings().LLM_API_KEY,
        max_tokens=config.max_tokens,
        requests_per_second=config.requests_per_second,
    )


def missing_setup() -> str | None:
    """Return why the environment cannot support a run, or None when it can.

    Both credentials are server-side settings, so nothing about one run's
    configuration can change the answer.

    Returns:
        A message naming the missing piece, or None.
    """
    settings = get_settings()
    if not settings.LLM_API_KEY:
        return "LLM_API_KEY is not set; no model call can be made."
    if not settings.TAVILY_API_KEY:
        return "TAVILY_API_KEY is not set; the search and look_up tools cannot run."
    return None


def unpriced_models(config: Configuration) -> list[str]:
    """Return the agent's models that have no price, in sorted order.

    An unpriced call counts at $0, so a run on such a model reports a Y that is
    too low and looks plausible.

    Args:
        config: The run configuration.

    Returns:
        Every role's model with no entry in the price table.
    """
    return sorted(
        {config.model_for(role) for role in ROLES if price_for(config.model_for(role)) is None}
    )


@dataclass(frozen=True)
class Accuracy:
    """The score of a subset, kept apart from questions nobody graded.

    A run where a fifth of the questions hit a rate limit is not a run with a
    low score, so accuracy over all questions and over completed questions are
    never collapsed into one number.

    Args:
        correct: Graded questions the judge marked correct.
        graded: Questions the judge graded.
        completed_correct: Correct questions whose run stopped with an answer.
        completed: Graded questions whose run stopped with an answer.
        ungraded: Questions whose judge call failed, left out of both numbers.
    """

    correct: int
    graded: int
    completed_correct: int
    completed: int
    ungraded: int


def score(results: list[ResultLine]) -> Accuracy:
    """Score the newest result of every question in a subset.

    Args:
        results: The newest result record of every question in the subset,
            including ones an earlier pass finished.

    Returns:
        The counts behind both accuracy numbers.
    """
    graded = [r for r in results if r["judge_error"] is None]
    completed = [r for r in graded if r["stop_reason"] == "answer"]
    return Accuracy(
        correct=sum(1 for r in graded if r["correct"]),
        graded=len(graded),
        completed_correct=sum(1 for r in completed if r["correct"]),
        completed=len(completed),
        ungraded=len(results) - len(graded),
    )


def subset_results(path: Path, rows: Sequence[Row | GoldenRow]) -> list[ResultLine]:
    """Return the newest recorded result of each question in the subset.

    Args:
        path: The results file.
        rows: The subset's questions.

    Returns:
        One record per question that has one, in subset order.
    """
    recorded = latest_results(path)
    return [recorded[row.question_id] for row in rows if row.question_id in recorded]


async def main(argv: list[str] | None = None) -> int:
    """Run the subset, write one JSON line per question, and report accuracy.

    Returns:
        0 on a finished pass, 1 when the setup is broken, the budget stopped
        the pass, or a question has no result line.
    """
    args = parse_args(argv)
    config = Configuration.from_runnable_config()

    problem = missing_setup()
    if problem:
        logger.error(f"refusing to start: {problem}")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    budget = Budget(
        limit=args.budget_usd,
        spent=await asyncio.to_thread(spent_usd, out_path.parent),
    )
    if budget.exhausted():
        logger.error(
            f"refusing to start: ${budget.spent:.4f} already spent of the "
            f"${budget.limit} budget"
        )
        return 1

    unpriced = unpriced_models(config)
    if unpriced:
        logger.error(
            f"refusing to start: no price for {', '.join(unpriced)}. "
            "Add it to MODEL_PRICES_USD_PER_MTOK."
        )
        return 1

    if args.benchmark == GOLDEN and config.answer_style != "report":
        logger.error(
            "refusing to start: the golden set judges a report, so it needs "
            "ANSWER_STYLE=report."
        )
        return 1

    baseline: Baseline | None = None
    if args.gate:
        if args.benchmark != GOLDEN:
            logger.error("refusing to start: --gate runs only with --benchmark golden")
            return 1
        try:
            baseline = Baseline.model_validate_json(
                await asyncio.to_thread(BASELINE_PATH.read_text, encoding="utf-8")
            )
        except (OSError, ValidationError):
            logger.exception(f"refusing to start: no usable baseline at {BASELINE_PATH}")
            return 1
        judge_model = judge_spec(args, config).model
        if judge_model != baseline.judge_model:
            logger.error(
                f"refusing to start: the baseline was judged by "
                f"{baseline.judge_model}, this run by {judge_model}. A pass rate "
                "from a different judge is not comparable."
            )
            return 1

    with tracing(get_settings()) as traced:
        logger.info(
            f"benchmark={args.benchmark} arm={config.arm} "
            f"endpoint={config.base_url_for('think')} "
            f"tracing={'on' if traced else 'off'}"
        )

        judge = build_chat_model(judge_spec(args, config))
        if not await known_answer_check(
            judge,
            Retrier(policy=config.backoff_policy()),
            config.max_error_chars,
        ):
            logger.error(
                "refusing to start: the grader disagreed with its own known-answer "
                "set. Every number behind a wrong grader is worthless."
            )
            return 1
        logger.info("grader agreed with its known-answer set")

        rows: Sequence[Row | GoldenRow]
        if args.benchmark == GOLDEN:
            rows = await asyncio.to_thread(load_golden)
            # Every result line records the size of the set it belongs to.
            args.subset_size = len(rows)
        else:
            rows = await asyncio.to_thread(
                load_subset,
                benchmark=args.benchmark,
                subset_size=args.subset_size,
                seed=args.seed,
                offset=args.offset,
                timeout=config.tool_timeout_seconds,
            )

        done = finished_question_ids(
            await asyncio.to_thread(latest_results, out_path)
        )
        pending = [row for row in rows if row.question_id not in done]
        logger.info(
            f"{len(pending)} to run, {len(rows) - len(pending)} already finished"
        )

        semaphore = asyncio.Semaphore(max(1, args.max_concurrency))
        lock = asyncio.Lock()

        written = await asyncio.gather(
            *(
                _run_and_write(
                    row, args, config, judge, semaphore, lock, out_path, budget
                )
                for row in pending
            )
        )

    not_started = written.count(None)
    results = await asyncio.to_thread(subset_results, out_path, rows)
    if not_started:
        logger.error(
            f"budget reached: ${budget.spent:.4f} spent of ${budget.limit}, "
            f"{not_started} questions not started"
        )
    elif len(results) != len(rows):
        logger.error(
            f"{len(rows)} questions but {len(results)} result lines: "
            "a question was lost"
        )
        return 1

    accuracy = score(results)
    if accuracy.ungraded:
        logger.warning(f"{accuracy.ungraded} questions left out: the judge call failed")
    logger.info(f"accuracy over all questions: {accuracy.correct}/{accuracy.graded}")
    if accuracy.completed:
        logger.info(
            "accuracy over completed questions: "
            f"{accuracy.completed_correct}/{accuracy.completed}"
        )
    else:
        logger.warning("no question completed with stop_reason=answer")

    logger.info(f"results in {out_path}")
    if baseline is not None:
        failure = gate_failure(accuracy, len(rows), baseline)
        if failure:
            logger.error(f"gate failed: {failure}")
            return 1
        logger.info(
            f"gate passed: {accuracy.correct}/{len(rows)} against the baseline "
            f"{baseline.passed}/{baseline.total}"
        )
    return 1 if not_started else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
