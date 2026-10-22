"""The eval runner: result lines, failure handling, resume and setup checks."""

import json

import httpx2
import pytest
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import APIConnectionError, APITimeoutError, NotFoundError
from langsmith.utils import LangSmithNotFoundError

from agents.research_agent.graph import graph as agent_graph
from core.config import get_settings
from evals.research_agent.dataset import GoldenRow
from evals.research_agent.judge import ClaimCheck, FactCheck, JudgeFailure, Judgement
from evals.research_agent.run import (
    Accuracy,
    Baseline,
    gate_failure,
    main,
    score,
    subset_results,
)
from tests.conftest import (
    TRACE_PROJECT_ID,
    AlwaysWrongJudge,
    CountingGraph,
    FakeChatModel,
    FakeGraph,
    FakeLangSmithClient,
    FakeSearch,
    GoldenJudge,
    Raises,
    TracedRun,
    decision,
    logged,
    trace_page,
    traced_judge,
)

# Every run LangSmith receives for one question answered in round 0: the graph
# under its run name, its node, the model call inside it, the router, and the
# judge that grades the answer.
QUESTION_RUN_NAMES = ("frames-q1", "think", "chat_model", "route_after_think", "judge")

# spec.md section 8.4. A result line holds these and nothing else.
RESULT_FIELDS = {
    "question_id",
    "benchmark",
    "arm",
    "correct",
    "judge_error",
    "rounds",
    "tokens",
    "usd",
    "seconds",
    "stop_reason",
    "malformed_count",
    "rate_limit_hits",
    "rate_limit_wait_seconds",
    "transient_retries",
    "transient_wait_seconds",
    "malformed_summaries",
    "reasoning_effort",
    "answer_style",
    "subset_size",
    "seed",
    "model",
    "trace_url",
    "per_round",
    "answer",
    "report",
    "tool_targets",
    "tool_target_errors",
}


@pytest.fixture
def earlier_line():
    """Return a helper that builds a complete result line an earlier pass wrote."""

    def build(question_id: str, **fields: object) -> str:
        line = {
            "question_id": question_id,
            "benchmark": "frames",
            "arm": "report",
            "correct": True,
            "judge_error": None,
            "rounds": 1,
            "tokens": 10,
            "usd": 0.001,
            "seconds": 1.0,
            "stop_reason": "answer",
            "malformed_count": 0,
            "rate_limit_hits": 0,
            "rate_limit_wait_seconds": 0.0,
            "transient_retries": 0,
            "transient_wait_seconds": 0.0,
            "malformed_summaries": 0,
            "reasoning_effort": "high",
            "answer_style": "short",
            "subset_size": 3,
            "seed": 0,
            "model": "a-model",
            "trace_url": None,
            "per_round": [],
            **fields,
        }
        return json.dumps(line) + "\n"

    return build


@pytest.fixture
def out_path(tmp_path):
    """Return the results file path for one run."""
    return tmp_path / "results.jsonl"


@pytest.fixture
def working_setup(monkeypatch):
    """Give the runner the keys and the priced models it refuses to start without."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key-not-used")
    monkeypatch.setenv("THINK_MODEL", "zai-org/GLM-5.3-Flash")
    monkeypatch.setenv("LOOK_UP_MODEL", "zai-org/GLM-5.3-Flash")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def correct_graph(rows):
    """Return a graph that answers every question correctly."""
    return FakeGraph({row.question: row.answer for row in rows})


@pytest.fixture
def install(monkeypatch, rows, judge):
    """Return a helper that replaces the graph, the judge and the dataset loader."""

    def patch(graph: FakeGraph, grader: object = judge) -> None:
        monkeypatch.setattr("evals.research_agent.run.graph", graph)
        monkeypatch.setattr("evals.research_agent.run.load_subset", lambda **kw: rows)
        monkeypatch.setattr(
            "evals.research_agent.run.build_chat_model", lambda spec: grader
        )

    return patch


async def run_main(out_path, subset_size: int, extra_args: list[str] | None = None) -> int:
    """Run the eval runner on FRAMES, writing to the given results file."""
    return await main(
        [
            "--benchmark",
            "frames",
            "--subset-size",
            str(subset_size),
            "--out",
            str(out_path),
            *(extra_args or []),
        ]
    )


def result_lines(out_path) -> list[dict]:
    """Return the parsed JSON lines of a results file."""
    return [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture
def wire(install, rows):
    """Return a helper that runs a graph through the runner.

    Returns:
        A callable taking a graph and returning the exit code and result lines.
    """

    async def run(graph, out_path, extra_args=None):
        install(graph)
        code = await run_main(out_path, len(rows), extra_args)
        return code, result_lines(out_path)

    return run


async def test_one_line_per_question(
    working_setup, wire, correct_graph, out_path, rows
):
    """spec.md section 8.4 pins the shape of a result line."""
    code, lines = await wire(correct_graph, out_path)

    assert code == 0
    assert len(lines) == len(rows)
    assert set(lines[0]) == RESULT_FIELDS


async def test_correct_answer_recorded(
    working_setup, wire, correct_graph, out_path
):
    _, lines = await wire(correct_graph, out_path)
    assert all(line["correct"] for line in lines)


@pytest.fixture
def scripted_agent(monkeypatch):
    """Return a helper that puts a scripted model and search behind the real graph.

    Every call the model answers reports 1000 input and 200 output tokens on
    zai-org/GLM-5.3-Flash, which is $0.00025 a call.
    """

    def install(replies, search: FakeSearch | None = None) -> FakeChatModel:
        model = FakeChatModel(structured_replies=replies, usage=(1000, 200))
        monkeypatch.setattr(
            "agents.research_agent.nodes.build_chat_model", lambda spec: model
        )
        monkeypatch.setattr(
            "agents.research_agent.nodes.search", search or FakeSearch()
        )
        return model

    return install


async def test_one_round_line_per_round(
    working_setup, wire, scripted_agent, out_path
):
    scripted_agent([decision("r", queries=["q"]), decision("r", answer="Paris")])

    _, lines = await wire(agent_graph, out_path)

    assert [entry["round"] for entry in lines[0]["per_round"]] == [0, 1]


async def test_round_line_holds_own_tokens(
    working_setup, wire, scripted_agent, out_path
):
    scripted_agent([decision("r", queries=["q"]), decision("r", answer="Paris")])

    _, lines = await wire(agent_graph, out_path)

    assert [
        (entry["prompt_tokens"], entry["completion_tokens"], entry["usd"])
        for entry in lines[0]["per_round"]
    ] == [(1000, 200, 0.00025), (1000, 200, 0.00025)]


async def test_malformed_retries_summed(
    working_setup, wire, scripted_agent, out_path
):
    scripted_agent([ValueError("Invalid JSON"), decision("r", answer="Paris")])

    _, lines = await wire(agent_graph, out_path)

    assert lines[0]["malformed_count"] == 1


async def test_no_wait_no_rate_limit_hits(
    working_setup, wire, correct_graph, out_path
):
    _, lines = await wire(correct_graph, out_path)

    assert lines[0]["rate_limit_hits"] == 0


async def test_rate_limit_waits_recorded(
    working_setup, wire, rows, out_path
):
    """A slow run and an expensive run are different results."""
    graph = CountingGraph({row.question: row.answer for row in rows})

    _, lines = await wire(graph, out_path)

    assert lines[0]["rate_limit_hits"] == 2
    assert lines[0]["rate_limit_wait_seconds"] == 1.5


async def test_transient_retries_recorded(
    working_setup, wire, rows, out_path
):
    """The transient family keeps its own counts, apart from rate limits."""
    graph = CountingGraph({row.question: row.answer for row in rows})

    _, lines = await wire(graph, out_path)

    assert (lines[0]["transient_retries"], lines[0]["transient_wait_seconds"]) == (1, 0.5)


async def test_resume_skips_old_finished_line(
    working_setup, wire, rows, out_path, correct_graph, earlier_line
):
    """An older line must not make a resumed pass pay for its question again."""
    older = json.loads(earlier_line(rows[0].question_id))
    del older["transient_retries"], older["transient_wait_seconds"]
    out_path.write_text(json.dumps(older) + "\n", encoding="utf-8")

    await wire(correct_graph, out_path)

    asked = {config["run_name"] for config in correct_graph.configs}
    assert f"frames-{rows[0].question_id}" not in asked


async def test_wrong_answer_recorded(
    working_setup, monkeypatch, wire, rows, out_path
):
    graph = FakeGraph({row.question: "something else entirely" for row in rows})
    _, lines = await wire(graph, out_path)
    assert not any(line["correct"] for line in lines)


async def test_raising_question_writes_failed_line(
    working_setup, wire, rows, out_path
):
    """A crash is a recorded result. The run never loses the other questions."""
    graph = FakeGraph(
        {row.question: row.answer for row in rows}, raises={rows[0].question}
    )

    _, lines = await wire(graph, out_path)

    assert len(lines) == len(rows)
    failed = next(line for line in lines if line["question_id"] == rows[0].question_id)
    assert failed["stop_reason"] == "error" and failed["correct"] is False


async def test_raising_question_records_spend(
    working_setup, wire, scripted_agent, out_path
):
    """A stopped run still made its calls, so its spend is not zero."""
    scripted_agent(
        [decision("r", queries=["q"])],
        search=FakeSearch(error=RuntimeError("search is down")),
    )

    _, lines = await wire(agent_graph, out_path)

    assert (lines[0]["stop_reason"], lines[0]["tokens"], lines[0]["usd"]) == (
        "error",
        1200,
        0.00025,
    )


async def test_line_sums_tool_targets(
    working_setup, wire, scripted_agent, out_path
):
    """tool_call_success_rate is tool_target_errors over tool_targets, per run."""
    scripted_agent(
        [
            decision("r", queries=["q1", "q2"]),
            decision("r", queries=["q3"]),
            decision("r", answer="Paris"),
        ],
        search=FakeSearch(
            batches=[(["hit", ""], [None, "Search failed."]), ([""], ["Search failed."])]
        ),
    )

    _, lines = await wire(agent_graph, out_path)

    assert (lines[0]["tool_targets"], lines[0]["tool_target_errors"]) == (3, 2)


async def test_line_saves_answer(working_setup, wire, scripted_agent, out_path):
    scripted_agent([decision("the final report", answer="Paris")])

    _, lines = await wire(agent_graph, out_path)

    assert lines[0]["answer"] == "Paris"


async def test_line_saves_final_report(working_setup, wire, scripted_agent, out_path):
    """Groundedness checks the answer against the sourced facts in this report."""
    scripted_agent([decision("the final report", answer="Paris")])

    _, lines = await wire(agent_graph, out_path)

    assert lines[0]["report"] == "the final report"


async def test_failed_line_saves_no_answer(
    working_setup, wire, scripted_agent, out_path
):
    """A run that raised wrote no answer, so none may be scored as one."""
    scripted_agent(
        [decision("r", queries=["q"])],
        search=FakeSearch(error=RuntimeError("search is down")),
    )

    _, lines = await wire(agent_graph, out_path)

    assert (lines[0]["answer"], lines[0]["report"]) == (None, None)


async def test_resume_skips_line_without_tool_counts(
    working_setup, wire, rows, out_path, correct_graph, earlier_line
):
    """A line from before the tool counts existed must not be paid for again."""
    old_round = {
        "round": 0,
        "prompt_chars": 10,
        "report_chars": 1,
        "observation_chars": 0,
        "action_name": None,
        "malformed_retries": 0,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "usd": 0.001,
    }
    out_path.write_text(
        earlier_line(rows[0].question_id, per_round=[old_round]), encoding="utf-8"
    )

    await wire(correct_graph, out_path)

    asked = {config["run_name"] for config in correct_graph.configs}
    assert f"frames-{rows[0].question_id}" not in asked


async def test_line_spend_matches_runner(
    working_setup, wire, scripted_agent, out_path
):
    scripted_agent([decision("r", queries=["q"]), decision("r", answer="Paris")])

    _, lines = await wire(agent_graph, out_path)

    assert (lines[0]["tokens"], lines[0]["usd"]) == (2400, 0.0005)


@pytest.fixture
def three_answers(scripted_agent):
    """Answer every question in one call, so each question spends $0.00025."""
    scripted_agent([decision("r", answer="Paris")] * 3)


async def test_budget_stops_new_questions(
    working_setup, wire, three_answers, out_path
):
    """The budget is the one guard between a runaway pass and the operator's money."""
    _, lines = await wire(agent_graph, out_path, ["--budget-usd", "0.0004"])

    assert [line["question_id"] for line in lines] == ["q1", "q2"]


async def test_budget_stop_exits_non_zero(
    working_setup, wire, three_answers, out_path
):
    code, _ = await wire(agent_graph, out_path, ["--budget-usd", "0.0004"])

    assert code == 1


async def test_budget_stop_logs_skipped_count(
    working_setup, wire, three_answers, out_path, error_log
):
    await wire(agent_graph, out_path, ["--budget-usd", "0.0004"])

    assert error_log == [
        "budget reached: $0.0005 spent of $0.0004, 1 questions not started"
    ]


async def test_budget_counts_every_results_file(
    working_setup, install, correct_graph, out_path, earlier_line
):
    """Each metric run writes its own file, and the budget covers all of them."""
    install(correct_graph)
    (out_path.parent / "pilot.jsonl").write_text(
        earlier_line("p1", usd=0.001), encoding="utf-8"
    )

    code = await run_main(out_path, 3, ["--budget-usd", "0.001"])

    assert (code, out_path.exists()) == (1, False)


async def test_spent_budget_skips_grader(
    working_setup, install, correct_graph, judge, out_path, earlier_line
):
    """The grader check is a paid call, so a spent budget stops it too."""
    install(correct_graph)
    (out_path.parent / "pilot.jsonl").write_text(
        earlier_line("p1", usd=0.001), encoding="utf-8"
    )

    await run_main(out_path, 3, ["--budget-usd", "0.001"])

    assert judge.prompts == []


async def test_spent_budget_logs_spend(
    working_setup, install, correct_graph, out_path, earlier_line, error_log
):
    install(correct_graph)
    (out_path.parent / "pilot.jsonl").write_text(
        earlier_line("p1", usd=0.001), encoding="utf-8"
    )

    await run_main(out_path, 3, ["--budget-usd", "0.001"])

    assert error_log == [
        "refusing to start: $0.0010 already spent of the $0.001 budget"
    ]


async def test_budget_stop_reports_accuracy(
    working_setup, wire, three_answers, out_path, info_log
):
    """The questions that ran were paid for, so their score must not vanish."""
    await wire(agent_graph, out_path, ["--budget-usd", "0.0004"])

    assert "accuracy over all questions: 1/2" in info_log


async def test_no_budget_runs_all(
    working_setup, wire, three_answers, out_path, earlier_line
):
    (out_path.parent / "pilot.jsonl").write_text(
        earlier_line("p1", usd=5.0), encoding="utf-8"
    )

    code, lines = await wire(agent_graph, out_path)

    assert (code, len(lines)) == (0, 3)


async def test_failed_judge_writes_judge_error(
    working_setup, monkeypatch, wire, correct_graph, out_path
):
    async def failing_grade(**kwargs):
        return JudgeFailure(error="RuntimeError: provider is down")

    monkeypatch.setattr("evals.research_agent.run.grade", failing_grade)

    _, lines = await wire(correct_graph, out_path)

    assert lines[0]["judge_error"] == "RuntimeError: provider is down"


async def test_bad_grader_stops_before_spend(
    working_setup, install, rows, out_path, correct_graph
):
    """Every number behind a wrong grader is worthless, so nothing runs."""
    install(correct_graph, grader=AlwaysWrongJudge())

    code = await run_main(out_path, len(rows))

    assert code == 1
    assert correct_graph.configs == []


async def test_refuses_without_model_key(
    working_setup, monkeypatch, out_path, error_log
):
    """The search key is set, so only the model key check can refuse."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    get_settings.cache_clear()

    code = await run_main(out_path, 1)

    assert code == 1
    assert error_log == [
        "refusing to start: LLM_API_KEY is not set; no model call can be made."
    ]


async def test_refuses_without_search_key(
    monkeypatch, out_path, error_log
):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    get_settings.cache_clear()

    code = await run_main(out_path, 1)

    assert code == 1
    assert error_log == [
        "refusing to start: TAVILY_API_KEY is not set; "
        "the search and look_up tools cannot run."
    ]


async def test_refuses_unpriced_model(
    working_setup, set_env, out_path, error_log
):
    """An unpriced call counts at $0, which would understate Y."""
    set_env(THINK_MODEL="mystery/model")

    code = await run_main(out_path, 1)

    assert code == 1
    assert error_log == [
        "refusing to start: no price for mystery/model. "
        "Add it to MODEL_PRICES_USD_PER_MTOK."
    ]


async def test_resume_skips_finished(
    working_setup, wire, rows, out_path, correct_graph, earlier_line
):
    out_path.write_text(earlier_line(rows[0].question_id), encoding="utf-8")
    await wire(correct_graph, out_path)
    asked = {config["run_name"] for config in correct_graph.configs}
    assert asked == {f"frames-{rows[1].question_id}", f"frames-{rows[2].question_id}"}


async def test_resume_accuracy_counts_earlier_pass(
    working_setup, wire, rows, out_path, correct_graph, earlier_line
):
    """X is the score of the whole subset, not of the last pass alone."""
    out_path.write_text(
        earlier_line(rows[0].question_id, correct=False), encoding="utf-8"
    )

    await wire(correct_graph, out_path)

    accuracy = score(subset_results(out_path, rows))
    assert (accuracy.correct, accuracy.graded) == (2, 3)


async def test_resume_regrades_judge_failure(
    working_setup, wire, rows, out_path, correct_graph, earlier_line
):
    out_path.write_text(
        earlier_line(
            rows[0].question_id,
            correct=False,
            judge_error="RuntimeError: provider is down",
        ),
        encoding="utf-8",
    )

    await wire(correct_graph, out_path)

    asked = {config["run_name"] for config in correct_graph.configs}
    assert f"frames-{rows[0].question_id}" in asked


async def test_resume_reruns_incomplete_line(
    working_setup, wire, rows, out_path, correct_graph
):
    """A line from another shape cannot prove the question finished."""
    partial = {"question_id": rows[0].question_id, "stop_reason": "answer"}
    out_path.write_text(json.dumps(partial) + "\n", encoding="utf-8")

    await wire(correct_graph, out_path)

    asked = {config["run_name"] for config in correct_graph.configs}
    assert f"frames-{rows[0].question_id}" in asked


async def test_resume_retries_a_question_that_errored(
    working_setup, wire, rows, out_path, correct_graph, earlier_line
):
    out_path.write_text(
        earlier_line(rows[0].question_id, stop_reason="error"), encoding="utf-8"
    )
    await wire(correct_graph, out_path)
    asked = {config["run_name"] for config in correct_graph.configs}
    assert any(rows[0].question_id in name for name in asked)


async def test_env_arm_recorded(
    working_setup, set_env, wire, correct_graph, out_path
):
    set_env(CARRY_FULL_HISTORY="true")

    _, lines = await wire(correct_graph, out_path)

    assert all(line["arm"] == "full_history" for line in lines)


def test_accuracy_skips_ungraded(earlier_line):
    """A failed judge call is not a wrong answer."""
    results = [
        json.loads(earlier_line("q1")),
        json.loads(earlier_line("q2", correct=False, judge_error="RuntimeError: down")),
    ]

    accuracy = score(results)

    assert (accuracy.correct, accuracy.graded, accuracy.ungraded) == (1, 1, 1)


def test_completed_accuracy_skips_stopped_run(earlier_line):
    """A run that hit its round limit is scored overall but not as completed."""
    results = [
        json.loads(earlier_line("q1")),
        json.loads(earlier_line("q2", stop_reason="max_rounds")),
    ]

    accuracy = score(results)

    assert (accuracy.completed_correct, accuracy.completed) == (1, 1)


GOLDEN_ROWS = [
    GoldenRow(
        question_id="g01",
        question="Who directed The Princess Bride?",
        required_facts=["the film is The Princess Bride", "Rob Reiner directed it"],
        wrong_claims=["Steven Spielberg directed it"],
    ),
    GoldenRow(
        question_id="g02",
        question="Who was the first IOC president?",
        required_facts=["Demetrios Vikelas was the first IOC president"],
        wrong_claims=[],
    ),
]


@pytest.fixture
def golden_setup(working_setup, set_env):
    """Run the golden set in the answer style it judges."""
    set_env(ANSWER_STYLE="report")


@pytest.fixture
def run_golden(monkeypatch, out_path, chat_model):
    """Return a helper that runs the golden set with a scripted checklist judge.

    Returns:
        A callable taking the judge's structured replies and returning the exit
        code and result lines.
    """

    async def run(replies: list[object], extra_args: list[str] | None = None):
        judge = GoldenJudge(chat_model(structured_replies=replies))
        graph = FakeGraph({row.question: f"report on {row.question}" for row in GOLDEN_ROWS})
        monkeypatch.setattr("evals.research_agent.run.graph", graph)
        monkeypatch.setattr("evals.research_agent.run.load_golden", lambda: GOLDEN_ROWS)
        monkeypatch.setattr("evals.research_agent.run.build_chat_model", lambda spec: judge)
        code = await main(
            ["--benchmark", "golden", "--out", str(out_path), *(extra_args or [])]
        )
        lines = result_lines(out_path) if out_path.exists() else []
        return code, lines

    return run


def checklist(stated: list[bool], made: list[bool]) -> Judgement:
    """Build a judgement that answers every item in order."""
    return Judgement(
        facts=[FactCheck(index=i, stated=value) for i, value in enumerate(stated)],
        claims=[ClaimCheck(index=i, made=value) for i, value in enumerate(made)],
    )


async def test_golden_complete_report_passes(golden_setup, run_golden):
    _, lines = await run_golden([checklist([True, True], [False]), checklist([True], [])])

    assert [line["correct"] for line in lines] == [True, True]


async def test_golden_missing_fact_fails(golden_setup, run_golden):
    _, lines = await run_golden([checklist([True, False], [False]), checklist([True], [])])

    assert [line["correct"] for line in lines] == [False, True]


async def test_golden_records_missing_facts(golden_setup, run_golden):
    """A regression is read from which facts went missing, not from a bare false."""
    replies = [checklist([True, False], [False]), checklist([False], [])]

    _, lines = await run_golden(replies)

    assert [line["missing_facts"] for line in lines] == [
        ["Rob Reiner directed it"],
        ["Demetrios Vikelas was the first IOC president"],
    ]


async def test_golden_records_claims_made(golden_setup, run_golden):
    _, lines = await run_golden([checklist([True, True], [True]), checklist([True], [])])

    assert [line["claims_made"] for line in lines] == [["Steven Spielberg directed it"], []]


async def test_golden_line_fields(golden_setup, run_golden):
    """Golden lines carry the checklist outcome on top of the section 8.4 fields."""
    _, lines = await run_golden([checklist([True, True], [False])])

    assert set(lines[0]) == RESULT_FIELDS | {"missing_facts", "claims_made"}


async def test_golden_judge_failure_recorded(golden_setup, run_golden):
    """An ungraded report must not read as a failed one, so it carries no checklist."""
    _, lines = await run_golden([Raises(RuntimeError("provider is down"))])

    assert [(line["judge_error"], "missing_facts" in line) for line in lines] == [
        ("RuntimeError: provider is down", False)
    ] * 2


async def test_golden_runs_whole_set(golden_setup, run_golden):
    _, lines = await run_golden([checklist([True, True], [False])], ["--subset-size", "1"])

    assert [(line["question_id"], line["subset_size"]) for line in lines] == [
        ("g01", 2),
        ("g02", 2),
    ]


async def test_golden_short_style_exits_non_zero(working_setup, run_golden):
    code, _ = await run_golden([checklist([True, True], [False])])

    assert code == 1


async def test_golden_short_style_runs_nothing(working_setup, run_golden):
    _, lines = await run_golden([checklist([True, True], [False])])

    assert lines == []


async def test_golden_short_style_logs_reason(working_setup, run_golden, error_log):
    await run_golden([checklist([True, True], [False])])

    assert error_log == [
        "refusing to start: the golden set judges a report, so it needs ANSWER_STYLE=report."
    ]


JUDGE_MODEL = "judge-model-x"


@pytest.fixture
def baseline_file(monkeypatch, tmp_path):
    """Return a helper that writes the baseline the gate reads."""

    def write(passed: int, total: int = 2, judge_model: str = JUDGE_MODEL) -> None:
        path = tmp_path / "baseline.json"
        path.write_text(
            json.dumps(
                {
                    "passed": passed,
                    "total": total,
                    "think_model": "zai-org/GLM-5.3-Flash",
                    "look_up_model": "zai-org/GLM-5.3-Flash",
                    "judge_model": judge_model,
                    "date": "2026-10-20",
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr("evals.research_agent.run.BASELINE_PATH", path)

    return write


GATE_ARGS = ["--gate", "--grader-model", JUDGE_MODEL]
BOTH_PASS = [checklist([True, True], [False]), checklist([True], [])]
ONE_FAILS = [checklist([True, False], [False]), checklist([True], [])]


def baseline(passed: int, total: int) -> Baseline:
    """Build a baseline judged by the test judge."""
    return Baseline(
        passed=passed,
        total=total,
        think_model="zai-org/GLM-5.3-Flash",
        look_up_model="zai-org/GLM-5.3-Flash",
        judge_model=JUDGE_MODEL,
        date="2026-10-20",
    )


def graded(correct: int, total: int) -> Accuracy:
    """Build the score of a pass where every question was graded."""
    return Accuracy(
        correct=correct, graded=total, completed_correct=correct, completed=total, ungraded=0
    )


def test_gate_allows_ten_point_drop():
    """22 of 30 against 25 of 30 is exactly 10 points, which a float can misplace."""
    assert gate_failure(graded(22, 30), 30, baseline(25, 30)) is None


def test_gate_fails_past_ten_points():
    assert gate_failure(graded(21, 30), 30, baseline(25, 30)) == (
        "pass rate 70.0% is more than 10 points below the baseline 25/30"
    )


def test_gate_fails_ungraded_questions():
    """A pass rate with holes in it is not comparable, so it cannot pass."""
    accuracy = Accuracy(correct=25, graded=29, completed_correct=25, completed=29, ungraded=1)

    assert gate_failure(accuracy, 30, baseline(25, 30)) == (
        "1 of 30 questions have no grade, so the pass rate cannot be compared"
    )


def test_gate_fails_changed_set_size():
    assert gate_failure(graded(30, 31), 31, baseline(25, 30)) == (
        "the golden set holds 31 questions but the baseline was taken on 30. "
        "Accept a new baseline first."
    )


async def test_gate_pass_exits_zero(golden_setup, run_golden, baseline_file):
    baseline_file(passed=2)

    code, _ = await run_golden(BOTH_PASS, GATE_ARGS)

    assert code == 0


async def test_gate_regression_exits_non_zero(golden_setup, run_golden, baseline_file):
    baseline_file(passed=2)

    code, _ = await run_golden(ONE_FAILS, GATE_ARGS)

    assert code == 1


async def test_gate_regression_logs_reason(
    golden_setup, run_golden, baseline_file, error_log
):
    baseline_file(passed=2)

    await run_golden(ONE_FAILS, GATE_ARGS)

    assert error_log == [
        "gate failed: pass rate 50.0% is more than 10 points below the baseline 2/2"
    ]


async def test_gate_regression_still_writes_lines(golden_setup, run_golden, baseline_file):
    """A failed gate keeps its result lines, so the misses can be read."""
    baseline_file(passed=2)

    _, lines = await run_golden(ONE_FAILS, GATE_ARGS)

    assert [line["correct"] for line in lines] == [False, True]


async def test_gate_without_baseline_runs_nothing(
    golden_setup, run_golden, monkeypatch, tmp_path
):
    monkeypatch.setattr("evals.research_agent.run.BASELINE_PATH", tmp_path / "missing.json")

    code, lines = await run_golden(BOTH_PASS, GATE_ARGS)

    assert (code, lines) == (1, [])


async def test_gate_other_judge_runs_nothing(golden_setup, run_golden, baseline_file):
    """A pass rate from a different judge measures the judge, not the agent."""
    baseline_file(passed=2, judge_model="some-other-judge")

    code, lines = await run_golden(BOTH_PASS, GATE_ARGS)

    assert (code, lines) == (1, [])


async def test_gate_other_judge_logs_reason(
    golden_setup, run_golden, baseline_file, error_log
):
    baseline_file(passed=2, judge_model="some-other-judge")

    await run_golden(BOTH_PASS, GATE_ARGS)

    assert error_log == [
        "refusing to start: the baseline was judged by some-other-judge, this run by "
        f"{JUDGE_MODEL}. A pass rate from a different judge is not comparable."
    ]


async def test_gate_refused_on_benchmark(
    working_setup, install, correct_graph, out_path, rows, baseline_file
):
    """A benchmark pass rate compared with a golden baseline means nothing."""
    baseline_file(passed=2, judge_model="zai-org/GLM-5.3-Flash")
    install(correct_graph)

    code = await run_main(out_path, len(rows), ["--gate"])

    assert (code, out_path.exists()) == (1, False)


def threads_by_question(runs: list[TracedRun]) -> dict[str, set[str | None]]:
    """Group the thread ids LangSmith received by the question each run names."""
    threads: dict[str, set[str | None]] = {}
    for run in runs:
        if run.question_id is not None:
            threads.setdefault(run.question_id, set()).add(run.thread_id)
    return threads


@pytest.fixture
def traced_pass(working_setup, traced_runs, install, scripted_agent, out_path, rows):
    """Return a helper that runs one traced pass of the real graph and judge.

    Returns:
        A callable taking the agent's scripted replies and returning every run
        LangSmith received so far.
    """

    async def run(replies: list[object]) -> list[TracedRun]:
        scripted_agent(replies)
        install(agent_graph, traced_judge())
        await run_main(out_path, len(rows))
        return traced_runs()

    return run


async def test_question_runs_share_thread(traced_pass, rows) -> None:
    """LangSmith shows one question as one thread only when every run agrees."""
    runs = await traced_pass([decision("r", answer="Paris")])

    sizes = {question: len(ids) for question, ids in threads_by_question(runs).items()}

    assert sizes == {row.question_id: 1 for row in rows}


async def test_question_runs_never_threadless(traced_pass) -> None:
    runs = await traced_pass([decision("r", answer="Paris")])

    assert [run.name for run in runs if run.question_id and not run.thread_id] == []


async def test_questions_get_distinct_threads(traced_pass, rows) -> None:
    runs = await traced_pass([decision("r", answer="Paris")])

    threads = set().union(*threads_by_question(runs).values())

    assert len(threads) == len(rows)


async def test_judge_joins_question_thread(traced_pass, rows) -> None:
    """A grade read apart from the run it graded cannot be checked."""
    runs = await traced_pass([decision("r", answer="Paris")])
    (thread,) = threads_by_question(runs)[rows[0].question_id]

    names = sorted(run.name for run in runs if run.thread_id == thread)

    assert names == sorted(QUESTION_RUN_NAMES)


async def test_retried_question_gets_new_thread(traced_pass, rows) -> None:
    """A retry filed under the failed attempt's thread hides which attempt scored."""
    await traced_pass([Raises(ValueError("bad request"))])
    runs = await traced_pass([decision("r", answer="Paris")])

    sizes = {question: len(ids) for question, ids in threads_by_question(runs).items()}

    assert sizes == {row.question_id: 2 for row in rows}


async def test_golden_judge_carries_thread(golden_setup, traced_runs, run_golden) -> None:
    await run_golden([checklist([True, True], [False]), checklist([True], [])])

    judged = [
        (run.question_id is not None, run.thread_id is not None)
        for run in traced_runs()
        if run.name == "golden_judge"
    ]

    assert judged == [(True, True)] * len(GOLDEN_ROWS)


# The pages LangSmith names for a traced pass, from the root run of each question.


def root_pages(fake: FakeLangSmithClient) -> dict[str, str]:
    """Return the page of each question's root run, keyed by question id."""
    wait_for_all_tracers()
    return {
        run["name"].removeprefix("frames-"): trace_page(TRACE_PROJECT_ID, run["id"], run["id"])
        for run in fake.created
        if run["name"].startswith("frames-")
    }


def trace_urls(out_path) -> dict[str, str | None]:
    """Return the trace URL each result line recorded, keyed by question id."""
    return {line["question_id"]: line["trace_url"] for line in result_lines(out_path)}


def lookup_error(kind: str) -> Exception:
    """Build the error the LangSmith runs API raises, with its real class."""
    request = httpx2.Request("GET", "https://api.smith.test/runs/url")
    if kind == "timeout":
        return APITimeoutError(request=request)
    if kind == "connection":
        return APIConnectionError(request=request)
    return NotFoundError(
        "run not found", response=httpx2.Response(404, request=request), body=None
    )


async def test_line_records_trace_url(traced_pass, langsmith_fake, out_path) -> None:
    """A result line with no trace link cannot be checked by anyone who reads it."""
    await traced_pass([decision("r", answer="Paris")])

    assert trace_urls(out_path) == root_pages(langsmith_fake)


async def test_failed_line_records_trace_url(traced_pass, langsmith_fake, out_path) -> None:
    """A run that raised is the one whose trace someone needs to read."""
    await traced_pass([Raises(ValueError("bad request"))])

    assert trace_urls(out_path) == root_pages(langsmith_fake)


@pytest.mark.parametrize("kind", ["timeout", "connection", "not_found"])
async def test_lookup_error_records_no_url(
    traced_pass, langsmith_fake, out_path, rows, kind
) -> None:
    langsmith_fake.runs.error = lookup_error(kind)

    await traced_pass([decision("r", answer="Paris")])

    assert trace_urls(out_path) == {row.question_id: None for row in rows}


async def test_lookup_error_keeps_answers(traced_pass, langsmith_fake, out_path) -> None:
    """A missing link is a missing link, never a lost result."""
    langsmith_fake.runs.error = lookup_error("timeout")

    await traced_pass([decision("r", answer="Paris")])

    assert [line["stop_reason"] for line in result_lines(out_path)] == ["answer"] * 3


async def test_lookup_error_logged(
    traced_pass, langsmith_fake, log_records, out_path
) -> None:
    langsmith_fake.runs.error = lookup_error("timeout")

    await traced_pass([decision("r", answer="Paris")])

    assert logged(log_records) == [
        ("ERROR", f"no trace URL for run {run_id}", True)
        for run_id in sorted(lookup["run_id"] for lookup in langsmith_fake.runs.lookups)
    ]


async def test_unreadable_project_records_no_url(
    traced_pass, langsmith_fake, out_path, rows
) -> None:
    langsmith_fake.project_error = LangSmithNotFoundError("Project default not found")

    await traced_pass([decision("r", answer="Paris")])

    assert trace_urls(out_path) == {row.question_id: None for row in rows}


async def test_unreadable_project_logged(traced_pass, langsmith_fake, log_records) -> None:
    langsmith_fake.project_error = LangSmithNotFoundError("Project default not found")

    await traced_pass([decision("r", answer="Paris")])

    assert logged(log_records) == [
        ("ERROR", "cannot read LangSmith project default; no trace URLs", True)
    ]


async def test_project_read_once_per_pass(traced_pass, langsmith_fake, set_env) -> None:
    """One project read per question would cost a request per question."""
    set_env(LANGSMITH_PROJECT="evals")

    await traced_pass([decision("r", answer="Paris")])

    assert langsmith_fake.project_reads == ["evals"]


async def test_one_lookup_per_question(traced_pass, langsmith_fake, rows) -> None:
    await traced_pass([decision("r", answer="Paris")])

    assert len(langsmith_fake.runs.lookups) == len(rows)


async def test_one_client_per_pass(traced_pass, langsmith_fake) -> None:
    """A client per question opens a connection pool per question."""
    await traced_pass([decision("r", answer="Paris")])

    assert len(langsmith_fake.builds) == 1


async def test_lookup_timeout_from_config(traced_pass, langsmith_fake, set_env) -> None:
    """The SDK call is the last stop before the network, so its timeout is the guard."""
    set_env(TOOL_TIMEOUT_SECONDS="7.5")

    await traced_pass([decision("r", answer="Paris")])

    assert {lookup["timeout"] for lookup in langsmith_fake.runs.lookups} == {7.5}


async def test_project_read_timeout_from_config(traced_pass, langsmith_fake, set_env) -> None:
    set_env(TOOL_TIMEOUT_SECONDS="7.5")

    await traced_pass([decision("r", answer="Paris")])

    assert [build["timeout_ms"] for build in langsmith_fake.builds] == [7500]


async def test_tracing_off_builds_no_client(
    working_setup, langsmith_fake, install, scripted_agent, out_path, rows
) -> None:
    scripted_agent([decision("r", answer="Paris")])
    install(agent_graph, traced_judge())

    await run_main(out_path, len(rows))

    assert langsmith_fake.builds == []


async def test_traced_pass_logs_no_warning(traced_pass, log_records) -> None:
    await traced_pass([decision("r", answer="Paris")])

    assert logged(log_records) == []
