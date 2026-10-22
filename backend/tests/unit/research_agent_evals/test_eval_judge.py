"""The benchmark grader, and the golden set judge with the verdict it decides."""

import pytest
from openai import ContentFilterFinishReasonError

from core.retry import Retrier
from evals.research_agent.judge import (
    KNOWN_ANSWERS,
    NO_VERDICT,
    ClaimCheck,
    FactCheck,
    GoldenVerdict,
    JudgeFailure,
    Judgement,
    Verdict,
    grade,
    judge_report,
    known_answer_check,
    parse_verdict,
)
from tests.conftest import (
    AlwaysWrongJudge,
    BlockJudge,
    BrokenJudge,
    FakeChatModel,
    Raises,
)

MAX_ERROR_CHARS = 500
QUESTION = "When was Python first released, and who created it?"
REPORT = "Python 0.9.0 came out in February 1991. Guido van Rossum created it."
FACTS = ["released in February 1991", "created by Guido van Rossum"]
CLAIMS = ["Python was first released in 1989"]


def judgement(stated: list[bool], made: list[bool]) -> Judgement:
    """Build a judgement that answers every item in order."""
    return Judgement(
        facts=[FactCheck(index=i, stated=value) for i, value in enumerate(stated)],
        claims=[ClaimCheck(index=i, made=value) for i, value in enumerate(made)],
    )


@pytest.fixture
def run_judge(retrier: Retrier):
    """Return a helper that judges the sample report with a scripted judge."""

    async def run(model: FakeChatModel, report: str = REPORT):
        return await judge_report(
            model, QUESTION, report, FACTS, CLAIMS, retrier, MAX_ERROR_CHARS
        )

    return run


async def test_complete_report_passes(
    chat_model, run_judge
):
    model = chat_model(structured_replies=[judgement([True, True], [False])])
    verdict = await run_judge(model)
    assert verdict == GoldenVerdict(passed=True, missing_facts=[], claims_made=[])


async def test_missing_fact_fails_named(chat_model, run_judge):
    model = chat_model(structured_replies=[judgement([True, False], [False])])
    verdict = await run_judge(model)
    assert verdict == GoldenVerdict(
        passed=False, missing_facts=["created by Guido van Rossum"], claims_made=[]
    )


async def test_wrong_claim_fails_named(chat_model, run_judge):
    model = chat_model(structured_replies=[judgement([True, True], [True])])
    verdict = await run_judge(model)
    assert verdict == GoldenVerdict(
        passed=False,
        missing_facts=[],
        claims_made=["Python was first released in 1989"],
    )


async def test_skipped_fact_is_failure(
    chat_model, run_judge
):
    skipped = Judgement(
        facts=[FactCheck(index=0, stated=True)],
        claims=[ClaimCheck(index=0, made=False)],
    )
    model = chat_model(structured_replies=[skipped])
    verdict = await run_judge(model)
    assert verdict == JudgeFailure(
        error="MalformedJudgement: Expected one answer for each of the 2 "
        "required facts, got indexes [0]."
    )


async def test_duplicate_fact_is_failure(
    chat_model, run_judge
):
    repeated = Judgement(
        facts=[FactCheck(index=0, stated=True), FactCheck(index=0, stated=True)],
        claims=[ClaimCheck(index=0, made=False)],
    )
    model = chat_model(structured_replies=[repeated])
    verdict = await run_judge(model)
    assert verdict == JudgeFailure(
        error="MalformedJudgement: Expected one answer for each of the 2 "
        "required facts, got indexes [0, 0]."
    )


async def test_schema_mismatch_is_failure(chat_model, run_judge):
    model = chat_model(structured_replies=[ValueError("not json")])
    verdict = await run_judge(model)
    assert verdict == JudgeFailure(
        error="StructuredOutputError: The response did not match the schema: not json"
    )


async def test_cut_off_reply_is_failure(chat_model, run_judge):
    model = chat_model(structured_replies=[Raises(ContentFilterFinishReasonError())])
    verdict = await run_judge(model)
    assert isinstance(verdict, JudgeFailure)


async def test_failure_hides_api_key(chat_model, run_judge):
    model = chat_model(
        structured_replies=[Raises(RuntimeError("bad request with test-key-not-used"))]
    )
    verdict = await run_judge(model)
    assert verdict == JudgeFailure(error="RuntimeError: bad request with [redacted]")


async def test_blank_report_skips_judge(
    chat_model, run_judge
):
    model = chat_model(structured_replies=[judgement([True, True], [False])])
    verdict = await run_judge(model, report="  \n")
    assert (verdict, model.structured_calls) == (
        GoldenVerdict(passed=False, missing_facts=FACTS, claims_made=[]),
        0,
    )


async def test_prompt_lists_facts_as_json(chat_model, run_judge):
    model = chat_model(structured_replies=[judgement([True, True], [False])])
    await run_judge(model)
    assert (
        '[required_facts]: ["released in February 1991", "created by Guido van Rossum"]'
        in model.prompts[0]
    )


@pytest.mark.parametrize(
    ("text", "correct"),
    [
        ("correct: yes, they match", True),
        ("correct: no", False),
        # The test is the whole word, so "yesterday" is not a yes.
        ("correct: yesterday", False),
    ],
    ids=["yes", "no", "yesterday"],
)
def test_only_yes_is_correct(text, correct):
    assert parse_verdict(text) == Verdict(correct=correct)


@pytest.mark.parametrize(
    "text",
    ["reasoning: the judge wandered off", "**correct:** yes"],
    ids=["missing field", "markdown field"],
)
def test_missing_verdict_is_failure(text):
    assert parse_verdict(text) == JudgeFailure(error=NO_VERDICT)


@pytest.mark.parametrize(
    ("question", "response", "gold", "expected"),
    KNOWN_ANSWERS,
    ids=[f"{gold}-{expected}" for _, _, gold, expected in KNOWN_ANSWERS],
)
async def test_grader_scores_known_cases(
    judge, retrier: Retrier, question, response, gold, expected
):
    verdict = await grade(judge, question, response, gold, retrier, MAX_ERROR_CHARS)
    assert verdict.correct is expected


async def test_known_check_spots_broken_judge(
    judge, retrier: Retrier
):
    """The check is the gate on the run, so both verdicts are one behaviour."""
    assert await known_answer_check(judge, retrier, MAX_ERROR_CHARS) is True
    assert await known_answer_check(AlwaysWrongJudge(), retrier, MAX_ERROR_CHARS) is False


async def test_raising_judge_is_failure(
    retrier: Retrier,
):
    verdict = await grade(BrokenJudge(), "q", "a", "gold", retrier, MAX_ERROR_CHARS)
    assert verdict == JudgeFailure(error="RuntimeError: provider is down")


async def test_known_check_counts_failed_call(
    retrier: Retrier,
):
    assert await known_answer_check(BrokenJudge(), retrier, MAX_ERROR_CHARS) is False


async def test_block_reply_graded_as_text(
    retrier: Retrier,
):
    """Only the text block counts, so the reasoning block's "no" is ignored."""
    verdict = await grade(BlockJudge(), "q", "Paris", "Paris", retrier, MAX_ERROR_CHARS)
    assert verdict.correct is True
