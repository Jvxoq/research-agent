r"""The LLM judges: the benchmark grader and the golden set judge.

The benchmark grader is ported from ``steel-dev/atlas``, file
``evals/browsecomp.ts`` (``judgePrompt``, ``readJudgeField`` and the
``/^yes\\b/`` correctness test). It compares one answer with a gold answer.

The golden set judge, spec.md section 8.5, never decides pass or fail. It
answers one yes-or-no question per required fact and per wrong claim, each tied
to the item's position in the golden line. This module checks that every item
got exactly one answer and then decides: a report passes when it states every required fact and makes none of
the wrong claims. A judge that skips an item fails its check, so a missing fact
can never pass by being left out.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger
from pydantic import BaseModel

from core.config import get_settings
from core.llm import read_structured
from core.retry import Retrier
from core.tools.text import cap, redact

GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""


@dataclass(frozen=True)
class Verdict:
    """What the judge decided about one answer."""

    correct: bool


@dataclass(frozen=True)
class JudgeFailure:
    """The judge gave no verdict, so the answer was never graded."""

    error: str


NO_VERDICT = "The judge reply had no correct field."


def judge_failure(error: Exception, max_error_chars: int) -> JudgeFailure:
    """Record a judge call that raised, without the key or an unbounded body.

    Args:
        error: What the judge call raised.
        max_error_chars: Ceiling on the recorded message.

    Returns:
        The failure, naming the error type and its redacted, capped message.
    """
    message = redact(f"{type(error).__name__}: {error}", (get_settings().LLM_API_KEY,))
    return JudgeFailure(error=cap(message, max_error_chars))


def read_judge_field(text: str, field: str) -> str | None:
    """Return one ``field: value`` line from the judge's reply.

    Args:
        text: The judge's raw response.
        field: The field name to read.

    Returns:
        The value, or None when the judge did not write that field.
    """
    match = re.search(rf"^{re.escape(field)}:\s*(.*)$", text, re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else None


def parse_verdict(text: str) -> Verdict | JudgeFailure:
    """Turn the judge's reply into a verdict.

    An answer counts as correct only when the ``correct`` field starts with
    "yes". A reply with no ``correct`` field is not a wrong answer. It is a
    failure, so the next pass grades that question again.

    Args:
        text: The judge's raw response.

    Returns:
        The parsed verdict, or a failure when the reply has no verdict.
    """
    correct_raw = read_judge_field(text, "correct")
    if correct_raw is None:
        return JudgeFailure(error=NO_VERDICT)
    return Verdict(correct=bool(re.match(r"^yes\b", correct_raw.lower())))


async def grade(
    judge: BaseChatModel,
    question: str,
    response: str,
    correct_answer: str,
    retrier: Retrier,
    max_error_chars: int,
    config: RunnableConfig | None = None,
) -> Verdict | JudgeFailure:
    """Ask the judge whether one answer matches the gold answer.

    Args:
        judge: The chat model acting as judge. Its own timeout bounds each call.
        question: The benchmark question.
        response: The agent's answer.
        correct_answer: The benchmark's gold answer.
        retrier: Backoff policy and counters for rate limits and transient
            failures. The judge's SDK retries nothing itself.
        max_error_chars: Ceiling on the error a failure records.
        config: Run configuration for the judge call, so its trace joins the
            thread of the run it grades.

    Returns:
        The judge's verdict, or a failure when the call raised or the reply
        held no verdict. A failure is not a wrong answer, so the caller must
        tell the two apart.
    """
    prompt = GRADER_TEMPLATE.format(
        question=question, response=response, correct_answer=correct_answer
    )
    try:
        reply = await retrier.run(
            lambda: judge.ainvoke([HumanMessage(content=prompt)], config=config), "judge"
        )
    except Exception as error:
        logger.exception("judge call failed")
        return judge_failure(error, max_error_chars)

    return parse_verdict(reply.text)


# Hand-built cases with an obvious right answer. A judge that disagrees with
# any of these is misconfigured, and every number behind it would be worthless.
KNOWN_ANSWERS: tuple[tuple[str, str, str, bool], ...] = (
    (
        "What is the capital of France?",
        "Exact Answer: Paris",
        "Paris",
        True,
    ),
    (
        "What is the capital of France?",
        "Exact Answer: Berlin",
        "Paris",
        False,
    ),
    (
        "In what year did the Apollo 11 mission land on the Moon?",
        "Explanation: Apollo 11 landed in the summer of 1969. Exact Answer: 1969",
        "1969",
        True,
    ),
    (
        "In what year did the Apollo 11 mission land on the Moon?",
        "Exact Answer: 1972",
        "1969",
        False,
    ),
    (
        "Who wrote the novel Frankenstein?",
        "Exact Answer: Mary Shelley",
        "Mary Shelley",
        True,
    ),
    (
        "Who wrote the novel Frankenstein?",
        "I was unable to determine the author.",
        "Mary Shelley",
        False,
    ),
)


async def known_answer_check(
    judge: BaseChatModel, retrier: Retrier, max_error_chars: int
) -> bool:
    """Report whether the judge grades a hand-built set exactly right.

    The runner calls this before it starts, so a misconfigured judge costs a few
    seconds rather than an hour of model spend.

    Args:
        judge: The chat model acting as judge.
        retrier: Backoff policy and counters for each judge call.
        max_error_chars: Ceiling on the error a failed call records.

    Returns:
        True when every case was graded as expected. A judge that gave no
        verdict counts as a disagreement.
    """
    agreed = True
    for question, response, gold, expected in KNOWN_ANSWERS:
        verdict = await grade(judge, question, response, gold, retrier, max_error_chars)
        if isinstance(verdict, JudgeFailure):
            agreed = False
            logger.error(f"grader check failed: {verdict.error} gold={gold!r}")
        elif verdict.correct is not expected:
            agreed = False
            logger.error(
                f"grader check failed: expected correct={expected} for "
                f"gold={gold!r}, judge said {verdict.correct}"
            )
    return agreed


JUDGE_TEMPLATE = """You are grading a research report against a checklist.

[question]: {question}

[report]:
<report>
{report}
</report>

[required_facts]: {required_facts}

[wrong_claims]: {wrong_claims}

For each entry in [required_facts], decide whether the report states it. A fact
counts as stated when the report says the same thing in any wording, and the
report does not contradict it elsewhere. A fact that is only hinted at, hedged
as a guess, or stated with a different value is not stated.

For each entry in [wrong_claims], decide whether the report makes it. A claim
counts as made when the report asserts it as true. A report that mentions the
claim only to reject it does not make it.

Judge only what the report says. Do not use your own knowledge of the topic.

Return one entry in "facts" for every required fact and one entry in "claims"
for every wrong claim. Each entry carries the item's index, counted from 0 in
the order given above."""


class FactCheck(BaseModel):
    """Whether the report states one required fact."""

    index: int
    stated: bool


class ClaimCheck(BaseModel):
    """Whether the report makes one wrong claim."""

    index: int
    made: bool


class Judgement(BaseModel):
    """What the judge model returns for one report."""

    facts: list[FactCheck]
    claims: list[ClaimCheck]


@dataclass(frozen=True)
class GoldenVerdict:
    """Whether one report passed, with the items that decided it.

    Args:
        passed: True when every required fact was stated and no wrong claim
            was made.
        missing_facts: Required facts the report did not state, in golden
            line order.
        claims_made: Wrong claims the report made, in golden line order.
    """

    passed: bool
    missing_facts: list[str]
    claims_made: list[str]


class MalformedJudgement(ValueError):
    """Raised when the judge did not answer every item exactly once."""


def _check_indexes(indexes: list[int], expected: int, label: str) -> None:
    """Check that a judge answered items 0 to ``expected - 1`` exactly once each.

    Args:
        indexes: The indexes the judge answered, in the order it gave them.
        expected: How many items the golden line holds.
        label: The item kind, named in the error.

    Raises:
        MalformedJudgement: If an index is missing, repeated or out of range.
    """
    if sorted(indexes) != list(range(expected)):
        raise MalformedJudgement(
            f"Expected one answer for each of the {expected} {label}, "
            f"got indexes {json.dumps(indexes)}."
        )


def _verdict(
    judgement: Judgement, required_facts: list[str], wrong_claims: list[str]
) -> GoldenVerdict:
    """Turn a checked judgement into a verdict.

    Args:
        judgement: The judge's answers.
        required_facts: The golden line's required facts.
        wrong_claims: The golden line's wrong claims.

    Returns:
        The verdict, decided here rather than by the judge.

    Raises:
        MalformedJudgement: If the judge skipped, repeated or invented an item.
    """
    _check_indexes(
        [check.index for check in judgement.facts], len(required_facts), "required facts"
    )
    _check_indexes(
        [check.index for check in judgement.claims], len(wrong_claims), "wrong claims"
    )
    stated = {check.index for check in judgement.facts if check.stated}
    made = {check.index for check in judgement.claims if check.made}
    missing_facts = [fact for i, fact in enumerate(required_facts) if i not in stated]
    claims_made = [claim for i, claim in enumerate(wrong_claims) if i in made]
    return GoldenVerdict(
        passed=not missing_facts and not claims_made,
        missing_facts=missing_facts,
        claims_made=claims_made,
    )


async def judge_report(
    judge: BaseChatModel,
    question: str,
    report: str,
    required_facts: list[str],
    wrong_claims: list[str],
    retrier: Retrier,
    max_error_chars: int,
    config: RunnableConfig | None = None,
) -> GoldenVerdict | JudgeFailure:
    """Ask the judge which facts and wrong claims one report contains.

    Args:
        judge: The chat model acting as judge. Its own timeout bounds each call.
        question: The golden question.
        report: The agent's answer, written in the ``report`` answer style.
        required_facts: Facts a passing report must state.
        wrong_claims: Claims a passing report must not make.
        retrier: Backoff policy and counters for rate limits and transient
            failures.
        max_error_chars: Ceiling on the error a failure records.
        config: Run configuration for the judge call, so its trace joins the
            thread of the run it grades.

    Returns:
        The verdict, or a failure when the call raised or the judge's answers
        did not cover every item exactly once. A failure is not a failed
        report, so the caller must tell the two apart. A blank report fails
        with every required fact missing, and no judge call is made.
    """
    if not report.strip():
        return GoldenVerdict(
            passed=False, missing_facts=list(required_facts), claims_made=[]
        )

    prompt = JUDGE_TEMPLATE.format(
        question=question,
        report=report,
        required_facts=json.dumps(required_facts),
        wrong_claims=json.dumps(wrong_claims),
    )
    # include_raw hands a parsing failure back as a value, so a schema message
    # quoting a word like "timeout" never reads as a transient network error.
    structured = judge.with_structured_output(
        Judgement, method="json_schema", include_raw=True
    )
    # A transport failure, a reply cut off by the provider, a schema mismatch
    # and a skipped item all leave the report ungraded, so each is a failure.
    try:
        result = await retrier.run(lambda: structured.ainvoke(prompt, config=config), "golden_judge")
        return _verdict(read_structured(result, Judgement), required_facts, wrong_claims)
    except Exception as error:
        logger.exception("golden judge gave no usable verdict")
        return judge_failure(error, max_error_chars)
