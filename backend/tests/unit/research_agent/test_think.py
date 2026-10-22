"""``think`` builds the workspace, calls the model once, and decides.

Every test calls the node as a plain function with a hand-built state. The
prompt is checked through the mocked model that received it, never by calling a
private helper.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from openai import ContentFilterFinishReasonError

from agents.research_agent.nodes import think
from agents.research_agent.schemas import Answered, Decision, FinishedThread
from core.config import get_settings
from tests.conftest import Raises, decision


class _RateLimited(Exception):
    """A provider rate-limit error, with the status code a real one carries."""

    status_code = 429


@pytest.fixture
def question() -> str:
    """The question every test asks."""
    return "In what year was the observatory founded?"


@pytest.fixture
def observation() -> dict[str, Any]:
    """One previous round's observation."""
    return {
        "call": {"name": "search", "queries": ["founding year"]},
        "results": ["the observatory opened in 1893"],
        "errors": [None],
        "capped": False,
    }


@pytest.fixture
def answering() -> Decision:
    """A decision that answers the question."""
    return decision("r", answer="1893")


@pytest.fixture
def searching() -> Decision:
    """A decision that asks for one search."""
    return decision("r", queries=["who founded it"])


@pytest.fixture
def forced() -> Answered:
    """What the provider returns for the last round's ``Answered`` schema."""
    return Answered(report="final report", answer="1893")


@pytest.fixture
def last_round(config) -> dict[str, Any]:
    """A config whose round limit is 3, for a state at round 3."""
    config["configurable"]["max_rounds"] = 3
    return config


async def test_the_prompt_carries_the_frozen_date(
    chat_model, patch_model, config, question, answering
) -> None:
    model = chat_model(structured_replies=[answering])
    patch_model(model)

    await think({"question": question}, config)

    assert "- Current Date: 2026-01-02\n" in model.prompts[0]


async def test_the_prompt_carries_the_question(
    chat_model, patch_model, config, question, answering
) -> None:
    model = chat_model(structured_replies=[answering])
    patch_model(model)

    await think({"question": question}, config)

    assert f"- Question: {question}\n" in model.prompts[0]


async def test_the_prompt_carries_the_last_report(
    chat_model, patch_model, config, question, answering
) -> None:
    model = chat_model(structured_replies=[answering])
    patch_model(model)

    await think({"question": question, "report": "what I knew already"}, config)

    assert "<report>\nwhat I knew already\n</report>" in model.prompts[0]


async def test_round_zero_marks_empty_slots(
    chat_model, patch_model, config, question, answering
) -> None:
    model = chat_model(structured_replies=[answering])
    patch_model(model)

    await think({"question": question}, config)

    assert "(none, this is the first round)" in model.prompts[0]


async def test_the_previous_observation_reaches_the_prompt(
    chat_model, patch_model, config, question, observation, answering
) -> None:
    model = chat_model(structured_replies=[answering])
    patch_model(model)

    await think({"question": question, "round": 1, "observation": observation}, config)

    assert "the observatory opened in 1893" in model.prompts[0]


async def test_an_answer_completes_the_run(
    chat_model, patch_model, config, question, answering
) -> None:
    model = chat_model(structured_replies=[answering])
    patch_model(model)

    update = await think({"question": question}, config)

    assert update["answer"] == "1893"
    assert update["stop_reason"] == "answer"


async def test_action_stored_as_plain_dict(
    chat_model, patch_model, config, question, searching
) -> None:
    model = chat_model(structured_replies=[searching])
    patch_model(model)

    update = await think({"question": question}, config)

    assert update["action"] == {"name": "search", "queries": ["who founded it"]}


async def test_the_round_counter_advances(
    chat_model, patch_model, config, question, searching
) -> None:
    model = chat_model(structured_replies=[searching])
    patch_model(model)

    update = await think({"question": question, "round": 4}, config)

    assert update["round"] == 5


async def test_stopped_round_advances_counter(
    chat_model, patch_model, last_round, question, forced
) -> None:
    """A stopped round made its model calls, so it counts toward rounds."""
    model = chat_model(structured_replies=[forced])
    patch_model(model)

    update = await think({"question": question, "round": 3}, last_round)

    assert update["round"] == 4


async def test_the_round_record_names_its_round(
    chat_model, patch_model, config, question
) -> None:
    model = chat_model(structured_replies=[decision("a longer report", answer="1893")])
    patch_model(model)

    update = await think({"question": question, "round": 2}, config)

    assert update["per_round"]["round"] == 2


async def test_round_record_measures_new_report(
    chat_model, patch_model, config, question
) -> None:
    model = chat_model(structured_replies=[decision("a longer report", answer="1893")])
    patch_model(model)

    update = await think({"question": question, "round": 2}, config)

    assert update["per_round"]["report_chars"] == len("a longer report")


async def test_mapping_validated_into_decision(
    chat_model, patch_model, config, question
) -> None:
    model = chat_model(structured_replies=[{"report": "r", "answer": "1893"}])
    patch_model(model)

    update = await think({"question": question}, config)

    assert update["answer"] == "1893"


async def test_long_batch_trimmed_to_ceiling(
    chat_model, patch_model, config, question
) -> None:
    model = chat_model(structured_replies=[decision("r", queries=["a", "b", "c", "d"])])
    patch_model(model)
    config["configurable"]["max_queries_per_search"] = 2

    update = await think({"question": question}, config)

    assert update["action"]["queries"] == ["a", "b"]


async def test_failed_check_stops_after_retries(
    chat_model, patch_model, config, question
) -> None:
    model = chat_model(structured_replies=[ValueError("Invalid JSON")])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 2

    update = await think({"question": question}, config)

    assert update["stop_reason"] == "malformed"
    assert len(model.prompts) == 3


async def test_a_retry_quotes_the_real_error(
    chat_model, patch_model, config, question, answering
) -> None:
    model = chat_model(structured_replies=[ValueError("Invalid JSON at line 1"), answering])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 1

    await think({"question": question}, config)

    assert (
        "Your last response failed the decision check: "
        "The response did not match the schema: Invalid JSON at line 1"
    ) in model.prompts[1]


async def test_retry_hides_api_key(
    chat_model, patch_model, config, question, answering, monkeypatch
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-the-real-provider-key")
    get_settings.cache_clear()
    model = chat_model(
        structured_replies=[ValueError("bad request with sk-the-real-provider-key"), answering]
    )
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 1

    await think({"question": question}, config)

    assert "bad request with [redacted]" in model.prompts[1]


async def test_a_retry_caps_a_long_error(
    chat_model, patch_model, config, question, answering
) -> None:
    model = chat_model(structured_replies=[ValueError("x" * 500), answering])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 1
    config["configurable"]["max_error_chars"] = 100
    capped = ("The response did not match the schema: " + "x" * 500)[:94] + "\n[cut]"

    await think({"question": question}, config)

    assert f"failed the decision check: {capped}\n" in model.prompts[1]


async def test_a_blank_report_fails_the_check(
    chat_model, patch_model, config, question, answering
) -> None:
    model = chat_model(structured_replies=[decision("   ", answer="1893"), answering])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 1

    await think({"question": question}, config)

    assert "failed the decision check: The report is blank." in model.prompts[1]


async def test_blank_answer_alone_fails_check(
    chat_model, patch_model, config, question
) -> None:
    model = chat_model(structured_replies=[decision("r", answer="  ")])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 0

    update = await think({"question": question}, config)

    assert update["stop_reason"] == "malformed"


async def test_blank_answer_with_action_runs_action(
    chat_model, patch_model, config, question
) -> None:
    model = chat_model(
        structured_replies=[decision("r", answer=" ", queries=["who founded it"])]
    )
    patch_model(model)

    update = await think({"question": question}, config)

    assert update["action"]["name"] == "search"


async def test_answer_wins_over_action(
    chat_model, patch_model, config, question
) -> None:
    model = chat_model(
        structured_replies=[decision("r", answer="1893", queries=["who founded it"])]
    )
    patch_model(model)

    update = await think({"question": question}, config)

    assert update["stop_reason"] == "answer"


async def test_no_answer_no_action_fails(
    chat_model, patch_model, config, question
) -> None:
    model = chat_model(structured_replies=[decision("r")])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 0

    update = await think({"question": question}, config)

    assert update["stop_reason"] == "malformed"


async def test_invalid_mapping_fails_check(
    chat_model, patch_model, config, question
) -> None:
    model = chat_model(structured_replies=[{"report": ["not", "text"]}])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 0

    update = await think({"question": question}, config)

    assert update["stop_reason"] == "malformed"


async def test_cut_off_response_fails_check(
    chat_model, patch_model, config, question
) -> None:
    model = chat_model(structured_replies=[Raises(ContentFilterFinishReasonError())])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 0

    update = await think({"question": question}, config)

    assert update["stop_reason"] == "malformed"


async def test_schema_error_not_backed_off(
    chat_model, patch_model, config, question, no_backoff_wait
) -> None:
    """The error text is the model's, so a network word in it is not a network error."""
    model = chat_model(structured_replies=[ValueError("report: lost connection to source")])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 0

    update = await think({"question": question}, config)

    assert update["stop_reason"] == "malformed"


async def test_corrections_accumulate(
    chat_model, patch_model, config, question
) -> None:
    """Resetting the prompt each retry throws away what the model was told."""
    model = chat_model(structured_replies=[ValueError("Invalid JSON")])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 2

    await think({"question": question}, config)

    assert model.prompts[2].count("failed the decision check") == 2


async def test_malformed_stop_asks_final_answer(
    chat_model, patch_model, config, question
) -> None:
    """A stopped run still gets one call to answer from the report."""
    model = chat_model(
        structured_replies=[ValueError("Invalid JSON")],
        replies=["1893, from what I could confirm"],
    )
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 0

    update = await think({"question": question, "report": "partial findings"}, config)

    assert update["answer"] == "1893, from what I could confirm"


async def test_plain_call_sees_last_observation(
    chat_model, patch_model, config, question, observation
) -> None:
    """The report from the last good round never folded this observation in."""
    model = chat_model(structured_replies=[ValueError("Invalid JSON")], replies=["an answer"])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 0

    await think(
        {"question": question, "report": "partial findings", "observation": observation},
        config,
    )

    assert "[1] the observatory opened in 1893" in model.prompts[-1]


async def test_plain_call_forbids_json(
    chat_model, patch_model, config, question
) -> None:
    """The model just failed the JSON format, so the last call asks for plain text."""
    model = chat_model(structured_replies=[ValueError("Invalid JSON")], replies=["an answer"])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 0

    await think({"question": question, "report": "partial findings"}, config)

    assert (
        "Do not write JSON, do not write a report, and do not call a tool. Answer the "
        "question now in plain text, from the Last Report and the Last Tool Response "
        "only." in model.prompts[-1]
    )


UNCONFIRMED = "In the answer, state plainly which parts the research could not confirm."


@pytest.mark.parametrize(
    ("style", "names_gaps"),
    [("report", True), ("short", False)],
    ids=["product-names-gaps", "benchmark-stays-bare"],
)
async def test_plain_call_gaps_follow_style(
    chat_model, patch_model, config, question, style, names_gaps
) -> None:
    """A benchmark grader compares a bare answer, so only a product answer names its gaps."""
    model = chat_model(structured_replies=[ValueError("Invalid JSON")], replies=["an answer"])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 0
    config["configurable"]["answer_style"] = style

    await think({"question": question, "report": "partial findings"}, config)

    assert (UNCONFIRMED in model.prompts[-1]) is names_gaps


@pytest.mark.parametrize(
    ("style", "names_gaps"),
    [("report", True), ("short", False)],
    ids=["product-names-gaps", "benchmark-stays-bare"],
)
async def test_last_round_gaps_follow_style(
    chat_model, patch_model, last_round, question, forced, style, names_gaps
) -> None:
    """A benchmark grader compares a bare answer, so only a product answer names its gaps."""
    model = chat_model(structured_replies=[forced])
    patch_model(model)
    last_round["configurable"]["answer_style"] = style

    await think({"question": question, "round": 3}, last_round)

    assert (UNCONFIRMED in model.prompts[0]) is names_gaps


async def test_plain_call_drops_corrections(
    chat_model, patch_model, config, question
) -> None:
    """The corrections ask for JSON, which the plain call forbids."""
    model = chat_model(structured_replies=[ValueError("Invalid JSON")], replies=["an answer"])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 1

    await think({"question": question, "report": "partial findings"}, config)

    assert "failed the decision check" not in model.prompts[-1]


async def test_last_round_malformed_drops_answer_note(
    chat_model, patch_model, last_round, question
) -> None:
    """The JSON stop note and the plain-text stop note must never meet."""
    model = chat_model(structured_replies=[ValueError("Invalid JSON")], replies=["an answer"])
    patch_model(model)
    last_round["configurable"]["max_decision_retries"] = 0

    await think({"question": question, "report": "partial findings", "round": 3}, last_round)

    assert "this round must answer" not in model.prompts[-1]


async def test_observation_alone_gets_final_call(
    chat_model, patch_model, config, question, observation
) -> None:
    """The full_history arm keeps no report, but its observation is still evidence."""
    model = chat_model(structured_replies=[ValueError("Invalid JSON")], replies=["1893"])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 0

    update = await think({"question": question, "observation": observation}, config)

    assert update["answer"] == "1893"


async def test_failed_final_call_returns_report(
    chat_model, patch_model, config, question, monkeypatch
) -> None:
    """The last call is a best effort. A failure must not lose the report."""
    model = chat_model(structured_replies=[ValueError("Invalid JSON")])
    patch_model(model)

    async def refuse(prompt: str, config: object = None) -> None:
        raise RuntimeError("the provider refused the final call")

    monkeypatch.setattr(model, "ainvoke", refuse)
    config["configurable"]["max_decision_retries"] = 0

    update = await think({"question": question, "report": "partial findings"}, config)

    assert update["answer"] == "partial findings"


async def test_no_report_skips_final_call(
    chat_model, patch_model, config, question
) -> None:
    """There is nothing to answer from, so the call would only cost money."""
    model = chat_model(structured_replies=[ValueError("Invalid JSON")])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 0

    await think({"question": question}, config)

    assert len(model.prompts) == 1


async def test_final_answer_blocks_read_as_text(
    chat_model, patch_model, config, question
) -> None:
    """Reasoning models reply with a list of blocks, not one string."""
    model = chat_model(
        structured_replies=[ValueError("Invalid JSON")],
        replies=[[{"type": "text", "text": "1893"}]],
    )
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 0

    update = await think({"question": question, "report": "partial findings"}, config)

    assert update["answer"] == "1893"


async def test_round_usage_adds_to_totals(
    chat_model, patch_model, config, question, answering
) -> None:
    model = chat_model(structured_replies=[answering], usage=(1000, 200))
    patch_model(model)
    before = {"input_tokens": 500, "output_tokens": 50}

    update = await think({"question": question, "usage": before}, config)

    assert update["usage"] == {"input_tokens": 1500, "output_tokens": 250}


async def test_last_round_prompt_has_stop_note(
    chat_model, patch_model, last_round, question, forced
) -> None:
    model = chat_model(structured_replies=[forced])
    patch_model(model)

    await think({"question": question, "round": 3}, last_round)

    assert "You are out of research rounds, so this round must answer." in model.prompts[0]


async def test_last_round_asks_answered_schema(
    chat_model, patch_model, last_round, question, forced
) -> None:
    """The schema, not the prompt, is what keeps the model from picking a tool."""
    model = chat_model(structured_replies=[forced])
    patch_model(model)

    await think({"question": question, "round": 3}, last_round)

    assert model.schemas == [Answered]


async def test_earlier_round_asks_decision_schema(
    chat_model, patch_model, last_round, question, searching
) -> None:
    model = chat_model(structured_replies=[searching])
    patch_model(model)

    await think({"question": question, "round": 2}, last_round)

    assert model.schemas == [Decision]


async def test_forced_answer_stops_max_rounds(
    chat_model, patch_model, last_round, question, forced
) -> None:
    """The caller must tell a forced answer from a finished one."""
    model = chat_model(structured_replies=[forced])
    patch_model(model)

    update = await think({"question": question, "round": 3}, last_round)

    assert update["stop_reason"] == "max_rounds"


async def test_forced_answer_reaches_caller(
    chat_model, patch_model, last_round, question, forced
) -> None:
    model = chat_model(structured_replies=[forced])
    patch_model(model)

    update = await think({"question": question, "round": 3}, last_round)

    assert update["answer"] == "1893"


async def test_forced_answer_keeps_new_report(
    chat_model, patch_model, last_round, question, forced
) -> None:
    """The new report is the one that folded in the last observation."""
    model = chat_model(structured_replies=[forced])
    patch_model(model)

    update = await think(
        {"question": question, "report": "old report", "round": 3}, last_round
    )

    assert update["report"] == "final report"


async def test_last_round_makes_no_extra_call(
    chat_model, patch_model, last_round, question, forced
) -> None:
    """The last decision call is the answer, so a second call only costs money."""
    model = chat_model(structured_replies=[forced])
    patch_model(model)

    await think({"question": question, "round": 3}, last_round)

    assert len(model.prompts) == 1


async def test_last_round_blank_answer_fails_check(
    chat_model, patch_model, last_round, question
) -> None:
    model = chat_model(
        structured_replies=[Answered(report="final report", answer="  ")],
        replies=["1893"],
    )
    patch_model(model)
    last_round["configurable"]["max_decision_retries"] = 0

    update = await think({"question": question, "round": 3}, last_round)

    assert update["stop_reason"] == "malformed"


async def test_last_round_retry_asks_for_answer(
    chat_model, patch_model, last_round, question, forced
) -> None:
    """The usual correction asks for an action, which this schema cannot hold."""
    model = chat_model(structured_replies=[ValueError("Invalid JSON"), forced])
    patch_model(model)

    await think({"question": question, "round": 3}, last_round)

    assert model.prompts[1].endswith(
        "Answer again as one JSON object with a non-blank report and a non-blank answer."
    )


async def test_stopped_round_records_observation(
    chat_model, patch_model, config, question, observation
) -> None:
    """A stop reason is not a reason to lose the round's accounting."""
    model = chat_model(structured_replies=[ValueError("Invalid JSON")])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 0

    update = await think({"question": question, "observation": observation}, config)

    assert update["per_round"]["observation_chars"] == len(
        "[1] the observatory opened in 1893"
    )


async def test_full_history_records_observations(
    chat_model, patch_model, config, set_env, question, observation, answering
) -> None:
    """The ablation charts compare what each arm actually put in the prompt."""
    model = chat_model(structured_replies=[answering])
    patch_model(model)
    set_env(CARRY_FULL_HISTORY="true")

    update = await think(
        {"question": question, "round": 2, "history": [observation, observation]}, config
    )

    call = '{"name": "search", "queries": ["founding year"]}'
    carried = (
        f"--- Round 1: {call} ---\n[1] the observatory opened in 1893\n"
        f"--- Round 2: {call} ---\n[1] the observatory opened in 1893"
    )
    assert update["per_round"]["observation_chars"] == len(carried)


async def test_full_history_prompt_has_past_observations(
    chat_model, patch_model, config, set_env, question, observation, answering
) -> None:
    model = chat_model(structured_replies=[answering])
    patch_model(model)
    set_env(CARRY_FULL_HISTORY="true")
    older = {**observation, "results": ["an older finding"]}

    await think(
        {"question": question, "round": 2, "history": [older, observation]}, config
    )

    prompt = model.prompts[0]
    assert "an older finding" in prompt
    assert "the observatory opened in 1893" in prompt


async def test_rate_limit_spares_malformed_retry(
    chat_model, patch_model, config, question, answering, no_backoff_wait
) -> None:
    model = chat_model(structured_replies=[Raises(_RateLimited("429")), answering])
    patch_model(model)
    config["configurable"]["max_decision_retries"] = 0

    update = await think({"question": question}, config)

    assert update["stop_reason"] == "answer"


async def test_served_round_counts_rate_limits(
    chat_model, patch_model, config, question, answering, no_backoff_wait
) -> None:
    """A served run has no caller counters, so state carries what it waited for."""
    model = chat_model(structured_replies=[Raises(_RateLimited("429")), answering])
    patch_model(model)
    spent = {
        "rate_limit_hits": 2,
        "rate_limit_wait_seconds": 3.0,
        "transient_retries": 0,
        "transient_wait_seconds": 0.0,
        "malformed_summaries": 0,
    }

    update = await think({"question": question, "retries": spent}, config)

    assert update["retries"]["rate_limit_hits"] == 3


async def test_last_tool_call_uses_action_shape(
    chat_model, patch_model, config, question, answering
) -> None:
    """The model copies the shape it is shown, so it must match the schema."""
    model = chat_model(structured_replies=[answering])
    patch_model(model)
    state = {
        "question": question,
        "action": {"name": "search", "queries": ["founding year"]},
    }

    await think(state, config)

    assert '{"name": "search", "queries": ["founding year"]}' in model.prompts[0]


async def test_round_leaves_env_unchanged(
    chat_model, patch_model, config, question, answering
) -> None:
    """Runs share one process, so a round that edits it changes other runs."""
    model = chat_model(structured_replies=[answering])
    patch_model(model)
    before = dict(os.environ)

    await think({"question": question}, config)

    assert dict(os.environ) == before


async def test_exhausted_transport_stops_with_error(
    chat_model, patch_model, config, question, no_backoff_wait
) -> None:
    model = chat_model(structured_replies=[Raises(TimeoutError("provider timed out"))])
    patch_model(model)
    config["configurable"]["max_retries"] = 0

    update = await think({"question": question}, config)

    assert update["stop_reason"] == "error"


async def test_transport_stop_skips_final_call(
    chat_model, patch_model, config, question, no_backoff_wait
) -> None:
    """The provider just failed, so one more call would fail too."""
    model = chat_model(structured_replies=[Raises(TimeoutError("provider timed out"))])
    patch_model(model)
    config["configurable"]["max_retries"] = 0

    await think({"question": question, "report": "partial findings"}, config)

    assert model.plain_calls == 0


async def test_transport_stop_answers_with_report(
    chat_model, patch_model, config, question, no_backoff_wait
) -> None:
    model = chat_model(structured_replies=[Raises(TimeoutError("provider timed out"))])
    patch_model(model)
    config["configurable"]["max_retries"] = 0

    update = await think({"question": question, "report": "partial findings"}, config)

    assert update["answer"] == "partial findings"


async def test_finished_thread_refuses_without_model_call(
    chat_model, patch_model, config, answering
) -> None:
    model = chat_model(structured_replies=[answering])
    patch_model(model)
    finished = {
        "question": "a new question",
        "report": "the old question's report",
        "round": 12,
        "stop_reason": "max_rounds",
    }

    with pytest.raises(FinishedThread):
        await think(finished, config)

    assert model.prompts == []
