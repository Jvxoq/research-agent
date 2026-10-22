"""Research agent nodes.

Two nodes and one router. ``think`` rebuilds the workspace and makes one model
call. ``act`` runs the single chosen action, whose batch runs concurrently
inside the tool. The workspace is built fresh every round, so state never grows
with the round count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from loguru import logger

from agents.research_agent.configuration import Configuration
from agents.research_agent.prompts import (
    ANSWER_INSTRUCTIONS,
    EMPTY_SLOT,
    LOOK_UP_PROMPT,
    ROUND_PROMPT,
    get_current_date,
    render_tools,
)
from agents.research_agent.schemas import (
    Acting,
    Action,
    Answered,
    Decision,
    FinishedThread,
    MalformedDecision,
    MissingThread,
    SearchAction,
    trim_action,
)
from agents.research_agent.state import (
    Observation,
    ResearchAgentState,
    ResearchAgentUpdate,
    RoundRecord,
    StopReason,
    ToolCall,
)
from core.config import get_settings
from core.cost import TokenTracker
from core.llm import (
    UNUSABLE_OUTPUT_ERRORS,
    StructuredOutputError,
    build_chat_model,
    read_structured,
)
from core.retry import Retrier, RetriesExhausted, counters_from_config
from core.tools.text import cap, redact
from core.tools.web_look_up import LookUpSettings, look_up
from core.tools.web_search import search

CORRECTION = (
    "\n\nYour last response failed the decision check: {error}\n"
    "Answer again as one JSON object. Write a non-blank report, then set exactly "
    "one of answer or action."
)

FINAL_CORRECTION = (
    "\n\nYour last response failed the decision check: {error}\n"
    "Answer again as one JSON object with a non-blank report and a non-blank answer."
)

# The last round's call asks for ``Answered``, so the provider enforces the "no
# tool" part. The note has to say so too, because ROUND_PROMPT lists the tools.
FINAL_ANSWER_NUDGE = (
    "\n\n## STOP NOTE\n"
    "You are out of research rounds, so this round must answer. Tools are no longer "
    'available. Output one JSON object with two fields, "report" and "answer". There '
    'is no "action" field. Fold the Last Tool Response into the report as usual, then '
    "answer from the report.{limits} Do not apologise and do not describe the run."
)

# Plain text, because the model reaching this note just failed the JSON format
# several times and would likely fail it again.
PLAIN_ANSWER_NOTE = (
    "\n\n## STOP NOTE\n"
    "Your responses kept failing the output format, so the research has stopped. "
    "Ignore the output contract above. Do not write JSON, do not write a report, and "
    "do not call a tool. Answer the question now in plain text, from the Last Report "
    "and the Last Tool Response only.{limits} Do not apologise and do not describe the "
    "run. Output only the answer text."
)

# A product answer names its own gaps for the person reading it. A benchmark
# answer stays bare, because the grader compares it with a short gold answer.
STOP_LIMITS = {
    "short": "",
    "report": " In the answer, state plainly which parts the research could not confirm.",
}

# A fixed message, because the caller reads it back over the API.
FINISHED_THREAD = "This thread already finished its research. Start a new thread."
MISSING_THREAD = "A run needs a thread_id. Start it on a thread."


@dataclass(frozen=True)
class _Round:
    """What one ``think`` round built before it had to stop.

    Args:
        index: The round number.
        prompt_chars: Size of the workspace the round built.
        workspace: The round's prompt, before any stop note or correction.
        configuration: This run's configuration.
        retrier: The run's backoff policy and counters.
        tracker: The round's token tracker, holding the run's totals so far.
        config: The node's config.
    """

    index: int
    prompt_chars: int
    workspace: str
    configuration: Configuration
    retrier: Retrier
    tracker: TokenTracker
    config: RunnableConfig


def _render_action(action: ToolCall | None) -> str:
    """Render a tool call for the prompt's Last Tool Call slot.

    Rendered as real JSON in the same flat shape as the ``action`` field the
    model fills, because the model copies the shape it is shown.
    """
    if action is None:
        return EMPTY_SLOT
    return json.dumps(action)


def _to_call(action: Action) -> ToolCall:
    """Turn a checked action into the flat tool call state carries."""
    if isinstance(action, SearchAction):
        return {"name": "search", "queries": action.queries}
    return {"name": "look_up", "urls": action.urls, "goal": action.goal}


def _render_observation(observation: Observation | None) -> str:
    """Render one observation for the prompt's Last Tool Response slot.

    Each batch entry shows its target, then its result or its error.

    Args:
        observation: Last round's observation, if any.

    Returns:
        Prompt text, or the empty-slot marker on the first round.
    """
    if observation is None:
        return EMPTY_SLOT

    blocks: list[str] = []
    for index, result in enumerate(observation["results"]):
        error = observation["errors"][index]
        blocks.append(f"[{index + 1}] ERROR: {error}" if error else f"[{index + 1}] {result}")
    return "\n\n".join(blocks) if blocks else EMPTY_SLOT


def _render_history(history: list[Observation]) -> str:
    """Render every past observation, for the ablation baseline only.

    Args:
        history: Every observation so far.

    Returns:
        Prompt text covering the whole run.
    """
    if not history:
        return EMPTY_SLOT

    blocks: list[str] = []
    for round_index, observation in enumerate(history):
        blocks.append(f"--- Round {round_index + 1}: {json.dumps(observation['call'])} ---")
        blocks.append(_render_observation(observation))
    return "\n".join(blocks)


def _render_observation_slot(
    state: ResearchAgentState, configuration: Configuration
) -> str:
    """Render the observation text this run's arm puts in the prompt.

    Args:
        state: Current graph state.
        configuration: This run's configuration.

    Returns:
        Every past observation in the `full_history` arm, otherwise the last one.
    """
    if configuration.carry_full_history:
        return _render_history(state.get("history", []))
    return _render_observation(state.get("observation"))


def _tool_target_counts(observation: Observation | None) -> tuple[int, int]:
    """Count the targets of the tool batch a round read, and those that failed.

    Every round after the first reads exactly one batch, the one the previous
    ``act`` wrote, so summing these over a run's rounds counts each batch once.

    Args:
        observation: The last tool batch, or None in the first round.

    Returns:
        The number of search queries or URLs in the batch, and how many of them
        ended in an error.
    """
    if observation is None:
        return 0, 0
    errors = observation["errors"]
    return len(errors), sum(1 for error in errors if error is not None)


def _build_workspace(state: ResearchAgentState, configuration: Configuration) -> str:
    """Build this round's prompt from the state.

    The `report` arm passes the evolving report plus one previous step. The
    `full_history` arm passes every step so far and no report. That single
    branch is the whole ablation.

    Args:
        state: Current graph state, with ``current_date`` already set.
        configuration: This run's configuration.

    Returns:
        The full prompt text for one round.
    """
    if configuration.carry_full_history:
        report = "(not carried in this arm)"
    else:
        report = state.get("report") or EMPTY_SLOT

    prompt = ROUND_PROMPT.format(
        max_report_chars=configuration.max_report_chars,
        answer_instructions=ANSWER_INSTRUCTIONS[configuration.answer_style],
        tools=render_tools(
            configuration.max_queries_per_search,
            configuration.max_urls_per_look_up,
            configuration.max_search_results,
        ),
        current_date=state["current_date"],
        question=state["question"],
        report=report,
        action=_render_action(state.get("action")),
        observation=_render_observation_slot(state, configuration),
    )

    if configuration.warn_prompt_chars and len(prompt) > configuration.warn_prompt_chars:
        logger.warning(
            f"workspace is {len(prompt)} chars, over warn_prompt_chars "
            f"({configuration.warn_prompt_chars})"
        )
    return prompt


async def _decide(
    prompt: str,
    configuration: Configuration,
    retrier: Retrier,
    run_config: RunnableConfig,
    must_answer: bool,
) -> tuple[Answered | Acting, list[str]]:
    """Make one structured model call and verify the decision it returns.

    The call itself is retried for the two transport families. The check is
    not, because a response that fails it is not a transport problem and
    waiting does not improve it.

    ``include_raw`` hands a parsing failure back as a value instead of raising
    it inside the retried call. A validation message quotes the model's own
    text, and a word like "connection" in that text would otherwise read as a
    transient network error and be backed off.

    Args:
        prompt: This round's workspace.
        configuration: This run's configuration.
        retrier: The run's backoff policy and counters.
        run_config: The node's config with the run's cost tracker attached.
        must_answer: Ask for ``Answered`` instead of ``Decision``, so the
            response cannot hold a tool call.

    Returns:
        The verified decision, and notes about anything that was repaired.

    Raises:
        MalformedDecision: If the response fails the decision check.
    """
    schema: type[Decision] | type[Answered] = Answered if must_answer else Decision
    model = build_chat_model(configuration.model_spec_for("think"))
    structured = model.with_structured_output(
        schema, method="json_schema", include_raw=True
    )

    try:
        result = await retrier.run(
            lambda: structured.ainvoke(prompt, run_config), "think"
        )
    except UNUSABLE_OUTPUT_ERRORS as error:
        raise MalformedDecision(
            f"The response stopped before the JSON object was complete: {error}"
        ) from error

    try:
        decision = read_structured(result, schema)
    except StructuredOutputError as error:
        raise MalformedDecision(str(error)) from error

    report = decision.report.strip()
    if not report:
        raise MalformedDecision("The report is blank.")

    notes: list[str] = []
    answer = (decision.answer or "").strip()
    if isinstance(decision, Answered):
        if not answer:
            raise MalformedDecision("The answer is blank.")
        return Answered(report=report, answer=answer), notes
    if answer:
        if decision.action is not None:
            notes.append("Both answer and action present; kept the answer.")
        return Answered(report=report, answer=answer), notes
    if decision.action is None:
        raise MalformedDecision("Neither a non-blank answer nor an action in the response.")
    action, trimmed = trim_action(
        decision.action,
        configuration.max_queries_per_search,
        configuration.max_urls_per_look_up,
    )
    notes.extend(trimmed)
    return Acting(report=report, action=action), notes


async def think(
    state: ResearchAgentState, config: RunnableConfig
) -> ResearchAgentUpdate:
    """Rebuild the workspace, call the model once, and decide what to do next.

    Fills the round-zero defaults itself, so round 0 runs the same code as every
    other round. A malformed response is retried without backoff, because
    waiting does not improve a model's formatting.

    Args:
        state: Current graph state.
        config: Runnable configuration for this run.

    Returns:
        The rewritten report, the chosen action, the round counter, the run's
        usage and retry totals, and a stop reason when the run must end.

    Raises:
        MissingThread: The configuration names no ``thread_id``.
        FinishedThread: The thread already holds a finished run. Its report,
            round and usage belong to another question.
    """
    # LangChain copies a configurable thread_id onto every traced child run,
    # and LangSmith groups a run's calls into one thread by it. A run without
    # one scatters its calls across the project.
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        raise MissingThread(MISSING_THREAD)

    # A run only reaches think while stop_reason is unset, so a set one means a
    # new run started on a thread whose research already ended.
    if state.get("stop_reason") is not None:
        raise FinishedThread(FINISHED_THREAD)

    configuration = Configuration.from_runnable_config(config)

    current_date = (
        state.get("current_date")
        or configuration.current_date_override
        or get_current_date()
    )
    round_index = state.get("round", 0)
    state = ResearchAgentState(
        {**state, "current_date": current_date, "round": round_index}
    )

    retrier = Retrier(
        policy=configuration.backoff_policy(),
        counters=counters_from_config(config, state.get("retries")),
    )
    tracker = TokenTracker(spent=state.get("usage"))

    workspace = _build_workspace(state, configuration)
    out_of_rounds = round_index >= configuration.max_rounds
    limits = STOP_LIMITS[configuration.answer_style]
    prompt = workspace + FINAL_ANSWER_NUDGE.format(limits=limits) if out_of_rounds else workspace
    correction = FINAL_CORRECTION if out_of_rounds else CORRECTION

    logger.info(
        f"think round={round_index} prompt_chars={len(prompt)} "
        f"waited={retrier.counters.to_dict()}"
    )
    this_round = _Round(
        index=round_index,
        prompt_chars=len(prompt),
        workspace=workspace,
        configuration=configuration,
        retrier=retrier,
        tracker=tracker,
        config=config,
    )

    run_config = merge_configs(config, {"callbacks": [tracker]})
    decision: Answered | Acting | None = None
    notes: list[str] = []
    malformed = 0
    last_error = ""
    attempt_prompt = prompt

    for attempt in range(configuration.max_decision_retries + 1):
        try:
            decision, notes = await _decide(
                attempt_prompt, configuration, retrier, run_config, out_of_rounds
            )
            break
        except MalformedDecision as error:
            malformed += 1
            # The message quotes the model's output and the provider's, so it is
            # observation text: redacted, then capped, before prompt or log.
            last_error = cap(
                redact(str(error), (get_settings().LLM_API_KEY,)),
                configuration.max_error_chars,
            )
            attempt_prompt += correction.format(error=last_error)
            logger.warning(f"malformed decision, attempt {attempt + 1}: {last_error}")
        except RetriesExhausted:
            logger.exception("the decision call spent its transport retry budget")
            return await _stop_run(state, "error", this_round, malformed)

    for note in notes:
        logger.warning(f"decision repaired: {note}")

    if decision is None:
        logger.error(f"no valid decision after retries: {last_error}")
        return await _stop_run(state, "malformed", this_round, malformed)

    tool_targets, tool_target_errors = _tool_target_counts(state.get("observation"))
    record: RoundRecord = {
        "round": round_index,
        "prompt_chars": len(prompt),
        "report_chars": len(decision.report),
        "observation_chars": len(_render_observation_slot(state, configuration)),
        "action_name": decision.action.name if isinstance(decision, Acting) else None,
        "malformed_retries": malformed,
        "tool_targets": tool_targets,
        "tool_target_errors": tool_target_errors,
    }

    update: ResearchAgentUpdate = {
        "current_date": current_date,
        "report": decision.report,
        "round": round_index + 1,
        "per_round": record,
        "usage": tracker.totals(),
        "retries": retrier.counters.to_dict(),
    }

    if isinstance(decision, Answered):
        update["answer"] = decision.answer
        update["action"] = None
        # A forced answer is not a finished one, and the caller must tell the two
        # apart.
        update["stop_reason"] = "max_rounds" if out_of_rounds else "answer"
        return update

    update["action"] = _to_call(decision.action)
    return update


async def _answer_in_plain_text(
    state: ResearchAgentState, this_round: _Round
) -> str | None:
    """Ask the model once for a plain-text answer from this round's workspace.

    The workspace still holds the last observation, which the report from the
    last good round never folded in.

    Args:
        state: Current graph state.
        this_round: What the stopped round built.

    Returns:
        The model's answer, or None when there is nothing to answer from or
        the call failed.
    """
    if not state.get("report") and state.get("observation") is None:
        return None

    prompt = this_round.workspace + PLAIN_ANSWER_NOTE.format(
        limits=STOP_LIMITS[this_round.configuration.answer_style]
    )
    run_config = merge_configs(this_round.config, {"callbacks": [this_round.tracker]})
    try:
        model = build_chat_model(this_round.configuration.model_spec_for("think"))
        reply = await this_round.retrier.run(
            lambda: model.ainvoke(prompt, run_config), "think.final"
        )
    except Exception:  # noqa: BLE001 - the report is the fallback
        logger.exception("the final answer call failed; falling back to the report")
        return None

    return reply.text.strip() or None


async def _stop_run(
    state: ResearchAgentState,
    reason: StopReason,
    this_round: _Round,
    malformed: int,
) -> ResearchAgentUpdate:
    """Return the update for a run that must stop without a usable decision.

    A ``malformed`` stop gets one last plain-text call, and the raw report is
    the fallback when that call fails too. An ``error`` stop makes no call,
    because the decision call itself spent its transport retries.

    Args:
        state: Current graph state, holding the report from the last good round.
        reason: The stop reason to record.
        this_round: What the stopped round built.
        malformed: Malformed responses seen this round.

    Returns:
        A state update ending the run, with the run's usage and retry totals.
    """
    report = state.get("report") or ""
    answer: str | None = None
    if reason == "malformed":
        answer = await _answer_in_plain_text(state, this_round)
    tool_targets, tool_target_errors = _tool_target_counts(state.get("observation"))
    record: RoundRecord = {
        "round": this_round.index,
        "prompt_chars": this_round.prompt_chars,
        "report_chars": len(report),
        "observation_chars": len(
            _render_observation_slot(state, this_round.configuration)
        ),
        "action_name": None,
        "malformed_retries": malformed,
        "tool_targets": tool_targets,
        "tool_target_errors": tool_target_errors,
    }
    return {
        "answer": (
            answer or report or "The research run stopped before it found an answer."
        ),
        "report": report,
        "action": None,
        "stop_reason": reason,
        # The stopped round still made its model calls, so it counts as a round.
        "round": this_round.index + 1,
        "per_round": record,
        "usage": this_round.tracker.totals(),
        "retries": this_round.retrier.counters.to_dict(),
    }


async def act(
    state: ResearchAgentState, config: RunnableConfig
) -> ResearchAgentUpdate:
    """Run the single action the model chose.

    The batch inside that action runs concurrently within the tool itself. One
    failing target fills its own error slot and the batch still returns.

    Args:
        state: Current graph state, holding the pending action.
        config: Runnable configuration for this run.

    Returns:
        The observation for this round, replacing the previous one, and the
        run's usage and retry totals.
    """
    configuration = Configuration.from_runnable_config(config)
    call = state.get("action")
    # route_after_think sends a run here only with an action pending.
    assert call is not None
    logger.info(f"act round={state.get('round')} tool={call['name']}")

    retrier = Retrier(
        policy=configuration.backoff_policy(),
        counters=counters_from_config(config, state.get("retries")),
    )
    tracker = TokenTracker(spent=state.get("usage"))

    results: list[str]
    errors: list[str | None]
    if call["name"] == "search":
        results, errors = await search(
            queries=call["queries"],
            max_results=configuration.max_search_results,
            concurrency=configuration.tool_concurrency,
            timeout_seconds=configuration.tool_timeout_seconds,
            deadline_seconds=configuration.target_deadline_seconds,
            api_key=get_settings().TAVILY_API_KEY,
            base_url=get_settings().TAVILY_BASE_URL,
            retrier=retrier,
        )
    else:
        batch = await look_up(
            urls=call["urls"],
            settings=LookUpSettings(
                goal=call["goal"],
                max_page_chars=configuration.look_up_max_page_chars,
                max_summary_chars=configuration.look_up_summary_max_chars,
                max_error_chars=configuration.max_error_chars,
                max_summary_retries=configuration.max_summary_retries,
                concurrency=configuration.tool_concurrency,
                timeout_seconds=configuration.tool_timeout_seconds,
                deadline_seconds=configuration.target_deadline_seconds,
                api_key=get_settings().TAVILY_API_KEY,
                base_url=get_settings().TAVILY_BASE_URL,
            ),
            summariser=build_chat_model(configuration.model_spec_for("look_up")),
            prompt_template=LOOK_UP_PROMPT,
            retrier=retrier,
            run_config=merge_configs(config, {"callbacks": [tracker]}),
        )
        results, errors = batch.results, batch.errors
        retrier.counters.malformed_summaries += batch.malformed_summaries

    capped = False
    for index, result in enumerate(results):
        if len(result) > configuration.max_tool_chars:
            logger.warning(
                f"{call['name']} result {index} hit max_tool_chars "
                f"({len(result)} chars)"
            )
            results[index] = cap(result, configuration.max_tool_chars)
            capped = True

    observation: Observation = {
        "call": call,
        "results": results,
        "errors": errors,
        "capped": capped,
    }

    update: ResearchAgentUpdate = {
        "observation": observation,
        "usage": tracker.totals(),
        "retries": retrier.counters.to_dict(),
    }
    if configuration.carry_full_history:
        update["history"] = [observation]
    return update


def route_after_think(state: ResearchAgentState) -> str:
    """Send the run to the tools, or end it.

    Args:
        state: Current graph state.

    Returns:
        ``"act"`` to run the chosen action, or ``"__end__"``.
    """
    if state.get("stop_reason") is not None:
        return "__end__"
    return "act" if state.get("action") else "__end__"
