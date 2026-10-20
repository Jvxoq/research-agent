"""Tests for token accounting."""

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from core.cost import TokenTracker


def _result(prompt: int, completion: int) -> LLMResult:
    """Build a minimal LLMResult carrying the usage a provider would return."""
    return LLMResult(
        generations=[[]],
        llm_output={
            "model_name": "claude-opus-5",
            "token_usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
            },
        },
    )


def test_tokens_accumulate_across_calls():
    tracker = TokenTracker()
    tracker.on_llm_end(_result(1000, 500))
    tracker.on_llm_end(_result(2000, 100))
    assert tracker.totals() == {"input_tokens": 3000, "output_tokens": 600}


def test_no_usage_counts_zero():
    """A provider that omits usage must not break the run."""
    tracker = TokenTracker()
    tracker.on_llm_end(LLMResult(generations=[[]], llm_output={"model_name": "x"}))
    assert tracker.totals() == {"input_tokens": 0, "output_tokens": 0}


def test_tracker_adds_to_prior_usage():
    """Each node builds its own tracker, so the totals must carry earlier nodes."""
    tracker = TokenTracker(spent={"input_tokens": 9_000, "output_tokens": 10})

    tracker.on_llm_end(_result(2_000, 0))

    assert tracker.totals() == {"input_tokens": 11_000, "output_tokens": 10}


def _message_usage_result(llm_output: dict[str, object]) -> LLMResult:
    """Build an LLMResult whose usage sits on the message, as some clients report it."""
    message = AIMessage(
        content="",
        usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
    )
    return LLMResult(
        generations=[[ChatGeneration(message=message)]], llm_output=llm_output
    )


def test_message_usage_counted_as_fallback():
    tracker = TokenTracker()

    tracker.on_llm_end(_message_usage_result({"model_name": "claude-opus-5"}))

    assert tracker.totals() == {"input_tokens": 120, "output_tokens": 30}


def test_zero_usage_keeps_message_usage():
    """A client can fill token_usage with zeros and put the real counts on the message."""
    tracker = TokenTracker()
    zeros = {"prompt_tokens": 0, "completion_tokens": 0}

    tracker.on_llm_end(_message_usage_result({"token_usage": zeros}))

    assert tracker.totals() == {"input_tokens": 120, "output_tokens": 30}
