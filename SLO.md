# SLO

An SLO (Service Level Objective) is a target for one number.
Each key below is a metric. Each value is its target.
"baseline" means the last accepted run on the same eval set.

```yaml
# Quality
task_accuracy: ">= baseline"
groundedness: ">= 95% of claims cite a source"
hallucination_rate: "<= 2% of answers"
tool_call_success_rate: ">= 98% of tool calls"
malformed_output_rate: "< 1% of model calls"
stop_reason_max_rounds: "< 5% of runs"
stop_reason_error: "< 1% of runs"
user_thumbs_up_rate: ">= 80% of rated answers"

# Speed and reliability
ttft_p95: "< 2 s"
latency_p50: "< 90 s"
latency_p95: "< 3 min"
latency_p99: "< 5 min"
rounds_per_task_p95: "<= 8"
run_success_rate: ">= 99% of runs"
timeout_rate: "< 0.5% of runs"
retries_rate_limit: "<= 1 per task on average"
retries_transient: "<= 1 per task on average"
retries_bad_response: "<= 1 per task on average"
rate_limit_wait_p95: "< 10 s per task"
throughput: "all speed SLOs hold at 50 concurrent users"

# Cost and efficiency
cost_per_task: "< $0.50"
cost_per_correct_answer: "< $0.50 / task_accuracy"
tokens_per_task: "<= baseline"
prompt_size_per_round: "last round size <= 1.1 x round 2 size, think prompt only (see Notes)"
model_calls_per_task_p95: "<= 25"
cache_hit_rate: "report only, no target (see Notes)"

# Security
prompt_injection_attack_success_rate: "< 5% on the poisoned-page set"
jailbreak_attack_success_rate: "< 5% on the jailbreak set"
secret_leak_rate: "0 in outputs, logs, state and traces"
pii_leak_rate: "0 in outputs, logs, state and traces"
unsafe_fetch_block_rate: "100% of private, loopback and link-local URLs"
unauthorized_tool_action_rate: "0 tool calls outside the allowed set"
caller_override_rate: "0 credentials, endpoints or raised ceilings taken from a request"
cross_user_access_rate: "0 reads of another user's thread"
uncapped_output_rate: "0 tool results or errors over the cap"
irreversible_action_confirmation_rate: "100% of irreversible actions"
open_critical_high_findings: "0 from red team and penetration test"
critical_high_dependency_cves: "0 known"
```

## Golden baseline

This is the standard. Every "baseline" in the targets above means these values.
A new run on the same eval set is compared against this table.

Run: the golden set, 30 FRAMES questions, 2026-10-20. Think and look-up model
`zai-org/GLM-5.3-Flash` at `reasoning_effort = low`, `answer_style = report`.
Judge `deepseek-ai/DeepSeek-V4-Flash-0731`. Lines in
`backend/results/golden/baseline_run.jsonl`. Model call counts and cached tokens
come from the LangSmith traces of the same run, because the result lines do not
record them.

| Metric | Golden value | How it was measured |
|---|---|---|
| task_accuracy | 96.7% (29 of 30) | `correct` per line. The miss is `g27`. |
| stop_reason_max_rounds | 0% (0 of 30) | `stop_reason` per line. All 30 stopped with `answer`. |
| stop_reason_error | 0% (0 of 30) | `stop_reason` per line. |
| run_success_rate | 100% (30 of 30) | Lines with `stop_reason = answer`. |
| malformed_output_rate | 1.4% (1 of 73 model calls) | `malformed_count` summed, over the call count from the traces. |
| latency_p50 | 13.2 s | `seconds` per line. |
| latency_p95 | 23.2 s | `seconds` per line. |
| latency_p99 | 28.4 s | `seconds` per line. |
| rounds_per_task_p95 | 3 | `rounds` per line. |
| retries_rate_limit | 0 per task | `rate_limit_hits` per line. |
| retries_transient | 0 per task | `transient_retries` per line. |
| retries_bad_response | 0.03 per task (1 in 30) | `malformed_count` per line. |
| rate_limit_wait_p95 | 0 s | `rate_limit_wait_seconds` per line. |
| cost_per_task | $0.0014 | `usd` per line. The whole run cost $0.042. |
| cost_per_correct_answer | $0.0015 | $0.042 / 29 correct. |
| tokens_per_task | 6,988 | `tokens` per line. The whole run used 209,642. |
| model_calls_per_task_p95 | 4 (max 5) | LLM runs per trace in LangSmith. |
| cache_hit_rate | 2% (3,520 of 178,554 input tokens) | `cache_read` per LLM run in LangSmith. |
| prompt_size_per_round | 1.04 (13.8k to 14.4k chars) | Not from the golden set, which never passes round 2. From BrowseComp question 776, spec.md section 9.1. |

The golden set does not measure the rest. `groundedness`, `hallucination_rate`,
`tool_call_success_rate`, `ttft_p95`, `timeout_rate`, `throughput`, `user_thumbs_up_rate`
and every security metric need their own eval.

## Metrics

Fill the Current column after each measured run. Write the date and the eval set next to it.

| Metric | What it measures | Current |
|---|---|---|
| task_accuracy | Share of eval questions answered correctly. This is X on FRAMES. | 96.7%, 29 of 30 (golden, 2026-10-20). FRAMES subset: 95%, 19 of 20 (FRAMES m8, 2026-09-20) |
| groundedness | Share of claims backed by a cited source. | not measured |
| hallucination_rate | Share of answers with a claim no source supports. | not measured |
| tool_call_success_rate | Valid tool calls divided by all tool calls. | not measured |
| malformed_output_rate | Share of model replies that fail the schema. | 1.4%, 1 of 73 calls (golden, 2026-10-20). Misses the target |
| stop_reason_max_rounds | Share of runs that ran out of rounds. | 0%, 0 of 30 (golden, 2026-10-20) |
| stop_reason_error | Share of runs that ended in an error. | 0%, 0 of 30 (golden, 2026-10-20) |
| user_thumbs_up_rate | Share of rated answers that real users liked. | not measured |
| ttft_p95 | Time to the first streamed update, for 95% of runs. | not measured |
| latency_p50 | Time to the final answer, for 50% of runs. | 13.2 s (golden, 2026-10-20) |
| latency_p95 | Time to the final answer, for 95% of runs. | 23.2 s (golden, 2026-10-20) |
| latency_p99 | Time to the final answer, for 99% of runs. | 28.4 s (golden, 2026-10-20) |
| rounds_per_task_p95 | Think-act rounds a run needs, for 95% of runs. | 3 (golden, 2026-10-20) |
| run_success_rate | Share of runs that return an answer with no error. | 100%, 30 of 30 (golden, 2026-10-20) |
| timeout_rate | Share of runs that hit a timeout. | not measured |
| retries_rate_limit | Retries after a provider rate limit, per task. | 0 per task (golden, 2026-10-20) |
| retries_transient | Retries after a network or server error, per task. | 0 per task (golden, 2026-10-20) |
| retries_bad_response | Retries after a reply that failed the schema, per task. | 0.03 per task (golden, 2026-10-20) |
| rate_limit_wait_p95 | Seconds spent waiting on rate limits, for 95% of tasks. | 0 s (golden, 2026-10-20) |
| throughput | Runs finished per minute at 10, 50 and 100 users. | not measured |
| cost_per_task | Dollars spent on one task. This is $Y. | $0.0014 (golden, 2026-10-20). $0.0019 (FRAMES m8, 2026-09-20) |
| cost_per_correct_answer | Total cost divided by correct answers. | $0.0015 (golden, 2026-10-20). $0.0020 (FRAMES m8, 2026-09-20) |
| tokens_per_task | Input and output tokens for one task. | 6,988 (golden, 2026-10-20) |
| prompt_size_per_round | Prompt tokens in each round. This shows Z. | 1.04, round 2 13.8k to round 8 14.4k chars (BrowseComp 776, 2026-09-20) |
| model_calls_per_task_p95 | Model calls in one task, for 95% of tasks. | 4, max 5 (golden, 2026-10-20, LangSmith traces) |
| cache_hit_rate | Share of input tokens served from the prompt cache. | 2% (golden, 2026-10-20, LangSmith traces) |
| prompt_injection_attack_success_rate | Share of pages with hidden instructions that change what the agent does. | 0% (0 of 29 exposed cases, `results/injection_run3.jsonl`, 2026-10-22) |
| jailbreak_attack_success_rate | Share of jailbreak questions that get a harmful answer. | not measured |
| secret_leak_rate | API keys or tokens found in outputs, logs, state or traces. | not measured |
| pii_leak_rate | Personal data found where it should not be. | not measured |
| unsafe_fetch_block_rate | Share of private or local URLs the fetcher refuses. | not measured |
| unauthorized_tool_action_rate | Tool calls the agent made outside its allowed set. | not measured |
| caller_override_rate | Credentials, endpoints or raised limits a request managed to set. | not measured |
| cross_user_access_rate | Times one user read another user's thread. | not measured |
| uncapped_output_rate | Tool results or error messages longer than their cap. | not measured |
| irreversible_action_confirmation_rate | Share of actions that cannot be undone and asked a human first. | not measured |
| open_critical_high_findings | Critical or high issues still open from a red team or penetration test. | not measured |
| critical_high_dependency_cves | Known critical or high vulnerabilities in installed packages. | not measured |

## Notes

**cache_hit_rate was lowered from ">= 50% of input tokens" to report only (2026-10-20).**
The provider caches the start of a prompt automatically. No parameter turns it on.
Only the fixed instructions at the top of the round prompt can be cached. They are
about 700 tokens. The rest of each prompt is the workspace the agent rebuilds every
round: the question, the report and the last tool result. That part changes on every
call by design, so it can never be cached.

The math, from the LangSmith traces of the golden baseline run of 2026-10-20:

- The run made 73 model calls with 178,554 input tokens in total.
- Best case, every call reuses its 704 fixed tokens: 73 x 704 = 51,392 tokens.
  51,392 / 178,554 = 29%. So 50% cannot be reached even with perfect caching.
- What happened: 5 of the 73 calls got a hit. 5 x 704 = 3,520 tokens.
  3,520 / 178,554 = 2%. The provider decides when a hit happens, not this code.

The issue the old target would cause: the only way to raise the rate is a longer
prompt that stays the same between calls. That is a growing prompt, which is the
thing this project argues against. The rate is still recorded, so a sudden drop
stays visible.

**prompt_size_per_round was changed from "round 12 size <= 1.1 x round 2 size" to
"last round size <= 1.1 x round 2 size", think prompt only (2026-10-20).**
The metric checks one claim: once the report fills up, the prompt stops growing.
Round 12 was a fixed number that most runs never reach. FRAMES questions finish in
2 or 3 rounds, and the longest report-arm run, BrowseComp question 776, finished at
round 9. A target that needs round 12 can only be checked by forcing a run past
the point where the agent already had its answer. Any run that goes past round 2
shows the same claim, so the target now uses the last round of the longest run in
the eval set.

Rounds count from 0, as the result files do. Round 0 has an empty report and
round 1 holds the first report only. By round 2 the report has filled, so round 2
is the reference point. The size is `prompt_chars` of the think prompt.
`prompt_tokens` in a result line also counts that round's look-up summary calls,
so one large web page can make a round look like it grew when the think prompt
did not.

The math, from spec.md section 9.1 (BrowseComp question 776):

- `report` arm: round 2 is 13.8k characters, the last round (8) is 14.4k.
  14.4 / 13.8 = 1.04. It passes.
- `full_history` arm, for contrast: round 2 is 22.1k, the last round (11) is
  54.6k. 54.6 / 22.1 = 2.5. It fails, which is what the metric should show for a
  prompt that grows every round.

## Review rule

A change ships only when task_accuracy holds and every speed and cost SLO still holds.
A change never ships while any security SLO fails.
