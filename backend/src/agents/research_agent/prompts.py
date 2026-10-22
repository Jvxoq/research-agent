"""Prompt templates for the research agent.

The round template follows Appendix E.2 of arXiv 2511.07327, with this
project's two tools and its own limits substituted in. Every limit arrives as a
placeholder, so no number is written here.
"""

from __future__ import annotations

from datetime import datetime

EMPTY_SLOT = "(none, this is the first round)"

# Instructions for a short benchmark answer. The grader compares against a short
# gold answer, so an essay confuses it.
SHORT_ANSWER_INSTRUCTIONS = """1. Give the exact answer and nothing else.
2. No preamble, no explanation, no sources, no markdown.
3. One line where one line will do.
4. Answer in the same language as the question."""

# Instructions for a product answer. The paper's own format.
REPORT_ANSWER_INSTRUCTIONS = """1. Answer in the same language as the question.
2. Start with a direct answer to the question.
3. Then give a rich, complete response that synthesizes all relevant
   information, formatted as markdown with paragraphs and subheadings.
4. Do not include URLs, images or tables that appeared in your reasoning.
5. Output only the answer text. Do not start with a phrase like
   "Here is my answer".
6. For statistical data with at least 3 items, use a markdown table. For fewer,
   describe them in text."""

ANSWER_INSTRUCTIONS = {
    "short": SHORT_ANSWER_INSTRUCTIONS,
    "report": REPORT_ANSWER_INSTRUCTIONS,
}

TOOL_DESCRIPTIONS = """- search
  Search the web. Action: {{"name": "search", "queries": ["...", "..."]}}
  Send up to {max_queries_per_search} queries in one call. Each returns
  {max_search_results} results as title, URL and snippet. Snippets are short.
  Use them to find pages worth reading, not as final evidence.

- look_up
  Read pages. Action: {{"name": "look_up", "urls": ["...", "..."], "goal": "..."}}
  Send up to {max_urls_per_look_up} URLs in one call, with one shared goal.
  The goal is a sentence saying what you want from these pages, for example
  "the year the company was founded". A vague goal returns a vague summary,
  so be specific. Only the goal-relevant parts of each page come back."""

ROUND_PROMPT = """You are a professional problem-solving agent with rigorous information verification capabilities and deep analytical thinking.

You work in a multi-round research loop. You see only the Question, the Last Report, the Last Tool Call, and the Last Tool Response. You must carry forward everything needed in your next report. Prefer another research round when gaps remain. Do not rush to answer.

## CRITICAL OUTPUT CONTRACT
Reason as long as you need first. Then output one JSON object with three fields:
1. "report": the rewritten report. Never blank.
2. "answer": the final answer, or null.
3. "action": one tool call, or null.
Set exactly one of "answer" and "action". Never both. Never neither.

## The report field
Write the report with these four sections. Stay under {max_report_chars} characters. If space runs short, keep in this priority: Confirmed Facts first, then Gaps, then History, then Plan. Compress old history to one line per prior tool call.

### 1. Confirmed Facts
List facts that answer the Question. Attach source to each fact. Write the fact itself, never a pointer like "see last response".

### 2. Inferences and Hypotheses
Separate "highly credible inferences" from "hypotheses to verify". Mark uncertainty plainly.

### 3. Tool History
Record the Last Tool Call and what its Response returned, including empty results, errors, or failures. A disappointing result is still a result. If Last Report, Last Tool Call, or Last Tool Response is empty or None, treat this as round 1 and say so.

### 4. Gaps and Next Plan
Name what still needs proof. Give one specific next step.

## Decision rule
Answer only when Confirmed Facts directly answer the Question and no key claim lacks a source. Otherwise call one tool.

If you answer, put only the answer text in "answer" and set "action" to null. Follow these rules for the answer:
{answer_instructions}

If you act, put one tool call in "action" and set "answer" to null. "name" must match Available Tools exactly, and the other fields follow that tool's shape. Do exactly one tool call per round. Never repeat an identical call that just failed or returned nothing. Change the query or the tool.

## Available Tools
{tools}

## Principles
1. Verify every source. Note verification status.
2. Decide from evidence, never from assumption.
3. Stay focused on the original Question.

## Input
- Current Date: {current_date}
- Question: {question}
- Last Status Report and Deep Analysis:
<report>
{report}
</report>
- Last Tool Call:
<tool_call>
{action}
</tool_call>
- Last Tool Response:
<tool_response>
{observation}
</tool_response>

Write the report and the answer in the same language as the Question."""

LOOK_UP_PROMPT = """Read the page below and pull out only what serves this goal.

Goal: {goal}

Fill these four fields. Keep them under {max_chars} characters in total.
- findings: one entry per useful fact. Keep figures, dates, names and quotes, with the page detail that proves each one. Use only what the page says.
- source_date: the visible publish or update date, or "unknown".
- coverage: FULL, PARTIAL or NONE.
- coverage_note: one line that says what is missing, or why the page does not match the goal.

Rules:
- Drop navigation, adverts, boilerplate, and unrelated text.
- If the page does not address the goal, leave findings empty and set coverage to NONE.
- If coverage is FULL or PARTIAL, give at least one finding.
- Never invent facts. Never summarize from memory.

Page URL: {url}

Page content:
{content}"""


def get_current_date() -> str:
    """Return today's date in the form the prompt expects."""
    return datetime.now().strftime("%B %d, %Y")


def render_tools(
    max_queries_per_search: int,
    max_urls_per_look_up: int,
    max_search_results: int,
) -> str:
    """Describe the available tools, with this run's batch limits.

    Building this at runtime keeps the limits the model sees in step with the
    configuration, so the two can never drift.

    Args:
        max_queries_per_search: Batch ceiling for a search action.
        max_urls_per_look_up: Batch ceiling for a look-up action.
        max_search_results: Results returned per query.

    Returns:
        The tool list as prompt text.
    """
    return TOOL_DESCRIPTIONS.format(
        max_queries_per_search=max_queries_per_search,
        max_urls_per_look_up=max_urls_per_look_up,
        max_search_results=max_search_results,
    )

