"""The fake provider and fake Tavily the server calls during a load test.

Handlers are called with mocked requests, so no socket opens.
"""

from __future__ import annotations

import asyncio
import json
from unittest import mock

import pytest
from aiohttp import streams, web
from aiohttp.test_utils import make_mocked_request
from openai.types.chat import ChatCompletion

from loadtest.fakes import PAGE_TEXT, Delays, build_app
from loadtest.scenario import ModelRequest, RunOutcome, Scenario
from tests.conftest import logged

NO_WAIT = Delays(model_seconds=0.0, search_seconds=0.0, extract_seconds=0.0, jitter=0.0)


def _echo(request: ModelRequest) -> str:
    if request.prompt == "refuse me":
        raise ValueError("this scenario cannot answer that")
    return f"{request.model}|{request.schema_name}|{request.prompt}"


def _outcome(state: object) -> RunOutcome:
    raise AssertionError("not called")


ECHO = Scenario(inputs=({"question": "q"},), reply=_echo, outcome=_outcome)


async def _call(path: str, body: bytes) -> tuple[int, object]:
    """Send one POST to the fakes and return the status and decoded body."""
    app = build_app(ECHO, NO_WAIT)
    payload = streams.StreamReader(
        mock.Mock(_reading_paused=False), 2**16, loop=asyncio.get_running_loop()
    )
    payload.feed_data(body)
    payload.feed_eof()
    request = make_mocked_request(
        "POST", path, payload=payload, app=app, headers={"Content-Type": "application/json"}
    )
    match = await app.router.resolve(request)
    response = await match.handler(request)
    assert isinstance(response, web.Response) and response.text is not None
    return response.status, json.loads(response.text)


def _chat(prompt: str = "the prompt", **extra: object) -> bytes:
    return json.dumps(
        {
            "model": "think-model",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": prompt},
            ],
            **extra,
        }
    ).encode()


async def test_completion_parses_as_openai() -> None:
    """The real OpenAI client must read the fake's reply, or every run fails."""
    status, body = await _call("/v1/chat/completions", _chat())

    completion = ChatCompletion.model_validate(body)

    assert (status, completion.choices[0].message.content) == (
        200,
        "think-model|None|the prompt",
    )


async def test_completion_reports_usage() -> None:
    """Nonzero usage keeps the server's cost accounting running under load.

    Tokens are a quarter of the characters: 40 prompt characters, and 57 in
    the reply "think-model|None|" plus the 40.
    """
    _, body = await _call("/v1/chat/completions", _chat(prompt="x" * 40))

    assert isinstance(body, dict)
    assert body["usage"] == {"prompt_tokens": 10, "completion_tokens": 14, "total_tokens": 24}


async def test_schema_name_reaches_scenario() -> None:
    """The scenario picks its reply by the schema the agent asked for."""
    body = _chat(response_format={"type": "json_schema", "json_schema": {"name": "Decision"}})

    _, reply = await _call("/v1/chat/completions", body)

    assert ChatCompletion.model_validate(reply).choices[0].message.content == (
        "think-model|Decision|the prompt"
    )


async def test_streamed_completion_refused() -> None:
    """The agent never streams a model call, so a streamed one means the server changed."""
    assert await _call("/v1/chat/completions", _chat(stream=True)) == (
        400,
        {"error": "the fake model does not stream"},
    )


async def test_scenario_refusal_is_400(log_records) -> None:
    status, body = await _call("/v1/chat/completions", _chat(prompt="refuse me"))

    assert (status, body) == (400, {"error": "this scenario cannot answer that"})
    assert logged(log_records) == [
        ("WARNING", "fake refused a request: this scenario cannot answer that", False)
    ]


@pytest.mark.parametrize(
    "raw,reason",
    [
        (b"", "the request body is not a JSON object"),
        (b'{"messa', "the request body is not a JSON object"),
        (b"null", "the request body is not a JSON object"),
        (b"[]", "the request body is not a JSON object"),
        (b'{"messages": "hi"}', "the request has no messages list"),
        (b'{"messages": []}', "the request has no user message with text content"),
        (
            b'{"messages": [{"role": "user", "content": 5}]}',
            "the request has no user message with text content",
        ),
        (b"\xff\xfe", "the request body is not a JSON object"),
    ],
    ids=["empty", "truncated", "null", "list", "messages-not-list", "no-messages", "number", "bad-utf8"],
)
async def test_bad_chat_body_is_400(raw: bytes, reason: str) -> None:
    """A broken request fails its own run with a 400, never a crash in the fake."""
    assert await _call("/v1/chat/completions", raw) == (400, {"error": reason})


async def test_search_returns_fixture_pages() -> None:
    status, body = await _call("/search", b'{"query": "who"}')

    assert (status, body) == (
        200,
        {
            "query": "who",
            "answer": None,
            "images": [],
            "follow_up_questions": None,
            "response_time": 0.1,
            "results": [
                {
                    "title": f"Fixture page {n}",
                    "url": f"https://example.com/loadtest/{n}",
                    "content": f"Snippet {n} for: who",
                    "score": 0.9,
                    "raw_content": None,
                }
                for n in (1, 2, 3)
            ],
        },
    )


@pytest.mark.parametrize(
    "urls", ['"https://example.com/a"', '["https://example.com/a"]'], ids=["string", "list"]
)
async def test_extract_returns_page_text(urls: str) -> None:
    """Tavily takes one URL or a list, so the fake takes both."""
    status, body = await _call("/extract", f'{{"urls": {urls}}}'.encode())

    assert (status, body) == (
        200,
        {
            "results": [{"url": "https://example.com/a", "raw_content": PAGE_TEXT}],
            "failed_results": [],
            "response_time": 0.1,
        },
    )


@pytest.mark.parametrize(
    "path,raw,reason",
    [
        ("/search", b"not json", "the search request has no query"),
        ("/search", b'{"query": 3}', "the search request has no query"),
        ("/extract", b"not json", "the extract request has no urls"),
        ("/extract", b'{"urls": [1]}', "the extract request has no urls"),
    ],
    ids=["search-not-json", "search-bad-query", "extract-not-json", "extract-bad-urls"],
)
async def test_bad_tool_body_is_400(path: str, raw: bytes, reason: str) -> None:
    assert await _call(path, raw) == (400, {"error": reason})
