"""A fake model provider and a fake Tavily, served on one local port.

The load test measures the server, not the provider. So the server under test
calls this process instead: ``BASE_URL`` points at its OpenAI-compatible
``/v1/chat/completions`` and ``TAVILY_BASE_URL`` at its ``/search`` and
``/extract``. Every reply waits a fixed delay with jitter, so a run takes a
known amount of provider time and the rest is the server's own.

The model's replies come from the agent's :class:`~loadtest.scenario.Scenario`.
Search and extract replies are the same for every agent.

Run it with ``python -m loadtest.fakes --graph <graph_id>``. The load runner
starts and stops it on its own.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from dataclasses import dataclass

from aiohttp import web
from loguru import logger

from loadtest.scenario import ModelRequest, Scenario, load_scenario

HEALTH_PATH = "/health"
SEARCH_RESULTS = 3
PAGE_TEXT = (
    "This page is a load test fixture. It states one fact that answers the "
    "question, and the fake model reads nothing from it."
)


@dataclass(frozen=True)
class Delays:
    """How long each fake reply waits before it is sent.

    Args:
        model_seconds: One chat completion.
        search_seconds: One search query.
        extract_seconds: One page extract.
        jitter: Random fraction added or removed, so replies spread out.
    """

    model_seconds: float
    search_seconds: float
    extract_seconds: float
    jitter: float

    async def wait(self, seconds: float) -> None:
        """Sleep ``seconds`` give or take the jitter fraction."""
        await asyncio.sleep(max(0.0, seconds * random.uniform(1 - self.jitter, 1 + self.jitter)))


def read_model_request(body: object) -> ModelRequest:
    """Narrow a chat completion body to what a scenario needs.

    Args:
        body: The decoded JSON request body.

    Returns:
        The structured-output schema name, if any, and the last user message.

    Raises:
        ValueError: When the body holds no user message with text content.
    """
    if not isinstance(body, dict):
        raise ValueError("the request body is not a JSON object")
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise ValueError("the request has no messages list")
    prompts = [
        message["content"]
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
    ]
    if not prompts:
        raise ValueError("the request has no user message with text content")

    model = body.get("model")
    schema_name: str | None = None
    response_format = body.get("response_format")
    if isinstance(response_format, dict):
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict) and isinstance(json_schema.get("name"), str):
            schema_name = json_schema["name"]
    return ModelRequest(
        model=model if isinstance(model, str) else "fake",
        schema_name=schema_name,
        prompt=prompts[-1],
    )


def completion(model: str, prompt: str, content: str) -> dict[str, object]:
    """Build an OpenAI chat completion response holding ``content``.

    Token counts are a quarter of the character counts, so the server's cost
    accounting sees plausible, nonzero usage.
    """
    prompt_tokens = len(prompt) // 4
    completion_tokens = len(content) // 4
    return {
        "id": f"chatcmpl-fake-{random.getrandbits(48):x}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": None},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def search_results(query: str) -> dict[str, object]:
    """Build a Tavily search response with a fixed set of example pages."""
    return {
        "query": query,
        "answer": None,
        "images": [],
        "follow_up_questions": None,
        "response_time": 0.1,
        "results": [
            {
                "title": f"Fixture page {index}",
                "url": f"https://example.com/loadtest/{index}",
                "content": f"Snippet {index} for: {query}",
                "score": 0.9,
                "raw_content": None,
            }
            for index in range(1, SEARCH_RESULTS + 1)
        ],
    }


def extract_results(urls: list[str]) -> dict[str, object]:
    """Build a Tavily extract response with the same text for every URL."""
    return {
        "results": [{"url": url, "raw_content": PAGE_TEXT} for url in urls],
        "failed_results": [],
        "response_time": 0.1,
    }


async def _json_body(request: web.Request) -> object:
    """Decode a request body, or None when it is not JSON."""
    try:
        body: object = await request.json()
    except ValueError:
        return None
    return body


def _refuse(reason: str) -> web.Response:
    """Answer 400, so a broken request fails its run instead of passing quietly."""
    logger.warning(f"fake refused a request: {reason}")
    return web.json_response({"error": reason}, status=400)


def build_app(scenario: Scenario, delays: Delays) -> web.Application:
    """Wire the three fake endpoints and the health check.

    Args:
        scenario: The agent whose prompts the fake model answers.
        delays: How long each kind of reply waits.

    Returns:
        The aiohttp application.
    """

    async def chat_completions(request: web.Request) -> web.Response:
        body = await _json_body(request)
        if isinstance(body, dict) and body.get("stream"):
            # The agent never streams a model call. A streamed request means the
            # server changed, and a silent non-streamed reply would hide it.
            return _refuse("the fake model does not stream")
        try:
            model_request = read_model_request(body)
            content = scenario.reply(model_request)
        except ValueError as error:
            return _refuse(str(error))
        await delays.wait(delays.model_seconds)
        return web.json_response(
            completion(model_request.model, model_request.prompt, content)
        )

    async def search(request: web.Request) -> web.Response:
        body = await _json_body(request)
        query = body.get("query") if isinstance(body, dict) else None
        if not isinstance(query, str):
            return _refuse("the search request has no query")
        await delays.wait(delays.search_seconds)
        return web.json_response(search_results(query))

    async def extract(request: web.Request) -> web.Response:
        body = await _json_body(request)
        raw = body.get("urls") if isinstance(body, dict) else None
        urls = [raw] if isinstance(raw, str) else raw
        if not isinstance(urls, list) or not all(isinstance(url, str) for url in urls):
            return _refuse("the extract request has no urls")
        await delays.wait(delays.extract_seconds)
        return web.json_response(extract_results(urls))

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/search", search)
    app.router.add_post("/extract", extract)
    app.router.add_get(HEALTH_PATH, health)
    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, help="Graph id whose scenario answers.")
    # The runner owns every default and always passes these.
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-delay-seconds", type=float, required=True)
    parser.add_argument("--search-delay-seconds", type=float, required=True)
    parser.add_argument("--extract-delay-seconds", type=float, required=True)
    parser.add_argument("--jitter", type=float, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Serve the fakes for one graph until the process is stopped."""
    args = parse_args(argv)
    delays = Delays(
        model_seconds=args.model_delay_seconds,
        search_seconds=args.search_delay_seconds,
        extract_seconds=args.extract_delay_seconds,
        jitter=args.jitter,
    )
    app = build_app(load_scenario(args.graph), delays)
    logger.info(f"fakes for {args.graph} on {args.host}:{args.port}")
    # One access log line per request would flood the runner's output.
    web.run_app(app, host=args.host, port=args.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
