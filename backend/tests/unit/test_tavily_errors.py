"""The Tavily integration returns failures as values. These raise them again."""

import aiohttp
import pytest

from core.retry import is_rate_limit, is_transient
from core.tools.tavily_errors import TavilyHTTPError, raise_returned_error


def test_clean_payload_unchanged() -> None:
    payload = {"results": [{"url": "https://example.com"}]}
    assert raise_returned_error(payload) is payload


def test_returned_429_raises_rate_limit() -> None:
    with pytest.raises(TavilyHTTPError) as raised:
        raise_returned_error({"error": Exception("Error 429: Too Many Requests")})
    assert is_rate_limit(raised.value)


def test_returned_5xx_raises_transient() -> None:
    with pytest.raises(TavilyHTTPError) as raised:
        raise_returned_error({"error": Exception("Error 503: Service Unavailable")})
    assert is_transient(raised.value)


def test_dropped_connection_raises_transient() -> None:
    with pytest.raises(ConnectionError) as raised:
        raise_returned_error({"error": aiohttp.ServerDisconnectedError()})
    assert is_transient(raised.value)


def test_other_errors_raised_as_is() -> None:
    error = ValueError("bad parameter")
    with pytest.raises(ValueError) as raised:
        raise_returned_error({"error": error})
    assert raised.value is error
