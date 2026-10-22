"""Raise the failures the Tavily integration hands back as values.

``langchain_tavily`` catches every failure of its HTTP call and returns
``{"error": exception}`` instead of raising. Read as a payload, a rate limit
looks like a query with no results, and the retrier never sees it.

The integration keeps the status only in the message it builds itself, as
``Error <status>: <reason>``. That text comes from the library, never from a
model or a page, so reading the status from it cannot misclassify model output.
"""

from __future__ import annotations

import re

import aiohttp

_STATUS_MESSAGE = re.compile(r"^Error (\d{3}):")


class TavilyHTTPError(RuntimeError):
    """A Tavily request that came back with a non-200 status."""

    def __init__(self, status_code: int) -> None:
        """Keep the status where the retrier reads it."""
        super().__init__(f"Tavily returned HTTP {status_code}")
        self.status_code = status_code


def raise_returned_error(payload: object) -> object:
    """Return a Tavily payload, or raise the failure it carries instead.

    Args:
        payload: Whatever the integration's ``ainvoke`` returned.

    Returns:
        The payload unchanged, when it carries no error.

    Raises:
        TavilyHTTPError: For a non-200 status, so a 429 or a 5xx is retried.
        ConnectionError: For a dropped or refused connection, so it is retried.
        Exception: The carried error itself, for anything else.
    """
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, Exception):
        return payload

    status = _STATUS_MESSAGE.match(str(error))
    if status is not None:
        raise TavilyHTTPError(int(status.group(1))) from error
    if isinstance(error, aiohttp.ClientConnectionError):
        raise ConnectionError(type(error).__name__) from error
    raise error
