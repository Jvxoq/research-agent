"""One place to cut tool text down to a limit, and to strip secrets from it.

Everything a tool returns lands in the next prompt, error messages included. A
provider exception can carry a whole response body, so an uncapped error is an
uncapped observation. That body sometimes echoes the request, credential and
all, and the observation is state the caller reads back over the API.
"""

from __future__ import annotations

from collections.abc import Iterable

CUT_MARK = "\n[cut]"


def cap(text: str, limit: int) -> str:
    """Cut text to a limit, marking where it was cut.

    Args:
        text: The text to cap.
        limit: Maximum characters to return, the cut mark included.

    Returns:
        The text, cut and marked when it was over the limit. Never longer than
        ``limit``.
    """
    if len(text) <= limit:
        return text
    if limit <= len(CUT_MARK):
        return text[:limit]
    return text[: limit - len(CUT_MARK)] + CUT_MARK


REDACTED = "[redacted]"


def redact(text: str, secrets: Iterable[str | None]) -> str:
    """Replace every occurrence of a secret with a marker.

    Args:
        text: Text about to reach a prompt, a log line or the caller.
        secrets: Credentials this call had access to. ``None`` and short
            values are skipped, because a two-character needle would blank
            out ordinary text.

    Returns:
        The same text with each secret replaced.
    """
    for secret in secrets:
        if secret and len(secret) >= 8:
            text = text.replace(secret, REDACTED)
    return text
