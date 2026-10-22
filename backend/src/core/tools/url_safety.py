"""Refuse URLs a model should not be able to make us fetch.

The agent passes URLs chosen by a language model straight to a fetcher. A model
can be talked into asking for a cloud metadata endpoint or a service on the
loopback interface, so the scheme and the resolved address are checked first.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = frozenset({"http", "https"})


class UnsafeUrl(ValueError):
    """Raised when a URL must not be fetched."""


def _is_blocked_address(host: str) -> bool:
    """Return True when a hostname resolves to any address we refuse.

    One bad address is enough. A name under an attacker's control can answer
    with a public address and a loopback address in the same reply, and the
    fetcher is free to pick either one.

    Args:
        host: The hostname or literal address from the URL.

    Returns:
        True when any resolved address is not globally routable or is
        multicast, or when the name does not resolve.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        # is_private misses shared address space, 100.64.0.0/10, where one
        # cloud serves its metadata. Multicast counts as global, so it is named.
        if not address.is_global or address.is_multicast:
            return True
    return False


def check_url(url: str) -> str:
    """Return the URL unchanged, or refuse it.

    Args:
        url: The URL a model asked to read.

    Returns:
        The same URL, when it is safe to fetch.

    Raises:
        UnsafeUrl: If the scheme is not http or https, if there is no host, if
            any address the host resolves to is not globally routable or is
            multicast, or if it does not resolve.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise UnsafeUrl(f"Refused scheme {parsed.scheme!r}. Use http or https.")
    if not parsed.hostname:
        raise UnsafeUrl("Refused a URL with no host.")
    if _is_blocked_address(parsed.hostname):
        raise UnsafeUrl(f"Refused {parsed.hostname!r}: not a public address.")
    return url
