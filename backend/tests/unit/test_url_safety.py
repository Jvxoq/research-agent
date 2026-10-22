"""Tests for the URL guard that fronts every model-chosen fetch.

The URL comes from a model, so the scheme and the resolved address decide
whether it may be fetched. DNS is mocked: these tests touch no network.
"""


import pytest

from core.tools.url_safety import UnsafeUrl, check_url
from tests.conftest import PUBLIC_ADDRESS


@pytest.mark.parametrize("url", ["http://example.com/page", "https://example.com/page"])
def test_a_public_address_is_returned_unchanged(resolves, url) -> None:
    """http and https to a public host are the only fetchable case."""
    resolves({"example.com": PUBLIC_ADDRESS})
    assert check_url(url) == url


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/data.csv"])
def test_non_http_scheme_refused(url) -> None:
    """file:// would read the host's disk. Only http and https are fetchable."""
    with pytest.raises(UnsafeUrl, match="scheme"):
        check_url(url)


def test_hostless_url_refused() -> None:
    """There is nothing to resolve, so there is nothing safe to fetch."""
    with pytest.raises(UnsafeUrl, match="no host"):
        check_url("http:///page")


@pytest.mark.parametrize(
    ("url", "addresses"),
    [
        # Services on the agent's own box.
        ("http://localhost:8123/admin", {"localhost": "127.0.0.1"}),
        # Writing the address out instead of a name changes nothing.
        ("http://127.0.0.1/admin", {"127.0.0.1": "127.0.0.1"}),
        # The cloud metadata endpoint hands credentials to whoever asks.
        (
            "http://169.254.169.254/latest/meta-data/",
            {"169.254.169.254": "169.254.169.254"},
        ),
        # Private ranges are the rest of the operator's network.
        ("http://10.0.0.5/internal", {"10.0.0.5": "10.0.0.5"}),
        # A public-looking name that resolves inward is still inward.
        ("https://internal.example.com/secrets", {"internal.example.com": "192.168.1.10"}),
        # Shared address space, where Alibaba Cloud serves its metadata.
        ("http://100.100.100.200/latest/meta-data/", {"100.100.100.200": "100.100.100.200"}),
        # Multicast counts as global, so it is refused by name.
        ("http://224.0.0.1/", {"224.0.0.1": "224.0.0.1"}),
        # A name with no address cannot be shown to be public.
        ("https://nowhere.invalid/page", {}),
    ],
)
def test_private_address_refused(resolves, url, addresses) -> None:
    resolves(addresses)
    with pytest.raises(UnsafeUrl, match="not a public address"):
        check_url(url)


def test_one_bad_address_refuses_host(resolves) -> None:
    """A name an attacker controls can answer with a public and a private
    address at once, and the fetcher may pick either."""
    resolves({"split.example.com": [PUBLIC_ADDRESS, "127.0.0.1"]})

    with pytest.raises(UnsafeUrl, match="not a public address"):
        check_url("https://split.example.com/page")
