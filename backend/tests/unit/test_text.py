"""Tests for the shared cap and redact helpers.

Both run on text that is about to become an observation, so both decide what
the next prompt and the caller get to see.
"""

from core.tools.text import cap, redact


def test_short_text_unchanged() -> None:
    assert cap("short", 100) == "short"


def test_long_text_cut_and_marked() -> None:
    assert cap("abcdefghijklmnop", 10) == "abcd\n[cut]"


def test_tiny_limit_cuts_without_mark() -> None:
    assert cap("abcdef", 3) == "abc"


def test_secret_is_redacted() -> None:
    assert redact("sent key sk-abcdefgh12", ["sk-abcdefgh12"]) == "sent key [redacted]"


def test_every_occurrence_is_replaced() -> None:
    assert redact("sk-abcdefgh12 sk-abcdefgh12", ["sk-abcdefgh12"]) == (
        "[redacted] [redacted]"
    )


def test_no_secret_leaves_text() -> None:
    assert redact("nothing here", [None]) == "nothing here"


def test_short_value_not_redacted() -> None:
    """A two-character needle would blank out ordinary words."""
    assert redact("an ordinary sentence", ["an"]) == "an ordinary sentence"
