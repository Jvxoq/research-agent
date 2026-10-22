"""Fixtures for the research agent nodes, so no test reaches a provider or the web."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def patch_model(monkeypatch):
    """Return a helper that installs a FakeChatModel into the nodes module."""

    def install(model) -> None:
        monkeypatch.setattr(
            "agents.research_agent.nodes.build_chat_model", lambda spec: model
        )

    return install


@pytest.fixture
def config() -> dict[str, Any]:
    """A run configuration with the date frozen, so a prompt test is stable."""
    return {
        "configurable": {
            "current_date_override": "2026-01-02",
            "thread_id": "thread-unit",
        }
    }
