"""Every workflow runs the exact action and uv code it was reviewed with.

A tag can be moved to other code, and the image job holds a token that writes to GHCR.
The image build depends on how `uv export` behaves, so uv itself is pinned too.
"""

import re

import pytest
import yaml

from tests.structure.conftest import BACKEND

WORKFLOWS_DIR = BACKEND.parent / ".github" / "workflows"

COMMIT_PIN = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
EXACT_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


def workflow_steps(text: str) -> list[dict]:
    """Return every step of every job in one workflow file."""
    workflow = yaml.safe_load(text)
    return [step for job in workflow["jobs"].values() for step in job["steps"]]


def unpinned_actions(text: str) -> list[str]:
    """Return every action a workflow uses by anything other than a full commit SHA."""
    return [
        step["uses"]
        for step in workflow_steps(text)
        if "uses" in step and not COMMIT_PIN.match(step["uses"])
    ]


def loose_uv_versions(text: str) -> list[str]:
    """Return the version of every setup-uv step that does not name one exact uv release."""
    versions = [
        str(step.get("with", {}).get("version", "unset"))
        for step in workflow_steps(text)
        if step.get("uses", "").startswith("astral-sh/setup-uv@")
    ]
    return [v for v in versions if not EXACT_VERSION.match(v)]


@pytest.fixture(scope="module")
def workflows() -> list[str]:
    paths = sorted(WORKFLOWS_DIR.glob("*.yml"))
    assert paths, f"no workflow files in {WORKFLOWS_DIR}"
    return [p.read_text(encoding="utf-8") for p in paths]


def test_actions_pinned_to_commit(workflows):
    assert [a for text in workflows for a in unpinned_actions(text)] == []


def test_uv_version_exact(workflows):
    assert [v for text in workflows for v in loose_uv_versions(text)] == []


def snippet(uses: str, version: str | None = None) -> str:
    step: dict[str, object] = {"uses": uses}
    if version is not None:
        step["with"] = {"version": version}
    return yaml.safe_dump({"jobs": {"build": {"steps": [step, {"run": "true"}]}}})


@pytest.mark.parametrize(
    "uses",
    [
        "actions/checkout@v4",
        "actions/checkout@main",
        "actions/checkout@11d5960",
        "actions/checkout@v4.4.0",
    ],
    ids=["major-tag", "branch", "short-sha", "exact-tag"],
)
def test_tag_pin_is_reported(uses):
    assert unpinned_actions(snippet(uses)) == [uses]


SETUP_UV = "astral-sh/setup-uv@caf0cab7a618c569241d31dcd442f54681755d39"


@pytest.mark.parametrize(
    ("version", "expected"),
    [("latest", ["latest"]), (None, ["unset"]), ("0.11", ["0.11"])],
    ids=["latest", "unset", "minor-only"],
)
def test_loose_uv_version_is_reported(version, expected):
    assert loose_uv_versions(snippet(SETUP_UV, version)) == expected
