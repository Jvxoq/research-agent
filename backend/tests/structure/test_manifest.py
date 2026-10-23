"""langgraph.json and src/agents must describe the same set of graphs."""

import importlib
import json
import pathlib

import pytest
from langgraph_cli.schemas import HttpConfig

from tests.structure.conftest import BACKEND, agent_packages

MANIFEST_PATH = BACKEND / "langgraph.json"

# Default server routes no caller uses. Left open, every signed-in user can reach them.
CLOSED_ROUTE_FLAGS = (
    "disable_assistants",
    "disable_store",
    "disable_ui",
    "disable_mcp",
    "disable_a2a",
    "disable_webhooks",
)

# The web app needs threads and runs. disable_meta stops /ok checking the database.
REQUIRED_OPEN_FLAGS = ("disable_threads", "disable_runs", "disable_meta")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def registered_packages(manifest: dict) -> set[str]:
    """Return the agent package name behind each graph the manifest lists."""
    return {
        pathlib.PurePosixPath(path.split(":")[0]).parent.name
        for path in manifest["graphs"].values()
    }


def http_block(manifest: dict) -> dict:
    """Return the manifest's ``http`` block, or an empty one when it has none."""
    return manifest.get("http", {})


def open_route_flags(manifest: dict) -> list[str]:
    """Return every route flag that should close a route group but does not."""
    http = http_block(manifest)
    return [flag for flag in CLOSED_ROUTE_FLAGS if http.get(flag) is not True]


def closed_required_flags(manifest: dict) -> list[str]:
    """Return every flag that closes a route this deployment needs."""
    http = http_block(manifest)
    return [flag for flag in REQUIRED_OPEN_FLAGS if http.get(flag) is True]


def unknown_http_flags(manifest: dict) -> list[str]:
    """Return every ``http`` key LangGraph's config schema does not define."""
    return sorted(set(http_block(manifest)) - set(HttpConfig.__annotations__))


def import_target(target: str) -> tuple[str, str, bool]:
    """Import the module a ``path:attribute`` target names.

    Returns:
        The module name, the attribute, and whether the module defines it.
    """
    rel, _, attribute = target.partition(":")
    path = (BACKEND / rel).resolve()
    assert path.is_file(), f"{rel} is missing"
    module_name = ".".join(path.relative_to(BACKEND / "src").with_suffix("").parts)
    module = importlib.import_module(module_name)
    return module_name, attribute, hasattr(module, attribute)


def test_the_manifest_exists():
    assert MANIFEST_PATH.is_file(), "backend/langgraph.json is missing"


def test_the_manifest_declares_its_dependencies(manifest):
    assert manifest.get("dependencies"), "langgraph.json declares no dependencies"


def test_every_agent_package_is_registered(manifest):
    missing = {p.name for p in agent_packages()} - registered_packages(manifest)
    assert not missing, f"agent packages missing from langgraph.json: {sorted(missing)}"


def test_the_manifest_lists_nothing_extra(manifest):
    extra = registered_packages(manifest) - {p.name for p in agent_packages()}
    assert not extra, f"langgraph.json lists graphs with no package: {sorted(extra)}"


def test_every_graph_path_resolves(manifest):
    for name, target in manifest["graphs"].items():
        module_name, attribute, defined = import_target(target)
        assert defined, (
            f"graph {name!r} names {attribute!r}, which {module_name} does not define"
        )


def test_the_auth_handler_resolves(manifest):
    auth = manifest.get("auth")
    assert auth and auth.get("path"), "langgraph.json registers no auth handler"

    module_name, attribute, defined = import_target(auth["path"])
    assert defined, f"auth path names {attribute!r}, which {module_name} does not define"


def test_the_env_file_is_declared(manifest):
    assert manifest.get("env"), "langgraph.json declares no env file"


def test_unused_routes_are_closed(manifest):
    assert open_route_flags(manifest) == []


@pytest.mark.parametrize(
    ("bad_manifest", "expected"),
    [
        (
            {"http": {flag: True for flag in CLOSED_ROUTE_FLAGS if flag != "disable_mcp"}},
            ["disable_mcp"],
        ),
        ({"http": {flag: False for flag in CLOSED_ROUTE_FLAGS}}, list(CLOSED_ROUTE_FLAGS)),
        ({}, list(CLOSED_ROUTE_FLAGS)),
    ],
    ids=["one-flag-missing", "flags-false", "no-http-block"],
)
def test_open_route_is_reported(bad_manifest, expected):
    assert open_route_flags(bad_manifest) == expected


def test_needed_routes_stay_open(manifest):
    """Closing threads or runs breaks the web app, and closing meta blinds the health check."""
    assert closed_required_flags(manifest) == []


def test_closed_needed_route_is_reported():
    bad_manifest = {"http": {"disable_runs": True, "disable_meta": True}}

    assert closed_required_flags(bad_manifest) == ["disable_runs", "disable_meta"]


def test_http_flags_are_server_fields(manifest):
    """A misspelled flag is ignored by the server, so its route group stays open."""
    assert unknown_http_flags(manifest) == []


def test_misspelled_http_flag_is_reported():
    bad_manifest = {"http": {"disable_assistant": True, "disable_mcp": True}}

    assert unknown_http_flags(bad_manifest) == ["disable_assistant"]
