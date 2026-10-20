"""langgraph.json and src/agents must describe the same set of graphs."""

import importlib
import json
import pathlib

import pytest

from tests.structure.conftest import BACKEND, agent_packages

MANIFEST_PATH = BACKEND / "langgraph.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def registered_packages(manifest: dict) -> set[str]:
    """Return the agent package name behind each graph the manifest lists."""
    return {
        pathlib.PurePosixPath(path.split(":")[0]).parent.name
        for path in manifest["graphs"].values()
    }


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
