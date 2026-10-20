"""Filesystem helpers for the structure suite.

These tests enforce the repository's architecture rules mechanically. They walk the tree and never
hardcode an agent name, so a correctly built new agent passes with no edit
here.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

BACKEND = pathlib.Path(__file__).resolve().parents[2]
SRC = BACKEND / "src"
AGENTS = SRC / "agents"
CORE = SRC / "core"

REQUIRED_AGENT_FILES = (
    "__init__.py",
    "graph.py",
    "state.py",
    "nodes.py",
    "prompts.py",
    "schemas.py",
    "configuration.py",
)


def agent_packages() -> list[pathlib.Path]:
    """Return every agent package directory under src/agents."""
    return sorted(
        p
        for p in AGENTS.iterdir()
        if p.is_dir() and not p.name.startswith((".", "__"))
    )


def python_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Return every Python source file under a directory."""
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def parse(path: pathlib.Path) -> ast.Module:
    """Parse one Python file into a syntax tree."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def imported_modules(tree: ast.Module) -> set[str]:
    """Return every dotted module name a file imports."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def configuration_classes() -> list[tuple[pathlib.Path, type]]:
    """Return each agent package paired with its Configuration class."""
    pairs = []
    for package in agent_packages():
        module = importlib.import_module(f"agents.{package.name}.configuration")
        pairs.append((package, module.Configuration))
    return pairs


def configuration_modules() -> list[tuple[pathlib.Path, object]]:
    """Return each agent package paired with its imported configuration module."""
    return [
        (package, importlib.import_module(f"agents.{package.name}.configuration"))
        for package in agent_packages()
    ]
