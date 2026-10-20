"""Every agent package has the same seven files and exposes a graph."""

import ast

import pytest

from tests.structure.conftest import (
    REQUIRED_AGENT_FILES,
    agent_packages,
    parse,
)


def test_at_least_one_agent_exists():
    assert agent_packages(), "src/agents holds no agent package"


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_the_seven_required_files_exist(package):
    missing = [f for f in REQUIRED_AGENT_FILES if not (package / f).is_file()]
    assert not missing, f"{package.name} is missing {missing}"


def module_level_names(tree: ast.Module) -> set[str]:
    """Return every name a module binds at its top level by assignment or import."""
    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update(alias.asname or alias.name for alias in node.names)
    return bound


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_graph_module_exposes_graph(package):
    bound = module_level_names(parse(package / "graph.py"))
    assert "graph" in bound, f"{package.name}/graph.py binds no module-level `graph`"


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_the_package_reexports_the_graph(package):
    bound = module_level_names(parse(package / "__init__.py"))
    assert "graph" in bound, f"{package.name}/__init__.py does not export `graph`"
