"""graph.py wires. nodes.py works. Neither does the other's job."""

import ast

import pytest

from tests.structure.conftest import SRC, agent_packages, imported_modules, parse

WIRING_NAMES = ("StateGraph", "add_node", "add_edge", "add_conditional_edges")


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_graph_defines_no_async_function(package):
    tree = parse(package / "graph.py")
    async_defs = [
        n.name for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)
    ]
    assert not async_defs, (
        f"{package.name}/graph.py defines node logic: {async_defs}"
    )


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_graph_has_no_model_calls(package):
    source = (package / "graph.py").read_text(encoding="utf-8")
    for banned in ("build_chat_model", "ainvoke(", "instructions", "with_structured_output"):
        assert banned not in source, (
            f"{package.name}/graph.py contains {banned!r}; that belongs in nodes.py"
        )


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_nodes_never_import_the_graph_api(package):
    path = package / "nodes.py"
    for module in imported_modules(parse(path)):
        assert module != "langgraph.graph" and not module.startswith(
            "langgraph.graph."
        ), f"{path.relative_to(SRC)} imports {module}; wiring belongs in graph.py"


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_nodes_build_no_edges(package):
    source = (package / "nodes.py").read_text(encoding="utf-8")
    for banned in WIRING_NAMES:
        assert banned not in source, (
            f"{package.name}/nodes.py contains {banned!r}; that belongs in graph.py"
        )
