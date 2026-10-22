"""Node functions are async and take state first, so a test can call them."""

import ast

import pytest

from tests.structure.conftest import agent_packages, parse

# Routing functions decide an edge. They do no input or output, so they stay
# synchronous. They are recognised by shape, never by name: a return annotation
# naming an edge rather than a state update.
ROUTER_RETURNS = {"str"}


def is_router(func) -> bool:
    """Return True when a function decides an edge rather than doing work."""
    if not 1 <= len(func.args.args) <= 2:
        return False
    return _names_an_edge(func.returns)


def _names_an_edge(returns) -> bool:
    """Return True when a return annotation names an edge, not a state update."""
    if isinstance(returns, ast.Name):
        return returns.id in ROUTER_RETURNS
    if isinstance(returns, ast.Subscript):
        return _names_an_edge(returns.value)
    if isinstance(returns, ast.BinOp) and isinstance(returns.op, ast.BitOr):
        return _names_an_edge(returns.left) or _names_an_edge(returns.right)
    return False


def public_functions(package):
    tree = parse(package / "nodes.py")
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    ]


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_nodes_module_defines_public_functions(package):
    assert public_functions(package), f"{package.name}/nodes.py defines no nodes"


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_every_worker_node_is_async(package):
    sync = [
        f.name
        for f in public_functions(package)
        if isinstance(f, ast.FunctionDef) and not is_router(f)
    ]
    assert not sync, f"{package.name}/nodes.py has synchronous nodes: {sync}"


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_every_node_takes_state_first(package):
    for func in public_functions(package):
        args = func.args.args
        assert args, f"{package.name}.{func.name} takes no arguments"
        assert args[0].arg == "state", (
            f"{package.name}.{func.name} takes {args[0].arg!r} first, not 'state'"
        )


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_every_node_is_annotated(package):
    for func in public_functions(package):
        assert func.returns is not None, (
            f"{package.name}.{func.name} has no return annotation"
        )
        for arg in func.args.args:
            assert arg.annotation is not None, (
                f"{package.name}.{func.name} argument {arg.arg!r} has no type hint"
            )


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_every_node_has_a_docstring(package):
    for func in public_functions(package):
        assert ast.get_docstring(func), (
            f"{package.name}.{func.name} has no docstring"
        )
