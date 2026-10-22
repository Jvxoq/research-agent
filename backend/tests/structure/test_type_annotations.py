"""Every annotation names a real shape, so the checker can see the contract.

A bare ``dict`` and an ``Any`` both switch type checking off at exactly the
place where a wrong value does the most damage: a node's return, a caller's
payload, a decoded response. These tests fail the build on both.
"""

import ast

import pytest

from tests.structure.conftest import SRC, agent_packages, parse, python_files

BARE_CONTAINERS = frozenset({"dict", "list", "tuple", "set", "frozenset"})

# The only signatures allowed to say ``Any``. Each takes a payload another
# library owns, whose shape this repository does not define.
ANY_ALLOWED = frozenset(
    {
        ("core/auth.py", "_own"),
        ("core/auth.py", "scope_threads"),
        ("core/auth.py", "scope_crons"),
        ("core/auth.py", "scope_store"),
    }
)


def _functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return every function defined in a file, nested ones included."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _signature_annotations(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.expr]:
    """Return every annotation in one function's signature, return type included."""
    args = function.args
    slots = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    found = [slot.annotation for slot in slots if slot is not None and slot.annotation]
    if function.returns is not None:
        found.append(function.returns)
    return found


def _all_annotations(tree: ast.Module) -> list[ast.expr]:
    """Return every annotation in a file, on a signature or on an attribute."""
    found = [
        node.annotation
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and node.annotation
    ]
    for function in _functions(tree):
        found.extend(_signature_annotations(function))
    return found


def _bare_containers(annotation: ast.expr) -> set[str]:
    """Return the unparameterised container names used inside one annotation.

    A ``Name`` under a ``Subscript`` is the subscripted type itself, so
    ``dict[str, int]`` yields nothing while a plain ``dict`` yields ``dict``.
    """
    subscripted = {
        node.value.id
        for node in ast.walk(annotation)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
    }
    used = {
        node.id
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name) and node.id in BARE_CONTAINERS
    }
    return used - subscripted


@pytest.mark.parametrize(
    "path", python_files(SRC), ids=lambda p: str(p.relative_to(SRC))
)
def test_no_annotation_uses_an_unparameterised_container(path):
    offenders = [
        f"line {annotation.lineno}: {ast.unparse(annotation)}"
        for annotation in _all_annotations(parse(path))
        if _bare_containers(annotation)
    ]
    assert not offenders, (
        f"{path.relative_to(SRC)} annotates a bare container, which types "
        f"nothing: {offenders}"
    )


@pytest.mark.parametrize(
    "path", python_files(SRC), ids=lambda p: str(p.relative_to(SRC))
)
def test_any_only_in_allowlist(path):
    relative = str(path.relative_to(SRC))
    offenders = [
        function.name
        for function in _functions(parse(path))
        if (relative, function.name) not in ANY_ALLOWED
        and any(
            isinstance(node, ast.Name) and node.id == "Any"
            for annotation in _signature_annotations(function)
            for node in ast.walk(annotation)
        )
    ]
    assert not offenders, (
        f"{relative} uses Any in the signature of {offenders}. Narrow it at the "
        "boundary, or add it to ANY_ALLOWED with a reason."
    )


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_nodes_return_named_update(package):
    offenders = [
        function.name
        for function in _functions(parse(package / "nodes.py"))
        if not function.name.startswith("_")
        and function.returns is not None
        and ast.unparse(function.returns).startswith("dict")
    ]
    assert not offenders, (
        f"{package.name}/nodes.py returns a raw dict from {offenders}. A node "
        "returns the agent's partial-state TypedDict, so a mistyped key fails."
    )
