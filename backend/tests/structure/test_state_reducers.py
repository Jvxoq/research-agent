"""A rebuilt workspace must never accumulate.

An agent whose round loop rebuilds its state each round declares the fields that
hold that workspace in ``ROUND_LOOP_FIELDS``. An additive reducer on one of them
turns the loop back into a growing context, which is the exact failure the
architecture exists to avoid, and nothing else in the codebase would catch it.

Agents that declare no ``ROUND_LOOP_FIELDS`` are skipped.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from tests.structure.conftest import agent_packages, parse

# Reducers that append rather than replace.
ADDITIVE_REDUCERS = frozenset({"add", "iadd", "concat", "add_messages"})


def _round_loop_fields(tree: ast.Module) -> set[str]:
    """Return the field names a state module declares as its round workspace.

    Args:
        tree: Parsed ``state.py``.

    Returns:
        The names in ``ROUND_LOOP_FIELDS``, or an empty set when absent.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if "ROUND_LOOP_FIELDS" not in targets:
            continue
        if not isinstance(node.value, (ast.Tuple, ast.List)):
            continue
        return {
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
    return set()


def _reducer_name(node: ast.expr) -> str | None:
    """Return the dotted name of a reducer expression, if it has one."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _reducer_name(node.func)
    return None


def _additive_fields(tree: ast.Module) -> set[str]:
    """Return every annotated field whose ``Annotated`` reducer appends.

    Args:
        tree: Parsed ``state.py``.

    Returns:
        Field names carrying an additive reducer.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(
            node.target, ast.Name
        ):
            continue
        annotation = node.annotation
        if not isinstance(annotation, ast.Subscript):
            continue
        if _reducer_name(annotation.value) != "Annotated":
            continue
        arguments = (
            annotation.slice.elts
            if isinstance(annotation.slice, ast.Tuple)
            else [annotation.slice]
        )
        for argument in arguments[1:]:
            if _reducer_name(argument) in ADDITIVE_REDUCERS:
                found.add(node.target.id)
    return found


def _state_modules() -> list[tuple[pathlib.Path, set[str], set[str]]]:
    """Return each agent's state module with its declared and additive fields."""
    modules = []
    for package in agent_packages():
        tree = parse(package / "state.py")
        modules.append((package, _round_loop_fields(tree), _additive_fields(tree)))
    return modules


@pytest.mark.parametrize(
    "package,declared,additive",
    _state_modules(),
    ids=[p.name for p in agent_packages()],
)
def test_round_fields_are_replaced(package, declared, additive):
    """A field in ROUND_LOOP_FIELDS must not carry an additive reducer."""
    assert declared, (
        f"{package.name}/state.py declares no ROUND_LOOP_FIELDS, so nothing guards "
        "its round workspace from accumulating."
    )

    offending = sorted(declared & additive)
    assert not offending, (
        f"{package.name}/state.py accumulates its round workspace: {offending}. "
        "These fields are rebuilt every round and must use a replace reducer."
    )


@pytest.mark.parametrize(
    "package,declared,additive",
    _state_modules(),
    ids=[p.name for p in agent_packages()],
)
def test_round_fields_exist_in_state(package, declared, additive):
    """Every name in ROUND_LOOP_FIELDS must exist as a field in state.py."""
    tree = parse(package / "state.py")
    annotated = {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    missing = sorted(declared - annotated)
    assert not missing, (
        f"{package.name}/state.py lists {missing} in ROUND_LOOP_FIELDS "
        "but declares no such field."
    )
