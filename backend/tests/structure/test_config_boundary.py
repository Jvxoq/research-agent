"""Only core/config.py touches the environment."""

import ast

from core.config import get_settings
from tests.structure.conftest import SRC, parse, python_files

CONFIG_MODULE = SRC / "core" / "config.py"


def environment_uses(tree: ast.Module) -> set[str]:
    """Return the os.environ or os.getenv uses in a file."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "os" and node.attr in {"environ", "getenv"}:
                found.add(f"os.{node.attr}")
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            found.update(
                f"os.{a.name}" for a in node.names if a.name in {"environ", "getenv"}
            )
    return found


def test_only_config_reads_env():
    offenders = {}
    for path in python_files(SRC):
        if path == CONFIG_MODULE:
            continue
        uses = environment_uses(parse(path))
        if uses:
            offenders[str(path.relative_to(SRC))] = sorted(uses)
    assert not offenders, (
        f"environment access outside core/config.py: {offenders}"
    )


def settings_classes(tree: ast.Module) -> set[str]:
    """Return the classes in a file that subclass pydantic-settings' BaseSettings."""
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(base, ast.Name) and base.id == "BaseSettings"
            for base in node.bases
        )
    }


def test_config_module_reads_env():
    """core/config.py reads the environment through a BaseSettings class, not os."""
    assert settings_classes(parse(CONFIG_MODULE)), (
        "core/config.py defines no BaseSettings class; has the environment moved?"
    )


def test_settings_are_cached():
    assert get_settings() is get_settings(), (
        "core/config.py must expose a cached get_settings()"
    )
