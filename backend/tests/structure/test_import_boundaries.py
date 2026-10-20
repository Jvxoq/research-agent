"""The dependency rule runs one way, and no HTTP framework enters the repo."""

import re
import tomllib

import pytest

from tests.structure.conftest import (
    CORE,
    SRC,
    agent_packages,
    imported_modules,
    parse,
    python_files,
)

FORBIDDEN_FRAMEWORKS = ("fastapi", "flask", "django", "starlette", "quart")

# The repository pins one provider-agnostic model factory in core/llm.py. An agent
# reaching for a provider SDK escapes it, and with it the per-node model
# settings and the cost ceiling.
PROVIDER_PACKAGES = frozenset(
    {
        "langchain_groq",
        "langchain_anthropic",
        "langchain_openai",
        "langchain_together",
        "langchain_tavily",
        "groq",
        "anthropic",
        "openai",
    }
)


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_an_agent_never_imports_another_agent(package):
    others = {p.name for p in agent_packages()} - {package.name}
    for path in python_files(package):
        for module in imported_modules(parse(path)):
            parts = module.split(".")
            if parts[0] == "agents" and len(parts) > 1:
                assert parts[1] not in others, (
                    f"{path.relative_to(SRC)} imports another agent: {module}"
                )


def test_core_never_imports_an_agent():
    for path in python_files(CORE):
        for module in imported_modules(parse(path)):
            assert not module.startswith("agents"), (
                f"{path.relative_to(SRC)} imports {module}; core must not know agents"
            )


def test_core_never_imports_evals():
    for path in python_files(CORE):
        for module in imported_modules(parse(path)):
            assert not module.startswith("evals"), (
                f"{path.relative_to(SRC)} imports {module}"
            )


def test_nothing_imports_an_http_framework():
    for path in python_files(SRC):
        for module in imported_modules(parse(path)):
            root = module.split(".")[0]
            assert root not in FORBIDDEN_FRAMEWORKS, (
                f"{path.relative_to(SRC)} imports {module}; "
                "LangGraph Server is the API for this repository"
            )


def declared_dependencies() -> list[str]:
    """Return every dependency the project declares, from all three blocks."""
    from tests.structure.conftest import BACKEND

    data = tomllib.loads((BACKEND / "pyproject.toml").read_text(encoding="utf-8"))
    declared = list(data["project"].get("dependencies", []))
    for group in data["project"].get("optional-dependencies", {}).values():
        declared.extend(group)
    for group in data.get("dependency-groups", {}).values():
        declared.extend(group)
    return declared


def test_no_http_framework_is_a_dependency():
    for requirement in declared_dependencies():
        name = re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0].strip().lower()
        assert name not in FORBIDDEN_FRAMEWORKS, (
            f"{name} is still a declared dependency"
        )


@pytest.mark.parametrize("package", agent_packages(), ids=lambda p: p.name)
def test_no_provider_sdk_in_agents(package):
    for path in python_files(package):
        leaked = {
            module.split(".")[0] for module in imported_modules(parse(path))
        } & PROVIDER_PACKAGES
        assert not leaked, (
            f"{path.relative_to(SRC)} imports provider SDKs directly: {sorted(leaked)}"
        )
