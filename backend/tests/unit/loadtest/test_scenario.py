"""Finding each agent's load scenario, and refusing an agent that has none."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import evals
from loadtest.scenario import MissingScenario, RunOutcome, Scenario, load_scenario

MANIFEST = Path(__file__).resolve().parents[3] / "langgraph.json"


def _manifest_graphs() -> list[str]:
    graphs = list(json.loads(MANIFEST.read_text())["graphs"])
    assert graphs, "langgraph.json lists no graphs"
    return graphs


def _outcome(state: object) -> RunOutcome:
    raise AssertionError("not called")


@pytest.fixture
def extra_agents(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Return a helper that adds an agent's eval package with a given load.py."""
    monkeypatch.setattr(evals, "__path__", [*evals.__path__, str(tmp_path)])

    def add(name: str, load_source: str) -> None:
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_text("")
        (package / "load.py").write_text(load_source)
        for module in (f"evals.{name}", f"evals.{name}.load"):
            monkeypatch.delitem(sys.modules, module, raising=False)

    return add


@pytest.mark.parametrize("graph_id", _manifest_graphs())
def test_every_manifest_graph_has_scenario(graph_id: str) -> None:
    """A graph the server serves but the load test cannot drive would go unmeasured."""
    assert isinstance(load_scenario(graph_id), Scenario)


def test_missing_scenario_refused() -> None:
    with pytest.raises(MissingScenario, match="no_such_agent has no evals/no_such_agent/load.py"):
        load_scenario("no_such_agent")


def test_scenario_without_constant_refused(extra_agents) -> None:
    extra_agents("bare_agent", "VALUE = 1\n")

    with pytest.raises(MissingScenario, match="defines no Scenario named SCENARIO"):
        load_scenario("bare_agent")


def test_broken_scenario_import_raises(extra_agents) -> None:
    """A bug inside load.py must surface, not pass as a missing scenario."""
    extra_agents("broken_agent", "import no_such_dependency_xyz\n")

    with pytest.raises(ModuleNotFoundError, match="no_such_dependency_xyz"):
        load_scenario("broken_agent")


def test_scenario_needs_inputs() -> None:
    """Run n sends inputs[n % len], so no inputs would crash the level mid-run."""
    with pytest.raises(ValueError, match="needs at least one input"):
        Scenario(inputs=(), reply=lambda request: "", outcome=_outcome)
