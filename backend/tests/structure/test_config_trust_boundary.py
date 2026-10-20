"""The request body may not name a credential, an endpoint, or a higher limit.

`configurable` arrives in the HTTP request. In production the caller is not the
operator, so a field it can set is a field an attacker can set. Three classes of
field must therefore never take a caller's value:

1. A credential. It would travel in the body and land in the checkpoint and the
   trace.
2. An endpoint. This server attaches its own key to whatever host it is told to
   call, so a caller-chosen endpoint is a key handed over.
3. A ceiling. Raising one spends the operator's provider and Tavily credit.

These tests read each agent's own declarations, so a new agent is covered
without an edit here.
"""

from __future__ import annotations

import pytest

from tests.structure.conftest import agent_packages, configuration_modules

IDS = [p.name for p in agent_packages()]

SECRET_MARKERS = ("api_key", "secret", "password", "credential", "auth_token")
ENDPOINT_MARKERS = ("base_url", "endpoint")


@pytest.mark.parametrize("package,module", configuration_modules(), ids=IDS)
def test_no_configuration_field_holds_a_secret(package, module):
    named = [
        name
        for name in module.Configuration.model_fields
        if any(marker in name.lower() for marker in SECRET_MARKERS)
    ]
    assert not named, (
        f"{package.name}/configuration.py declares credential fields {named}. "
        "Read them from core.config.get_settings instead."
    )


@pytest.mark.parametrize("package,module", configuration_modules(), ids=IDS)
def test_every_endpoint_field_is_server_only(package, module):
    server_only = getattr(module, "SERVER_ONLY_FIELDS", frozenset())
    endpoints = {
        name
        for name in module.Configuration.model_fields
        if any(marker in name.lower() for marker in ENDPOINT_MARKERS)
    }
    assert endpoints <= set(server_only), (
        f"{package.name}/configuration.py lets a caller set "
        f"{sorted(endpoints - set(server_only))}."
    )


@pytest.mark.parametrize("package,module", configuration_modules(), ids=IDS)
def test_max_fields_are_guarded(package, module):
    guarded = set(getattr(module, "CEILING_FIELDS", frozenset())) | set(
        getattr(module, "SERVER_ONLY_FIELDS", frozenset())
    )
    limits = {
        name for name in module.Configuration.model_fields if name.startswith("max_")
    }
    assert limits <= guarded, (
        f"{package.name}/configuration.py lets a caller raise "
        f"{sorted(limits - guarded)}."
    )


def field_defaults(module: object, names: frozenset[str]) -> dict[str, object]:
    """Return the declared default of each named Configuration field."""
    fields = module.Configuration.model_fields
    return {name: fields[name].default for name in sorted(names)}


def numeric_ceilings(module: object) -> dict[str, float]:
    """Return each ceiling field whose default is a number a request could exceed."""
    return {
        name: default
        for name, default in field_defaults(
            module, getattr(module, "CEILING_FIELDS", frozenset())
        ).items()
        if isinstance(default, (int, float)) and not isinstance(default, bool)
    }


@pytest.mark.parametrize("package,module", configuration_modules(), ids=IDS)
def test_server_only_ignores_request(package, module):
    defaults = field_defaults(module, getattr(module, "SERVER_ONLY_FIELDS", frozenset()))
    requested = {
        name: (not default) if isinstance(default, bool) else "https://attacker.example/v1"
        for name, default in defaults.items()
    }

    resolved = module.Configuration.from_runnable_config({"configurable": requested})

    assert {name: getattr(resolved, name) for name in defaults} == defaults


@pytest.mark.parametrize("package,module", configuration_modules(), ids=IDS)
def test_ceiling_refuses_higher_request(package, module):
    ceilings = numeric_ceilings(module)
    requested = {name: type(default)(default * 2 + 500) for name, default in ceilings.items()}

    resolved = module.Configuration.from_runnable_config({"configurable": requested})

    assert {name: getattr(resolved, name) for name in ceilings} == ceilings


@pytest.mark.parametrize("package,module", configuration_modules(), ids=IDS)
def test_ceiling_accepts_lower_request(package, module):
    lower = {
        name: type(default)(default / 2)
        for name, default in numeric_ceilings(module).items()
        if default > 0
    }

    resolved = module.Configuration.from_runnable_config({"configurable": lower})

    assert {name: getattr(resolved, name) for name in lower} == lower
