"""Every per-run configuration field must also be settable by environment.

`Configuration.from_runnable_config` reads a field from `Settings` by its
uppercased name. A field added to an agent's configuration and forgotten in
`core/config.py` silently loses environment support, with no error anywhere.
"""

import pytest

from core.config import Settings
from tests.structure.conftest import agent_packages, configuration_classes


@pytest.mark.parametrize(
    "package,configuration",
    configuration_classes(),
    ids=[p.name for p in agent_packages()],
)
def test_config_fields_have_settings(package, configuration):
    expected = {name.upper() for name in configuration.model_fields}
    missing = sorted(expected - set(Settings.model_fields))
    assert not missing, (
        f"{package.name}/configuration.py declares fields with no counterpart "
        f"in core/config.py: {missing}"
    )


@pytest.mark.parametrize(
    "package,configuration",
    configuration_classes(),
    ids=[p.name for p in agent_packages()],
)
def test_an_optional_settings_field_means_unset(package, configuration):
    """A Settings field must default to None, or an unset variable wins anyway."""
    wrong = [
        name
        for name in configuration.model_fields
        if name.upper() in Settings.model_fields
        and Settings.model_fields[name.upper()].default is not None
    ]
    assert not wrong, (
        f"these Settings fields shadow the agent's own defaults: {sorted(wrong)}"
    )
