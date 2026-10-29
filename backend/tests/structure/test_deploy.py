"""The image CI pushes and the Render service that runs it.

The image must run the versions the tests ran against and carry no local secret.
render.yaml must name every variable production needs and hold none of their values.
"""

import shlex
import tomllib

import pytest
import yaml
from langgraph_cli.config import config_to_docker, validate_config_file

from core.config import Settings
from tests.structure.conftest import BACKEND

MANIFEST_PATH = BACKEND / "langgraph.json"
DOCKERIGNORE_PATH = BACKEND / ".dockerignore"
RENDER_PATH = BACKEND.parent / "render.yaml"

# The install needs these and nothing else. Everything else in backend/, .env above
# all, stays out of the build context.
ALLOWED_IN_IMAGE = frozenset({"pyproject.toml", "uv.lock", "src/"})

# Read by LangGraph Server itself, so core/config.py has no field for them.
SERVER_VARIABLES = frozenset({"DATABASE_URI", "REDIS_URI", "N_JOBS_PER_WORKER"})


@pytest.fixture(scope="module")
def render_env() -> list[dict]:
    blueprint = yaml.safe_load(RENDER_PATH.read_text(encoding="utf-8"))
    (service,) = blueprint["services"]
    assert service["envVars"], "render.yaml declares no variables"
    return service["envVars"]


def locked_version(package: str) -> str:
    """Return the version uv.lock pins for one package."""
    lock = tomllib.loads((BACKEND / "uv.lock").read_text(encoding="utf-8"))
    return next(p["version"] for p in lock["package"] if p["name"] == package)


def dockerignore_leaks(text: str) -> list[str]:
    """Return every reason the build context could carry more than the install needs."""
    rules = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    leaks = [] if rules[:1] == ["*"] else ["first rule does not refuse everything"]
    return leaks + [r for r in rules if r.startswith("!") and r[1:] not in ALLOWED_IN_IMAGE]


def run_words(lines: list[str], command: str) -> list[tuple[int, list[str]]]:
    """Return the line number and words of every RUN line that starts with a command."""
    return [
        (i, shlex.split(line))
        for i, line in enumerate(lines)
        if line.startswith(f"RUN {command} ")
    ]


def lock_install_problems(dockerfile: str) -> list[str]:
    """Return every reason the Dockerfile skips installing uv.lock's pins.

    The CLI's own install step comes last and keeps a version that already fits, so
    the pins must be in place before it. CI checks the built image as well.
    """
    lines = dockerfile.splitlines()
    cli_install = next(i for i, line in enumerate(lines) if "for dep in /deps/*" in line)
    exports = run_words(lines, "uv export")
    installs = [
        (i, words) for i, words in run_words(lines, "uv pip install") if "-r" in words
    ]
    if not exports or not installs:
        return ["no install from the exported lock"]

    (export_at, export), (install_at, install) = exports[0], installs[0]
    problems = [] if "--frozen" in export else ["export may re-resolve the lock"]
    if export[export.index("--output-file") + 1] != install[install.index("-r") + 1]:
        problems.append("install reads another file")
    if max(export_at, install_at) > cli_install:
        problems.append("pins arrive after the install")
    return problems


def baked_values(env_vars: list[dict]) -> list[str]:
    """Return every variable render.yaml gives a value instead of leaving to Render."""
    return [v["key"] for v in env_vars if v != {"key": v["key"], "sync": False}]


def unread_keys(env_vars: list[dict]) -> list[str]:
    """Return every variable neither LangGraph Server nor core/config.py reads."""
    known = SERVER_VARIABLES | set(Settings.model_fields)
    return sorted(v["key"] for v in env_vars if v["key"] not in known)


def test_image_server_matches_lock():
    """CI tests against the locked server, so the image must run that same one."""
    config = validate_config_file(MANIFEST_PATH)

    assert config["api_version"] == locked_version("langgraph-api")


def test_image_installs_from_lock():
    config = validate_config_file(MANIFEST_PATH)
    dockerfile, _ = config_to_docker(MANIFEST_PATH, config)

    assert lock_install_problems(dockerfile) == []


CLI_INSTALL = 'RUN for dep in /deps/*; do (cd "$dep" && uv pip install --system -e .); done'
EXPORT = "RUN uv export --frozen --output-file /lock/pins.txt"
PINS = "RUN uv pip install --system -r /lock/pins.txt"


@pytest.mark.parametrize(
    ("dockerfile", "expected"),
    [
        ("\n".join([EXPORT, CLI_INSTALL]), ["no install from the exported lock"]),
        (
            "\n".join([EXPORT.replace(" --frozen", ""), PINS, CLI_INSTALL]),
            ["export may re-resolve the lock"],
        ),
        (
            "\n".join([EXPORT, PINS.replace("pins", "other"), CLI_INSTALL]),
            ["install reads another file"],
        ),
        ("\n".join([EXPORT, CLI_INSTALL, PINS]), ["pins arrive after the install"]),
    ],
    ids=["no-pins-install", "not-frozen", "other-file", "pins-too-late"],
)
def test_unpinned_install_is_reported(dockerfile, expected):
    assert lock_install_problems(dockerfile) == expected


def test_build_context_holds_only_install_files():
    """langgraph build copies backend/ whole, and backend/.env holds real keys."""
    text = DOCKERIGNORE_PATH.read_text(encoding="utf-8")

    assert dockerignore_leaks(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("*\n!src/\n!.env\n", ["!.env"]),
        ("!src/\n", ["first rule does not refuse everything"]),
        ("", ["first rule does not refuse everything"]),
    ],
    ids=["env-let-in", "no-refuse-all", "empty"],
)
def test_build_context_leak_is_reported(text, expected):
    assert dockerignore_leaks(text) == expected


def test_render_holds_no_values(render_env):
    """A value here lands in git, and Render's environment is where values belong."""
    assert baked_values(render_env) == []


@pytest.mark.parametrize(
    "env_vars",
    [
        [{"key": "N_JOBS_PER_WORKER", "value": "50"}, {"key": "BASE_URL", "sync": False}],
        [{"key": "N_JOBS_PER_WORKER"}, {"key": "BASE_URL", "sync": False}],
    ],
    ids=["value", "no-sync"],
)
def test_baked_value_is_reported(env_vars):
    assert baked_values(env_vars) == ["N_JOBS_PER_WORKER"]


def test_render_keys_are_read(render_env):
    """A misspelled name is ignored, so the server runs on its default."""
    assert unread_keys(render_env) == []


def test_misspelled_render_key_is_reported():
    env_vars = [{"key": "N_JOB_PER_WORKER", "sync": False}, {"key": "REDIS_URI", "sync": False}]

    assert unread_keys(env_vars) == ["N_JOB_PER_WORKER"]


def test_render_declares_server_variables(render_env):
    """Without these the server cannot start, or runs 10 runs at once."""
    declared = {v["key"] for v in render_env}

    assert sorted(SERVER_VARIABLES - declared) == []
