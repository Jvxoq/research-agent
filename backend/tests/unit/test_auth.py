"""Who the server lets in, and what they are allowed to see."""

import jwt
import pytest
from langgraph_sdk import Auth

from core.auth import (
    DEV_IDENTITY,
    authenticate,
    reset_development_warning,
    scope_crons,
    scope_runs,
    scope_store,
    scope_threads,
)
from tests.conftest import FakeAuthContext

SECRET = "test-signing-secret-that-is-long-enough-for-hs256-and-for-hs512-too"
ISSUER = "talk-to-web"
AUDIENCE = "research"
# Fixed instants, so no test depends on today's date.
FAR_FUTURE = 4_102_444_800
LONG_PAST = 946_684_800


@pytest.fixture(autouse=True)
def _reset_warning():
    reset_development_warning()
    yield
    reset_development_warning()


@pytest.fixture
def jwt_env(set_env):
    """Configure the server the way production verifies the web app's tokens."""
    set_env(
        ENVIRONMENT="production",
        JWT_SECRET=SECRET,
        JWT_ISSUER=ISSUER,
        JWT_AUDIENCE=AUDIENCE,
    )


def make_token(secret: str = SECRET, algorithm: str = "HS256", **overrides) -> str:
    claims = {"sub": "42", "iss": ISSUER, "aud": AUDIENCE, "exp": FAR_FUTURE}
    claims.update(overrides)
    return jwt.encode(
        {k: v for k, v in claims.items() if v is not None}, secret, algorithm=algorithm
    )


def bearer(token: str) -> dict:
    return {"authorization": f"Bearer {token}"}


async def refused(headers) -> Auth.exceptions.HTTPException:
    with pytest.raises(Auth.exceptions.HTTPException) as raised:
        await authenticate(headers)
    return raised.value


async def test_valid_token_uses_subject(jwt_env):
    user = await authenticate(bearer(make_token(sub="42")))

    assert user["identity"] == "42"


async def test_byte_headers_are_accepted(jwt_env):
    token = make_token().encode()

    user = await authenticate({b"authorization": b"Bearer " + token})

    assert user["identity"] == "42"


@pytest.mark.parametrize(
    "token",
    [
        make_token(secret="a-different-secret-that-is-also-long-enough"),
        make_token(iss="someone-else"),
        make_token(aud="talk-to-web-browser"),
        make_token(aud=None),
        make_token(iss=None),
        make_token(sub=None),
        make_token(exp=None),
        make_token(sub=""),
        make_token(algorithm="HS512"),
        jwt.encode({"sub": "42", "iss": ISSUER, "aud": AUDIENCE, "exp": FAR_FUTURE}, None, algorithm="none"),
        "not-a-jwt",
    ],
    ids=[
        "wrong-secret",
        "wrong-issuer",
        "wrong-audience",
        "no-audience",
        "no-issuer",
        "no-subject",
        "no-expiry",
        "blank-subject",
        "other-algorithm",
        "unsigned",
        "garbage",
    ],
)
async def test_unverified_token_refused(jwt_env, token):
    error = await refused(bearer(token))

    assert (error.status_code, error.detail) == (401, "Invalid or missing bearer token")


async def test_expired_token_refused(jwt_env):
    error = await refused(bearer(make_token(exp=LONG_PAST)))

    assert (error.status_code, error.detail) == (401, "Token expired")


async def test_a_missing_header_is_refused(jwt_env):
    error = await refused({})

    assert error.status_code == 401


async def test_a_non_bearer_scheme_is_refused(jwt_env):
    error = await refused({"authorization": f"Basic {make_token()}"})

    assert error.status_code == 401


async def test_secret_without_audience_fails_closed(set_env):
    """Without an audience, a token meant for another service would pass."""
    set_env(ENVIRONMENT="production", JWT_SECRET=SECRET, JWT_ISSUER=ISSUER)

    error = await refused(bearer(make_token()))

    assert (error.status_code, error.detail) == (
        401,
        "Server authentication is not configured",
    )


async def test_dev_without_secret_gets_local_user(set_env):
    set_env(ENVIRONMENT="development")

    user = await authenticate({})

    assert user["identity"] == DEV_IDENTITY


async def test_production_without_a_secret_fails_closed(set_env):
    set_env(ENVIRONMENT="production")

    error = await refused({})

    assert error.status_code == 401


async def test_no_secret_no_env_fails_closed():
    """The default is production, so a forgotten variable never opens the door."""
    error = await refused({})

    assert error.status_code == 401


async def test_any_non_development_environment_fails_closed(set_env):
    set_env(ENVIRONMENT="staging")

    error = await refused(bearer(make_token()))

    assert error.status_code == 401


SCOPED = (scope_threads, scope_runs, scope_crons, scope_store)


@pytest.mark.parametrize("handler", SCOPED, ids=lambda h: h.__name__)
async def test_resources_scoped_to_owner(handler):
    """Stamping and filtering are one handler call, so one test covers both."""
    value: dict = {}

    filter_ = await handler(ctx=FakeAuthContext(), value=value)

    assert value["metadata"]["owner"] == "user-42"
    assert filter_ == {"owner": "user-42"}


async def test_existing_metadata_is_kept():
    value = {"metadata": {"source": "cli"}}

    await scope_threads(ctx=FakeAuthContext(), value=value)

    assert value["metadata"] == {"source": "cli", "owner": "user-42"}
