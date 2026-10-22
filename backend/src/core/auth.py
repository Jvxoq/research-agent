"""Authentication and per-user thread scoping for the LangGraph Server.

The handler verifies the web application's JWT from the ``Authorization``
header, then stamps the user it names on the threads, runs, crons and store
records that user writes, and filters those same four resources on read.
Assistants stay unfiltered, because they come from ``langgraph.json`` and
belong to the deployment.

A tenant is one user. The owner is the token's ``sub`` claim, and nothing else
in the request can change it. A token must carry a valid signature, an unexpired
``exp``, and the configured issuer and audience.

Local development stays usable. When no secret is configured and
``ENVIRONMENT`` is ``development``, the handler returns a fixed ``local-dev``
identity and warns once. Any other environment fails closed.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from functools import lru_cache
from typing import Any, cast

import jwt
from langgraph_sdk import Auth
from loguru import logger

from core.config import get_settings

DEV_IDENTITY = "local-dev"

auth = Auth()


def _bearer_token(headers: Mapping[str | bytes, str | bytes]) -> str | None:
    """Return the bearer token from an Authorization header.

    Args:
        headers: Request headers, keyed as bytes or as strings.

    Returns:
        The token, or None when the header is absent or malformed.
    """
    for key, value in headers.items():
        name = key.decode() if isinstance(key, bytes) else key
        if name.lower() != "authorization":
            continue
        raw = value.decode() if isinstance(value, bytes) else value
        scheme, _, token = raw.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token
        return None
    return None


@lru_cache(maxsize=1)
def _warn_open_development() -> None:
    """Report the open development door once per process."""
    logger.warning(
        "JWT_SECRET is not set - accepting unauthenticated "
        f"requests as {DEV_IDENTITY!r} because ENVIRONMENT is development"
    )


def reset_development_warning() -> None:
    """Let the next unauthenticated development request warn again."""
    _warn_open_development.cache_clear()


def _allow_unauthenticated_dev() -> bool:
    """Report whether this process may run without a secret, and warn once."""
    if get_settings().ENVIRONMENT != "development":
        return False

    _warn_open_development()
    return True


def _unauthorized(detail: str) -> Auth.exceptions.HTTPException:
    """Build the 401 every refused request gets."""
    return Auth.exceptions.HTTPException(status_code=401, detail=detail)


def _user(identity: str) -> Auth.types.MinimalUserDict:
    """Describe an authenticated caller the way LangGraph Server expects."""
    return {
        "identity": identity,
        "display_name": identity,
        "is_authenticated": True,
        "permissions": [],
    }


@auth.authenticate
async def authenticate(
    headers: Mapping[str | bytes, str | bytes],
) -> Auth.types.MinimalUserDict:
    """Resolve the caller from the JWT in the Authorization header.

    Args:
        headers: Request headers as delivered by the server.

    Returns:
        The authenticated caller, named by the token's ``sub`` claim.

    Raises:
        Auth.exceptions.HTTPException: 401 when the server has no complete JWT
            setup, or the token is missing, expired or fails verification.
    """
    settings = get_settings()
    secret, issuer, audience = (
        settings.JWT_SECRET,
        settings.JWT_ISSUER,
        settings.JWT_AUDIENCE,
    )

    if not secret:
        if _allow_unauthenticated_dev():
            return _user(DEV_IDENTITY)
        raise _unauthorized("Server authentication is not configured")
    if not issuer or not audience:
        # A token checked without an audience is accepted from any service that
        # shares the secret, including tokens the browser already holds.
        logger.error("JWT_ISSUER and JWT_AUDIENCE must be set with JWT_SECRET")
        raise _unauthorized("Server authentication is not configured")

    token = _bearer_token(headers)
    if token is None:
        raise _unauthorized("Invalid or missing bearer token")

    try:
        claims = jwt.decode(
            token,
            secret,
            # Exactly the configured algorithm. Trusting the token's own `alg`
            # header lets "none" or an HMAC-for-RSA swap verify a forgery.
            algorithms=[settings.JWT_ALGORITHM],
            issuer=issuer,
            audience=audience,
            options={"require": ["exp", "sub", "iss", "aud"]},
        )
    except jwt.ExpiredSignatureError:
        raise _unauthorized("Token expired") from None
    except jwt.PyJWTError as error:
        # The reason stays in the log. Returning it tells a forger what to fix.
        logger.info(f"Rejected a bearer token: {type(error).__name__}")
        raise _unauthorized("Invalid or missing bearer token") from None

    subject = claims["sub"]
    if not isinstance(subject, str) or not subject.strip():
        raise _unauthorized("Invalid or missing bearer token")
    return _user(subject)


def _own(
    ctx: Auth.types.AuthContext, value: MutableMapping[str, Any]
) -> dict[str, str]:
    """Stamp the caller on a write and filter a read to that caller.

    Args:
        ctx: The authenticated context for this request.
        value: The payload the server is about to act on.

    Returns:
        A filter restricting the request to the caller's own resources.
    """
    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters


@auth.on.threads
async def scope_threads(
    *, ctx: Auth.types.AuthContext, value: MutableMapping[str, Any]
) -> dict[str, str]:
    """Restrict threads to the caller who created them."""
    return _own(ctx, value)


@auth.on.threads.create_run
async def scope_runs(
    *, ctx: Auth.types.AuthContext, value: Auth.types.RunsCreate
) -> dict[str, str]:
    """Restrict runs to the caller who started them."""
    # The SDK types this one action as a TypedDict. At runtime it is the same
    # mutable payload every other handler receives.
    return _own(ctx, cast(MutableMapping[str, Any], value))


@auth.on.crons
async def scope_crons(
    *, ctx: Auth.types.AuthContext, value: MutableMapping[str, Any]
) -> dict[str, str]:
    """Restrict scheduled runs to the caller who created them."""
    return _own(ctx, value)


@auth.on.store
async def scope_store(
    *, ctx: Auth.types.AuthContext, value: MutableMapping[str, Any]
) -> dict[str, str]:
    """Restrict stored records to the caller who wrote them."""
    return _own(ctx, value)
