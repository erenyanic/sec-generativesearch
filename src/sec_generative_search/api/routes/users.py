"""Admin-tier user management routes.

Surface:

- ``POST /api/admin/users`` — mint a single-use enrolment token bound
  to a username. Admin shares the token (or the convenience URL) with
  the user out-of-band. **The admin never sees the user password.**
- ``DELETE /api/admin/users/{user_id}`` — wipe a user row + their
  vault. Forces fresh enrolment if the same username re-appears.
- ``POST /api/admin/users/{user_id}/unlock`` — clear the lockout state
  early (the soft lock auto-clears after 15 minutes; this is the
  operator-driven early-clear seam).

Wire contract:

- Admin tier (``admin_route_dependencies``) — both ``X-API-Key`` and
  ``X-Admin-Key`` must validate.
- Routes refuse with a ``503 user_tier_disabled`` envelope when
  ``app.state.user_store`` is ``None`` (no SQLCipher / no pepper).
- The enrolment-token response carries the token and an opaque
  ``enrol_url`` only; the route does NOT inject the deployment
  hostname (it cannot know it reliably behind a reverse proxy) — the
  admin SPA builds the absolute URL client-side.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Request

from sec_generative_search.api.dependencies import (
    admin_route_dependencies,
    get_user_store,
    require_user_store,
)
from sec_generative_search.api.errors import database_error, http_error
from sec_generative_search.api.schemas import (
    AdminUserCreateRequest,
    AdminUserCreateResponse,
    AdminUserDeleteResponse,
    AdminUserUnlockResponse,
)
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    DatabaseError,
)
from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.core.security import mask_secret
from sec_generative_search.core.user_auth import (
    mint_enrolment_token,
    verify_enrolment_token,
)
from sec_generative_search.database.users import UserStore

__all__ = ["router"]


logger = get_logger(__name__)


# ``admin_route_dependencies`` returns ``[verify_api_key, verify_admin_key]``
# — both must hit. Attached at router-level so every route on this
# router carries the same two-tier gate.
router = APIRouter(dependencies=admin_route_dependencies())


@router.post(
    "/users",
    response_model=AdminUserCreateResponse,
    status_code=201,
    summary="Mint an enrolment token for a fresh username",
)
async def create_user(
    request: Request,
    body: AdminUserCreateRequest,
    store: UserStore | None = Depends(get_user_store),
) -> AdminUserCreateResponse:
    """Mint a single-use signed enrolment token bound to ``body.username``.

    The token is HMAC'd under the deployment pepper and expires after
    :data:`~sec_generative_search.core.user_auth.DEFAULT_ENROLMENT_TTL_SECONDS`
    (30 minutes by default).  If the username already corresponds to
    an enrolled user, the route refuses with ``409 username_exists`` —
    the admin must DELETE the existing user first.
    """
    user_store = require_user_store(store)
    pepper = get_settings().api.auth_pepper

    if user_store.get_by_username(body.username) is not None:
        raise http_error(
            status_code=409,
            error="username_exists",
            message="A user with this username is already enrolled.",
            hint=("DELETE /api/admin/users/{id} first if the enrolment must be re-issued."),
        )

    try:
        token = mint_enrolment_token(body.username, pepper)
    except ConfigurationError as exc:
        raise http_error(
            status_code=503,
            error="user_tier_disabled",
            message="Server pepper is not configured.",
        ) from exc

    # Decode the token's payload to surface the expiry — we already
    # know the values used at mint time, but reading them back via the
    # public verify path keeps the contract single-sourced (no two
    # places to update if the envelope format changes).
    payload = verify_enrolment_token(token, pepper)

    audit_log(
        "user_enrol_token_minted",
        detail=(f"username_tail={mask_secret(body.username)} expires_at={payload.expires_at}"),
    )

    # ``enrol_url`` carries only the *path* portion; the SPA builds the
    # absolute URL because the API process cannot know the deployment
    # hostname reliably behind a reverse proxy.
    return AdminUserCreateResponse(
        username=body.username,
        enrolment_token=token,
        expires_at=payload.expires_at,
        enrol_url=f"/enrol?token={token}",
    )


@router.delete(
    "/users/{user_id}",
    response_model=AdminUserDeleteResponse,
    summary="Delete a user row + their encrypted vault",
)
async def delete_user(
    user_id: int = Path(..., ge=1),
    store: UserStore | None = Depends(get_user_store),
) -> AdminUserDeleteResponse:
    """Hard-delete the user row.

    No undo — the encrypted vault goes with it. The recovery path is a
    fresh enrolment under the same username (which will mint a fresh
    salt + KEK, so the old ciphertext was already unreadable even if
    an attacker had it).
    """
    user_store = require_user_store(store)
    try:
        deleted = user_store.delete_user(user_id)
    except DatabaseError as exc:
        raise database_error() from exc
    if not deleted:
        raise http_error(
            status_code=404,
            error="user_not_found",
            message=f"User {user_id} does not exist.",
        )
    return AdminUserDeleteResponse(deleted=True, user_id=user_id)


@router.post(
    "/users/{user_id}/unlock",
    response_model=AdminUserUnlockResponse,
    summary="Clear the lockout state for a user",
)
async def unlock_user(
    user_id: int = Path(..., ge=1),
    store: UserStore | None = Depends(get_user_store),
) -> AdminUserUnlockResponse:
    """Clear ``failed_login_count`` + ``locked_until`` for ``user_id``.

    Idempotent — a never-locked user returns ``unlocked=True``
    regardless. A non-existent user returns 404.
    """
    user_store = require_user_store(store)
    if user_store.get_by_id(user_id) is None:
        raise http_error(
            status_code=404,
            error="user_not_found",
            message=f"User {user_id} does not exist.",
        )
    try:
        cleared = user_store.unlock(user_id)
    except DatabaseError as exc:
        raise database_error() from exc
    return AdminUserUnlockResponse(unlocked=cleared, user_id=user_id)
