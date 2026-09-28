"""Server-side ``session_id`` minting and revocation.

A weak or forgeable ``session_id`` collapses the entire credential
isolation model: one tenant's guess gives them another tenant's keys.

Cookie contract enforced here:

        - **Server-minted only** — ``secrets.token_urlsafe(32)`` (≥256 bits
            of entropy). Any value supplied by a browser is ignored on
            ``POST /api/session``; mint always replaces.
        - **HTTP-only** — JavaScript cannot read the cookie.
        - **``Secure``** — browser refuses to send over plain HTTP except
            from ``localhost``/``127.0.0.1`` (modern browsers treat localhost
            as a secure context).
        - **``SameSite=Strict``** — eliminates the cross-site request
      forgery vector for credentialed endpoints.
    - **Rotated on auth-state change** — call ``POST /api/session``
      again on login/logout; the prior session is invalidated and a
      fresh ``session_id`` is issued.
    - **Never logged raw** — only ``mask_secret``-tailed forms hit the
      audit log (already enforced by the credentials module).
    - **Never in URLs / query strings / referrers** — the cookie path
      is the only transport surface.

Logout (``POST /api/session/logout``) clears the in-memory store entry
for the cookie's ``session_id`` and expires the cookie immediately.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from sec_generative_search.api.dependencies import (
    SESSION_COOKIE_NAME,
    clear_session_cookie,
    client_ip,
    extract_session_id,
    get_edgar_identity_store,
    get_session_store,
    mint_session_id,
    set_session_cookie,
)
from sec_generative_search.api.errors import http_error
from sec_generative_search.api.schemas import (
    EdgarIdentityClearResponse,
    EdgarIdentityRegisterResponse,
    EdgarIdentityRequest,
    SessionLogoutResponse,
    SessionResponse,
)
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.credentials import CredentialStore
from sec_generative_search.core.edgar_identity import (
    EdgarIdentity,
    InMemorySessionEdgarIdentityStore,
)
from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.core.security import mask_secret

__all__ = ["router"]


logger = get_logger(__name__)


router = APIRouter()


# Module-level seam the tests patch.  The implementation (and the 32-byte
# / 256-bit ``secrets.token_urlsafe`` floor — never weaken) is shared with
# ``auth.py`` in ``api/dependencies.py``.
_mint_session_id = mint_session_id


@router.post(
    "/session",
    response_model=SessionResponse,
    tags=["session"],
    summary="Mint a server-side session_id",
    status_code=201,
)
async def mint_session(
    request: Request,
    response: Response,
    store: CredentialStore = Depends(get_session_store),
    edgar_store: InMemorySessionEdgarIdentityStore = Depends(get_edgar_identity_store),
) -> SessionResponse:
    """Issue a fresh ``session_id`` cookie.

    Idempotent rotation semantics: when the request already carries a
    ``session_id`` cookie, the prior session's stored credentials *and*
    EDGAR identity are cleared from their in-memory stores before the
    new id is minted.  An authenticating client sends the call once and
    the rotation happens transparently.

    The newly minted ``session_id`` is returned exclusively via the
    ``Set-Cookie`` header — the JSON body never carries it.
    """
    prior = extract_session_id(request)
    cleared = 0
    if prior is not None:
        cleared = store.clear(prior)
        # Clear any per-session EDGAR identity for the prior session so
        # the rotated cookie cannot reach back into the previous user's
        # EDGAR scope.  The clear is silent on miss (idempotent).
        edgar_store.delete(prior)
        if cleared:
            logger.info(
                "Rotated session %s — cleared %d credential(s)",
                mask_secret(prior),
                cleared,
            )

    session_id = _mint_session_id()
    audit_log(
        "session_mint",
        detail=(
            f"session_id_tail={mask_secret(session_id)} "
            f"client_ip={client_ip(request)} "
            f"rotated={'yes' if prior is not None else 'no'}"
        ),
    )

    # Sliding TTL — the in-memory store evicts on idle; the cookie
    # ``Max-Age`` matches the operator-configurable session TTL.
    ttl_seconds = get_settings().api.session_ttl_seconds
    set_session_cookie(response, session_id, max_age=ttl_seconds)

    return SessionResponse(
        issued=True,
        cookie_name=SESSION_COOKIE_NAME,
        expires_in_seconds=ttl_seconds,
    )


@router.post(
    "/session/logout",
    response_model=SessionLogoutResponse,
    tags=["session"],
    summary="Invalidate the current session_id",
)
async def logout_session(
    request: Request,
    response: Response,
    store: CredentialStore = Depends(get_session_store),
    edgar_store: InMemorySessionEdgarIdentityStore = Depends(get_edgar_identity_store),
) -> SessionLogoutResponse:
    """Clear the session's credentials, EDGAR identity, and expire the cookie.

    Idempotent — a request with no cookie returns ``cleared=0`` and the
    expired-cookie ``Set-Cookie`` is still emitted defensively.
    """
    session_id = extract_session_id(request)
    cleared = 0
    edgar_cleared = False
    if session_id is not None:
        cleared = store.clear(session_id)
        edgar_cleared = edgar_store.delete(session_id)
        audit_log(
            "session_logout",
            detail=(
                f"session_id_tail={mask_secret(session_id)} "
                f"cleared={cleared} edgar_cleared={edgar_cleared}"
            ),
        )

    clear_session_cookie(response)
    return SessionLogoutResponse(
        cleared_credentials=cleared,
        cleared_edgar_identity=edgar_cleared,
    )


# ---------------------------------------------------------------------------
# Per-session EDGAR identity
# ---------------------------------------------------------------------------


def _require_active_session_id(request: Request) -> str:
    """Return the validated ``session_id`` cookie or raise 401.

    EDGAR-identity registration is meaningless without an active
    server-minted session — the cookie is the only key under which the
    identity can be looked up by downstream routes.
    """
    session_id = extract_session_id(request)
    if session_id is None:
        raise http_error(
            status_code=401,
            error="session_required",
            message="No active session.",
            hint="Mint a session via POST /api/session before registering an EDGAR identity.",
        )
    return session_id


@router.post(
    "/session/edgar",
    response_model=EdgarIdentityRegisterResponse,
    tags=["session"],
    summary="Register an EDGAR identity for the active session",
    status_code=201,
)
async def register_edgar_identity(
    request: Request,
    body: EdgarIdentityRequest,
    edgar_store: InMemorySessionEdgarIdentityStore = Depends(get_edgar_identity_store),
) -> EdgarIdentityRegisterResponse:
    """Bind ``(name, email)`` to the current ``session_id``.

    The body is validated by Pydantic at the schema boundary and again
    by :class:`EdgarIdentity.from_strings` for shape (control chars,
    syntactically valid email).  Validation errors raise the unified
    400 envelope and never echo the supplied value back.

    Audit-log: emits a single ``edgar_identity_set`` line carrying only
    the masked session-id tail.  Name and email are NEVER logged.
    """
    session_id = _require_active_session_id(request)

    try:
        identity = EdgarIdentity.from_strings(body.name, body.email)
    except ValueError as exc:
        raise http_error(
            status_code=400,
            error="invalid_edgar_identity",
            message="EDGAR identity failed validation.",
            hint=str(exc),
        ) from exc

    edgar_store.set(session_id, identity)
    return EdgarIdentityRegisterResponse(registered=True)


@router.delete(
    "/session/edgar",
    response_model=EdgarIdentityClearResponse,
    tags=["session"],
    summary="Clear the active session's EDGAR identity",
)
async def clear_edgar_identity(
    request: Request,
    edgar_store: InMemorySessionEdgarIdentityStore = Depends(get_edgar_identity_store),
) -> EdgarIdentityClearResponse:
    """Drop the EDGAR identity for the current ``session_id``.

    Idempotent — a request with no cookie or no stored identity returns
    ``cleared=False`` without raising.  This route does NOT expire the
    session cookie; ``POST /api/session/logout`` is the single seam that
    invalidates the whole session.
    """
    session_id = extract_session_id(request)
    cleared = False
    if session_id is not None:
        cleared = edgar_store.delete(session_id)
    return EdgarIdentityClearResponse(cleared=cleared)
