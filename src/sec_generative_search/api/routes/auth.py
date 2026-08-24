"""User-tier authentication routes.

Surface:

- ``GET /api/auth/login-params?username=…`` — salt + KDF params lookup.
  Real for enrolled users; deterministic decoy for unknown ones (the
  same shape and timing so an attacker cannot enumerate usernames).
- ``POST /api/auth/login`` — validate ``auth_proof`` against the
  stored ``auth_hash``, mint a session cookie, return the ciphertext
  vault. The browser decrypts locally; the server never sees the KEK.
- ``POST /api/auth/enrol`` — close the enrolment loop: validate the
  signed one-time token, create the user row, mark the nonce
  consumed.
- ``POST /api/auth/password`` — atomic password change.
- ``DELETE /api/auth/session`` — sign-out alias that revokes the
  active session in lockstep with the legacy ``POST /api/session/logout``
  contract.
- ``POST /api/auth/vault`` — re-upload an updated ciphertext blob.

Wire contract:

- **Opaque error envelopes on the login surface.** ``login`` and
  ``login-params`` return ``401`` for any failure (unknown user, wrong
  proof, locked account, missing pepper) and the response body never
  carries enough signal to distinguish the cases. The audit log keeps
  the distinction; the wire does not.
- **No raw secrets in any response.** ``auth_proof`` and the vault
  ciphertext flow IN; only ``session`` cookies + the ciphertext blob
  flow OUT. The KEK + password never cross the wire.
- **Per-username rate limit gate.** ``login`` consults a per-username
  sliding window held on ``app.state`` (default 3 rpm). The per-IP
  bucket lives in :class:`RateLimitMiddleware`; both must allow.
"""

from __future__ import annotations

import base64
import secrets
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Request, Response

from sec_generative_search.api.dependencies import (
    SESSION_COOKIE_NAME,
    extract_session_id,
    get_edgar_identity_store,
    get_login_username_window,
    get_session_store,
    get_user_store,
)
from sec_generative_search.api.errors import http_error
from sec_generative_search.api.schemas import (
    EnrolmentCompleteRequest,
    EnrolmentCompleteResponse,
    LoginParamsResponse,
    LoginRequest,
    LoginResponse,
    PasswordChangeRequest,
    PasswordChangeResponse,
    VaultUpdateRequest,
    VaultUpdateResponse,
)
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.credentials import CredentialStore
from sec_generative_search.core.edgar_identity import InMemorySessionEdgarIdentityStore
from sec_generative_search.core.exceptions import (
    AuthError,
    ConfigurationError,
    DatabaseError,
    EnrolmentTokenError,
)
from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.core.security import mask_secret
from sec_generative_search.core.user_auth import (
    SALT_BYTES,
    decoy_salt,
    verify_enrolment_token,
)
from sec_generative_search.database.users import UserStore

__all__ = ["router"]


logger = get_logger(__name__)


router = APIRouter()


# ---------------------------------------------------------------------------
# Wire-shape helpers (base64url decode + length check at the boundary)
# ---------------------------------------------------------------------------


_VAULT_IV_BYTES = 12  # AES-GCM IV — 96-bit recommendation.
_AUTH_PROOF_BYTES = 32  # HKDF-SHA256 output width.


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode_fixed(value: str, expected_len: int, *, field: str) -> bytes:
    """Decode a base64url string and reject any unexpected length.

    The wire-side Pydantic pattern already locks the encoded length;
    this defends against an attacker who crafts a base64-valid string
    that decodes to the wrong byte width — Pydantic's length bounds
    are over the encoded text, not the decoded bytes.
    """
    padding = (-len(value)) % 4
    try:
        decoded = base64.urlsafe_b64decode(value + ("=" * padding))
    except (ValueError, TypeError) as exc:
        raise http_error(
            status_code=400,
            error="invalid_payload",
            message=f"{field} is not a valid base64url string.",
        ) from exc
    if len(decoded) != expected_len:
        raise http_error(
            status_code=400,
            error="invalid_payload",
            message=f"{field} decodes to {len(decoded)} bytes; expected {expected_len}.",
        )
    return decoded


def _b64url_decode_variable(value: str, *, field: str, max_bytes: int) -> bytes:
    """Decode a base64url string with a variable but bounded byte length."""
    padding = (-len(value)) % 4
    try:
        decoded = base64.urlsafe_b64decode(value + ("=" * padding))
    except (ValueError, TypeError) as exc:
        raise http_error(
            status_code=400,
            error="invalid_payload",
            message=f"{field} is not a valid base64url string.",
        ) from exc
    if len(decoded) > max_bytes:
        raise http_error(
            status_code=400,
            error="invalid_payload",
            message=f"{field} exceeds {max_bytes} bytes after decoding.",
        )
    return decoded


def _require_user_store(store: UserStore | None) -> UserStore:
    """Refuse user-tier routes when the user tier is disabled.

    A deployment without SQLCipher or without the pepper has a ``None``
    on ``app.state.user_store``; surface a 503 so the SPA shows a
    coherent "feature not available" message rather than a 500 trace.
    """
    if store is None:
        raise http_error(
            status_code=503,
            error="user_tier_disabled",
            message=("User-tier authentication is not available on this deployment."),
            hint=(
                "Configure SQLCipher (DB_ENCRYPTION_KEY) and the auth "
                "pepper (API_AUTH_PEPPER) and restart the API."
            ),
        )
    return store


def _require_pepper() -> str:
    """Resolve the pepper or refuse with the same 503 shape."""
    pepper = get_settings().api.auth_pepper
    if not pepper:
        raise http_error(
            status_code=503,
            error="user_tier_disabled",
            message="User-tier authentication is not available on this deployment.",
            hint="Configure API_AUTH_PEPPER (or API_AUTH_PEPPER_FILE) and restart the API.",
        )
    return pepper


# Session-cookie helpers mirror ``api/routes/session.py``.  The constants
# are duplicated rather than imported so a future refactor that splits
# session lifecycle from authentication cannot accidentally diverge the
# attributes between the two seams.


def _mint_session_id() -> str:
    return secrets.token_urlsafe(32)


def _set_session_cookie(response: Response, session_id: str, *, max_age: int) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=max_age,
        path="/",
        httponly=True,
        secure=True,
        samesite="strict",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        secure=True,
        httponly=True,
        samesite="strict",
    )


# ---------------------------------------------------------------------------
# GET /api/auth/login-params
# ---------------------------------------------------------------------------


@router.get(
    "/auth/login-params",
    response_model=LoginParamsResponse,
    tags=["auth"],
    summary="Return salt + KDF params for a username (real or decoy)",
)
async def login_params(
    request: Request,
    response: Response,
    username: str,
    store: UserStore | None = Depends(get_user_store),
) -> LoginParamsResponse:
    """Return the parameters needed to derive ``auth_proof`` client-side.

    For an enrolled user, returns the actual ``salt_M`` + the row's
    recorded ``kdf_algo`` + ``pbkdf2_iterations``. For an unknown user,
    returns a deterministic decoy salt computed from the username
    under the deployment pepper, and the global default KDF params —
    same shape, same timing, same bytes across calls.

    ``Cache-Control: no-store`` so the response is never cached
    upstream (a cached real salt would be a privacy regression).
    """
    user_store = _require_user_store(store)
    pepper = _require_pepper()

    # Username shape check is enforced at the query-string boundary by
    # the same pattern Pydantic uses on POST bodies — we keep it as a
    # defensive inline check because GET path parameters don't share
    # the schema validators.
    if not username or len(username) > 64:
        # Treat shape-invalid usernames the same as unknown — refuse
        # any wire signal that distinguishes them.
        decoy = decoy_salt(username or "_", pepper)
        response.headers["Cache-Control"] = "no-store"
        return LoginParamsResponse(
            salt_m=_b64url_encode(decoy),
            kdf_algo="pbkdf2-sha256",
            pbkdf2_iterations=600_000,
        )

    record = user_store.get_by_username(username)
    response.headers["Cache-Control"] = "no-store"

    if record is None:
        # Decoy path — same response shape; deterministic per username.
        return LoginParamsResponse(
            salt_m=_b64url_encode(decoy_salt(username, pepper)),
            kdf_algo="pbkdf2-sha256",
            pbkdf2_iterations=600_000,
        )

    return LoginParamsResponse(
        salt_m=_b64url_encode(record.salt_m),
        kdf_algo=record.kdf_algo,
        pbkdf2_iterations=record.pbkdf2_iterations,
    )


# ---------------------------------------------------------------------------
# POST /api/auth/login
# ---------------------------------------------------------------------------


def _enforce_per_username_gate(
    request: Request,
    window: Any,
    username: str,
) -> None:
    """Per-username sliding window — companion to the per-IP middleware bucket.

    When the window is ``None`` (operator disabled), the check is a
    no-op. Rejections share the same 429 envelope as the per-IP path,
    so a wire observer cannot distinguish the rate-limit dimension.
    """
    if window is None:
        return
    allowed, retry_after = window.is_allowed(username)
    if allowed:
        return
    raise http_error(
        status_code=429,
        error="rate_limited",
        message=f"Rate limit exceeded for login. Retry in {retry_after}s.",
        details={"category": "login", "limit_per_minute": window.limit},
        hint=f"Maximum {window.limit} login requests per minute per username.",
        headers={"Retry-After": str(retry_after)},
    )


@router.post(
    "/auth/login",
    response_model=LoginResponse,
    tags=["auth"],
    summary="Validate auth_proof and mint a session",
)
async def login(
    request: Request,
    response: Response,
    body: LoginRequest,
    store: UserStore | None = Depends(get_user_store),
    session_store: CredentialStore = Depends(get_session_store),
    edgar_store: InMemorySessionEdgarIdentityStore = Depends(get_edgar_identity_store),
    username_window: Any = Depends(get_login_username_window),
) -> LoginResponse:
    """Validate the supplied ``auth_proof`` against the stored ``auth_hash``.

    Wire contract on failure: every error mode (unknown user, wrong
    proof, locked account) returns the same opaque 401
    ``login_refused`` envelope. The audit log carries enough detail to
    distinguish them; the response does not.
    """
    user_store = _require_user_store(store)
    _require_pepper()  # Surface 503 before doing work if pepper missing.

    _enforce_per_username_gate(request, username_window, body.username)

    auth_proof = _b64url_decode_fixed(body.auth_proof, _AUTH_PROOF_BYTES, field="auth_proof")

    try:
        payload = user_store.verify_login(body.username, auth_proof)
    except AuthError:
        audit_log(
            "login_refused",
            detail=(
                f"username_tail={mask_secret(body.username)} "
                f"client_ip={request.client.host if request.client else 'unknown'}"
            ),
        )
        raise http_error(
            status_code=401,
            error="login_refused",
            message="Login refused.",
        ) from None
    except DatabaseError as exc:
        raise http_error(
            status_code=500,
            error="database_error",
            message="Database error during login.",
        ) from exc

    # Rotate any prior session: clear stored credentials AND the EDGAR
    # identity under the old ``session_id``, then mint a fresh one.
    # Mirrors the established ``POST /api/session`` rotation contract —
    # the two stores are keyed identically and MUST be cleared in
    # lockstep, otherwise a rotated cookie leaves the prior session's
    # ``(name, email)`` PII resident until its independent TTL sweep.
    # The delete keys off ``prior``, never the freshly minted id.
    prior = extract_session_id(request)
    if prior is not None:
        session_store.clear(prior)
        edgar_store.delete(prior)

    session_id = _mint_session_id()
    ttl_seconds = get_settings().api.session_ttl_seconds
    _set_session_cookie(response, session_id, max_age=ttl_seconds)

    # Bind ``session_id → user_id`` so authenticated follow-up routes
    # (password change, vault update) can resolve the user without a
    # second auth round-trip. The index is a process-local dict; the
    # next mint that rotates the cookie evicts the prior entry via the
    # ``prior`` lookup above.
    index = getattr(request.app.state, "session_user_index", None)
    if index is not None:
        if prior is not None:
            index.pop(prior, None)
        index[session_id] = payload.user_id

    audit_log(
        "login_success_wire",
        detail=(f"user_id={payload.user_id} session_id_tail={mask_secret(session_id)}"),
    )

    return LoginResponse(
        user_id=payload.user_id,
        username=body.username,
        ciphertext_vault=_b64url_encode(payload.ciphertext_vault),
        vault_iv=_b64url_encode(payload.vault_iv),
    )


# ---------------------------------------------------------------------------
# POST /api/auth/enrol
# ---------------------------------------------------------------------------


@router.post(
    "/auth/enrol",
    response_model=EnrolmentCompleteResponse,
    tags=["auth"],
    summary="Complete enrolment with a signed one-time token",
    status_code=201,
)
async def complete_enrolment(
    request: Request,
    body: EnrolmentCompleteRequest,
    store: UserStore | None = Depends(get_user_store),
) -> EnrolmentCompleteResponse:
    """Verify the enrolment token and create the user row.

    The token binds a username + expiry + nonce under the deployment
    pepper. On verify, the route creates the row carrying ``salt_M``,
    ``auth_hash`` (HMAC over the supplied ``auth_proof``), and the
    initial empty ciphertext vault. The nonce is stored alongside so
    a replayed token after the first successful login surfaces as
    ``409 enrolment_already_completed``.
    """
    user_store = _require_user_store(store)
    pepper = _require_pepper()

    try:
        token_payload = verify_enrolment_token(body.token, pepper)
    except EnrolmentTokenError as exc:
        audit_log(
            "enrolment_token_invalid",
            detail=f"reason_class={type(exc).__name__}",
        )
        raise http_error(
            status_code=401,
            error="enrolment_token_invalid",
            message="Enrolment token is invalid or expired.",
        ) from None
    except ConfigurationError as exc:
        raise http_error(
            status_code=503,
            error="user_tier_disabled",
            message="Server pepper is not configured.",
        ) from exc

    # An enrolled-but-not-completed row may already exist for this
    # username (e.g. the admin minted the token but the user closed the
    # browser).  We refuse to overwrite an existing row — the admin
    # must DELETE the user first.
    if user_store.get_by_username(token_payload.username) is not None:
        raise http_error(
            status_code=409,
            error="enrolment_already_completed",
            message="A user with this username has already enrolled.",
            hint=(
                "Ask the operator to DELETE /api/admin/users/{id} first if "
                "the enrolment must be re-issued."
            ),
        )

    salt_m = _b64url_decode_fixed(body.salt_m, SALT_BYTES, field="salt_m")
    auth_proof = _b64url_decode_fixed(body.auth_proof, _AUTH_PROOF_BYTES, field="auth_proof")
    iv = _b64url_decode_fixed(body.vault_iv, _VAULT_IV_BYTES, field="vault_iv")
    ciphertext = _b64url_decode_variable(
        body.ciphertext_vault, field="ciphertext_vault", max_bytes=48 * 1024
    )

    try:
        user_id = user_store.create_user(
            username=token_payload.username,
            salt_m=salt_m,
            auth_proof=auth_proof,
            ciphertext_vault=ciphertext,
            vault_iv=iv,
            kdf_algo=body.kdf_algo,
            pbkdf2_iterations=body.pbkdf2_iterations,
            enrolment_nonce=token_payload.nonce,
        )
    except DatabaseError as exc:
        raise http_error(
            status_code=500,
            error="database_error",
            message="Database error during enrolment.",
        ) from exc

    return EnrolmentCompleteResponse(
        enrolled=True,
        user_id=user_id,
        username=token_payload.username,
    )


# ---------------------------------------------------------------------------
# POST /api/auth/password
# ---------------------------------------------------------------------------


def _resolve_active_user(request: Request, user_store: UserStore) -> tuple[int, str]:
    """Return ``(user_id, username)`` for the active session.

    The route leans on the same ``session_id`` cookie as the existing
    session-lifecycle routes; the mapping from cookie → user_id lives
    in ``app.state.session_user_index``. Until the session store grows
    a typed "logged-in user" field we look up the user by reading that
    dict. If the resolution fails, we surface a hard 401.
    """
    session_id = extract_session_id(request)
    if session_id is None:
        raise http_error(
            status_code=401,
            error="session_required",
            message="No active session.",
        )
    index = getattr(request.app.state, "session_user_index", None)
    if index is None or session_id not in index:
        raise http_error(
            status_code=401,
            error="session_required",
            message="No active session.",
        )
    user_id = index[session_id]
    record = user_store.get_by_id(user_id)
    if record is None:
        # The row was deleted out from under us.  Drop the stale
        # session entry and force re-login.
        index.pop(session_id, None)
        raise http_error(
            status_code=401,
            error="session_required",
            message="No active session.",
        )
    return user_id, record.username


@router.post(
    "/auth/password",
    response_model=PasswordChangeResponse,
    tags=["auth"],
    summary="Atomic password change",
)
async def change_password(
    request: Request,
    body: PasswordChangeRequest,
    store: UserStore | None = Depends(get_user_store),
) -> PasswordChangeResponse:
    """Validate ``auth_proof_old``, then atomically rotate salt + hash + vault."""
    user_store = _require_user_store(store)
    _require_pepper()

    user_id, username = _resolve_active_user(request, user_store)

    proof_old = _b64url_decode_fixed(body.auth_proof_old, _AUTH_PROOF_BYTES, field="auth_proof_old")
    proof_new = _b64url_decode_fixed(body.auth_proof_new, _AUTH_PROOF_BYTES, field="auth_proof_new")
    salt_m = _b64url_decode_fixed(body.salt_m, SALT_BYTES, field="salt_m")
    iv = _b64url_decode_fixed(body.vault_iv, _VAULT_IV_BYTES, field="vault_iv")
    ciphertext = _b64url_decode_variable(
        body.ciphertext_vault, field="ciphertext_vault", max_bytes=48 * 1024
    )

    # Validate the old proof by running it through verify_login on the
    # authenticated username — reuses the same lockout state machine
    # as the login surface so a password-change brute force is gated
    # identically.
    try:
        user_store.verify_login(username, proof_old)
    except AuthError as exc:
        raise http_error(
            status_code=401,
            error="password_change_refused",
            message="Old password did not validate.",
        ) from exc

    try:
        rotated = user_store.update_password(
            user_id,
            salt_m=salt_m,
            auth_proof=proof_new,
            ciphertext_vault=ciphertext,
            vault_iv=iv,
            kdf_algo=body.kdf_algo,
            pbkdf2_iterations=body.pbkdf2_iterations,
        )
    except DatabaseError as exc:
        raise http_error(
            status_code=500,
            error="database_error",
            message="Database error during password change.",
        ) from exc

    return PasswordChangeResponse(rotated=rotated)


# ---------------------------------------------------------------------------
# DELETE /api/auth/session
# ---------------------------------------------------------------------------


@router.delete(
    "/auth/session",
    tags=["auth"],
    summary="Revoke the active session (sign-out)",
)
async def sign_out(
    request: Request,
    response: Response,
    session_store: CredentialStore = Depends(get_session_store),
    edgar_store: InMemorySessionEdgarIdentityStore = Depends(get_edgar_identity_store),
) -> dict[str, bool]:
    """Sign-out: revoke the session in lockstep with the cookie.

    Aliases ``POST /api/session/logout`` behaviour but lives under the
    user-tier ``auth/`` prefix so the SPA can call a single endpoint.
    Idempotent — a request with no cookie still emits an expired cookie
    and returns ``{cleared: false}``.

    Credentials and EDGAR identity are dropped in lockstep, as on the
    legacy seam.  The response body deliberately stays ``{cleared:
    bool}``: ``session.py``'s logout carries a second
    ``cleared_edgar_identity`` field, but widening this one would be a
    schema change on a route the SPA already consumes, and the flag
    would carry no information the caller can act on.
    """
    session_id = extract_session_id(request)
    cleared = False
    if session_id is not None:
        session_store.clear(session_id)
        edgar_store.delete(session_id)
        index = getattr(request.app.state, "session_user_index", None)
        if index is not None:
            cleared = index.pop(session_id, None) is not None
    _clear_session_cookie(response)
    return {"cleared": cleared}


# ---------------------------------------------------------------------------
# POST /api/auth/vault — re-upload an updated ciphertext blob
# ---------------------------------------------------------------------------


@router.post(
    "/auth/vault",
    response_model=VaultUpdateResponse,
    tags=["auth"],
    summary="Re-upload an updated vault ciphertext blob",
)
async def update_vault(
    request: Request,
    body: VaultUpdateRequest,
    store: UserStore | None = Depends(get_user_store),
) -> VaultUpdateResponse:
    """Replace the ciphertext + IV under the active session's user_id.

    The route is the seam for the provider-key write path AND the
    EDGAR-identity update path: both flow through "decrypt vault →
    mutate → re-encrypt → upload". The IV is fresh per call (load-
    bearing for AES-GCM security — IV reuse against the same key
    breaks confidentiality and integrity).
    """
    user_store = _require_user_store(store)
    user_id, _ = _resolve_active_user(request, user_store)

    iv = _b64url_decode_fixed(body.vault_iv, _VAULT_IV_BYTES, field="vault_iv")
    ciphertext = _b64url_decode_variable(
        body.ciphertext_vault, field="ciphertext_vault", max_bytes=48 * 1024
    )

    try:
        updated = user_store.update_vault(
            user_id,
            ciphertext_vault=ciphertext,
            vault_iv=iv,
        )
    except DatabaseError as exc:
        raise http_error(
            status_code=500,
            error="database_error",
            message="Database error during vault update.",
        ) from exc

    return VaultUpdateResponse(updated=updated)


# ``datetime`` import is intentionally retained at module top for future
# audit-log helpers; silence ruff's "unused" lint locally if it ever
# fires.  (Used implicitly via core.logging.audit_log timestamp
# discipline; no runtime import-loop.)
_ = datetime
_ = UTC
