"""Security tests for the user-tier auth surface.

Coverage:

- ``GET /api/auth/login-params`` returns identical shapes (and the
  same bytes across calls) for real and unknown usernames so the
  decoy envelope is the load-bearing username-enumeration defence.
- ``POST /api/auth/login`` collapses every failure mode (unknown user,
  wrong proof, locked account) into the same opaque ``401`` envelope
  — the wire never distinguishes the cases even though the audit log
  does.
- Per-username sliding window rejects bursts even when per-IP is
  disabled (proves the gate is independent of the middleware bucket).
- Successful login mints a session cookie carrying the HttpOnly /
  Secure / SameSite=Strict attribute trio, mirrors the established
  ``POST /api/session`` contract.
- ``POST /api/auth/enrol`` happy path + token replay + signature
  tampering.
- ``POST /api/auth/vault`` requires an active session and rejects
  malformed IVs / oversize ciphertexts.

The fixtures build a fresh FastAPI app per test, force the registry
into the ``encrypted=True`` state (SQLCipher is not in CI's install
set), and attach a real :class:`UserStore` plus the per-username
sliding window onto ``app.state``.
"""

from __future__ import annotations

import base64
import os
import secrets
from collections.abc import Iterator
from http.cookies import SimpleCookie
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.api.dependencies import SESSION_COOKIE_NAME
from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.edgar_identity import (
    EdgarIdentity,
    InMemorySessionEdgarIdentityStore,
)
from sec_generative_search.core.user_auth import (
    SALT_BYTES,
    mint_enrolment_token,
)
from sec_generative_search.database.metadata import MetadataRegistry
from sec_generative_search.database.users import UserStore

_PEPPER = "pepper-not-a-secret"


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


@pytest.fixture
def auth_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Reset settings + chdir + configure pepper + DB encryption knobs."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chroma").mkdir()
    for key in list(os.environ.keys()):
        if key.startswith(("API_", "DB_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("API_AUTH_PEPPER", _PEPPER)
    monkeypatch.setenv("DB_ENCRYPTION_KEY", "test-encryption-key-not-a-secret")
    monkeypatch.setenv("DB_CHROMA_PATH", "./chroma")
    monkeypatch.setenv("DB_METADATA_DB_PATH", "./meta.sqlite")
    # Relax per-IP login bucket so tests can hit the per-username gate
    # without tripping the IP limiter first.
    monkeypatch.setenv("API_RATE_LIMIT_LOGIN", "0")
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def auth_app(auth_env: None):
    from sec_generative_search.api.app import create_app

    app = create_app()

    # Open a registry against the test DB and force-encrypt (SQLCipher
    # is unavailable in CI; the wire-level encryption is exercised in
    # ``test_metadata``).
    registry = MetadataRegistry()
    registry._encrypted = True

    # Build the real UserStore + per-username window.
    user_store = UserStore(registry)

    from sec_generative_search.api.middleware import _SlidingWindow

    # Window with a tiny limit so a couple of bursts exercise the gate.
    username_window = _SlidingWindow(3)

    app.state.registry = registry
    app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=3600)
    app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=3600)
    app.state.encrypted_credential_store = None
    app.state.user_store = user_store
    app.state.login_username_window = username_window
    app.state.session_user_index = {}

    yield app

    registry.close()


@pytest.fixture
def auth_client(auth_app) -> Iterator[TestClient]:
    yield TestClient(auth_app, base_url="https://testserver")


def _enrol_user(
    auth_app,
    *,
    username: str = "alice",
    auth_proof: bytes = b"a" * 32,
) -> int:
    store: UserStore = auth_app.state.user_store
    return store.create_user(
        username=username,
        salt_m=b"s" * SALT_BYTES,
        auth_proof=auth_proof,
        ciphertext_vault=b"ct-initial",
        vault_iv=b"i" * 12,
        kdf_algo="pbkdf2-sha256",
        pbkdf2_iterations=600_000,
    )


# ---------------------------------------------------------------------------
# GET /api/auth/login-params — decoy salt enumeration defence
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestLoginParamsDecoy:
    def test_known_user_returns_real_salt(self, auth_app, auth_client: TestClient) -> None:
        user_id = _enrol_user(auth_app)  # noqa: F841
        response = auth_client.get("/api/auth/login-params?username=alice")
        assert response.status_code == 200
        body = response.json()
        # The real salt is b"s" * 16, base64url-encoded.
        assert body["salt_m"] == _b64(b"s" * SALT_BYTES)
        assert body["kdf_algo"] == "pbkdf2-sha256"
        assert body["pbkdf2_iterations"] == 600_000

    def test_unknown_user_returns_decoy_salt(self, auth_client: TestClient) -> None:
        response = auth_client.get("/api/auth/login-params?username=mallory")
        assert response.status_code == 200
        body = response.json()
        # Decoy is 16 bytes — base64url 22 chars (no padding).
        assert len(body["salt_m"]) >= 22
        assert body["kdf_algo"] == "pbkdf2-sha256"

    def test_decoy_salt_is_deterministic_across_calls(self, auth_client: TestClient) -> None:
        """Load-bearing for enumeration defence: an attacker who hits the
        endpoint twice with the same unknown username must see the same
        salt. A varying decoy would give 'no such user' away in one
        round-trip."""
        first = auth_client.get("/api/auth/login-params?username=mallory").json()
        second = auth_client.get("/api/auth/login-params?username=mallory").json()
        assert first["salt_m"] == second["salt_m"]

    def test_decoy_distinct_per_username(self, auth_client: TestClient) -> None:
        a = auth_client.get("/api/auth/login-params?username=mallory").json()
        b = auth_client.get("/api/auth/login-params?username=eve").json()
        assert a["salt_m"] != b["salt_m"]

    def test_real_and_decoy_response_shapes_are_identical(
        self, auth_app, auth_client: TestClient
    ) -> None:
        """The unknown and real branches must serialise to the same keys
        — otherwise a wire observer can fingerprint registration state
        even without comparing salt bytes."""
        _enrol_user(auth_app)
        real = auth_client.get("/api/auth/login-params?username=alice").json()
        decoy = auth_client.get("/api/auth/login-params?username=ghost").json()
        assert set(real.keys()) == set(decoy.keys())

    def test_response_carries_no_store_cache_header(self, auth_client: TestClient) -> None:
        """A cached real salt is a privacy regression — the response
        must explicitly opt out of upstream caching."""
        response = auth_client.get("/api/auth/login-params?username=alice")
        assert response.headers["Cache-Control"] == "no-store"


# ---------------------------------------------------------------------------
# POST /api/auth/login — opaque failures + lockout
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestLoginOpaqueFailure:
    def test_login_success_mints_session_cookie(self, auth_app, auth_client: TestClient) -> None:
        _enrol_user(auth_app, auth_proof=b"p" * 32)
        response = auth_client.post(
            "/api/auth/login",
            json={"username": "alice", "auth_proof": _b64(b"p" * 32)},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["username"] == "alice"
        assert body["ciphertext_vault"] == _b64(b"ct-initial")
        # Cookie must carry HttpOnly + Secure + SameSite=Strict.
        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie
        assert "secure" in cookie
        assert "samesite=strict" in cookie

    def test_unknown_user_returns_opaque_401(self, auth_client: TestClient) -> None:
        response = auth_client.post(
            "/api/auth/login",
            json={"username": "ghost", "auth_proof": _b64(b"p" * 32)},
        )
        assert response.status_code == 401
        body = response.json()
        assert body["error"] == "login_refused"
        # The error must NOT distinguish the failure mode.
        assert "unknown" not in body["message"].lower()
        assert "no such" not in body["message"].lower()

    def test_wrong_proof_returns_same_envelope(self, auth_app, auth_client: TestClient) -> None:
        _enrol_user(auth_app, auth_proof=b"p" * 32)
        unknown = auth_client.post(
            "/api/auth/login",
            json={"username": "ghost", "auth_proof": _b64(b"q" * 32)},
        ).json()
        wrong = auth_client.post(
            "/api/auth/login",
            json={"username": "alice", "auth_proof": _b64(b"q" * 32)},
        ).json()
        # The opaque-failure contract: response bodies must match shape +
        # content so the wire never distinguishes the cases.
        assert unknown["error"] == wrong["error"] == "login_refused"
        assert unknown["message"] == wrong["message"]

    def test_response_never_echoes_auth_proof(self, auth_app, auth_client: TestClient) -> None:
        _enrol_user(auth_app, auth_proof=b"p" * 32)
        proof = b"super-secret-proof" + b"x" * 14
        response = auth_client.post(
            "/api/auth/login",
            json={"username": "alice", "auth_proof": _b64(proof)},
        )
        assert "super-secret-proof" not in response.text


@pytest.mark.security
class TestLoginPerUsernameRateLimit:
    def test_burst_per_username_returns_429(self, auth_app, auth_client: TestClient) -> None:
        """Three failed attempts in quick succession fit; the fourth must
        be rate-limited by the per-username sliding window."""
        _enrol_user(auth_app, auth_proof=b"p" * 32)
        # The fixture window is sized at 3 rpm.
        for _ in range(3):
            r = auth_client.post(
                "/api/auth/login",
                json={"username": "alice", "auth_proof": _b64(b"q" * 32)},
            )
            assert r.status_code == 401
        # Fourth must hit the per-username bucket → 429.
        r4 = auth_client.post(
            "/api/auth/login",
            json={"username": "alice", "auth_proof": _b64(b"q" * 32)},
        )
        assert r4.status_code == 429
        assert r4.json()["details"]["category"] == "login"
        assert "Retry-After" in r4.headers

    def test_per_username_bucket_keyed_per_username(
        self, auth_app, auth_client: TestClient
    ) -> None:
        """A different username must NOT inherit Alice's bucket — the
        per-username window must isolate per-key."""
        _enrol_user(auth_app, username="alice", auth_proof=b"p" * 32)
        _enrol_user(auth_app, username="bob", auth_proof=b"p" * 32)
        # Burn Alice's bucket.
        for _ in range(3):
            auth_client.post(
                "/api/auth/login",
                json={"username": "alice", "auth_proof": _b64(b"q" * 32)},
            )
        # Bob still gets the full bucket.
        r = auth_client.post(
            "/api/auth/login",
            json={"username": "bob", "auth_proof": _b64(b"p" * 32)},
        )
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/auth/enrol — happy path + token replay + tamper
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestEnrolmentRoute:
    def test_enrolment_happy_path(self, auth_app, auth_client: TestClient) -> None:
        token = mint_enrolment_token("alice", _PEPPER)
        body = {
            "token": token,
            "salt_m": _b64(b"s" * SALT_BYTES),
            "auth_proof": _b64(b"p" * 32),
            "ciphertext_vault": _b64(b"ct"),
            "vault_iv": _b64(b"i" * 12),
            "kdf_algo": "pbkdf2-sha256",
            "pbkdf2_iterations": 600_000,
        }
        response = auth_client.post("/api/auth/enrol", json=body)
        assert response.status_code == 201
        assert response.json()["enrolled"] is True

    def test_tampered_token_rejected(self, auth_client: TestClient) -> None:
        token = mint_enrolment_token("alice", "different-pepper")
        body = {
            "token": token,
            "salt_m": _b64(b"s" * SALT_BYTES),
            "auth_proof": _b64(b"p" * 32),
            "ciphertext_vault": _b64(b"ct"),
            "vault_iv": _b64(b"i" * 12),
            "kdf_algo": "pbkdf2-sha256",
            "pbkdf2_iterations": 600_000,
        }
        response = auth_client.post("/api/auth/enrol", json=body)
        assert response.status_code == 401
        assert response.json()["error"] == "enrolment_token_invalid"

    def test_replay_after_completion_is_409(self, auth_app, auth_client: TestClient) -> None:
        token = mint_enrolment_token("alice", _PEPPER)
        body = {
            "token": token,
            "salt_m": _b64(b"s" * SALT_BYTES),
            "auth_proof": _b64(b"p" * 32),
            "ciphertext_vault": _b64(b"ct"),
            "vault_iv": _b64(b"i" * 12),
            "kdf_algo": "pbkdf2-sha256",
            "pbkdf2_iterations": 600_000,
        }
        first = auth_client.post("/api/auth/enrol", json=body)
        assert first.status_code == 201
        # Replay the same token (and same body) — must hit the
        # username-exists branch.
        second = auth_client.post("/api/auth/enrol", json=body)
        assert second.status_code == 409
        assert second.json()["error"] == "enrolment_already_completed"

    def test_invalid_base64_payload_rejected(self, auth_client: TestClient) -> None:
        """Pydantic length bounds defend the encoded width; the route
        adds a defence-in-depth check on the *decoded* width to refuse
        an attacker who base64-encodes the wrong number of bytes."""
        token = mint_enrolment_token("alice", _PEPPER)
        body = {
            "token": token,
            "salt_m": _b64(b"s" * 10),  # decodes to 10 bytes — wrong width
            "auth_proof": _b64(b"p" * 32),
            "ciphertext_vault": _b64(b"ct"),
            "vault_iv": _b64(b"i" * 12),
            "kdf_algo": "pbkdf2-sha256",
            "pbkdf2_iterations": 600_000,
        }
        response = auth_client.post("/api/auth/enrol", json=body)
        # Either Pydantic's length pattern catches it (422) or the
        # route's decoder catches it (400). Both are valid refusals;
        # both must reject the payload.
        assert response.status_code in {400, 422}


# ---------------------------------------------------------------------------
# POST /api/auth/vault — authenticated re-upload
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestVaultUpdate:
    def _login(self, auth_app, auth_client: TestClient) -> None:
        _enrol_user(auth_app, auth_proof=b"p" * 32)
        r = auth_client.post(
            "/api/auth/login",
            json={"username": "alice", "auth_proof": _b64(b"p" * 32)},
        )
        assert r.status_code == 200

    def test_unauthenticated_call_rejected(self, auth_client: TestClient) -> None:
        response = auth_client.post(
            "/api/auth/vault",
            json={
                "ciphertext_vault": _b64(b"new-ct"),
                "vault_iv": _b64(b"j" * 12),
            },
        )
        assert response.status_code == 401
        assert response.json()["error"] == "session_required"

    def test_authenticated_round_trip(self, auth_app, auth_client: TestClient) -> None:
        self._login(auth_app, auth_client)
        response = auth_client.post(
            "/api/auth/vault",
            json={
                "ciphertext_vault": _b64(b"updated-ciphertext"),
                "vault_iv": _b64(b"j" * 12),
            },
        )
        assert response.status_code == 200
        assert response.json()["updated"] is True
        # The next login (rotated session) must return the updated blob.
        r = auth_client.post(
            "/api/auth/login",
            json={"username": "alice", "auth_proof": _b64(b"p" * 32)},
        )
        assert r.json()["ciphertext_vault"] == _b64(b"updated-ciphertext")

    def test_wrong_iv_length_rejected(self, auth_app, auth_client: TestClient) -> None:
        self._login(auth_app, auth_client)
        # An 8-byte IV decodes valid base64 but the route enforces
        # exactly 12 bytes (AES-GCM requirement).
        response = auth_client.post(
            "/api/auth/vault",
            json={
                "ciphertext_vault": _b64(b"updated-ciphertext"),
                "vault_iv": _b64(b"\x00" * 8),
            },
        )
        # Pydantic locks the encoded length; if a string passes Pydantic
        # but the byte width is wrong, the decoder catches it.
        assert response.status_code in {400, 422}


# ---------------------------------------------------------------------------
# DELETE /api/auth/session — sign-out clears session_id binding
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSignOut:
    def test_signout_clears_session_index(self, auth_app, auth_client: TestClient) -> None:
        _enrol_user(auth_app, auth_proof=b"p" * 32)
        auth_client.post(
            "/api/auth/login",
            json={"username": "alice", "auth_proof": _b64(b"p" * 32)},
        )
        # The session_user_index must have an entry.
        assert auth_app.state.session_user_index
        # Sign out.
        r = auth_client.delete("/api/auth/session")
        assert r.status_code == 200
        assert r.json()["cleared"] is True
        # The index is now empty.
        assert not auth_app.state.session_user_index

    def test_signout_without_session_is_idempotent(self, auth_client: TestClient) -> None:
        r = auth_client.delete("/api/auth/session")
        assert r.status_code == 200
        assert r.json()["cleared"] is False


# ---------------------------------------------------------------------------
# 503 user-tier-disabled envelope when UserStore is not configured
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestUserTierDisabled:
    def test_login_without_user_store_returns_503(self, auth_app, auth_client: TestClient) -> None:
        auth_app.state.user_store = None
        r = auth_client.post(
            "/api/auth/login",
            json={"username": "alice", "auth_proof": _b64(b"p" * 32)},
        )
        assert r.status_code == 503
        assert r.json()["error"] == "user_tier_disabled"

    def test_enrol_without_user_store_returns_503(self, auth_app, auth_client: TestClient) -> None:
        auth_app.state.user_store = None
        token = mint_enrolment_token("alice", _PEPPER)
        r = auth_client.post(
            "/api/auth/enrol",
            json={
                "token": token,
                "salt_m": _b64(b"s" * SALT_BYTES),
                "auth_proof": _b64(b"p" * 32),
                "ciphertext_vault": _b64(b"ct"),
                "vault_iv": _b64(b"i" * 12),
                "kdf_algo": "pbkdf2-sha256",
                "pbkdf2_iterations": 600_000,
            },
        )
        assert r.status_code == 503
        assert r.json()["error"] == "user_tier_disabled"


# ---------------------------------------------------------------------------
# EDGAR-identity lockstep on the user-tier session seams (L1)
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestEdgarIdentityLockstep:
    """The user-tier router must clear EDGAR identity with the session.

    Documented invariant (AGENT.md §API, DEPLOYMENT.md §4.4): *session
    rotation and logout MUST clear EDGAR identity in lockstep with
    credentials*.  ``api/routes/session.py`` honoured this on both its
    seams; the parallel user-tier router in ``api/routes/auth.py`` did
    not, leaving a ``(name, email)`` Tier-3 PII pair resident in the
    in-memory store until its independent TTL sweep evicted it.

    Every test below seeds the store under a **real**
    ``secrets.token_urlsafe(32)`` value: ``extract_session_id`` shape-
    checks the cookie and returns ``None`` for anything else, so a
    made-up id would make the route a no-op and the assertion vacuous.
    Each test therefore also asserts the identity *is* present before
    the call.
    """

    @staticmethod
    def _minted_session_id(response) -> str:
        """Read the freshly minted id out of the response ``Set-Cookie``.

        Reading the client jar instead would be ambiguous once a test
        has seeded a prior cookie of its own — httpx raises
        ``CookieConflict`` when two cookies share a name across domains.
        """
        jar = SimpleCookie()
        jar.load(response.headers["set-cookie"])
        return jar[SESSION_COOKIE_NAME].value

    @staticmethod
    def _seed_identity(auth_app, session_id: str) -> None:
        store: InMemorySessionEdgarIdentityStore = auth_app.state.edgar_identity_store
        store.set(session_id, EdgarIdentity(name="Ada Lovelace", email="ada@example.com"))

    def test_login_clears_prior_session_edgar_identity(
        self, auth_app, auth_client: TestClient
    ) -> None:
        _enrol_user(auth_app, auth_proof=b"p" * 32)
        prior = secrets.token_urlsafe(32)
        self._seed_identity(auth_app, prior)
        auth_client.cookies.set(SESSION_COOKIE_NAME, prior)

        store = auth_app.state.edgar_identity_store
        # Non-vacuity guard: the identity must really be resident first.
        assert store.get(prior) is not None

        response = auth_client.post(
            "/api/auth/login",
            json={"username": "alice", "auth_proof": _b64(b"p" * 32)},
        )
        assert response.status_code == 200
        assert store.get(prior) is None

    def test_login_does_not_carry_identity_onto_the_new_session(
        self, auth_app, auth_client: TestClient
    ) -> None:
        """The delete must key off ``prior``, never the freshly minted id."""
        _enrol_user(auth_app, auth_proof=b"p" * 32)
        prior = secrets.token_urlsafe(32)
        self._seed_identity(auth_app, prior)
        auth_client.cookies.set(SESSION_COOKIE_NAME, prior)

        response = auth_client.post(
            "/api/auth/login",
            json={"username": "alice", "auth_proof": _b64(b"p" * 32)},
        )
        assert response.status_code == 200
        minted = self._minted_session_id(response)
        assert minted != prior
        assert auth_app.state.edgar_identity_store.get(minted) is None

    def test_signout_clears_session_edgar_identity(self, auth_app, auth_client: TestClient) -> None:
        _enrol_user(auth_app, auth_proof=b"p" * 32)
        login = auth_client.post(
            "/api/auth/login",
            json={"username": "alice", "auth_proof": _b64(b"p" * 32)},
        )
        session_id = self._minted_session_id(login)
        self._seed_identity(auth_app, session_id)

        store = auth_app.state.edgar_identity_store
        assert store.get(session_id) is not None

        response = auth_client.delete("/api/auth/session")
        assert response.status_code == 200
        assert store.get(session_id) is None

    def test_signout_response_shape_is_unchanged(self, auth_app, auth_client: TestClient) -> None:
        """Rule **L**: the sign-out body stays ``{cleared: bool}``.

        ``session.py::logout_session`` deliberately carries a second
        ``cleared_edgar_identity`` field; ``auth.py::sign_out`` does
        not.  Mirroring it here would be a schema widening, so the
        asymmetry is pinned rather than left to drift.
        """
        _enrol_user(auth_app, auth_proof=b"p" * 32)
        login = auth_client.post(
            "/api/auth/login",
            json={"username": "alice", "auth_proof": _b64(b"p" * 32)},
        )
        self._seed_identity(auth_app, self._minted_session_id(login))

        body = auth_client.delete("/api/auth/session").json()
        assert set(body) == {"cleared"}
