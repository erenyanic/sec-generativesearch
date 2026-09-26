"""Security tests for the admin user-management surface.

Coverage:

- Two-tier guard: every route refuses without both ``X-API-Key`` AND
  ``X-Admin-Key``.
- ``POST /api/admin/users`` mints a single-use enrolment token bound
  to the requested username; refuses ``409 username_exists`` for an
  already-enrolled user.
- ``DELETE /api/admin/users/{user_id}`` wipes the vault.
- ``POST /api/admin/users/{user_id}/unlock`` clears the lockout state.
- ``503 user_tier_disabled`` envelope when the user store is missing.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.edgar_identity import InMemorySessionEdgarIdentityStore
from sec_generative_search.core.user_auth import (
    SALT_BYTES,
    verify_enrolment_token,
)
from sec_generative_search.database.metadata import MetadataRegistry
from sec_generative_search.database.users import UserStore

_PEPPER = "pepper-not-a-secret"
_API_KEY = "test-api-key-not-a-secret"  # pragma: allowlist secret
_ADMIN_KEY = "test-admin-key-not-a-secret"  # pragma: allowlist secret


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


@pytest.fixture
def admin_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chroma").mkdir()
    for key in list(os.environ.keys()):
        if key.startswith(("API_", "DB_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("API_KEY", _API_KEY)
    monkeypatch.setenv("API_ADMIN_KEY", _ADMIN_KEY)
    monkeypatch.setenv("API_AUTH_PEPPER", _PEPPER)
    monkeypatch.setenv("DB_ENCRYPTION_KEY", "test-encryption-key-not-a-secret")
    monkeypatch.setenv("DB_CHROMA_PATH", "./chroma")
    monkeypatch.setenv("DB_METADATA_DB_PATH", "./meta.sqlite")
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def admin_app(admin_env: None):
    from sec_generative_search.api.app import create_app

    app = create_app()
    registry = MetadataRegistry()
    registry._encrypted = True
    user_store = UserStore(registry)
    app.state.registry = registry
    app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=3600)
    app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=3600)
    app.state.encrypted_credential_store = None
    app.state.user_store = user_store
    app.state.login_username_window = None
    yield app
    registry.close()


@pytest.fixture
def admin_client(admin_app) -> Iterator[TestClient]:
    yield TestClient(admin_app, base_url="https://testserver")


def _admin_headers() -> dict[str, str]:
    return {"X-API-Key": _API_KEY, "X-Admin-Key": _ADMIN_KEY}


def _seed_user(admin_app, *, username: str = "alice") -> int:
    store: UserStore = admin_app.state.user_store
    return store.create_user(
        username=username,
        salt_m=b"s" * SALT_BYTES,
        auth_proof=b"p" * 32,
        ciphertext_vault=b"ct",
        vault_iv=b"i" * 12,
        kdf_algo="pbkdf2-sha256",
        pbkdf2_iterations=600_000,
    )


# ---------------------------------------------------------------------------
# Two-tier auth guard
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestAdminTierGuard:
    def test_create_user_without_admin_key_rejected(self, admin_client: TestClient) -> None:
        r = admin_client.post(
            "/api/admin/users",
            json={"username": "alice"},
            headers={"X-API-Key": _API_KEY},
        )
        # Either 401 (no admin key) or 403 (admin key validation). Both
        # are valid refusals; the wire MUST reject without both keys.
        assert r.status_code in {401, 403}

    def test_create_user_without_any_key_rejected(self, admin_client: TestClient) -> None:
        r = admin_client.post(
            "/api/admin/users",
            json={"username": "alice"},
        )
        assert r.status_code == 401

    def test_create_user_with_only_admin_key_returns_401(self, admin_client: TestClient) -> None:
        """The two-tier order matters: api-key-missing should land BEFORE
        admin-key-validation so a leaked admin key alone cannot probe
        the destructive surface for shape information."""
        r = admin_client.post(
            "/api/admin/users",
            json={"username": "alice"},
            headers={"X-Admin-Key": _ADMIN_KEY},
        )
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Enrolment token mint
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestCreateUserRoute:
    def test_happy_path_mints_token(self, admin_client: TestClient) -> None:
        r = admin_client.post(
            "/api/admin/users",
            json={"username": "alice"},
            headers=_admin_headers(),
        )
        assert r.status_code == 201
        body = r.json()
        assert body["username"] == "alice"
        # The token must verify under the deployment pepper.
        payload = verify_enrolment_token(body["enrolment_token"], _PEPPER)
        assert payload.username == "alice"
        assert body["expires_at"] == payload.expires_at
        # enrol_url is a path only — no scheme or host.
        assert body["enrol_url"].startswith("/enrol?token=")

    def test_existing_username_returns_409(self, admin_app, admin_client: TestClient) -> None:
        _seed_user(admin_app)
        r = admin_client.post(
            "/api/admin/users",
            json={"username": "alice"},
            headers=_admin_headers(),
        )
        assert r.status_code == 409
        assert r.json()["error"] == "username_exists"


# ---------------------------------------------------------------------------
# Delete + unlock
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestDeleteUserRoute:
    def test_delete_happy_path(self, admin_app, admin_client: TestClient) -> None:
        user_id = _seed_user(admin_app)
        r = admin_client.delete(
            f"/api/admin/users/{user_id}",
            headers=_admin_headers(),
        )
        assert r.status_code == 200
        body = r.json()
        assert body["deleted"] is True
        assert body["user_id"] == user_id

    def test_delete_unknown_returns_404(self, admin_client: TestClient) -> None:
        r = admin_client.delete(
            "/api/admin/users/99999",
            headers=_admin_headers(),
        )
        assert r.status_code == 404
        assert r.json()["error"] == "user_not_found"


@pytest.mark.security
class TestUnlockUserRoute:
    def test_unlock_happy_path(self, admin_app, admin_client: TestClient) -> None:
        user_id = _seed_user(admin_app)
        store: UserStore = admin_app.state.user_store
        # Burn the bucket to lock the account.
        import contextlib

        from sec_generative_search.core.exceptions import AuthError

        for _ in range(10):
            with contextlib.suppress(AuthError):
                store.verify_login("alice", b"wrong-proof-xxxxxxxxxxxxxxxxxxxx")
        record = store.get_by_id(user_id)
        assert record is not None and record.locked_until is not None
        # Admin unlock clears the state.
        r = admin_client.post(
            f"/api/admin/users/{user_id}/unlock",
            headers=_admin_headers(),
        )
        assert r.status_code == 200
        assert r.json()["unlocked"] is True
        record = store.get_by_id(user_id)
        assert record is not None and record.locked_until is None

    def test_unlock_unknown_returns_404(self, admin_client: TestClient) -> None:
        r = admin_client.post(
            "/api/admin/users/99999/unlock",
            headers=_admin_headers(),
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# 503 user-tier-disabled
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestAdminTierDisabled:
    def test_create_user_without_user_store_returns_503(
        self, admin_app, admin_client: TestClient
    ) -> None:
        admin_app.state.user_store = None
        r = admin_client.post(
            "/api/admin/users",
            json={"username": "alice"},
            headers=_admin_headers(),
        )
        assert r.status_code == 503
        assert r.json()["error"] == "user_tier_disabled"
