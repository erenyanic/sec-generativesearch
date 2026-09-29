"""Integration tests for API key access control."""

from __future__ import annotations

import ast
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from sec_generative_search.api.dependencies import (
    admin_route_dependencies,
    verify_admin_key,
    verify_api_key,
)
from sec_generative_search.api.errors import install_error_handlers
from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.logging import LOGGER_NAME

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_app_with(dependencies: list, *, env: dict[str, str] | None = None) -> FastAPI:
    """Build a hermetic test app exposing a single sentinel route."""
    if env is not None:
        for key, value in env.items():
            os.environ[key] = value
    reload_settings()

    app = FastAPI()
    install_error_handlers(app)
    # Stand-in singletons for shared app state.
    app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
    app.state.encrypted_credential_store = None

    router = APIRouter()

    @router.get("/sentinel", dependencies=dependencies)
    async def sentinel() -> dict[str, bool]:
        return {"ok": True}

    app.include_router(router)
    return app


@pytest.fixture
def env_clear() -> Iterator[None]:
    """Clear API_KEY / API_ADMIN_KEY between tests so each starts clean."""
    for key in ("API_KEY", "API_ADMIN_KEY"):
        os.environ.pop(key, None)
    reload_settings()
    yield
    for key in ("API_KEY", "API_ADMIN_KEY"):
        os.environ.pop(key, None)
    reload_settings()


@pytest.mark.security
class TestVerifyApiKey:
    def test_passes_when_unconfigured_scenario_a(self, env_clear: None) -> None:
        app = _build_app_with([Depends(verify_api_key)])
        client = TestClient(app)
        response = client.get("/sentinel")
        assert response.status_code == 200
        assert response.json() == {"ok": True}

    def test_rejects_missing_header(self, env_clear: None) -> None:
        app = _build_app_with(
            [Depends(verify_api_key)],
            env={"API_KEY": "shared-team-key"},  # pragma: allowlist secret
        )
        client = TestClient(app)
        response = client.get("/sentinel")
        assert response.status_code == 401
        body = response.json()
        assert body["error"] == "unauthorised"
        # Hint must steer the caller to the X-API-Key header.
        assert "X-API-Key" in body["hint"]

    def test_rejects_wrong_header(self, env_clear: None) -> None:
        app = _build_app_with(
            [Depends(verify_api_key)],
            env={"API_KEY": "shared-team-key"},  # pragma: allowlist secret
        )
        client = TestClient(app)
        response = client.get("/sentinel", headers={"X-API-Key": "rotated"})
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"

    def test_accepts_correct_header(self, env_clear: None) -> None:
        secret = "shared-team-key"  # pragma: allowlist secret
        app = _build_app_with([Depends(verify_api_key)], env={"API_KEY": secret})
        client = TestClient(app)
        response = client.get("/sentinel", headers={"X-API-Key": secret})
        assert response.status_code == 200

    def test_denial_emits_security_audit(
        self,
        env_clear: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        app = _build_app_with(
            [Depends(verify_api_key)],
            env={"API_KEY": "shared-team-key"},  # pragma: allowlist secret
        )
        client = TestClient(app)

        # caplog hooks the root logger, so re-enable propagation for the
        # duration of this test only.
        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                client.get("/sentinel")  # no header → 401
        finally:
            package_logger.propagate = prior_propagate

        audit = [r for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()]
        assert any("api_key_denied" in r.getMessage() for r in audit)
        assert any("/sentinel" in r.getMessage() for r in audit)

    def test_audit_never_logs_raw_key(
        self,
        env_clear: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        app = _build_app_with(
            [Depends(verify_api_key)],
            env={"API_KEY": "real-secret-12345"},  # pragma: allowlist secret
        )
        client = TestClient(app)

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                # Send a wrong key shaped like a real one.
                client.get("/sentinel", headers={"X-API-Key": "wrong-but-distinct-9999"})
        finally:
            package_logger.propagate = prior_propagate

        # Neither the configured key NOR the supplied (wrong) key may
        # appear anywhere in the audit-log line.
        for record in caplog.records:
            message = record.getMessage()
            assert "real-secret-12345" not in message
            assert "wrong-but-distinct-9999" not in message


@pytest.mark.security
class TestVerifyAdminKey:
    def test_passes_when_unconfigured_scenario_a(self, env_clear: None) -> None:
        app = _build_app_with([Depends(verify_admin_key)])
        client = TestClient(app)
        response = client.get("/sentinel")
        assert response.status_code == 200

    def test_rejects_missing_header(self, env_clear: None) -> None:
        app = _build_app_with(
            [Depends(verify_admin_key)],
            env={"API_ADMIN_KEY": "secret-admin-key"},  # pragma: allowlist secret
        )
        client = TestClient(app)
        response = client.get("/sentinel")
        assert response.status_code == 403
        body = response.json()
        assert body["error"] == "admin_required"
        assert "X-Admin-Key" in body["hint"]

    def test_rejects_wrong_header(self, env_clear: None) -> None:
        app = _build_app_with(
            [Depends(verify_admin_key)],
            env={"API_ADMIN_KEY": "secret-admin-key"},  # pragma: allowlist secret
        )
        client = TestClient(app)
        response = client.get("/sentinel", headers={"X-Admin-Key": "wrong"})
        assert response.status_code == 403

    def test_accepts_correct_header(self, env_clear: None) -> None:
        secret = "secret-admin-key"  # pragma: allowlist secret
        app = _build_app_with(
            [Depends(verify_admin_key)],
            env={"API_ADMIN_KEY": secret},
        )
        client = TestClient(app)
        response = client.get("/sentinel", headers={"X-Admin-Key": secret})
        assert response.status_code == 200


@pytest.mark.security
class TestAdminRouteDependencies:
    """The canonical helper enforces both tiers in the documented order."""

    def test_returns_two_dependencies_in_order(self) -> None:
        deps = admin_route_dependencies()
        assert len(deps) == 2
        assert all(hasattr(d, "dependency") for d in deps)
        assert deps[0].dependency is verify_api_key
        assert deps[1].dependency is verify_admin_key

    def test_returns_fresh_list_each_call(self) -> None:
        first = admin_route_dependencies()
        first.append("polluted")  # type: ignore[arg-type]
        second = admin_route_dependencies()
        assert second != first
        assert len(second) == 2

    def test_scenario_a_unrestricted(self, env_clear: None) -> None:
        app = _build_app_with(admin_route_dependencies())
        client = TestClient(app)
        assert client.get("/sentinel").status_code == 200

    def test_api_key_only_configured_blocks_at_read_tier(
        self,
        env_clear: None,
    ) -> None:
        app = _build_app_with(
            admin_route_dependencies(),
            env={"API_KEY": "shared-team-key"},  # pragma: allowlist secret
        )
        client = TestClient(app)

        response = client.get("/sentinel")
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"

        response = client.get(
            "/sentinel",
            headers={"X-API-Key": "shared-team-key"},  # pragma: allowlist secret
        )
        assert response.status_code == 200

    def test_admin_key_only_configured_still_requires_admin_header(
        self,
        env_clear: None,
    ) -> None:
        app = _build_app_with(
            admin_route_dependencies(),
            env={"API_ADMIN_KEY": "secret-admin-key"},  # pragma: allowlist secret
        )
        client = TestClient(app)
        response = client.get("/sentinel")
        assert response.status_code == 403
        assert response.json()["error"] == "admin_required"

    def test_both_keys_configured_require_both_headers(
        self,
        env_clear: None,
    ) -> None:
        app = _build_app_with(
            admin_route_dependencies(),
            env={
                "API_KEY": "shared-team-key",  # pragma: allowlist secret
                "API_ADMIN_KEY": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        client = TestClient(app)

        assert client.get("/sentinel").status_code == 401

        response = client.get(
            "/sentinel",
            headers={"X-API-Key": "shared-team-key"},  # pragma: allowlist secret
        )
        assert response.status_code == 403
        assert response.json()["error"] == "admin_required"

        response = client.get(
            "/sentinel",
            headers={"X-Admin-Key": "secret-admin-key"},  # pragma: allowlist secret
        )
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"

        response = client.get(
            "/sentinel",
            headers={
                "X-API-Key": "shared-team-key",  # pragma: allowlist secret
                "X-Admin-Key": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        assert response.status_code == 200

    def test_admin_denial_emits_security_audit(
        self,
        env_clear: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        app = _build_app_with(
            admin_route_dependencies(),
            env={
                "API_KEY": "shared-team-key",  # pragma: allowlist secret
                "API_ADMIN_KEY": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        client = TestClient(app)

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                client.get(
                    "/sentinel",
                    headers={
                        "X-API-Key": "shared-team-key",  # pragma: allowlist secret
                        "X-Admin-Key": "wrong",
                    },
                )
        finally:
            package_logger.propagate = prior_propagate

        audit = [r for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()]
        assert any("admin_denied" in r.getMessage() for r in audit)
        assert any("/sentinel" in r.getMessage() for r in audit)


def _get_with_audit(
    caplog: pytest.LogCaptureFixture, app: FastAPI, headers: dict[str, str | bytes]
) -> tuple[Response, list[str]]:
    """``GET /sentinel`` without re-raising server errors (a 500 must show up
    as a status), returning the response and its ``SECURITY_AUDIT`` lines."""
    client = TestClient(app, raise_server_exceptions=False)
    package_logger = logging.getLogger(LOGGER_NAME)
    prior_propagate = package_logger.propagate
    package_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            response = client.get("/sentinel", headers=headers)
    finally:
        package_logger.propagate = prior_propagate
    audit = [r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()]
    return response, audit


@pytest.mark.security
class TestNonAsciiCredentialIsAPlainMismatch:
    """A non-ASCII header value is a wrong key: the tier's own denial status
    and audit line.  A bare ``hmac.compare_digest`` on ``str`` raised
    ``TypeError`` there — an unaudited ``500`` with a traceback."""

    # Starlette decodes header bytes as latin-1, so each byte arrives as "é".
    _NON_ASCII = b"\xe9" * 24

    def test_api_key_tier_is_an_audited_401(
        self, env_clear: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        app = _build_app_with(
            [Depends(verify_api_key)],
            env={"API_KEY": "shared-team-key"},  # pragma: allowlist secret
        )
        response, audit = _get_with_audit(caplog, app, {"X-API-Key": self._NON_ASCII})
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"
        assert any("api_key_denied" in message for message in audit)

    def test_admin_tier_is_an_audited_403(
        self, env_clear: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        app = _build_app_with(
            admin_route_dependencies(),
            env={
                "API_KEY": "shared-team-key",  # pragma: allowlist secret
                "API_ADMIN_KEY": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        response, audit = _get_with_audit(
            caplog,
            app,
            {
                "X-API-Key": "shared-team-key",  # pragma: allowlist secret
                "X-Admin-Key": self._NON_ASCII,
            },
        )
        assert response.status_code == 403
        assert response.json()["error"] == "admin_required"
        assert any("admin_denied" in message for message in audit)

    def test_both_tiers_compare_through_secure_compare(
        self, env_clear: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pins the constant-time primitive: an ``==`` would pass every
        status assertion above while re-opening the timing oracle."""
        import sec_generative_search.api.dependencies as deps

        seen: list[tuple[object, object]] = []
        real = deps.secure_compare

        def _spy(a: object, b: object) -> bool:
            seen.append((a, b))
            return real(a, b)  # type: ignore[arg-type]

        monkeypatch.setattr(deps, "secure_compare", _spy)
        app = _build_app_with(
            admin_route_dependencies(),
            env={
                "API_KEY": "shared-team-key",  # pragma: allowlist secret
                "API_ADMIN_KEY": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        response = TestClient(app).get(
            "/sentinel",
            headers={
                "X-API-Key": "shared-team-key",  # pragma: allowlist secret
                "X-Admin-Key": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        assert response.status_code == 200
        assert ("shared-team-key", "shared-team-key") in seen
        assert ("secret-admin-key", "secret-admin-key") in seen

    def test_only_core_security_calls_compare_digest(self) -> None:
        """Every secret comparison goes through ``secure_compare``, which
        owns the ``str`` / ``bytes`` / ``None`` edges exactly once."""
        src = Path(__file__).resolve().parents[2] / "src" / "sec_generative_search"
        offenders = [
            f"{path.relative_to(src)}:{node.lineno}"
            for path in sorted(src.rglob("*.py"))
            if path.relative_to(src).as_posix() != "core/security.py"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if (isinstance(node, ast.Attribute) and node.attr == "compare_digest")
            or (
                isinstance(node, ast.ImportFrom)
                and any(alias.name == "compare_digest" for alias in node.names)
            )
        ]
        assert offenders == []
