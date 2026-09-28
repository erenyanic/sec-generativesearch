"""Security tests for the dependency providers and resolver factory.

The suite covers API/admin authentication, session cookie handling,
provider-key header parsing, and resolver-chain precedence.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi import FastAPI, Request

from sec_generative_search.api.dependencies import (
    ADMIN_USER_ID,
    SESSION_COOKIE_NAME,
    is_admin_request,
    parse_provider_key_headers,
    request_scoped_resolver,
)
from sec_generative_search.core.credentials import InMemorySessionCredentialStore


def _request(*, headers: dict[str, str] | None = None, cookies: dict[str, str] | None = None):
    """Build a minimal FastAPI Request with the given headers + cookies."""
    raw_headers: list[tuple[bytes, bytes]] = []
    for name, value in (headers or {}).items():
        raw_headers.append((name.lower().encode(), value.encode()))
    if cookies:
        cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
        raw_headers.append((b"cookie", cookie_str.encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "raw_path": b"/",
        "query_string": b"",
        "headers": raw_headers,
    }
    return Request(scope)


@pytest.mark.security
class TestIsAdminRequest:
    def test_no_admin_key_configured_returns_true(self, monkeypatch) -> None:
        monkeypatch.delenv("API_ADMIN_KEY", raising=False)
        from sec_generative_search.config.settings import reload_settings

        reload_settings()
        request = _request()
        assert is_admin_request(request) is True

    def test_admin_key_required_and_provided(self, monkeypatch) -> None:
        admin_secret = "super-admin-key-1234"  # pragma: allowlist secret
        monkeypatch.setenv("API_ADMIN_KEY", admin_secret)
        from sec_generative_search.config.settings import reload_settings

        reload_settings()
        request = _request(headers={"X-Admin-Key": admin_secret})
        assert is_admin_request(request) is True

    def test_admin_key_required_and_wrong(self, monkeypatch) -> None:
        monkeypatch.setenv("API_ADMIN_KEY", "real-admin-key")  # pragma: allowlist secret
        from sec_generative_search.config.settings import reload_settings

        reload_settings()
        request = _request(headers={"X-Admin-Key": "wrong"})
        assert is_admin_request(request) is False

    def test_admin_key_required_and_missing(self, monkeypatch) -> None:
        monkeypatch.setenv("API_ADMIN_KEY", "real-admin-key")  # pragma: allowlist secret
        from sec_generative_search.config.settings import reload_settings

        reload_settings()
        request = _request()
        assert is_admin_request(request) is False


@pytest.mark.security
class TestParseProviderKeyHeaders:
    def test_extracts_lowercase_provider(self) -> None:
        headers = {"X-Provider-Key-openai": "sk-test"}
        assert parse_provider_key_headers(headers) == {"openai": "sk-test"}

    def test_case_insensitive_header_name(self) -> None:
        # HTTP header names are case-insensitive.
        headers = {"x-PROVIDER-key-openai": "sk-test"}
        assert parse_provider_key_headers(headers) == {"openai": "sk-test"}

    def test_empty_value_dropped(self) -> None:
        headers = {"X-Provider-Key-openai": ""}
        assert parse_provider_key_headers(headers) == {}

    def test_unrelated_headers_ignored(self) -> None:
        headers = {
            "X-API-Key": "secret",
            "Authorization": "Bearer x",
            "Cookie": "sec_rag_session=abc",
        }
        assert parse_provider_key_headers(headers) == {}

    def test_dotted_or_special_provider_rejected(self) -> None:
        headers = {
            "X-Provider-Key-../etc/passwd": "x",
            "X-Provider-Key-foo bar": "x",
            "X-Provider-Key-foo.bar": "x",
        }
        assert parse_provider_key_headers(headers) == {}


@pytest.mark.security
class TestRequestScopedResolverChain:
    def _attach_state(
        self,
        request: Request,
        *,
        session_store=None,
        encrypted_store=None,
        admin_key: str | None = None,
    ) -> None:
        """Wire a tiny stand-in app onto the request for the resolver."""
        app = FastAPI()
        app.state.session_store = session_store
        app.state.encrypted_credential_store = encrypted_store
        # The resolver reads ``request.app.state``; Starlette's Request
        # discovers ``app`` via ``scope['app']``.
        request.scope["app"] = app
        if admin_key is not None:
            import os

            os.environ["API_ADMIN_KEY"] = admin_key
            from sec_generative_search.config.settings import reload_settings

            reload_settings()

    def test_chain_falls_through_to_admin_env(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-admin-env-1234")  # pragma: allowlist secret
        request = _request()
        self._attach_state(request)
        resolver = request_scoped_resolver(request)
        assert resolver("openai") == "sk-admin-env-1234"  # pragma: allowlist secret
        # Unknown provider with no env var → None.
        assert resolver("nonexistent-provider") is None

    def test_session_store_shadows_admin_env(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-admin-env")  # pragma: allowlist secret
        store = InMemorySessionCredentialStore(ttl_seconds=300)
        sid = "A" * 43  # valid shape per ``extract_session_id``
        store.set(sid, "openai", "sk-session-key-from-store")  # pragma: allowlist secret

        request = _request(cookies={SESSION_COOKIE_NAME: sid})
        self._attach_state(request, session_store=store)

        resolver = request_scoped_resolver(request)
        # Session tier wins over admin-env tier.
        assert resolver("openai") == "sk-session-key-from-store"  # pragma: allowlist secret

    def test_forged_cookie_falls_through(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-admin")  # pragma: allowlist secret
        store = InMemorySessionCredentialStore(ttl_seconds=300)
        # A forged but syntactically invalid cookie — must NOT key the store.
        request = _request(cookies={SESSION_COOKIE_NAME: "tooshort"})
        self._attach_state(request, session_store=store)

        resolver = request_scoped_resolver(request)
        # Falls through to admin env.
        assert resolver("openai") == "sk-admin"  # pragma: allowlist secret

    def test_header_tier_shadows_session_and_admin(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-admin-env")  # pragma: allowlist secret
        store = InMemorySessionCredentialStore(ttl_seconds=300)
        sid = "B" * 43
        store.set(sid, "openai", "sk-session")  # pragma: allowlist secret

        request = _request(
            headers={"X-Provider-Key-openai": "sk-from-header"},
            cookies={SESSION_COOKIE_NAME: sid},
        )
        self._attach_state(request, session_store=store)

        resolver = request_scoped_resolver(request)
        # Header tier wins over session and admin-env.
        assert resolver("openai") == "sk-from-header"  # pragma: allowlist secret

    def test_header_tier_only_used_for_named_provider(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-admin-env-openai")  # pragma: allowlist secret
        anthropic_env = "sk-admin-env-anthropic"  # pragma: allowlist secret
        monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_env)
        request = _request(headers={"X-Provider-Key-openai": "sk-header-openai"})
        self._attach_state(request)
        resolver = request_scoped_resolver(request)
        # Named tier — header wins for openai, falls through for anthropic.
        assert resolver("openai") == "sk-header-openai"  # pragma: allowlist secret
        assert resolver("anthropic") == anthropic_env

    def test_encrypted_tier_only_for_admin(self, monkeypatch) -> None:
        # Stand-in store: ``CredentialStore`` Protocol is structural.
        class StubEncrypted:
            def get(self, key_id: str, provider: str) -> str | None:
                return "sk-encrypted-admin" if key_id == ADMIN_USER_ID else None

            def set(self, *a, **kw) -> None: ...
            def delete(self, *a, **kw) -> bool:
                return False

            def list_providers(self, *a, **kw) -> set:
                return set()

            def clear(self, *a, **kw) -> int:
                return 0

        admin_secret = "real-admin-key"  # pragma: allowlist secret
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Non-admin request — encrypted tier MUST be skipped.
        non_admin = _request()
        self._attach_state(
            non_admin,
            encrypted_store=StubEncrypted(),
            admin_key=admin_secret,
        )
        resolver = request_scoped_resolver(non_admin)
        assert resolver("openai") is None  # encrypted tier was skipped

        # Admin request — encrypted tier MUST be consulted.
        admin = _request(headers={"X-Admin-Key": admin_secret})
        self._attach_state(
            admin,
            encrypted_store=StubEncrypted(),
            admin_key=admin_secret,
        )
        resolver = request_scoped_resolver(admin)
        assert resolver("openai") == "sk-encrypted-admin"  # pragma: allowlist secret


# ---------------------------------------------------------------------------
# F33 — shared route helpers have exactly one home
# ---------------------------------------------------------------------------


class TestClientIp:
    def test_missing_peer_reads_unknown(self) -> None:
        from types import SimpleNamespace

        from sec_generative_search.api.dependencies import client_ip

        assert client_ip(SimpleNamespace(client=None)) == "unknown"  # type: ignore[arg-type]
        peer = SimpleNamespace(client=SimpleNamespace(host="203.0.113.7"))
        assert client_ip(peer) == "203.0.113.7"  # type: ignore[arg-type]


@pytest.mark.security
class TestSharedRouteHelpersAreNotReForked:
    """Five copies of the audit-line client IP (plus three inline), two of
    the session-cookie helpers and nine ``500 database_error`` envelopes
    (four different wordings) were consolidated.  A re-fork is how cookie attributes or the
    audit key silently diverge between routers — keep one copy each."""

    @staticmethod
    def _api_modules() -> list[tuple[str, ast.AST]]:
        import sec_generative_search.api as api_pkg

        root = Path(api_pkg.__file__).parent
        return [
            (str(path.relative_to(root)), ast.parse(path.read_text(encoding="utf-8")))
            for path in sorted(root.rglob("*.py"))
        ]

    def test_session_cookie_is_written_only_by_the_shared_helpers(self) -> None:
        offenders = [
            f"{name}:{node.lineno}"
            for name, tree in self._api_modules()
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr in {"set_cookie", "delete_cookie"}
            and name != "dependencies.py"
        ]
        assert offenders == []

    def test_peer_address_is_read_only_by_client_ip(self) -> None:
        offenders = [
            f"{name}:{node.lineno}"
            for name, tree in self._api_modules()
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "host"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "client"
            and name != "dependencies.py"
        ]
        assert offenders == []

    def test_database_error_envelope_is_built_only_in_errors(self) -> None:
        offenders = [
            f"{name}:{node.lineno}"
            for name, tree in self._api_modules()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and node.value == "database_error"
            and name != "errors.py"
        ]
        assert offenders == []
