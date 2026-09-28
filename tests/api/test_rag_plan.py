"""Tests for the RAG query-understanding route.

Strategy
--------

The route is exercised through the standard ``api_client`` /
``api_client_factory`` fixtures from :mod:`tests.api.conftest`.  We
*never* hit a real LLM — :func:`build_llm_provider` and
:func:`understand_query` are monkey-patched to controllable in-process
stubs so the tests stay hermetic and fast.

Coverage focuses on:

    - schema guards (query length, provider slug, model length, unknown
      fields rejected via ``extra="forbid"``);
    - provider resolution (body fields override settings, settings
      defaults apply when both are omitted, unknown provider rejected);
    - resolver-chain wiring (``X-Provider-Key-{provider}`` header is
      forwarded so the route runs on the caller's key, not admin env);
    - error mapping (``ConfigurationError`` → 400 ``provider_key_required``,
      ``ProviderAuthError`` → 401, transient ``ProviderError`` → 503,
      generic ``ProviderError`` → 502);
    - audit-log emission (``rag_plan`` action, no raw query in detail);
    - rate-limit classification (the request hits the ``rag`` bucket);
    - auth gate (read-tier — admin-key NOT required).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.api.app import create_app
from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.edgar_identity import InMemorySessionEdgarIdentityStore
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.logging import LOGGER_NAME
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.query_understanding import QueryPlan

# ---------------------------------------------------------------------------
# Stubs for build_llm_provider + understand_query
# ---------------------------------------------------------------------------


@dataclass
class _StubLLM:
    """Minimal stand-in returned by the patched ``build_llm_provider``."""

    provider_name: str = "openai"
    closed: bool = False

    def close(self) -> None:
        self.closed = True


@dataclass
class _StubUnderstandRecorder:
    """Captures ``understand_query`` calls and returns a controllable plan."""

    plan: QueryPlan | None = None
    raise_with: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    last_llm: Any = None

    def __call__(
        self,
        query: str,
        *,
        llm: Any,
        model: str,
        structured_output_supported: bool = False,
        **kwargs: Any,
    ) -> QueryPlan:
        # Retain the built LLM so tests can assert the route closed it
        # after query understanding.
        self.last_llm = llm
        self.calls.append(
            {
                "query": query,
                "model": model,
                "structured_output_supported": structured_output_supported,
                "llm_provider_name": getattr(llm, "provider_name", None),
            }
        )
        if self.raise_with is not None:
            raise self.raise_with
        return self.plan or QueryPlan(raw_query=query)


@dataclass
class _StubBuildRecorder:
    """Captures ``build_llm_provider`` invocations + the resolved key."""

    raise_with: Exception | None = None
    llm: _StubLLM = field(default_factory=_StubLLM)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        provider_name: str,
        *,
        api_key_resolver: Any,
    ) -> _StubLLM:
        # Probe the resolver for the named provider so the test can
        # assert that the route forwarded the resolver chain (header /
        # session / admin-env) all the way through.
        resolved = api_key_resolver(provider_name)
        self.calls.append(
            {
                "provider": provider_name,
                "resolved_key": resolved,
            }
        )
        if self.raise_with is not None:
            raise self.raise_with
        # Stash the provider name on the stub so understand_query stub
        # can echo it on the recorded call.
        return _StubLLM(provider_name=provider_name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_rag_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in list(os.environ.keys()):
        if key.startswith(("API_", "LLM_")):
            monkeypatch.delenv(key, raising=False)
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def rag_app_factory(monkeypatch: pytest.MonkeyPatch):
    """Build a fresh app with build_llm_provider + understand_query stubbed."""

    def factory(
        *,
        plan: QueryPlan | None = None,
        understand_raise: Exception | None = None,
        build_raise: Exception | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[Any, _StubBuildRecorder, _StubUnderstandRecorder]:
        if env:
            for key, value in env.items():
                monkeypatch.setenv(key, value)
        reload_settings()

        understand_stub = _StubUnderstandRecorder(plan=plan, raise_with=understand_raise)
        build_stub = _StubBuildRecorder(raise_with=build_raise)

        # Patch the imported names in the route module — direct
        # ``build_llm_provider`` / ``understand_query`` references. The
        # capability probe is left untouched; ProviderRegistry returns a
        # safe ``ProviderCapability`` for known LLM providers.
        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.build_llm_provider",
            build_stub,
        )
        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.understand_query",
            understand_stub,
        )

        app = create_app()
        app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
        app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=300)
        app.state.encrypted_credential_store = None
        return app, build_stub, understand_stub

    return factory


def _sample_plan() -> QueryPlan:
    return QueryPlan(
        raw_query="What are AAPL's iPhone segment risks?",
        detected_language="en",
        query_en="What are AAPL's iPhone segment risks?",
        tickers=["AAPL"],
        form_types=["10-K"],
        date_range=("2023-01-01", "2024-01-01"),
        intent="Identify risks for the iPhone segment",
        suggested_answer_mode=AnswerMode.ANALYTICAL,
    )


# ---------------------------------------------------------------------------
# Happy-path delegation
# ---------------------------------------------------------------------------


class TestRagPlanDelegation:
    def test_returns_plan_for_valid_query(self, rag_app_factory) -> None:
        app, _build, understand = rag_app_factory(plan=_sample_plan())
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/rag/plan",
            json={"query": "What are AAPL's iPhone segment risks?"},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )

        assert response.status_code == 200
        body = response.json()
        plan = body["plan"]
        assert plan["raw_query"] == "What are AAPL's iPhone segment risks?"
        assert plan["tickers"] == ["AAPL"]
        assert plan["form_types"] == ["10-K"]
        assert plan["date_range"] == ["2023-01-01", "2024-01-01"]
        assert plan["suggested_answer_mode"] == "analytical"
        assert body["provider"] == "openai"
        # Default settings.llm.default_model is None → empty string.
        assert body["model"] == ""
        # The route forwarded the user query into understand_query.
        assert understand.calls[0]["query"] == "What are AAPL's iPhone segment risks?"

    def test_response_strips_internal_only_fields(self, rag_app_factory) -> None:
        app, _build, _understand = rag_app_factory(plan=_sample_plan())
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/plan",
            json={"query": "any"},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == {"plan", "provider", "model"}
        assert set(body["plan"].keys()) == {
            "raw_query",
            "detected_language",
            "query_en",
            "tickers",
            "form_types",
            "date_range",
            "intent",
            "suggested_answer_mode",
        }

    def test_body_provider_overrides_settings(
        self, rag_app_factory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_DEFAULT_PROVIDER", "openai")
        app, build, _understand = rag_app_factory(plan=_sample_plan())
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/rag/plan",
            json={"query": "any", "provider": "anthropic"},
            headers={"X-Provider-Key-anthropic": "sk-ant-1234"},  # pragma: allowlist secret
        )

        assert response.status_code == 200
        assert response.json()["provider"] == "anthropic"
        assert build.calls[0]["provider"] == "anthropic"

    def test_body_model_overrides_settings(self, rag_app_factory) -> None:
        app, _build, understand = rag_app_factory(plan=_sample_plan())
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/rag/plan",
            json={"query": "any", "model": "gpt-5.4-mini"},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )

        assert response.status_code == 200
        assert response.json()["model"] == "gpt-5.4-mini"
        assert understand.calls[0]["model"] == "gpt-5.4-mini"

    def test_header_resolver_forwards_user_key(self, rag_app_factory) -> None:
        app, build, _understand = rag_app_factory(plan=_sample_plan())
        client = TestClient(app, base_url="https://testserver")

        secret = "sk-USER-PROVIDED-KEY-1234"  # pragma: allowlist secret
        client.post(
            "/api/rag/plan",
            json={"query": "any"},
            headers={"X-Provider-Key-openai": secret},
        )

        # The route MUST run the LLM construction against the chained
        # resolver — header tier first.  Without that, an admin-env
        # fallback would silently mask a missing header.
        assert build.calls[0]["resolved_key"] == secret


# ---------------------------------------------------------------------------
# Schema guards
# ---------------------------------------------------------------------------


class TestRagPlanSchemaGuards:
    def test_missing_query_rejected(self, rag_app_factory) -> None:
        app, _build, understand = rag_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/rag/plan", json={})
        assert response.status_code == 422
        # No LLM call must have occurred when the schema rejects the body.
        assert understand.calls == []

    def test_empty_query_rejected(self, rag_app_factory) -> None:
        app, _build, _understand = rag_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/rag/plan", json={"query": ""})
        assert response.status_code == 422

    def test_oversize_query_rejected(self, rag_app_factory) -> None:
        app, _build, _understand = rag_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/rag/plan", json={"query": "x" * 2000})
        assert response.status_code == 422

    def test_uppercase_provider_rejected(self, rag_app_factory) -> None:
        app, _build, _understand = rag_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/plan",
            json={"query": "any", "provider": "OPENAI"},
        )
        assert response.status_code == 422

    def test_unknown_field_rejected(self, rag_app_factory) -> None:
        app, _build, _understand = rag_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/plan",
            json={"query": "any", "depth": 7},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestRagPlanErrorMapping:
    def test_unknown_provider_400(self, rag_app_factory) -> None:
        app, _build, _understand = rag_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/plan",
            json={"query": "any", "provider": "definitely-not-real"},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "unknown_provider"

    def test_missing_key_maps_to_400(self, rag_app_factory) -> None:
        app, _build, _understand = rag_app_factory(
            build_raise=ConfigurationError("no key resolved"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/plan",
            json={"query": "any"},
        )
        assert response.status_code == 400
        body = response.json()
        assert body["error"] == "provider_key_required"
        # Hint must steer the caller toward the resolver chain options.
        assert "X-Provider-Key" in body["hint"]

    def test_provider_auth_error_maps_to_401(self, rag_app_factory) -> None:
        app, _build, _understand = rag_app_factory(
            understand_raise=ProviderAuthError("rejected by upstream"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/plan",
            json={"query": "any"},
            headers={"X-Provider-Key-openai": "sk-bad-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 401
        assert response.json()["error"] == "provider_unauthorized"

    def test_rate_limit_maps_to_503(self, rag_app_factory) -> None:
        app, _build, _understand = rag_app_factory(
            understand_raise=ProviderRateLimitError("upstream limited"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/plan",
            json={"query": "any"},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 503
        body = response.json()
        assert body["error"] == "provider_unavailable"
        assert "do not rotate" in body["hint"].lower()

    def test_timeout_maps_to_503(self, rag_app_factory) -> None:
        app, _build, _understand = rag_app_factory(
            understand_raise=ProviderTimeoutError("timed out"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/plan",
            json={"query": "any"},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 503

    def test_connection_error_maps_to_503(self, rag_app_factory) -> None:
        # Unreachable endpoint during planning → transient 503 with
        # endpoint-focused guidance (chiefly an unstarted local_llm server).
        app, _build, _understand = rag_app_factory(
            understand_raise=ProviderConnectionError("could not reach endpoint"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/plan",
            json={"query": "any"},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 503
        body = response.json()
        assert body["error"] == "provider_unavailable"
        assert "reach" in body["message"].lower()
        assert "local_llm" in body["hint"].lower()

    def test_generic_provider_error_maps_to_502(self, rag_app_factory) -> None:
        app, _build, _understand = rag_app_factory(
            understand_raise=ProviderError("transport blew up"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/plan",
            json={"query": "any"},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 502
        assert response.json()["error"] == "provider_error"


# ---------------------------------------------------------------------------
# Privacy / audit-log no-leak contract
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRagPlanPrivacyContract:
    def test_audit_log_carries_metadata_not_query(
        self,
        rag_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        app, _build, _understand = rag_app_factory(plan=_sample_plan())
        client = TestClient(app, base_url="https://testserver")

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                client.post(
                    "/api/rag/plan",
                    json={"query": "PROPRIETARY-WATCHLIST-MNEMONIC-ZZZ"},
                    headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
                )
        finally:
            package_logger.propagate = prior_propagate

        audit = [r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()]
        assert any("rag_plan" in line for line in audit)
        # Audit lines MUST NOT carry the raw query.
        assert all("PROPRIETARY-WATCHLIST-MNEMONIC-ZZZ" not in line for line in audit)

    def test_error_envelope_does_not_echo_provider_key(
        self,
        rag_app_factory,
    ) -> None:
        app, _build, _understand = rag_app_factory(
            understand_raise=ProviderError("transport blew up"),
        )
        client = TestClient(app, base_url="https://testserver")
        secret = "sk-NEVER-ECHO-ME-1234"  # pragma: allowlist secret
        response = client.post(
            "/api/rag/plan",
            json={"query": "any"},
            headers={"X-Provider-Key-openai": secret},
        )
        assert response.status_code == 502
        assert secret not in response.text


# ---------------------------------------------------------------------------
# Auth tier — read-tier (no admin gate)
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRagPlanAuthGate:
    def test_api_key_required_when_configured(self, rag_app_factory) -> None:
        app, _build, _understand = rag_app_factory(
            plan=_sample_plan(),
            env={"API_KEY": "shared-team-key"},  # pragma: allowlist secret
        )
        client = TestClient(app, base_url="https://testserver")

        unauthed = client.post(
            "/api/rag/plan",
            json={"query": "any"},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert unauthed.status_code == 401
        assert unauthed.json()["error"] == "unauthorised"

        ok = client.post(
            "/api/rag/plan",
            json={"query": "any"},
            headers={
                "X-API-Key": "shared-team-key",  # pragma: allowlist secret
                "X-Provider-Key-openai": "sk-1234",  # pragma: allowlist secret
            },
        )
        assert ok.status_code == 200

    def test_admin_key_not_required(self, rag_app_factory) -> None:
        app, _build, _understand = rag_app_factory(
            plan=_sample_plan(),
            env={
                "API_KEY": "shared-team-key",  # pragma: allowlist secret
                "API_ADMIN_KEY": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/plan",
            json={"query": "any"},
            headers={
                "X-API-Key": "shared-team-key",  # pragma: allowlist secret
                "X-Provider-Key-openai": "sk-1234",  # pragma: allowlist secret
            },
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Rate-limit classification
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRagPlanRateLimitClassification:
    def test_plan_path_classifies_as_rag(self) -> None:
        from sec_generative_search.api.middleware import _rate_category_for_scope

        assert (
            _rate_category_for_scope({"type": "http", "path": "/api/rag/plan", "method": "POST"})
            == "rag"
        )

    def test_plan_respects_per_ip_window(self, rag_app_factory) -> None:
        app, _build, _understand = rag_app_factory(
            plan=_sample_plan(),
            env={"API_RATE_LIMIT_RAG": "3"},
        )
        client = TestClient(app, base_url="https://testserver")
        statuses: list[int] = []
        for _ in range(6):
            r = client.post(
                "/api/rag/plan",
                json={"query": "any"},
                headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
            )
            statuses.append(r.status_code)
        assert statuses.count(200) == 3
        assert 429 in statuses


@pytest.mark.security
class TestRagPlanClientLifecycle:
    """The plan route closes the per-request LLM on every exit path."""

    def test_client_closed_after_success(self, rag_app_factory) -> None:
        app, _build, understand = rag_app_factory(plan=_sample_plan())
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/plan",
            json={"query": "What are AAPL's iPhone segment risks?"},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        assert understand.last_llm.closed is True

    def test_client_closed_after_provider_error(self, rag_app_factory) -> None:
        app, _build, understand = rag_app_factory(
            understand_raise=ProviderRateLimitError("upstream limited"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/plan",
            json={"query": "any"},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )
        assert response.status_code == 503
        # The finally still runs even though query-understanding raised.
        assert understand.last_llm.closed is True
