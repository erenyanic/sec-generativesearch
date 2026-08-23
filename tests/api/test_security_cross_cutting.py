"""Cross-cutting security regression tests.

This module collects route-spanning security invariants that are not
owned end-to-end by the individual route test modules. The existing
route-level tests already cover three of the six dimensions in depth:

    - Resolver-chain ordering — :mod:`tests.api.test_dependencies`
      exhaustively walks the header → session → encrypted-user (admin
      gated) → admin-env precedence, including the forged-cookie
      fall-through and the admin gate on the encrypted tier.
    - Two-tier admin-key gating — :mod:`tests.api.test_auth_tiers`
      exercises the full ``API_KEY`` x ``API_ADMIN_KEY`` matrix with
      ``admin_route_dependencies()``.
    - Validate-route per-session rate-limit shadow —
      :class:`tests.api.test_providers.TestValidateRouteRateLimit`
      asserts the per-session bucket fires independently of the per-IP
      bucket.

This module fills the remaining gaps in a single file so the cross-
cutting checks stay together:

    1. Validate route: the **per-IP** bucket fires independently of the
       per-session bucket, *and* both windows must allow the request.
    2. Header redaction is honoured **across routes** at the audit-log
       layer — secrets supplied via ``Authorization`` / ``X-API-Key`` /
       ``X-Admin-Key`` / ``X-Provider-Key-*`` / ``X-Edgar-Name`` /
       ``X-Edgar-Email`` never reach an audit-log record from any
       business surface.
    3. SSE streaming generation never leaks a stored EDGAR identity
       (name / email) — neither into the response body nor into the
       ``rag_stream`` / ``rag_stream_completed`` audit lines.
    4. ``session_id`` only flows through the server-set cookie. The
       resolver chain ignores attempts to plant the id via
       ``X-Session-Id`` header, query string, or ``Authorization``
       header.

Strategy
--------

Every test in this file rides on the hermetic fixtures from
``tests/api/conftest.py`` (no lifespan, stubbed singletons) and reuses
an in-process orchestrator stub for the SSE case so no real LLM or
embedder is touched.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from sec_generative_search.api.app import create_app
from sec_generative_search.api.dependencies import (
    SESSION_COOKIE_NAME,
    request_scoped_resolver,
)
from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.edgar_identity import (
    EdgarIdentity,
    InMemorySessionEdgarIdentityStore,
)
from sec_generative_search.core.logging import LOGGER_NAME
from sec_generative_search.core.types import (
    Citation,
    FilingIdentifier,
    GenerationResult,
    TokenUsage,
)
from sec_generative_search.providers.registry import (
    ProviderEntry,
    ProviderRegistry,
    ProviderSurface,
)
from sec_generative_search.rag.orchestrator import StreamEvent
from sec_generative_search.rag.query_understanding import QueryPlan

# ---------------------------------------------------------------------------
# Test-only credential / identity payloads
# ---------------------------------------------------------------------------

# Distinctive sentinel strings so a leak in any log line / response body
# / SSE frame is trivially greppable. Each one is long enough to dodge
# ``mask_secret``'s under-8-char full-redaction rule, so any partial leak
# (e.g. tail-show) still asserts.
_API_KEY = "shared-team-key-XYZ"  # pragma: allowlist secret
_ADMIN_KEY = "secret-admin-key-ABC"  # pragma: allowlist secret
_PROVIDER_KEY = "sk-NEVER-LEAK-PROVIDER-KEY"  # pragma: allowlist secret
_EDGAR_NAME = "PROPRIETARY_NAME_SENTINEL"
_EDGAR_EMAIL = "leak-sentinel@example.invalid"


# ---------------------------------------------------------------------------
# Stub LLM + orchestrator (shared by the SSE no-leak test)
# ---------------------------------------------------------------------------


@dataclass
class _StubLLM:
    provider_name: str = "openai"

    def close(self) -> None:
        """No-op teardown — the F5(a) contract requires every provider
        double to expose ``close()`` (the ``/stream`` producer thread calls
        it in its ``finally``)."""


@dataclass
class _StubOrchestrator:
    retrieval: Any = None
    llm: Any = None
    events: list[StreamEvent] = field(default_factory=list)

    def generate_stream(self, plan: QueryPlan, **_: Any) -> Iterator[StreamEvent]:
        yield from self.events


def _default_citation() -> Citation:
    return Citation(
        chunk_id="chunk-1",
        filing_id=FilingIdentifier(
            ticker="AAPL",
            form_type="10-K",
            filing_date=date(2023, 9, 30),
            accession_number="0000320193-23-000077",
        ),
        section_path="Part I > Item 1A > Risk Factors",
        text_span="A non-secret excerpt.",
        similarity=0.84,
        display_index=1,
    )


def _default_final_result() -> GenerationResult:
    return GenerationResult(
        answer="Stub answer [1].",
        provider="openai",
        model="gpt-5.4-mini",
        prompt_version="rag-prompt-v1",
        citations=[_default_citation()],
        retrieved_chunks=[],
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        latency_seconds=0.1,
        streamed=True,
    )


def _sample_plan_payload() -> dict[str, Any]:
    return {
        "raw_query": "What are AAPL's iPhone segment risks?",
        "detected_language": "en",
        "query_en": "What are AAPL's iPhone segment risks?",
        "tickers": ["AAPL"],
        "form_types": ["10-K"],
        "date_range": ["2023-01-01", "2024-01-01"],
        "intent": "Identify risks for the iPhone segment",
        "suggested_answer_mode": "analytical",
    }


# ---------------------------------------------------------------------------
# Stub validate-route provider registry
# ---------------------------------------------------------------------------


class _StubValidateProvider:
    """Always-valid stand-in matching ``BaseLLMProvider.validate_key``."""

    def __init__(self, *args, **kwargs) -> None: ...

    def validate_key(self) -> bool:
        return True

    def close(self) -> None: ...


@pytest.fixture
def _patch_validate_registry(monkeypatch: pytest.MonkeyPatch):
    """Point ``ProviderRegistry.get_entry`` at the always-valid stub.

    Scoped to the validate-route tests so the rest of the file still
    sees the real registry.
    """
    fake_entry = ProviderEntry(
        name="openai",
        surface=ProviderSurface.LLM,
        provider_cls=_StubValidateProvider,
    )

    def _get_entry(name, surface):
        if name == "openai" and surface == ProviderSurface.LLM:
            return fake_entry
        raise KeyError(f"No provider registered for name='{name}', surface='{surface.value}'.")

    monkeypatch.setattr(
        ProviderRegistry,
        "get_entry",
        classmethod(lambda cls, n, s: _get_entry(n, s)),
    )


# ---------------------------------------------------------------------------
# 1. Validate route — per-IP bucket fires independently of per-session
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestValidateRoutePerIpRateLimit:
    """The per-IP and per-session windows are independent buckets.

    A previous regression hid the per-IP fire when the per-session
    counter was permissive — both must allow the request.
    """

    def test_per_ip_bucket_fires_when_per_session_permissive(
        self,
        api_client_factory,
        _patch_validate_registry,
    ) -> None:
        # Tight per-IP, generous per-session. The per-IP window must
        # reject before the per-session bucket would even notice.
        client = api_client_factory(
            API_RATE_LIMIT_VALIDATE="3",
            API_RATE_LIMIT_VALIDATE_PER_SESSION="100",
        )

        statuses: list[int] = []
        for _ in range(8):
            r = client.post(
                "/api/providers/validate",
                json={"provider": "openai", "api_key": "sk-test-1234"},  # pragma: allowlist secret
            )
            statuses.append(r.status_code)

        # At least one 429 must appear — the per-IP bucket of 3 was the
        # binding constraint, not the per-session bucket of 100.
        assert 429 in statuses
        # Some 200s must precede the limiter kicking in.
        assert 200 in statuses

    def test_both_windows_must_allow(
        self,
        api_client_factory,
        _patch_validate_registry,
    ) -> None:
        # Both windows tight. Either can fire first; what matters is
        # that the route never exceeds the *minimum* of the two budgets
        # — never exceeds the per-IP limit, never the per-session limit.
        client = api_client_factory(
            API_RATE_LIMIT_VALIDATE="5",
            API_RATE_LIMIT_VALIDATE_PER_SESSION="3",
        )
        # Mint a session so the per-session bucket has a key.
        client.post("/api/session")

        success_count = 0
        for _ in range(10):
            r = client.post(
                "/api/providers/validate",
                json={"provider": "openai", "api_key": "sk-test-1234"},  # pragma: allowlist secret
            )
            if r.status_code == 200:
                success_count += 1

        # Effective budget is min(5, 3) = 3. The combined gate cannot
        # let more requests through than the smaller bucket.
        assert success_count <= 3


# ---------------------------------------------------------------------------
# 2. Cross-route header redaction at the audit-log layer
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestCrossRouteAuditNoLeak:
    """Audit-log lines never carry credential-shaped header values.

    Hits one read-tier route (``/api/session``) and one destructive
    admin-tier route (``DELETE /api/filings/{accession}``) so the
    invariant is verified across at least two surfaces. The route-level
    audit lines (``session_minted``, ``api_key_denied``, ``admin_denied``)
    have been individually tested; this assertion is the *cross-cutting*
    check that no header value lands in any audit line from any route.
    """

    def test_no_credential_header_lands_in_audit_log(
        self,
        api_client_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        client = api_client_factory(
            API_KEY=_API_KEY,
            API_ADMIN_KEY=_ADMIN_KEY,
        )

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
                # Mint a session (authenticated). Carries every secret
                # header shape so a leaky audit-log line would surface
                # at least one of them.
                client.post(
                    "/api/session",
                    headers={
                        "X-API-Key": _API_KEY,
                        "X-Admin-Key": _ADMIN_KEY,
                        "X-Provider-Key-openai": _PROVIDER_KEY,
                        "X-Edgar-Name": _EDGAR_NAME,
                        "X-Edgar-Email": _EDGAR_EMAIL,
                        "Authorization": f"Bearer {_PROVIDER_KEY}",
                    },
                )
                # Trigger a 401 on a destructive route — the
                # ``api_key_denied`` audit line must not echo the
                # supplied (wrong) key. Use a wrong API key shaped like
                # the real one so the check is not trivially satisfied.
                client.delete(
                    "/api/filings/0000320193-23-000077",
                    headers={
                        "X-API-Key": "wrong-but-distinct-sentinel-ZZZ",
                        "X-Admin-Key": _ADMIN_KEY,
                        "X-Edgar-Name": _EDGAR_NAME,
                        "X-Edgar-Email": _EDGAR_EMAIL,
                    },
                )
        finally:
            package_logger.propagate = prior_propagate

        audit_messages = [
            r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()
        ]
        # We must have produced at least one audit line so the assertion
        # below is actually exercising something.
        assert audit_messages, "expected at least one SECURITY_AUDIT line"

        for message in audit_messages:
            assert _API_KEY not in message
            assert _ADMIN_KEY not in message
            assert _PROVIDER_KEY not in message
            assert _EDGAR_NAME not in message
            assert _EDGAR_EMAIL not in message
            assert "wrong-but-distinct-sentinel-ZZZ" not in message


# ---------------------------------------------------------------------------
# 3. SSE no-leak of stored EDGAR identity
# ---------------------------------------------------------------------------


@pytest.fixture
def rag_stream_app(monkeypatch: pytest.MonkeyPatch):
    """Build a fresh app with build_llm_provider + RAGOrchestrator stubbed.

    Slimmer than ``test_rag_stream.rag_stream_app_factory`` — we only
    need a single happy-path stream for the no-leak check.
    """
    for key in list(os.environ.keys()):
        if key.startswith(("API_", "LLM_")):
            monkeypatch.delenv(key, raising=False)
    reload_settings()

    stub_orch = _StubOrchestrator(events=[StreamEvent(final=_default_final_result())])

    def _build_llm(*args, **kwargs) -> _StubLLM:
        # Resolve through the supplied resolver so the test still proves
        # the resolver chain is exercised on this surface, even though
        # the resolved value is then ignored by the stub LLM.
        kwargs["api_key_resolver"]("openai")
        return _StubLLM()

    def _orch_factory(**kwargs: Any) -> _StubOrchestrator:
        stub_orch.retrieval = kwargs.get("retrieval")
        stub_orch.llm = kwargs.get("llm")
        return stub_orch

    monkeypatch.setattr(
        "sec_generative_search.api.routes.rag.build_llm_provider",
        _build_llm,
    )
    monkeypatch.setattr(
        "sec_generative_search.api.routes.rag.RAGOrchestrator",
        _orch_factory,
    )

    app = create_app()
    app.state.retrieval_service = object()
    app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
    app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=300)
    app.state.encrypted_credential_store = None
    return app


@pytest.mark.security
class TestSseNoLeakOfEdgarIdentity:
    """A stored EDGAR identity never appears in the SSE stream output
    or in the ``rag_stream`` / ``rag_stream_completed`` audit lines.

    Background: EDGAR identity is PII (Tier 3) and the access-log layer
    fully suppresses ``X-Edgar-*`` headers. The streaming surface adds a
    second exposure path — the orchestrator could plausibly thread the
    identity through prompt assembly or token-usage diagnostics and
    surface it as a delta or in the final payload. This test pins the
    invariant at the route boundary.
    """

    def test_stream_body_and_audit_log_omit_edgar_identity(
        self,
        rag_stream_app,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        client = TestClient(rag_stream_app, base_url="https://testserver")

        # Mint a session and register an EDGAR identity on it so the
        # identity store is non-empty for the lifetime of the request.
        client.post("/api/session")
        sid = client.cookies.get(SESSION_COOKIE_NAME)
        assert sid is not None
        rag_stream_app.state.edgar_identity_store.set(
            sid,
            EdgarIdentity(name=_EDGAR_NAME, email=_EDGAR_EMAIL),
        )

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                response = client.post(
                    "/api/rag/stream",
                    json={"plan": _sample_plan_payload()},
                    headers={
                        "X-Provider-Key-openai": _PROVIDER_KEY,
                        # And also pass them as headers — the route
                        # boundary is the test target.
                        "X-Edgar-Name": _EDGAR_NAME,
                        "X-Edgar-Email": _EDGAR_EMAIL,
                    },
                )
        finally:
            package_logger.propagate = prior_propagate

        assert response.status_code == 200

        # Response body (SSE frames) MUST NOT carry the identity.
        assert _EDGAR_NAME not in response.text
        assert _EDGAR_EMAIL not in response.text
        # Provider key must not leak through the SSE body either.
        assert _PROVIDER_KEY not in response.text

        # Audit-log lines for both stream open and stream close must be
        # present and identity-free.
        audit_lines = [
            r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()
        ]
        assert any("rag_stream" in line for line in audit_lines)
        assert any("rag_stream_completed" in line for line in audit_lines)
        for line in audit_lines:
            assert _EDGAR_NAME not in line
            assert _EDGAR_EMAIL not in line


# ---------------------------------------------------------------------------
# 4. session_id only flows via the cookie surface
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSessionIdHeaderPathRejection:
    """A ``session_id`` planted via header / query string / Authorization
    does not authenticate against the per-session credential store.

    The store is the only thing that gives a ``session_id`` value
    meaning at the resolver-chain layer — a cookie that is not also
    backed by a store entry resolves to no credentials. This test
    inverts the check: we install a credential under a known
    ``session_id``, then send that same id through every non-cookie
    surface and prove the resolver does not pick it up.
    """

    def test_x_session_id_header_is_not_honoured(
        self,
        api_client,
        api_app,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Plant a credential under a valid-shape session_id directly in
        # the store (bypass the mint route entirely so no cookie is set).
        planted_sid = "Q" * 43
        api_app.state.session_store.set(
            planted_sid,
            "openai",
            "sk-planted-credential-MUST-NOT-RESOLVE",  # pragma: allowlist secret
        )
        # Admin-env fallback so the resolver always returns *something*
        # when the chain falls through. Distinct from the planted value
        # so we can tell which tier won.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-admin-env-correct-value")
        reload_settings()

        # Attempt to bind the session via a non-cookie surface. The
        # resolver chain reads ``extract_session_id`` which only consults
        # cookies, so each of these must fall through to admin-env.
        for header_name in ("X-Session-Id", "X-Session", "Session-Id"):
            scope = {
                "type": "http",
                "method": "GET",
                "path": "/",
                "raw_path": b"/",
                "query_string": f"session_id={planted_sid}".encode(),
                "headers": [
                    (header_name.lower().encode(), planted_sid.encode()),
                ],
            }
            request = Request(scope)
            request.scope["app"] = api_app
            resolver = request_scoped_resolver(request)
            resolved = resolver("openai")
            # The planted credential MUST NOT resolve — only admin-env.
            assert resolved == "sk-admin-env-correct-value"

    def test_authorization_header_cannot_carry_session_id(
        self,
        api_client,
        api_app,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # ``Authorization`` is a known credential surface for some APIs;
        # we deliberately do NOT treat it as a session-id transport.
        planted_sid = "R" * 43
        api_app.state.session_store.set(
            planted_sid,
            "openai",
            "sk-MUST-NOT-LEAK-via-authorization",  # pragma: allowlist secret
        )
        monkeypatch.setenv("OPENAI_API_KEY", "sk-admin-env-correct-value")
        reload_settings()

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "raw_path": b"/",
            "query_string": b"",
            "headers": [
                (b"authorization", f"Bearer {planted_sid}".encode()),
            ],
        }
        request = Request(scope)
        request.scope["app"] = api_app
        resolver = request_scoped_resolver(request)
        assert resolver("openai") == "sk-admin-env-correct-value"

    def test_query_string_session_id_is_not_honoured_by_route(
        self,
        api_client,
        api_app,
    ) -> None:
        # Same idea via the real route surface: a logout call that
        # carries ``?session_id=...`` in the URL has no cookie, so the
        # resolver chain treats it as anonymous. The route is the right
        # surface to test against because the URL goes through Starlette
        # parsing and the route still must not pick it up.
        planted_sid = "S" * 43
        api_app.state.session_store.set(
            planted_sid,
            "openai",
            "sk-MUST-NOT-LEAK-via-query",  # pragma: allowlist secret
        )

        # No cookie set on the client; the planted session_id is only
        # in the query string.
        response = api_client.post(f"/api/session/logout?session_id={planted_sid}")
        assert response.status_code == 200
        # The route reports zero clears because no cookie was supplied
        # — confirming the URL parameter was never treated as a
        # ``session_id`` source.
        assert response.json()["cleared_credentials"] == 0
        # And the planted credential is still present in the store —
        # the logout call did NOT find it.
        assert api_app.state.session_store.get(planted_sid, "openai") is not None


# ---------------------------------------------------------------------------
# 5. Research identifiers (ticker / accession) never reach the log stream
#    under LOG_REDACT_QUERIES (finding M3)
# ---------------------------------------------------------------------------

# A ticker shaped like a real one but unmistakable in a log line, and the
# canonical accession shape. Both are the Tier-3 "research pattern" the
# no-persistence model keeps out of storage — the log stream must not
# re-expose them.
_TICKER_SENTINEL = "ZQXW"
_ACCESSION_SENTINEL = "0000320193-23-000077"


@dataclass
class _LogProbeRetrievalService:
    """Retrieval stub for the redaction probe — returns one hit."""

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]:
        return []


@dataclass
class _LogProbeRegistry:
    """Registry stub exposing exactly one filing under the sentinels."""

    record: Any

    def get_filing(self, accession_number: str) -> Any:
        if accession_number == self.record.accession_number:
            return self.record
        return None


@dataclass
class _LogProbeStore:
    """FilingStore stub that reports a successful single delete."""

    deleted: list[str] = field(default_factory=list)

    def delete_filing(self, accession_number: str) -> bool:
        self.deleted.append(accession_number)
        return True


@dataclass
class _LogProbeTaskManager:
    """TaskManager stub for the ingest create path."""

    created: list[dict[str, Any]] = field(default_factory=list)

    def create_task(self, **kwargs: Any) -> str:
        kwargs.pop("edgar_identity_resolver", None)
        self.created.append(kwargs)
        return f"{len(self.created):032x}"


@pytest.fixture
def json_log_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[Any]:
    """Boot the **real** package logging stack into a JSON-lines file.

    ``caplog`` attaches to the root logger, but the package logger sets
    ``propagate = False`` and the redaction filter lives on the
    *handlers* ``configure_logging`` builds (same placement as
    :class:`CorrelationIdFilter`, and for the same reason). Capturing
    through ``caplog`` would therefore observe **unfiltered** records
    and make this lock vacuous. Driving the production
    ``configure_logging`` path into a real file handler exercises
    filter + handler wiring + formatter exactly as a Scenario-B/C
    deployment does.
    """
    from sec_generative_search.core import logging as sgs_logging

    log_file = tmp_path / "app.jsonl"
    monkeypatch.setenv("LOG_FILE_PATH", str(log_file))
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_REDACT_QUERIES", "true")

    package_logger = logging.getLogger(LOGGER_NAME)
    prior_handlers = list(package_logger.handlers)
    prior_level = package_logger.level
    prior_configured = sgs_logging._logging_configured

    # ``configure_logging`` early-returns on the module global; without
    # this reset the new handler is never built and the assertions below
    # would pass vacuously.
    package_logger.handlers.clear()
    sgs_logging._logging_configured = False
    sgs_logging.configure_logging(use_rich=False)

    try:
        yield log_file
    finally:
        for handler in list(package_logger.handlers):
            handler.close()
        package_logger.handlers[:] = prior_handlers
        package_logger.setLevel(prior_level)
        sgs_logging._logging_configured = prior_configured


@pytest.mark.security
class TestResearchIdentifierLogRedaction:
    """``LOG_REDACT_QUERIES=true`` must cover tickers **and** accessions.

    The flag is a documented Scenario-B/C default and ``LOG_FORMAT=json``
    ships the stream to an aggregator (a third-party one in Scenario C).
    An operator who enables it must not still be shipping every ingested,
    searched, and deleted research target in plaintext.

    Drives the three surfaces the finding names — an ingest create, a
    ``POST /api/search`` carrying a ticker filter, and a
    ``DELETE /api/filings/{accession}`` — through the real logging stack
    and asserts no raw identifier survives in any emitted record.
    """

    def _drive_all_three_surfaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sec_generative_search.database import FilingRecord

        for key in list(os.environ):
            if key.startswith("API_") or key.startswith("EDGAR_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("EDGAR_IDENTITY_NAME", "Test User")
        monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "test@example.com")
        reload_settings()

        record = FilingRecord(
            id=1,
            ticker=_TICKER_SENTINEL,
            form_type="10-K",
            filing_date="2023-09-30",
            accession_number=_ACCESSION_SENTINEL,
            chunk_count=12,
            ingested_at="2024-01-01T00:00:00Z",
        )

        app = create_app()
        app.state.retrieval_service = _LogProbeRetrievalService()
        app.state.registry = _LogProbeRegistry(record=record)
        app.state.filing_store = _LogProbeStore()
        app.state.task_manager = _LogProbeTaskManager()
        app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
        app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=300)
        app.state.encrypted_credential_store = None

        client = TestClient(app, base_url="https://testserver")

        created = client.post(
            "/api/ingest/add",
            json={"tickers": [_TICKER_SENTINEL], "form_types": ["10-K"]},
        )
        assert created.status_code == 202, created.text

        searched = client.post(
            "/api/search",
            json={"query": "segment revenue", "ticker": _TICKER_SENTINEL},
        )
        assert searched.status_code == 200, searched.text

        deleted = client.delete(f"/api/filings/{_ACCESSION_SENTINEL}")
        assert deleted.status_code == 200, deleted.text

    @staticmethod
    def _emitted_lines(log_file) -> list[str]:
        text = log_file.read_text(encoding="utf-8")
        return [line for line in text.splitlines() if line.strip()]

    def test_no_raw_ticker_or_accession_in_any_emitted_record(
        self,
        json_log_file,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._drive_all_three_surfaces(monkeypatch)

        lines = self._emitted_lines(json_log_file)
        assert lines, "expected the routes to emit at least one log record"

        # Non-vacuity: the three audit lines must actually be present,
        # otherwise the absence assertions below prove nothing.
        joined = "\n".join(lines)
        for action in ("ingest_task_created", "search_executed", "delete_filing"):
            assert action in joined, f"expected a {action} audit line"

        for line in lines:
            payload = json.loads(line)
            message = payload["message"]
            assert _TICKER_SENTINEL not in message, f"raw ticker leaked: {message}"
            assert _ACCESSION_SENTINEL not in message, f"raw accession leaked: {message}"
            # The dash-free rendering must not survive either.
            assert _ACCESSION_SENTINEL.replace("-", "") not in message

        # And the redaction actually fired (rather than the identifiers
        # simply being dropped from every line).
        assert "<redacted:" in joined

    def test_scenario_a_default_keeps_identifiers_readable(
        self,
        json_log_file,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With the flag off (Scenario A) the operator keeps full detail.

        The control is deliberately opt-in: a single-user local install
        gets grep-able logs. Pinning this stops a future change from
        turning redaction on unconditionally, which would be a silent
        operability regression.
        """
        monkeypatch.setenv("LOG_REDACT_QUERIES", "false")
        self._drive_all_three_surfaces(monkeypatch)

        joined = "\n".join(self._emitted_lines(json_log_file))
        assert _TICKER_SENTINEL in joined
        assert _ACCESSION_SENTINEL in joined
