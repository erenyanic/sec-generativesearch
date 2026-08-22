"""Tests for the retrieval-only search route.

Strategy
--------

The route surface is exercised via the ``api_client`` test fixture
inherited from :mod:`tests.api.conftest`.  The production lifespan is
skipped — :class:`RetrievalService` is replaced on ``app.state`` with a
controllable in-process stub so the tests never touch ChromaDB or an
embedder.

Coverage focuses on:

    - schema guards (query length, top_k bounds, list-filter bounds,
      ISO date pattern, accepted content-type values);
    - error mapping (:class:`SearchError` → 400, :class:`DatabaseError`
      → 500, :class:`ProviderError` → 502) and the privacy contract
      that the body NEVER echoes the raw query;
    - delegation to :class:`RetrievalService`: the route forwards
      every documented field, applies no extra filtering, and returns
      the service's result list verbatim;
    - audit-log emission on success (``search_executed`` line, no query);
    - rate-limit classification (the request hits the ``search`` bucket).
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
    DatabaseError,
    ProviderError,
    SearchError,
)
from sec_generative_search.core.logging import LOGGER_NAME
from sec_generative_search.core.types import ContentType, RetrievalResult
from sec_generative_search.search.retrieval import RetrievalService

# ---------------------------------------------------------------------------
# In-process retrieval stub
# ---------------------------------------------------------------------------


@dataclass
class _StubRetrievalService:
    """Minimal stand-in for :class:`RetrievalService`.

    Records the kwargs of every ``retrieve`` call so tests can assert the
    route's delegation contract without touching the real service.  The
    return value and exception behaviour are overridable per-test.
    """

    results: list[RetrievalResult] = field(default_factory=list)
    raise_with: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def retrieve(self, query: str, **kwargs: Any) -> list[RetrievalResult]:
        self.calls.append({"query": query, **kwargs})
        if self.raise_with is not None:
            raise self.raise_with
        return list(self.results)


def _result(
    *,
    chunk_id: str = "chunk-1",
    content: str = "Apple's iPhone segment grew 9% YoY.",
    path: str = "Part I > Item 1 > Business",
    ticker: str = "AAPL",
    form_type: str = "10-K",
    similarity: float = 0.82,
    rerank_score: float | None = None,
    accession: str | None = "0000320193-23-000077",
    filing_date: str | None = "2023-09-30",
    token_count: int = 128,
) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        path=path,
        content_type=ContentType.TEXT,
        ticker=ticker,
        form_type=form_type,
        similarity=similarity,
        filing_date=filing_date,
        accession_number=accession,
        chunk_id=chunk_id,
        token_count=token_count,
        truncated=False,
        section_boundaries=("Part I", "Item 1", "Business"),
        rerank_score=rerank_score,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_search_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip API_* env vars so each test sees a clean baseline."""
    for key in list(os.environ.keys()):
        if key.startswith("API_"):
            monkeypatch.delenv(key, raising=False)
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def search_app_factory(monkeypatch: pytest.MonkeyPatch):
    """Build a fresh app with a stub :class:`RetrievalService` attached."""

    def factory(
        *,
        results: list[RetrievalResult] | None = None,
        raise_with: Exception | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[Any, _StubRetrievalService]:
        if env:
            for key, value in env.items():
                monkeypatch.setenv(key, value)
        reload_settings()

        app = create_app()
        service = _StubRetrievalService(
            results=list(results or []),
            raise_with=raise_with,
        )
        app.state.retrieval_service = service
        app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
        app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=300)
        app.state.encrypted_credential_store = None
        return app, service

    return factory


# ---------------------------------------------------------------------------
# Happy-path delegation
# ---------------------------------------------------------------------------


class TestSearchDelegation:
    def test_returns_hits_for_valid_query(self, search_app_factory) -> None:
        results = [_result(chunk_id="chunk-A"), _result(chunk_id="chunk-B")]
        app, service = search_app_factory(results=results)
        client = TestClient(app, base_url="https://testserver")

        response = client.post("/api/search", json={"query": "revenue concentration risk"})

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        ids = [hit["chunk_id"] for hit in body["hits"]]
        assert ids == ["chunk-A", "chunk-B"]
        assert service.calls[0]["query"] == "revenue concentration risk"

    def test_forwards_every_documented_filter(self, search_app_factory) -> None:
        app, service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")

        client.post(
            "/api/search",
            json={
                "query": "supply chain",
                "top_k": 7,
                "min_similarity": 0.4,
                "ticker": ["AAPL", "MSFT"],
                "form_type": "10-K",
                "accession_number": ["0000320193-23-000077"],
                "start_date": "2022-01-01",
                "end_date": "2024-01-01",
                "max_per_section": 2,
                "max_per_filing": 3,
                "rerank_over_fetch_factor": 6,
                "context_token_budget": 4000,
            },
        )

        assert len(service.calls) == 1
        call = service.calls[0]
        assert call["top_k"] == 7
        assert call["min_similarity"] == 0.4
        assert call["ticker"] == ["AAPL", "MSFT"]
        assert call["form_type"] == "10-K"
        assert call["accession_number"] == ["0000320193-23-000077"]
        assert call["start_date"] == "2022-01-01"
        assert call["end_date"] == "2024-01-01"
        assert call["max_per_section"] == 2
        assert call["max_per_filing"] == 3
        assert call["rerank_over_fetch_factor"] == 6
        assert call["context_token_budget"] == 4000

    def test_omitted_optional_fields_pass_none(self, search_app_factory) -> None:
        app, service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")

        client.post("/api/search", json={"query": "minimum body"})

        call = service.calls[0]
        # The retrieval service handles ``None`` defaults itself; the
        # route MUST NOT substitute its own values.
        assert call["top_k"] is None
        assert call["min_similarity"] is None
        assert call["ticker"] is None
        assert call["context_token_budget"] is None
        # The diversity caps + over-fetch also default to None so the
        # service resolves them from settings.search rather than the
        # route hardcoding 0/0/4.
        assert call["max_per_section"] is None
        assert call["max_per_filing"] is None
        assert call["rerank_over_fetch_factor"] is None

    def test_response_strips_internal_only_fields(self, search_app_factory) -> None:
        app, _service = search_app_factory(results=[_result(chunk_id="chunk-1", rerank_score=0.91)])
        client = TestClient(app, base_url="https://testserver")

        response = client.post("/api/search", json={"query": "any"})
        assert response.status_code == 200
        hit = response.json()["hits"][0]

        # Wire fields are exactly the documented set — no surprise leaks.
        assert set(hit.keys()) == {
            "chunk_id",
            "content",
            "path",
            "content_type",
            "ticker",
            "form_type",
            "filing_date",
            "accession_number",
            "similarity",
            "rerank_score",
            "token_count",
            "truncated",
            "section_boundaries",
        }
        assert hit["rerank_score"] == 0.91
        assert hit["section_boundaries"] == ["Part I", "Item 1", "Business"]
        assert hit["content_type"] == "text"


# ---------------------------------------------------------------------------
# Schema guards
# ---------------------------------------------------------------------------


class TestSearchSchemaGuards:
    def test_missing_query_rejected(self, search_app_factory) -> None:
        app, service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/search", json={})
        assert response.status_code == 422
        # Service MUST NOT be touched on a schema-rejected request.
        assert service.calls == []

    def test_empty_query_rejected(self, search_app_factory) -> None:
        app, _service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/search", json={"query": ""})
        assert response.status_code == 422

    def test_oversize_query_rejected(self, search_app_factory) -> None:
        app, _service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/search", json={"query": "x" * 2000})
        assert response.status_code == 422

    def test_top_k_upper_bound_rejected(self, search_app_factory) -> None:
        app, _service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/search", json={"query": "any", "top_k": 100})
        assert response.status_code == 422

    def test_top_k_zero_rejected(self, search_app_factory) -> None:
        app, _service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/search", json={"query": "any", "top_k": 0})
        assert response.status_code == 422

    @pytest.mark.security
    def test_diversity_cap_upper_bound_rejected(self, search_app_factory) -> None:
        # A cap above 50 is refused so a request cannot ask the service
        # to retain an unbounded slice per section/filing.
        app, service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")
        for cap_field in ("max_per_section", "max_per_filing"):
            response = client.post("/api/search", json={"query": "any", cap_field: 51})
            assert response.status_code == 422
        assert service.calls == []

    def test_diversity_cap_negative_rejected(self, search_app_factory) -> None:
        app, _service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/search", json={"query": "any", "max_per_section": -1})
        assert response.status_code == 422

    @pytest.mark.security
    def test_rerank_over_fetch_factor_bounds_rejected(self, search_app_factory) -> None:
        # The over-fetch multiplier is bounded 1..10. The upper bound is
        # the candidate-explosion guard: top_k (<=50) * factor (<=10)
        # caps the fetch at 500, so a single request cannot pin Chroma.
        app, service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")
        for value in (0, 11):
            response = client.post(
                "/api/search",
                json={"query": "any", "rerank_over_fetch_factor": value},
            )
            assert response.status_code == 422
        assert service.calls == []

    def test_min_similarity_out_of_range_rejected(self, search_app_factory) -> None:
        app, _service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")
        for value in (-0.5, 1.5):
            response = client.post(
                "/api/search",
                json={"query": "any", "min_similarity": value},
            )
            assert response.status_code == 422

    def test_malformed_iso_date_rejected_at_schema(self, search_app_factory) -> None:
        app, service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")
        # Schema regex rejects "2024/01/01" before the route reaches the
        # service's ``date.fromisoformat`` semantic check.
        response = client.post(
            "/api/search",
            json={"query": "any", "start_date": "2024/01/01"},
        )
        assert response.status_code == 422
        assert service.calls == []

    def test_filter_list_too_long_rejected(self, search_app_factory) -> None:
        app, _service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/search",
            json={"query": "any", "ticker": [f"T{i}" for i in range(60)]},
        )
        assert response.status_code == 422

    def test_empty_filter_list_rejected(self, search_app_factory) -> None:
        app, _service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/search",
            json={"query": "any", "ticker": []},
        )
        assert response.status_code == 422

    def test_unknown_field_rejected(self, search_app_factory) -> None:
        app, service = search_app_factory()
        client = TestClient(app, base_url="https://testserver")
        # ``extra="forbid"`` on the base schema catches typos and
        # client-side drift loudly.
        response = client.post(
            "/api/search",
            json={"query": "any", "depth": 7},
        )
        assert response.status_code == 422
        assert service.calls == []


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestSearchErrorMapping:
    def test_search_error_maps_to_400(self, search_app_factory) -> None:
        app, _service = search_app_factory(
            raise_with=SearchError("Empty retrieval query", details="should not leak to body"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/search", json={"query": "  whitespace  "})
        assert response.status_code == 400
        body = response.json()
        assert body["error"] == "invalid_query"
        # Internal driver detail MUST NOT leak to the client.
        assert "should not leak to body" not in (body.get("hint") or "")
        assert "should not leak to body" not in (body.get("details") or {}).__repr__()

    def test_provider_error_maps_to_502(self, search_app_factory) -> None:
        app, _service = search_app_factory(
            raise_with=ProviderError("upstream embedder broke"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/search", json={"query": "anything"})
        assert response.status_code == 502
        assert response.json()["error"] == "provider_error"

    def test_database_error_maps_to_500(self, search_app_factory) -> None:
        app, _service = search_app_factory(
            raise_with=DatabaseError("disk gone", details="stub-detail"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/search", json={"query": "anything"})
        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "database_error"
        # Same redaction rule as the filings routes.
        assert "stub-detail" not in (body.get("hint") or "")


# ---------------------------------------------------------------------------
# Privacy / no-leak contract
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSearchPrivacyContract:
    def test_response_body_does_not_echo_query(self, search_app_factory) -> None:
        app, _service = search_app_factory(results=[_result()])
        client = TestClient(app, base_url="https://testserver")
        secret_query = "PROPRIETARY-TICKER-WATCHLIST-MNEMONIC-AAA"  # pragma: allowlist secret
        response = client.post("/api/search", json={"query": secret_query})
        # The response MUST NOT carry the raw query — neither in hits
        # (chunk content is independent of the query) nor in metadata.
        assert response.status_code == 200
        assert secret_query not in response.text

    def test_400_envelope_does_not_echo_query(self, search_app_factory) -> None:
        app, _service = search_app_factory(
            raise_with=SearchError("Invalid start_date", details="got '2024/13/99'"),
        )
        client = TestClient(app, base_url="https://testserver")
        secret_query = "PROPRIETARY-TICKER-WATCHLIST-MNEMONIC-AAA"  # pragma: allowlist secret
        response = client.post("/api/search", json={"query": secret_query})
        assert response.status_code == 400
        # Neither the message nor the hint quotes the query content.
        body = response.json()
        assert secret_query not in response.text
        # And the driver-supplied detail string MUST NOT leak either.
        assert "2024/13/99" not in (body.get("hint") or "")

    def test_audit_log_carries_metadata_not_query(
        self,
        search_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        app, _service = search_app_factory(results=[_result()])
        client = TestClient(app, base_url="https://testserver")

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                client.post(
                    "/api/search",
                    json={"query": "PROPRIETARY-TICKER-WATCHLIST"},
                )
        finally:
            package_logger.propagate = prior_propagate

        audit = [r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()]
        assert any("search_executed" in line for line in audit)
        # Audit lines MUST NOT carry the raw query.
        assert all("PROPRIETARY-TICKER-WATCHLIST" not in line for line in audit)


# ---------------------------------------------------------------------------
# Auth tier — search is read-tier
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSearchAuthGate:
    def test_api_key_required_when_configured(self, search_app_factory) -> None:
        app, _service = search_app_factory(
            results=[_result()],
            env={"API_KEY": "shared-team-key"},  # pragma: allowlist secret
        )
        client = TestClient(app, base_url="https://testserver")

        unauthed = client.post("/api/search", json={"query": "any"})
        assert unauthed.status_code == 401
        assert unauthed.json()["error"] == "unauthorised"

        ok = client.post(
            "/api/search",
            json={"query": "any"},
            headers={"X-API-Key": "shared-team-key"},  # pragma: allowlist secret
        )
        assert ok.status_code == 200

    def test_admin_key_not_required(self, search_app_factory) -> None:
        # Search is read-tier — a leaked-but-not-elevated API key MUST
        # still reach the route.  Without this, ``API_ADMIN_KEY`` would
        # silently become a global gate.
        app, _service = search_app_factory(
            results=[_result()],
            env={
                "API_KEY": "shared-team-key",  # pragma: allowlist secret
                "API_ADMIN_KEY": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/search",
            json={"query": "any"},
            headers={"X-API-Key": "shared-team-key"},  # pragma: allowlist secret
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Rate-limit classification
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSearchRateLimitClassification:
    def test_search_path_classifies_as_search(self) -> None:
        from sec_generative_search.api.middleware import _classify_path

        assert _classify_path("/api/search", "POST") == "search"

    def test_search_respects_per_ip_window(self, search_app_factory) -> None:
        # Tight per-IP limit — fourth request inside the same minute
        # MUST be rejected with 429.
        app, _service = search_app_factory(
            results=[_result()],
            env={"API_RATE_LIMIT_SEARCH": "3"},
        )
        client = TestClient(app, base_url="https://testserver")
        statuses: list[int] = []
        for _ in range(6):
            r = client.post("/api/search", json={"query": "any"})
            statuses.append(r.status_code)
        assert statuses.count(200) == 3
        assert 429 in statuses


# ---------------------------------------------------------------------------
# Error mapping through the REAL RetrievalService  (M1 regression lock)
# ---------------------------------------------------------------------------


class _Vector:
    """Minimal stand-in for the 1-D ``np.ndarray`` an embedder returns."""

    @staticmethod
    def tolist() -> list[float]:
        return [0.1, 0.2, 0.3]


@dataclass
class _FakeEmbedder:
    """Embedder double: returns a vector or raises the supplied error."""

    raise_with: Exception | None = None

    def embed_query(self, query: str) -> _Vector:
        if self.raise_with is not None:
            raise self.raise_with
        return _Vector()


@dataclass
class _FakeChroma:
    """ChromaDB client double: returns hits or raises the supplied error."""

    raise_with: Exception | None = None

    def query(self, **kwargs: Any) -> list[Any]:
        if self.raise_with is not None:
            raise self.raise_with
        return []


@pytest.mark.security
class TestRealRetrievalServiceErrorMapping:
    """Drive the production :class:`RetrievalService` end-to-end.

    :class:`TestSearchErrorMapping` above raises the typed error from a
    *stub* ``retrieve``, so it never exercises ``_fetch_candidates`` —
    the very method that used to re-wrap ``DatabaseError`` /
    ``ProviderError`` into ``SearchError``.  These cases wire the real
    service over fake embedder / Chroma doubles so the wrapping
    behaviour is what is under test.
    """

    @staticmethod
    def _app_with_real_service(
        search_app_factory,
        *,
        embedder_raises: Exception | None = None,
        chroma_raises: Exception | None = None,
    ):
        app, _stub = search_app_factory()
        app.state.retrieval_service = RetrievalService(
            embedder=_FakeEmbedder(raise_with=embedder_raises),
            chroma_client=_FakeChroma(raise_with=chroma_raises),
            token_counter=len,
        )
        return app

    def test_embedder_provider_error_reaches_the_502_arm(self, search_app_factory) -> None:
        app = self._app_with_real_service(
            search_app_factory,
            embedder_raises=ProviderError(
                "embedder upstream failed",
                details="https://embed.internal.example/v1 returned 500",
            ),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/search", json={"query": "revenue concentration"})

        assert response.status_code == 502
        body = response.json()
        assert body["error"] == "provider_error"
        # Upstream hostnames / driver strings never reach the caller.
        assert "embed.internal.example" not in response.text

    def test_storage_database_error_reaches_the_500_arm(self, search_app_factory) -> None:
        app = self._app_with_real_service(
            search_app_factory,
            chroma_raises=DatabaseError(
                "collection read failed",
                details="sqlite3.OperationalError: database is locked at /app/data/metadata.sqlite",
            ),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/search", json={"query": "revenue concentration"})

        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "database_error"
        # The SQLite path and driver text are operator-log-only.
        assert "/app/data/metadata.sqlite" not in response.text
        assert "OperationalError" not in response.text

    def test_unexpected_error_still_collapses_to_400(self, search_app_factory) -> None:
        # A genuinely untyped failure keeps the historical SearchError
        # category — the narrowing must not swallow the catch-all.
        app = self._app_with_real_service(
            search_app_factory,
            chroma_raises=RuntimeError("unexpected driver state 0xdeadbeef"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/search", json={"query": "revenue concentration"})

        assert response.status_code == 400
        assert response.json()["error"] == "invalid_query"
        # ...but the driver string must not ride the envelope out.
        assert "0xdeadbeef" not in response.text

    def test_caller_fault_search_error_body_is_a_fixed_message(self, search_app_factory) -> None:
        # Whitespace-only query clears the schema guard (min_length=1) and
        # is rejected by the service.  The 400 body carries a fixed
        # message — never ``SearchError.details``.
        app = self._app_with_real_service(search_app_factory)
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/search", json={"query": "   "})

        assert response.status_code == 400
        body = response.json()
        assert body["error"] == "invalid_query"
        assert "whitespace-only" not in response.text
        assert body["message"] == "The search query could not be processed."
