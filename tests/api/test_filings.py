"""Tests for the filing management routes.

Strategy
--------

The route surface is exercised via the ``api_client`` test fixture and
runs against in-process stubs for :class:`MetadataRegistry` and
:class:`FilingStore` attached to ``app.state``.  Real SQLite / ChromaDB
construction is the lifespan's job; these tests focus on the route
contract:

    - schema guards (path / query / body validators)
    - read vs destructive auth tiers
    - 404 / 400 / 500 error envelopes
    - audit-log emission for every destructive call
    - ChromaDB-first ordering left to :class:`FilingStore` — the routes
      delegate, so we assert the delegation, not the ordering itself.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.api.app import create_app
from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.exceptions import DatabaseError
from sec_generative_search.core.logging import LOGGER_NAME
from sec_generative_search.database import FilingRecord

# ---------------------------------------------------------------------------
# In-process stubs for the storage seam
# ---------------------------------------------------------------------------


@dataclass
class _StubRegistry:
    """Minimal stand-in for :class:`MetadataRegistry`.

    Implements only the methods the filings routes call.  Behaviour is
    overridable per-test via the public attributes (``records``,
    ``raise_on_get``, ``raise_on_list``).
    """

    records: list[FilingRecord]
    raise_on_get: bool = False
    raise_on_list: bool = False

    def get_filing(self, accession_number: str) -> FilingRecord | None:
        if self.raise_on_get:
            raise DatabaseError("simulated", details="stub")
        for r in self.records:
            if r.accession_number == accession_number:
                return r
        return None

    def get_filings_by_accessions(self, accession_numbers: list[str]) -> list[FilingRecord]:
        wanted = set(accession_numbers)
        return [r for r in self.records if r.accession_number in wanted]

    def list_filings(
        self,
        ticker: str | None = None,
        form_type: str | None = None,
        *,
        sort_by: str = "filing_date",
        order: str = "desc",
        limit: int | None = None,
        offset: int = 0,
    ) -> list[FilingRecord]:
        if self.raise_on_list:
            raise DatabaseError("simulated", details="stub")
        out = list(self.records)
        if ticker:
            out = [r for r in out if r.ticker == ticker]
        if form_type:
            out = [r for r in out if r.form_type == form_type]
        # Mirror the real registry: sort in SQL-equivalent order with a
        # deterministic ``id`` tie-breaker, then apply limit/offset.
        reverse = order == "desc"
        out.sort(key=lambda r: r.id, reverse=reverse)
        out.sort(key=lambda r: getattr(r, sort_by), reverse=reverse)
        if offset:
            out = out[offset:]
        if limit is not None:
            out = out[:limit]
        return out


@dataclass
class _StubStore:
    """Minimal stand-in for :class:`FilingStore`.

    Records each call so tests can assert delegation occurred without
    inspecting ChromaDB / SQLite directly.
    """

    deleted_single: list[str]
    batch_calls: list[list[str]]
    cleared_calls: int = 0
    raise_on_delete: bool = False
    raise_on_clear: bool = False
    cleared_chunks: int = 0
    cleared_filings: int = 0

    def delete_filing(self, accession_number: str) -> bool:
        if self.raise_on_delete:
            raise DatabaseError("simulated", details="stub")
        self.deleted_single.append(accession_number)
        return True

    def delete_filings_batch(self, accession_numbers: list[str]) -> int:
        if self.raise_on_delete:
            raise DatabaseError("simulated", details="stub")
        self.batch_calls.append(list(accession_numbers))
        return len(accession_numbers)

    def clear_all(self) -> tuple[int, int]:
        if self.raise_on_clear:
            raise DatabaseError("simulated", details="stub")
        self.cleared_calls += 1
        return self.cleared_chunks, self.cleared_filings


def _record(
    *,
    accession: str = "0000320193-23-000077",
    ticker: str = "AAPL",
    form_type: str = "10-K",
    filing_date: str = "2023-09-30",
    chunks: int = 12,
    ingested_at: str = "2024-01-01T00:00:00Z",
    record_id: int = 1,
) -> FilingRecord:
    return FilingRecord(
        id=record_id,
        ticker=ticker,
        form_type=form_type,
        filing_date=filing_date,
        accession_number=accession,
        chunk_count=chunks,
        ingested_at=ingested_at,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def filings_app_factory(monkeypatch: pytest.MonkeyPatch):
    """Build a fresh app with stub storage attached to ``app.state``."""

    def factory(
        *,
        records: list[FilingRecord] | None = None,
        raise_on_get: bool = False,
        raise_on_list: bool = False,
        raise_on_delete: bool = False,
        raise_on_clear: bool = False,
        cleared_chunks: int = 0,
        cleared_filings: int = 0,
        env: dict[str, str] | None = None,
    ) -> tuple[Any, _StubRegistry, _StubStore]:
        if env:
            for key, value in env.items():
                monkeypatch.setenv(key, value)
        reload_settings()

        app = create_app()
        registry = _StubRegistry(
            records=list(records or []),
            raise_on_get=raise_on_get,
            raise_on_list=raise_on_list,
        )
        store = _StubStore(
            deleted_single=[],
            batch_calls=[],
            raise_on_delete=raise_on_delete,
            raise_on_clear=raise_on_clear,
            cleared_chunks=cleared_chunks,
            cleared_filings=cleared_filings,
        )
        app.state.registry = registry
        app.state.filing_store = store
        app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
        app.state.encrypted_credential_store = None
        return app, registry, store

    return factory


@pytest.fixture(autouse=True)
def _clean_filings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip API_* env vars so each test sees a clean baseline."""
    for key in list(os.environ.keys()):
        if key.startswith("API_"):
            monkeypatch.delenv(key, raising=False)
    reload_settings()
    yield
    reload_settings()


# ---------------------------------------------------------------------------
# Read-tier tests
# ---------------------------------------------------------------------------


class TestListFilings:
    def test_returns_empty_when_registry_empty(self, filings_app_factory) -> None:
        app, _registry, _store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.get("/api/filings/")
        assert response.status_code == 200
        assert response.json() == {"filings": [], "total": 0}

    def test_returns_records_default_sort(self, filings_app_factory) -> None:
        records = [
            _record(
                accession="0000000001-23-000001",
                filing_date="2022-01-01",
                record_id=1,
            ),
            _record(
                accession="0000000002-23-000002",
                filing_date="2024-01-01",
                record_id=2,
            ),
            _record(
                accession="0000000003-23-000003",
                filing_date="2023-01-01",
                record_id=3,
            ),
        ]
        app, _registry, _store = filings_app_factory(records=records)
        client = TestClient(app, base_url="https://testserver")
        response = client.get("/api/filings/")
        assert response.status_code == 200
        body = response.json()
        # Default sort is filing_date DESC.
        dates = [f["filing_date"] for f in body["filings"]]
        assert dates == ["2024-01-01", "2023-01-01", "2022-01-01"]
        assert body["total"] == 3
        # ``id`` MUST NOT appear on the wire — the schema drops it.
        assert all("id" not in f for f in body["filings"])

    def test_filter_by_ticker_uppercases(self, filings_app_factory) -> None:
        records = [
            _record(ticker="AAPL", accession="0000000001-23-000001"),
            _record(ticker="MSFT", accession="0000000002-23-000002"),
        ]
        app, _registry, _store = filings_app_factory(records=records)
        client = TestClient(app, base_url="https://testserver")
        response = client.get("/api/filings/?ticker=aapl")
        assert response.status_code == 200
        tickers = [f["ticker"] for f in response.json()["filings"]]
        assert tickers == ["AAPL"]

    def test_unknown_sort_is_rejected(self, filings_app_factory) -> None:
        app, _registry, _store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.get("/api/filings/?sort_by=invalid")
        assert response.status_code == 422

    def _paging_records(self) -> list[FilingRecord]:
        # Five filings with descending-distinct dates so the default sort
        # order is unambiguous for the paging assertions.
        return [
            _record(
                accession=f"000000000{i}-23-00000{i}",
                filing_date=f"2024-01-0{5 - i}",
                record_id=i + 1,
            )
            for i in range(5)
        ]

    def test_limit_returns_single_page(self, filings_app_factory) -> None:
        app, _registry, _store = filings_app_factory(records=self._paging_records())
        client = TestClient(app, base_url="https://testserver")
        response = client.get("/api/filings/?limit=2")
        assert response.status_code == 200
        body = response.json()
        assert len(body["filings"]) == 2
        # ``total`` is the count of returned rows (page size), per contract.
        assert body["total"] == 2
        dates = [f["filing_date"] for f in body["filings"]]
        assert dates == ["2024-01-05", "2024-01-04"]

    def test_offset_returns_next_page(self, filings_app_factory) -> None:
        app, _registry, _store = filings_app_factory(records=self._paging_records())
        client = TestClient(app, base_url="https://testserver")
        page1 = client.get("/api/filings/?limit=2&offset=0").json()["filings"]
        page2 = client.get("/api/filings/?limit=2&offset=2").json()["filings"]
        # Non-overlapping windows.
        a1 = {f["accession_number"] for f in page1}
        a2 = {f["accession_number"] for f in page2}
        assert a1.isdisjoint(a2)
        assert [f["filing_date"] for f in page2] == ["2024-01-03", "2024-01-02"]

    def test_omitting_limit_returns_full_list_unchanged(self, filings_app_factory) -> None:
        # Backward-compatibility guard: the pre-pagination SPA sends no
        # ``limit`` and must still receive every row.
        app, _registry, _store = filings_app_factory(records=self._paging_records())
        client = TestClient(app, base_url="https://testserver")
        body = client.get("/api/filings/").json()
        assert body["total"] == 5
        assert len(body["filings"]) == 5

    def test_limit_zero_is_rejected(self, filings_app_factory) -> None:
        app, _registry, _store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        assert client.get("/api/filings/?limit=0").status_code == 422

    def test_negative_offset_is_rejected(self, filings_app_factory) -> None:
        app, _registry, _store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        assert client.get("/api/filings/?offset=-1").status_code == 422

    def test_oversize_limit_is_rejected(self, filings_app_factory) -> None:
        app, _registry, _store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        assert client.get("/api/filings/?limit=10001").status_code == 422

    def test_database_error_returns_500_envelope(self, filings_app_factory) -> None:
        app, _registry, _store = filings_app_factory(raise_on_list=True)
        client = TestClient(app, base_url="https://testserver")
        response = client.get("/api/filings/")
        assert response.status_code == 500
        body = response.json()
        assert body["error"] == "database_error"
        # Internal driver detail MUST NOT leak to the client.
        assert "stub" not in (body.get("hint") or "")
        assert body["details"] is None


class TestGetFiling:
    def test_returns_record_when_found(self, filings_app_factory) -> None:
        records = [_record(accession="0000320193-23-000077")]
        app, _registry, _store = filings_app_factory(records=records)
        client = TestClient(app, base_url="https://testserver")
        response = client.get("/api/filings/0000320193-23-000077")
        assert response.status_code == 200
        assert response.json()["accession_number"] == "0000320193-23-000077"

    def test_returns_404_envelope_when_missing(self, filings_app_factory) -> None:
        app, _registry, _store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.get("/api/filings/0000000001-23-000001")
        assert response.status_code == 404
        body = response.json()
        assert body["error"] == "not_found"
        assert "GET /api/filings/" in body["hint"]

    def test_rejects_malformed_accession(self, filings_app_factory) -> None:
        app, _registry, _store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        # Path-bound regex catches malformed accession numbers as 422.
        response = client.get("/api/filings/not-an-accession")
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Destructive-tier tests
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestDeleteFilingAuthGate:
    """The destructive routes MUST require both API_KEY and API_ADMIN_KEY."""

    def test_blocked_at_read_tier_when_only_api_key_supplied(self, filings_app_factory) -> None:
        records = [_record(accession="0000320193-23-000077")]
        app, _registry, store = filings_app_factory(
            records=records,
            env={
                "API_KEY": "shared-team-key",  # pragma: allowlist secret
                "API_ADMIN_KEY": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        client = TestClient(app, base_url="https://testserver")

        # No headers — read tier rejects first (401, not 403).
        response = client.delete("/api/filings/0000320193-23-000077")
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"
        assert store.deleted_single == []

        # API key only — admin tier rejects.
        response = client.delete(
            "/api/filings/0000320193-23-000077",
            headers={"X-API-Key": "shared-team-key"},  # pragma: allowlist secret
        )
        assert response.status_code == 403
        assert response.json()["error"] == "admin_required"
        assert store.deleted_single == []

    def test_admin_key_only_still_blocked_at_read_tier(self, filings_app_factory) -> None:
        records = [_record(accession="0000320193-23-000077")]
        app, _registry, _store = filings_app_factory(
            records=records,
            env={
                "API_KEY": "shared-team-key",  # pragma: allowlist secret
                "API_ADMIN_KEY": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        client = TestClient(app, base_url="https://testserver")
        # X-Admin-Key alone surfaces 401 (not 403) — order is load-bearing.
        response = client.delete(
            "/api/filings/0000320193-23-000077",
            headers={
                "X-Admin-Key": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"

    def test_both_headers_pass_through(self, filings_app_factory) -> None:
        records = [_record(accession="0000320193-23-000077")]
        app, _registry, store = filings_app_factory(
            records=records,
            env={
                "API_KEY": "shared-team-key",  # pragma: allowlist secret
                "API_ADMIN_KEY": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.delete(
            "/api/filings/0000320193-23-000077",
            headers={
                "X-API-Key": "shared-team-key",  # pragma: allowlist secret
                "X-Admin-Key": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        assert response.status_code == 200
        assert store.deleted_single == ["0000320193-23-000077"]


class TestDeleteFiling:
    def test_returns_404_when_not_in_registry(self, filings_app_factory) -> None:
        app, _registry, store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.delete("/api/filings/0000000001-23-000001")
        assert response.status_code == 404
        # The store MUST NOT be touched on a 404 — pre-check pays off.
        assert store.deleted_single == []

    def test_deletes_through_filing_store(self, filings_app_factory) -> None:
        records = [_record(accession="0000320193-23-000077", chunks=42)]
        app, _registry, store = filings_app_factory(records=records)
        client = TestClient(app, base_url="https://testserver")
        response = client.delete("/api/filings/0000320193-23-000077")
        assert response.status_code == 200
        assert response.json() == {
            "accession_number": "0000320193-23-000077",
            "chunks_deleted": 42,
        }
        # Routes must delegate to FilingStore; never touch chroma /
        # registry directly.
        assert store.deleted_single == ["0000320193-23-000077"]

    def test_database_error_during_delete_returns_500(self, filings_app_factory) -> None:
        records = [_record(accession="0000320193-23-000077")]
        app, _registry, _store = filings_app_factory(
            records=records,
            raise_on_delete=True,
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.delete("/api/filings/0000320193-23-000077")
        assert response.status_code == 500
        assert response.json()["error"] == "database_error"

    @pytest.mark.security
    def test_emits_security_audit_with_metadata(
        self,
        filings_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        records = [
            _record(
                accession="0000320193-23-000077",
                ticker="AAPL",
                form_type="10-K",
                chunks=42,
            )
        ]
        app, _registry, _store = filings_app_factory(records=records)
        client = TestClient(app, base_url="https://testserver")

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                client.delete("/api/filings/0000320193-23-000077")
        finally:
            package_logger.propagate = prior_propagate

        audit_lines = [
            r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()
        ]
        assert any("delete_filing" in line for line in audit_lines)
        assert any("accession=0000320193-23-000077" in line for line in audit_lines)
        assert any("chunks=42" in line for line in audit_lines)


class TestDeleteByIds:
    def test_deletes_known_reports_unknown(self, filings_app_factory) -> None:
        records = [
            _record(accession="0000000001-23-000001", chunks=10),
            _record(accession="0000000002-23-000002", chunks=20),
        ]
        app, _registry, store = filings_app_factory(records=records)
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/filings/delete-by-ids",
            json={
                "accession_numbers": [
                    "0000000001-23-000001",
                    "0000000002-23-000002",
                    "0000000099-23-000099",  # unknown
                ]
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["filings_deleted"] == 2
        assert body["chunks_deleted"] == 30
        assert body["not_found"] == ["0000000099-23-000099"]
        # Single batch call against FilingStore.
        assert len(store.batch_calls) == 1
        assert sorted(store.batch_calls[0]) == [
            "0000000001-23-000001",
            "0000000002-23-000002",
        ]

    def test_no_matches_short_circuits_without_store_call(self, filings_app_factory) -> None:
        app, _registry, store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/filings/delete-by-ids",
            json={"accession_numbers": ["0000000001-23-000001"]},
        )
        assert response.status_code == 200
        assert response.json()["filings_deleted"] == 0
        assert store.batch_calls == []

    def test_malformed_accession_rejected_by_schema(self, filings_app_factory) -> None:
        app, _registry, _store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/filings/delete-by-ids",
            json={"accession_numbers": ["nope"]},
        )
        assert response.status_code == 422

    def test_oversize_batch_rejected(self, filings_app_factory) -> None:
        app, _registry, _store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        # 501 well-formed accessions (> 500 cap).
        accessions = [f"{n:010d}-23-000001" for n in range(501)]
        response = client.post(
            "/api/filings/delete-by-ids",
            json={"accession_numbers": accessions},
        )
        assert response.status_code == 422

    def test_empty_batch_rejected(self, filings_app_factory) -> None:
        app, _registry, _store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/filings/delete-by-ids",
            json={"accession_numbers": []},
        )
        assert response.status_code == 422

    def test_duplicates_collapsed(self, filings_app_factory) -> None:
        records = [_record(accession="0000000001-23-000001", chunks=5)]
        app, _registry, store = filings_app_factory(records=records)
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/filings/delete-by-ids",
            json={
                "accession_numbers": [
                    "0000000001-23-000001",
                    "0000000001-23-000001",
                ]
            },
        )
        assert response.status_code == 200
        # The duplicate is collapsed before the batch hits FilingStore.
        assert len(store.batch_calls) == 1
        assert store.batch_calls[0] == ["0000000001-23-000001"]


class TestBulkDelete:
    def test_requires_at_least_one_filter(self, filings_app_factory) -> None:
        app, _registry, store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/filings/bulk-delete", json={})
        assert response.status_code == 400
        body = response.json()
        assert body["error"] == "validation_error"
        # Hint MUST mention the confirm-gated full wipe alternative.
        assert "?confirm=true" in body["hint"]
        assert store.batch_calls == []

    def test_no_matches_returns_zeros(self, filings_app_factory) -> None:
        app, _registry, store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/filings/bulk-delete",
            json={"ticker": "AAPL"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body == {
            "filings_deleted": 0,
            "chunks_deleted": 0,
            "tickers_affected": [],
        }
        assert store.batch_calls == []

    def test_matching_filings_deleted_and_audited(
        self,
        filings_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        records = [
            _record(ticker="AAPL", accession="0000000001-23-000001", chunks=5),
            _record(ticker="AAPL", accession="0000000002-23-000002", chunks=7),
            _record(ticker="MSFT", accession="0000000003-23-000003", chunks=11),
        ]
        app, _registry, store = filings_app_factory(records=records)
        client = TestClient(app, base_url="https://testserver")

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                response = client.post(
                    "/api/filings/bulk-delete",
                    json={"ticker": "AAPL"},
                )
        finally:
            package_logger.propagate = prior_propagate

        assert response.status_code == 200
        body = response.json()
        assert body["filings_deleted"] == 2
        assert body["chunks_deleted"] == 12
        assert body["tickers_affected"] == ["AAPL"]
        # Exactly one batch call hit FilingStore.
        assert len(store.batch_calls) == 1
        assert sorted(store.batch_calls[0]) == [
            "0000000001-23-000001",
            "0000000002-23-000002",
        ]
        audit = [r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()]
        assert any("bulk_delete" in line for line in audit)


class TestClearAll:
    def test_requires_confirm_flag(self, filings_app_factory) -> None:
        app, _registry, store = filings_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.delete("/api/filings/")
        assert response.status_code == 400
        body = response.json()
        assert body["error"] == "confirmation_required"
        assert "?confirm=true" in body["hint"]
        assert store.cleared_calls == 0

    @pytest.mark.security
    def test_demo_mode_blocks_even_with_confirm(self, filings_app_factory) -> None:
        """Demo mode (Scenario C) disables operator clear-all unconditionally.

        On a public demo the corpus is reset by the nightly Cloud Run
        ``Job`` (which drives the operator-only CLI, bypassing demo
        mode). A human-driven ``DELETE /api/filings/?confirm=true`` MUST
        be refused with ``403 demo_mode`` even with the confirm flag, so
        a leaked admin key cannot wipe the shared demo corpus out of band.
        """
        app, _registry, store = filings_app_factory(env={"API_DEMO_MODE": "true"})
        client = TestClient(app, base_url="https://testserver")
        response = client.delete("/api/filings/?confirm=true")
        assert response.status_code == 403
        body = response.json()
        assert body["error"] == "demo_mode"
        # The store is never reached — the gate fires before any write.
        assert store.cleared_calls == 0

    def test_clears_through_filing_store(self, filings_app_factory) -> None:
        app, _registry, store = filings_app_factory(cleared_chunks=99, cleared_filings=7)
        client = TestClient(app, base_url="https://testserver")
        response = client.delete("/api/filings/?confirm=true")
        assert response.status_code == 200
        assert response.json() == {"filings_deleted": 7, "chunks_deleted": 99}
        assert store.cleared_calls == 1

    @pytest.mark.security
    def test_emits_security_audit(
        self,
        filings_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        app, _registry, _store = filings_app_factory(cleared_chunks=12, cleared_filings=3)
        client = TestClient(app, base_url="https://testserver")

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                client.delete("/api/filings/?confirm=true")
        finally:
            package_logger.propagate = prior_propagate

        audit = [r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()]
        assert any("clear_all" in line for line in audit)


# ---------------------------------------------------------------------------
# Rate-limit classification (the destructive POSTs inherit the delete bucket)
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestFilingsRateLimitClassification:
    """The batch-delete POSTs MUST share the destructive rate-limit bucket.

    Without this, an attacker could side-step the per-IP DELETE limit by
    switching from ``DELETE /{accession}`` to ``POST /delete-by-ids``.
    """

    def test_post_delete_by_ids_classifies_as_delete(self) -> None:
        from sec_generative_search.api.middleware import _rate_category_for_scope

        assert (
            _rate_category_for_scope(
                {"type": "http", "path": "/api/filings/delete-by-ids", "method": "POST"}
            )
            == "delete"
        )

    def test_post_bulk_delete_classifies_as_delete(self) -> None:
        from sec_generative_search.api.middleware import _rate_category_for_scope

        assert (
            _rate_category_for_scope(
                {"type": "http", "path": "/api/filings/bulk-delete", "method": "POST"}
            )
            == "delete"
        )

    def test_get_listing_classifies_as_general(self) -> None:
        from sec_generative_search.api.middleware import _rate_category_for_scope

        assert (
            _rate_category_for_scope({"type": "http", "path": "/api/filings/", "method": "GET"})
            == "general"
        )

    def test_delete_route_classifies_as_delete(self) -> None:
        from sec_generative_search.api.middleware import _rate_category_for_scope

        assert (
            _rate_category_for_scope(
                {"type": "http", "path": "/api/filings/0000320193-23-000077", "method": "DELETE"}
            )
            == "delete"
        )

    def test_destructive_post_respects_delete_rate_limit(self, filings_app_factory) -> None:
        records = [_record(accession="0000000001-23-000001")]
        app, _registry, _store = filings_app_factory(
            records=records,
            env={"API_RATE_LIMIT_DELETE": "2"},
        )
        client = TestClient(app, base_url="https://testserver")

        statuses: list[int] = []
        for _ in range(5):
            response = client.post(
                "/api/filings/delete-by-ids",
                json={"accession_numbers": ["0000000001-23-000001"]},
            )
            statuses.append(response.status_code)
        # The destructive bucket caps the burst at 2 successes.
        assert 429 in statuses
