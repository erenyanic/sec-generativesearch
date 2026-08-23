"""Tests for :class:`sec_generative_search.database.MetadataRegistry`.

Covers the full CRUD surface, SQL-level aggregate statistics, FIFO
eviction helpers, filing-limit enforcement, task-history persistence
with the redaction controls required for team/cloud deployments, the
concurrency guarantees of ``register_filing_if_new``, and the
SQLCipher-key handling path.

The tests drive a real SQLite database under ``tmp_path`` —
``MetadataRegistry`` is thin enough over the driver that mocking would
hide the very contracts (parameter binding, WAL pragma, integrity
errors) we care about.
"""

from __future__ import annotations

import io
import logging
import sqlite3
import threading
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from sec_generative_search.core.exceptions import (
    DatabaseError,
    FilingLimitExceededError,
)
from sec_generative_search.core.logging import LOGGER_NAME, AccessionRedactionFilter
from sec_generative_search.core.types import FilingIdentifier
from sec_generative_search.database import (
    DatabaseStatistics,
    FilingRecord,
    MetadataRegistry,
    TickerStatistics,
)
from sec_generative_search.database.metadata import _scrub_error_message

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_db_path: str) -> MetadataRegistry:
    """Registry backed by the per-test SQLite file."""
    return MetadataRegistry(db_path=tmp_db_path)


@pytest.fixture
def stored_filing(
    registry: MetadataRegistry,
    sample_filing_id: FilingIdentifier,
) -> FilingIdentifier:
    """Register ``sample_filing_id`` with 42 chunks and return the identifier."""
    registry.register_filing(sample_filing_id, chunk_count=42)
    return sample_filing_id


# ---------------------------------------------------------------------------
# Registration and duplicate detection
# ---------------------------------------------------------------------------


class TestRegisterFiling:
    def test_register_and_count(
        self,
        registry: MetadataRegistry,
        sample_filing_id: FilingIdentifier,
    ) -> None:
        registry.register_filing(sample_filing_id, chunk_count=59)
        assert registry.count() == 1

    def test_register_duplicate_raises(
        self,
        registry: MetadataRegistry,
        sample_filing_id: FilingIdentifier,
    ) -> None:
        registry.register_filing(sample_filing_id, chunk_count=5)
        with pytest.raises(DatabaseError, match="already exists"):
            registry.register_filing(sample_filing_id, chunk_count=5)


class TestIsDuplicate:
    def test_true_for_registered(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        assert registry.is_duplicate(stored_filing.accession_number) is True

    def test_false_for_absent(self, registry: MetadataRegistry) -> None:
        assert registry.is_duplicate("missing-accession") is False


class TestRegisterFilingIfNew:
    """``register_filing_if_new`` atomically checks and inserts."""

    def test_new_filing_registered(
        self,
        registry: MetadataRegistry,
        sample_filing_id: FilingIdentifier,
    ) -> None:
        assert registry.register_filing_if_new(sample_filing_id, chunk_count=42) is True
        assert registry.count() == 1

    def test_existing_filing_returns_false(
        self,
        registry: MetadataRegistry,
        sample_filing_id: FilingIdentifier,
    ) -> None:
        registry.register_filing(sample_filing_id, chunk_count=42)
        assert registry.register_filing_if_new(sample_filing_id, chunk_count=42) is False
        assert registry.count() == 1

    def test_atomic_under_contention(self, registry: MetadataRegistry) -> None:
        """Only one thread should succeed registering the same filing."""
        fid = FilingIdentifier(
            ticker="AAPL",
            form_type="10-K",
            filing_date=date(2024, 1, 1),
            accession_number="ACC-RACE",
        )
        results: list[bool] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(10)

        def try_register() -> None:
            try:
                barrier.wait()
                results.append(registry.register_filing_if_new(fid, chunk_count=10))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=try_register) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert results.count(True) == 1
        assert results.count(False) == 9
        assert registry.count() == 1

    def test_distinct_filings_all_succeed(self, registry: MetadataRegistry) -> None:
        errors: list[BaseException] = []

        def register(i: int) -> None:
            try:
                fid = FilingIdentifier(
                    ticker="AAPL",
                    form_type="10-K",
                    filing_date=date(2020 + i, 1, 1),
                    accession_number=f"ACC-CONC-{i}",
                )
                registry.register_filing_if_new(fid, chunk_count=i)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert registry.count() == 10


class TestGetExistingAccessions:
    def test_empty_input_short_circuits(self, registry: MetadataRegistry) -> None:
        assert registry.get_existing_accessions([]) == set()

    def test_none_match(self, registry: MetadataRegistry) -> None:
        assert registry.get_existing_accessions(["X", "Y"]) == set()

    def test_partial_match(self, registry: MetadataRegistry) -> None:
        fid1 = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-1")
        fid2 = FilingIdentifier("MSFT", "10-K", date(2024, 3, 1), "ACC-2")
        registry.register_filing(fid1, chunk_count=10)
        registry.register_filing(fid2, chunk_count=20)

        result = registry.get_existing_accessions(["ACC-1", "ACC-MISSING", "ACC-2"])
        assert result == {"ACC-1", "ACC-2"}


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


class TestGetFiling:
    def test_found(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        record = registry.get_filing(stored_filing.accession_number)
        assert record is not None
        assert isinstance(record, FilingRecord)
        assert record.ticker == "AAPL"
        assert record.form_type == "10-K"
        assert record.chunk_count == 42
        assert record.filing_date == "2023-11-03"

    def test_not_found(self, registry: MetadataRegistry) -> None:
        assert registry.get_filing("nope") is None


class TestGetFilingsByAccessions:
    def test_empty_short_circuits(self, registry: MetadataRegistry) -> None:
        assert registry.get_filings_by_accessions([]) == []

    def test_returns_known_only(self, registry: MetadataRegistry) -> None:
        fid = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-KNOWN")
        registry.register_filing(fid, chunk_count=3)
        records = registry.get_filings_by_accessions(["ACC-KNOWN", "ACC-MISSING"])
        assert {r.accession_number for r in records} == {"ACC-KNOWN"}


class TestListFilings:
    def test_descending_by_filing_date(self, registry: MetadataRegistry) -> None:
        fid_old = FilingIdentifier("AAPL", "10-K", date(2022, 1, 1), "ACC-OLD")
        fid_new = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-NEW")
        registry.register_filing(fid_old, chunk_count=1)
        registry.register_filing(fid_new, chunk_count=2)

        filings = registry.list_filings()
        assert filings[0].accession_number == "ACC-NEW"
        assert filings[1].accession_number == "ACC-OLD"

    def test_ticker_filter_case_insensitive(self, registry: MetadataRegistry) -> None:
        fid1 = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-A")
        fid2 = FilingIdentifier("MSFT", "10-K", date(2024, 1, 1), "ACC-M")
        registry.register_filing(fid1, chunk_count=1)
        registry.register_filing(fid2, chunk_count=1)

        result = registry.list_filings(ticker="aapl")
        assert [r.accession_number for r in result] == ["ACC-A"]

    def test_combined_filters(self, registry: MetadataRegistry) -> None:
        fid1 = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-1")
        fid2 = FilingIdentifier("AAPL", "10-Q", date(2024, 6, 1), "ACC-2")
        registry.register_filing(fid1, chunk_count=1)
        registry.register_filing(fid2, chunk_count=2)

        result = registry.list_filings(ticker="AAPL", form_type="10-K")
        assert [r.accession_number for r in result] == ["ACC-1"]

    def test_limit_bounds_returned_rows(self, registry: MetadataRegistry) -> None:
        for i in range(5):
            fid = FilingIdentifier("AAPL", "10-K", date(2024, 1, i + 1), f"ACC-{i}")
            registry.register_filing(fid, chunk_count=1)

        page = registry.list_filings(limit=2)
        assert len(page) == 2
        # Newest first (filing_date DESC): 2024-01-05, 2024-01-04.
        assert [r.filing_date for r in page] == ["2024-01-05", "2024-01-04"]

    def test_offset_walks_pages_without_overlap_or_gap(self, registry: MetadataRegistry) -> None:
        for i in range(5):
            fid = FilingIdentifier("AAPL", "10-K", date(2024, 1, i + 1), f"ACC-{i}")
            registry.register_filing(fid, chunk_count=1)

        full = registry.list_filings()
        paged: list[str] = []
        for offset in (0, 2, 4):
            paged.extend(r.accession_number for r in registry.list_filings(limit=2, offset=offset))
        # Walking the corpus two rows at a time reproduces the full order
        # exactly — no row skipped, none duplicated.
        assert paged == [r.accession_number for r in full]

    def test_id_tiebreaker_keeps_pagination_stable_on_equal_filing_date(
        self, registry: MetadataRegistry
    ) -> None:
        # Three filings share one filing_date (distinct ticker/accession),
        # so ordering must fall through to the deterministic ``id``
        # tie-breaker rather than an arbitrary storage order.
        for i, ticker in enumerate(("AAA", "BBB", "CCC")):
            fid = FilingIdentifier(ticker, "10-K", date(2024, 1, 1), f"ACC-{i}")
            registry.register_filing(fid, chunk_count=1)

        full = [r.accession_number for r in registry.list_filings()]
        # filing_date ties → id DESC (registration order 0,1,2 → ids 1,2,3).
        assert full == ["ACC-2", "ACC-1", "ACC-0"]

        page1 = registry.list_filings(limit=2, offset=0)
        page2 = registry.list_filings(limit=2, offset=2)
        assert [r.accession_number for r in page1] == ["ACC-2", "ACC-1"]
        assert [r.accession_number for r in page2] == ["ACC-0"]

    def test_sort_by_column_ascending(self, registry: MetadataRegistry) -> None:
        registry.register_filing(
            FilingIdentifier("MSFT", "10-K", date(2024, 1, 1), "ACC-M"), chunk_count=1
        )
        registry.register_filing(
            FilingIdentifier("AAPL", "10-K", date(2024, 1, 2), "ACC-A"), chunk_count=1
        )
        result = registry.list_filings(sort_by="ticker", order="asc")
        assert [r.ticker for r in result] == ["AAPL", "MSFT"]

    def test_unknown_sort_by_raises_value_error(self, registry: MetadataRegistry) -> None:
        with pytest.raises(ValueError, match="Unsortable column"):
            registry.list_filings(sort_by="id")  # not an exposed sort column

    def test_invalid_order_raises_value_error(self, registry: MetadataRegistry) -> None:
        with pytest.raises(ValueError, match="Invalid sort order"):
            registry.list_filings(order="sideways")

    def test_negative_limit_or_offset_raises_value_error(self, registry: MetadataRegistry) -> None:
        with pytest.raises(ValueError, match="limit must be non-negative"):
            registry.list_filings(limit=-1)
        with pytest.raises(ValueError, match="offset must be non-negative"):
            registry.list_filings(offset=-1)


class TestListOldestFilings:
    def test_orders_by_ingested_at_ascending(self, registry: MetadataRegistry) -> None:
        fid_a = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-A")
        fid_b = FilingIdentifier("AAPL", "10-K", date(2024, 2, 1), "ACC-B")
        fid_c = FilingIdentifier("AAPL", "10-K", date(2024, 3, 1), "ACC-C")
        registry.register_filing(fid_a, chunk_count=1)
        registry.register_filing(fid_b, chunk_count=1)
        registry.register_filing(fid_c, chunk_count=1)

        oldest = registry.list_oldest_filings(limit=2)
        assert [r.accession_number for r in oldest] == ["ACC-A", "ACC-B"]


def _backdate_ingested_at(
    registry: MetadataRegistry,
    accession_number: str,
    days_ago: int,
) -> None:
    """Rewrite a filing's ``ingested_at`` so it appears ``days_ago`` old.

    The registry stamps ``ingested_at`` with ``datetime.now(UTC)`` at
    insertion, so the only way to test retention thresholds without
    sleeping is to backdate the column in-place.  Goes through the
    same persistent connection the registry holds so the writes are
    visible to subsequent registry calls.
    """
    sql = (
        "UPDATE filings SET ingested_at = datetime('now', '-' || ? || ' days') "
        "WHERE accession_number = ?"
    )
    with registry._lock, registry._conn:
        registry._conn.execute(sql, (days_ago, accession_number))


class TestListExpiredFilings:
    """Cutoff-driven retention enumeration; the read primitive consumed
    by :meth:`FilingStore.evict_expired`."""

    def test_returns_only_rows_older_than_cutoff(
        self,
        registry: MetadataRegistry,
    ) -> None:
        fresh = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-FRESH")
        old = FilingIdentifier("AAPL", "10-Q", date(2024, 1, 1), "ACC-OLD")
        registry.register_filing(fresh, chunk_count=10)
        registry.register_filing(old, chunk_count=5)

        # Backdate "old" to 100 days ago.
        _backdate_ingested_at(registry, "ACC-OLD", days_ago=100)

        result = registry.list_expired_filings(max_age_days=90)
        assert [r.accession_number for r in result] == ["ACC-OLD"]

    def test_orders_oldest_first(self, registry: MetadataRegistry) -> None:
        fid_a = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-A")
        fid_b = FilingIdentifier("AAPL", "10-Q", date(2024, 1, 1), "ACC-B")
        registry.register_filing(fid_a, chunk_count=1)
        registry.register_filing(fid_b, chunk_count=1)

        _backdate_ingested_at(registry, "ACC-A", days_ago=120)
        _backdate_ingested_at(registry, "ACC-B", days_ago=200)

        result = registry.list_expired_filings(max_age_days=90)
        assert [r.accession_number for r in result] == ["ACC-B", "ACC-A"]

    def test_empty_when_nothing_expired(self, registry: MetadataRegistry) -> None:
        fresh = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-FRESH")
        registry.register_filing(fresh, chunk_count=10)

        assert registry.list_expired_filings(max_age_days=30) == []

    def test_empty_database_returns_empty_list(
        self,
        registry: MetadataRegistry,
    ) -> None:
        assert registry.list_expired_filings(max_age_days=30) == []

    def test_record_carries_chunk_count_for_eviction_report(
        self,
        registry: MetadataRegistry,
    ) -> None:
        """``FilingStore.evict_expired`` sums ``chunk_count`` across the
        listed records to populate :class:`EvictionReport`; this test
        pins that the column survives the round-trip."""
        fid = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-OLD")
        registry.register_filing(fid, chunk_count=37)
        _backdate_ingested_at(registry, "ACC-OLD", days_ago=100)

        records = registry.list_expired_filings(max_age_days=90)
        assert len(records) == 1
        assert records[0].chunk_count == 37

    @pytest.mark.security
    def test_zero_max_age_rejected(self, registry: MetadataRegistry) -> None:
        """Defence-in-depth: zero would invert the WHERE clause and
        delete every recent filing.  Eviction is disabled at the
        settings layer (DB_RETENTION_MAX_AGE_DAYS=0), not by passing 0
        through the registry primitive."""
        with pytest.raises(ValueError, match="must be positive"):
            registry.list_expired_filings(max_age_days=0)

    @pytest.mark.security
    def test_negative_max_age_rejected(
        self,
        registry: MetadataRegistry,
    ) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            registry.list_expired_filings(max_age_days=-30)


class TestCount:
    def test_filters_apply(self, registry: MetadataRegistry) -> None:
        fid1 = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-1")
        fid2 = FilingIdentifier("MSFT", "10-K", date(2024, 1, 1), "ACC-2")
        registry.register_filing(fid1, chunk_count=1)
        registry.register_filing(fid2, chunk_count=2)

        assert registry.count() == 2
        assert registry.count(ticker="AAPL") == 1
        assert registry.count(form_type="10-K") == 2


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestGetStatistics:
    def test_empty_database(self, registry: MetadataRegistry) -> None:
        stats = registry.get_statistics()
        assert isinstance(stats, DatabaseStatistics)
        assert stats.filing_count == 0
        assert stats.tickers == []
        assert stats.form_breakdown == {}
        assert stats.ticker_breakdown == []

    def test_single_filing(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        stats = registry.get_statistics()
        assert stats.filing_count == 1
        assert stats.tickers == ["AAPL"]
        assert stats.form_breakdown == {"10-K": 1}
        assert len(stats.ticker_breakdown) == 1
        row = stats.ticker_breakdown[0]
        assert isinstance(row, TickerStatistics)
        assert row.ticker == "AAPL"
        assert row.filings == 1
        assert row.chunks == 42
        assert row.forms == ["10-K"]

    def test_multi_ticker_aggregation(self, registry: MetadataRegistry) -> None:
        fid1 = FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-1")
        fid2 = FilingIdentifier("AAPL", "10-Q", date(2024, 6, 1), "ACC-2")
        fid3 = FilingIdentifier("MSFT", "10-K", date(2024, 3, 1), "ACC-3")
        registry.register_filing(fid1, chunk_count=10)
        registry.register_filing(fid2, chunk_count=20)
        registry.register_filing(fid3, chunk_count=30)

        stats = registry.get_statistics()
        assert stats.filing_count == 3
        assert stats.tickers == ["AAPL", "MSFT"]
        assert stats.form_breakdown == {"10-K": 2, "10-Q": 1}

        assert len(stats.ticker_breakdown) == 2
        aapl, msft = stats.ticker_breakdown
        assert aapl.ticker == "AAPL"
        assert aapl.filings == 2
        assert aapl.chunks == 30
        assert aapl.forms == ["10-K", "10-Q"]
        assert msft.chunks == 30


# ---------------------------------------------------------------------------
# Delete operations
# ---------------------------------------------------------------------------


class TestRemoveFiling:
    def test_removes_existing(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        assert registry.remove_filing(stored_filing.accession_number) is True
        assert registry.count() == 0

    def test_returns_false_when_absent(self, registry: MetadataRegistry) -> None:
        assert registry.remove_filing("missing") is False


class TestRemoveFilingsBatch:
    def test_empty_noop(self, registry: MetadataRegistry) -> None:
        assert registry.remove_filings_batch([]) == 0

    def test_removes_multiple(self, registry: MetadataRegistry) -> None:
        for i in range(3):
            registry.register_filing(
                FilingIdentifier("AAPL", "10-K", date(2024, i + 1, 1), f"ACC-{i}"),
                chunk_count=1,
            )
        removed = registry.remove_filings_batch(["ACC-0", "ACC-2", "MISSING"])
        assert removed == 2
        assert {r.accession_number for r in registry.list_filings()} == {"ACC-1"}


class TestClearAll:
    def test_clears_rows(self, registry: MetadataRegistry) -> None:
        for i in range(3):
            registry.register_filing(
                FilingIdentifier("AAPL", "10-K", date(2024, i + 1, 1), f"ACC-{i}"),
                chunk_count=1,
            )
        assert registry.clear_all() == 3
        assert registry.count() == 0


# ---------------------------------------------------------------------------
# Filing limit (FIFO eviction trigger)
# ---------------------------------------------------------------------------


class TestCheckFilingLimit:
    def test_under_limit_passes(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        registry.check_filing_limit()

    def test_at_limit_raises(
        self,
        tmp_db_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exercise the limit gate by forcing a tiny max-filings ceiling.

        Setting ``_max_filings`` directly on the registry is deliberate:
        the Pydantic settings singleton is loaded at import time, and
        reloading it after ``monkeypatch.setenv`` still reads a cached
        default inside the test runner.  Overriding the registry's
        attribute bypasses that noise and exercises the contract —
        ``current >= self._max_filings`` raising
        :class:`FilingLimitExceededError`.
        """
        reg = MetadataRegistry(db_path=tmp_db_path)
        reg._max_filings = 1
        reg.register_filing(
            FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-1"),
            chunk_count=1,
        )
        with pytest.raises(FilingLimitExceededError) as exc_info:
            reg.check_filing_limit()
        assert exc_info.value.current_count == 1
        assert exc_info.value.max_filings == 1


# ---------------------------------------------------------------------------
# Persistent connection and thread safety
# ---------------------------------------------------------------------------


class TestPersistentConnection:
    def test_wal_journal_enabled(self, registry: MetadataRegistry) -> None:
        with registry._lock:
            row = registry._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0].lower() == "wal"

    def test_close_prevents_further_ops(self, registry: MetadataRegistry) -> None:
        registry.close()
        with pytest.raises(DatabaseError):
            registry.count()

    def test_concurrent_registrations_persist(self, registry: MetadataRegistry) -> None:
        errors: list[BaseException] = []

        def register(i: int) -> None:
            try:
                registry.register_filing(
                    FilingIdentifier(
                        ticker="AAPL",
                        form_type="10-K",
                        filing_date=date(2020 + i, 1, 1),
                        accession_number=f"ACC-THREAD-{i}",
                    ),
                    chunk_count=i,
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert registry.count() == 8


# ---------------------------------------------------------------------------
# Task history persistence and privacy controls
# ---------------------------------------------------------------------------


def _save_sample_task(
    registry: MetadataRegistry,
    *,
    task_id: str = "task-1",
    status: str = "completed",
    tickers: list[str] | None = None,
    form_types: list[str] | None = None,
    results: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> None:
    registry.save_task_history(
        task_id,
        status=status,
        tickers=tickers or ["AAPL"],
        form_types=form_types or ["10-K"],
        results=results or [{"accession_number": "ACC-1", "status": "ok"}],
        error=error,
        started_at="2026-04-20T10:00:00+00:00",
        completed_at="2026-04-20T10:00:05+00:00",
        filings_done=1,
    )


class TestTaskHistoryRoundtrip:
    def test_save_and_retrieve(self, registry: MetadataRegistry) -> None:
        _save_sample_task(registry, tickers=["AAPL", "MSFT"])
        row = registry.get_task_history("task-1")
        assert row is not None
        assert row["task_id"] == "task-1"
        assert row["status"] == "completed"
        assert row["form_types"] == ["10-K"]
        assert row["results"] == [{"accession_number": "ACC-1", "status": "ok"}]
        assert row["filings_done"] == 1

    def test_missing_task_returns_none(self, registry: MetadataRegistry) -> None:
        assert registry.get_task_history("never-saved") is None

    def test_save_is_idempotent(self, registry: MetadataRegistry) -> None:
        _save_sample_task(registry)
        _save_sample_task(registry, status="failed")
        row = registry.get_task_history("task-1")
        assert row is not None
        assert row["status"] == "failed"


# ---------------------------------------------------------------------------
# Security / privacy controls
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSQLInjectionSafety:
    """Parameter-bound queries reject malicious input instead of executing it."""

    def test_ticker_injection_matches_nothing(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        assert registry.list_filings(ticker="' OR 1=1 --") == []
        # Table still intact.
        assert registry.count() == 1

    def test_form_type_injection_matches_nothing(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        assert registry.list_filings(form_type="'; DROP TABLE filings; --") == []
        assert registry.count() == 1

    def test_accession_injection_returns_none(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        assert registry.is_duplicate("' OR '1'='1") is False

    def test_sort_by_injection_is_rejected_not_executed(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        # The ``ORDER BY`` column is the one clause built by f-string, so
        # its input must never reach the SQL text. A crafted ``sort_by``
        # is rejected against the whitelist *before* any query runs — it
        # is never interpolated, so the DROP cannot execute.
        with pytest.raises(ValueError, match="Unsortable column"):
            registry.list_filings(sort_by="filing_date; DROP TABLE filings; --")
        # Table still intact.
        assert registry.count() == 1

    def test_order_injection_is_rejected_not_executed(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        with pytest.raises(ValueError, match="Invalid sort order"):
            registry.list_filings(order="ASC; DROP TABLE filings; --")
        assert registry.count() == 1
        assert registry.get_filing("' UNION SELECT * FROM filings --") is None

    def test_count_injection_returns_zero(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        assert registry.count(ticker="' OR 1=1 --") == 0
        assert registry.count(form_type="'; DROP TABLE filings; --") == 0
        assert registry.count() == 1

    def test_batch_injection_returns_empty(
        self,
        registry: MetadataRegistry,
        stored_filing: FilingIdentifier,
    ) -> None:
        result = registry.get_existing_accessions(["' OR '1'='1", "'; DROP TABLE filings; --"])
        assert result == set()
        assert registry.count() == 1


@pytest.mark.security
class TestErrorScrubbing:
    """`_scrub_error_message` strips tickers and accession numbers."""

    def test_removes_ticker_case_insensitive(self) -> None:
        out = _scrub_error_message("Failed to ingest AAPL filing", ["AAPL"])
        assert "AAPL" not in (out or "")
        assert "[TICKER]" in (out or "")

    def test_removes_accession_number(self) -> None:
        out = _scrub_error_message(
            "Accession 0000320193-23-000077 failed",
            tickers=[],
        )
        assert "0000320193-23-000077" not in (out or "")
        assert "[ACCESSION]" in (out or "")

    def test_handles_none(self) -> None:
        assert _scrub_error_message(None, tickers=["AAPL"]) is None

    def test_accession_without_dashes(self) -> None:
        out = _scrub_error_message("000032019323000077 bombed", tickers=[])
        assert "000032019323000077" not in (out or "")
        assert "[ACCESSION]" in (out or "")

    def test_skips_empty_ticker_strings(self) -> None:
        """Empty strings in the ticker list must not contribute to the regex.

        ``_scrub_error_message`` filters via ``[t for t in tickers if t]``, so
        an empty string is treated as "no ticker list" — the original text is
        returned untouched.  The whitespace-only entry is a regex-legal
        alternation because ``re.escape(" ")`` leaves it unchanged and the
        message contains literal spaces; the scrub replaces them, which is
        acceptable noise but verifies the empty-string branch in isolation.
        """
        out = _scrub_error_message("Plain error text", tickers=[""])
        assert out == "Plain error text"


@pytest.mark.security
class TestTaskHistoryPrivacy:
    """``DB_TASK_HISTORY_PERSIST_TICKERS`` gates ticker persistence."""

    def test_tickers_stripped_by_default(
        self,
        registry: MetadataRegistry,
    ) -> None:
        """Default (off) — ``tickers`` column stores null, never the literal list."""
        _save_sample_task(
            registry,
            tickers=["AAPL", "MSFT"],
            error="AAPL request for 0000320193-23-000077 rate-limited",
        )
        row = registry.get_task_history("task-1")
        assert row is not None
        # Tickers absent from persisted payload.
        assert row["tickers"] == []
        # Error scrubbed of ticker + accession identifiers.
        assert "AAPL" not in row["error"]
        assert "0000320193-23-000077" not in row["error"]
        assert "[TICKER]" in row["error"]
        assert "[ACCESSION]" in row["error"]

    def test_tickers_persist_when_opted_in(
        self,
        tmp_db_path: str,
    ) -> None:
        """Patch the settings-reader attribute directly — the persist-tickers
        toggle is read via ``get_settings().database.task_history_persist_tickers``
        at save time, so flipping the cached singleton's attribute exercises the
        opt-in branch without fighting pydantic-settings env-var caching.
        """
        from sec_generative_search.config import settings as settings_module

        settings = settings_module.get_settings()
        original = settings.database.task_history_persist_tickers
        settings.database.task_history_persist_tickers = True
        try:
            reg = MetadataRegistry(db_path=tmp_db_path)
            _save_sample_task(reg, tickers=["AAPL", "MSFT"])
            row = reg.get_task_history("task-1")
            assert row is not None
            assert row["tickers"] == ["AAPL", "MSFT"]
        finally:
            settings.database.task_history_persist_tickers = original


@pytest.mark.security
class TestErrorScrubbingTripwires:
    """Pin ``_scrub_error_message`` end-to-end behaviour under both
    ``DB_TASK_HISTORY_PERSIST_TICKERS`` settings.

    The scrub is the load-bearing privacy control on the ``error``
    column and must fire regardless of the ticker-persistence flag. The
    unit tests on ``_scrub_error_message`` (above) cover the regex; this
    class drives the function via :meth:`MetadataRegistry.save_task_history`
    so a refactor that accidentally couples the scrub to the persist
    toggle surfaces here, not in a broader integration run.
    """

    @staticmethod
    def _set_persist_tickers(value: bool) -> bool:
        """Flip the cached settings singleton's persist-tickers flag and
        return the prior value so the test can restore it in ``finally``.

        Mirrors the pattern used by ``TestTaskHistoryPrivacy`` — settings
        are re-read inside ``save_task_history`` via ``get_settings()``,
        and patching the cached attribute side-steps pydantic-settings
        env-var caching.
        """
        from sec_generative_search.config import settings as settings_module

        settings = settings_module.get_settings()
        prior = settings.database.task_history_persist_tickers
        settings.database.task_history_persist_tickers = value
        return prior

    @staticmethod
    def _restore_persist_tickers(value: bool) -> None:
        from sec_generative_search.config import settings as settings_module

        settings_module.get_settings().database.task_history_persist_tickers = value

    # ------------------------------------------------------------------
    # Persist=True: error column must STILL be scrubbed.
    # ------------------------------------------------------------------

    def test_error_scrubbed_even_when_tickers_persisted(
        self,
        tmp_db_path: str,
    ) -> None:
        """Opting into ticker persistence must not weaken the error scrub.

        Tickers may legitimately appear in the ``tickers`` column under
        opt-in, but the ``error`` column carries free-form upstream text
        that an operator on Scenario B/C inspects for triage; ticker /
        accession identifiers in there leak research patterns to anyone
        with task-history read access.
        """
        prior = self._set_persist_tickers(True)
        try:
            reg = MetadataRegistry(db_path=tmp_db_path)
            _save_sample_task(
                reg,
                tickers=["AAPL", "MSFT"],
                error=(
                    "AAPL request for 0000320193-23-000077 rate-limited; "
                    "MSFT 0000789019-22-000011 also failed"
                ),
            )
            row = reg.get_task_history("task-1")
            assert row is not None
            # Tickers persisted on the column (opt-in honoured).
            assert row["tickers"] == ["AAPL", "MSFT"]
            # But the error column is still fully scrubbed.
            assert "AAPL" not in row["error"]
            assert "MSFT" not in row["error"]
            assert "0000320193-23-000077" not in row["error"]
            assert "0000789019-22-000011" not in row["error"]
            assert row["error"].count("[TICKER]") == 2
            assert row["error"].count("[ACCESSION]") == 2
        finally:
            self._restore_persist_tickers(prior)

    # ------------------------------------------------------------------
    # Persist=False: same scrub guarantee, plus tickers column is NULL.
    # ------------------------------------------------------------------

    def test_error_scrubbed_when_tickers_not_persisted(
        self,
        registry: MetadataRegistry,
    ) -> None:
        """Default privacy posture — both layers active simultaneously.

        Pinned separately from ``TestTaskHistoryPrivacy.test_tickers_stripped_by_default``
        so a regression in either layer (column null OR error scrub) is
        attributable to a single failing test rather than a compound assertion.
        """
        _save_sample_task(
            registry,
            tickers=["AAPL"],
            error="AAPL ingest of 0000320193-23-000077 failed (HTTP 429)",
        )
        row = registry.get_task_history("task-1")
        assert row is not None
        # Layer 1: tickers column is empty (stored as NULL → decoded as []).
        assert row["tickers"] == []
        # Layer 2: error scrubbed of both ticker and accession.
        assert "AAPL" not in row["error"]
        assert "0000320193-23-000077" not in row["error"]
        assert "[TICKER]" in row["error"]
        assert "[ACCESSION]" in row["error"]

    # ------------------------------------------------------------------
    # Edge cases that historically catch regex regressions.
    # ------------------------------------------------------------------

    def test_case_insensitive_ticker_redaction(
        self,
        registry: MetadataRegistry,
    ) -> None:
        """Lower-case / mixed-case ticker mentions must redact too.

        Edgartools / urllib3 sometimes echo the ticker in lower case in
        their error strings; the scrub uses ``re.IGNORECASE``, but a
        naive refactor that built a case-sensitive alternation would
        silently leak the mention.
        """
        _save_sample_task(
            registry,
            tickers=["AAPL"],
            error="lower-case aapl and Mixed Aapl both should redact",
        )
        row = registry.get_task_history("task-1")
        assert row is not None
        assert "aapl" not in row["error"].lower()
        assert row["error"].count("[TICKER]") == 2

    def test_ticker_word_boundary_discipline(
        self,
        registry: MetadataRegistry,
    ) -> None:
        """Substring containment must NOT match.

        ``\\b{ticker}\\b`` is the contract; a substring like ``MSFTX`` or
        ``XMSFT`` is a different identifier (often deliberate noise from
        an upstream tool) and must survive the scrub. Tripwires the
        word-boundary anchors against accidental removal.
        """
        _save_sample_task(
            registry,
            tickers=["MSFT"],
            error="Token MSFTX and prefix XMSFT must remain; standalone MSFT must redact",
        )
        row = registry.get_task_history("task-1")
        assert row is not None
        assert "MSFTX" in row["error"]
        assert "XMSFT" in row["error"]
        # Standalone MSFT redacted exactly once.
        assert row["error"].count("[TICKER]") == 1

    def test_mixed_accession_formats_in_one_error(
        self,
        registry: MetadataRegistry,
    ) -> None:
        """Both dashed and undashed accession spellings redact in one pass.

        ``_ACCESSION_RE`` matches optional dashes; if a future tightening
        accidentally splits the two forms across separate patterns, this
        test breaks before it ships.
        """
        _save_sample_task(
            registry,
            tickers=[],
            error="Failed: 0000320193-23-000077 and also 000078901922000011",
        )
        row = registry.get_task_history("task-1")
        assert row is not None
        assert "0000320193-23-000077" not in row["error"]
        assert "000078901922000011" not in row["error"]
        assert row["error"].count("[ACCESSION]") == 2

    def test_no_identifiers_passes_through_untouched(
        self,
        registry: MetadataRegistry,
    ) -> None:
        """Generic upstream errors with no PII must reach the operator
        verbatim — over-aggressive scrubbing would degrade triage value.
        """
        original = "Connection reset by peer (errno=104) after 30s"
        _save_sample_task(registry, tickers=["AAPL"], error=original)
        row = registry.get_task_history("task-1")
        assert row is not None
        assert row["error"] == original

    def test_scrub_invocation_is_unconditional(self) -> None:
        """White-box tripwire: the scrub call site in ``save_task_history``
        must NOT be wrapped in a ``persist_tickers`` conditional.

        Historically, the ticker-persist toggle is the kind of flag a
        well-intentioned refactor reaches for to ``optimise away`` work
        when ``tickers`` is null; that would re-couple the scrub to the
        flag and silently regress the privacy contract. Reading the
        source for the gating expression directly catches that drift in
        unit tests instead of a Scenario-B/C smoke run.
        """
        import inspect

        from sec_generative_search.database.metadata import MetadataRegistry

        source = inspect.getsource(MetadataRegistry.save_task_history)
        # The scrub line is guarded only on `error` truthiness, never on
        # the persist-tickers flag.
        assert "_scrub_error_message(error, tickers) if error else None" in source
        # Defensive: no persist_tickers token may share a line with the
        # scrub call.
        for line in source.splitlines():
            if "_scrub_error_message" in line:
                assert "persist_tickers" not in line, (
                    "Error scrub must not be gated on DB_TASK_HISTORY_PERSIST_TICKERS"
                )


@pytest.mark.security
class TestEncryptionKeyHandling:
    """SQLCipher key wiring — safe fallback and no key leakage via repr/str."""

    @staticmethod
    def _enable_propagation() -> None:
        """Re-enable log propagation so pytest's ``caplog`` captures records.

        ``configure_logging`` sets ``propagate = False`` on the package
        logger (see ``tests/core/test_logging.py::TestAuditLog``). Tests
        that want records surfaced through ``caplog`` must opt back in
        and restore the flag in a ``finally`` block.
        """
        import logging

        logging.getLogger("sec_generative_search").propagate = True

    @staticmethod
    def _reset_propagation() -> None:
        import logging

        logging.getLogger("sec_generative_search").propagate = False

    def test_fallback_to_sqlite3_without_pysqlcipher(
        self,
        tmp_db_path: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With a key but no pysqlcipher3 installed, the registry warns and
        falls back to plain sqlite3 instead of silently storing data unprotected.

        The warning is the load-bearing signal for the operator; absence of
        pysqlcipher3 is a deploy-time error, not a runtime-silent one.
        """
        import logging

        self._enable_propagation()
        try:
            with caplog.at_level(logging.WARNING, logger="sec_generative_search"):
                registry = MetadataRegistry(
                    db_path=tmp_db_path,
                    encryption_key="unit-test-key-not-a-secret",
                )
            assert registry.encrypted is False
            assert any(
                "pysqlcipher3 is not installed" in record.getMessage() for record in caplog.records
            )
            # Plain sqlite3 is the actual driver — no SQLCipher classes.
            assert registry._sqlite_module is sqlite3
        finally:
            self._reset_propagation()

    def test_key_never_appears_in_log_output(
        self,
        tmp_db_path: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The debug message on registry init must not echo the raw key."""
        import logging

        self._enable_propagation()
        key = "never-log-this-secret-value"
        try:
            with caplog.at_level(logging.DEBUG, logger="sec_generative_search"):
                MetadataRegistry(db_path=tmp_db_path, encryption_key=key)
            for record in caplog.records:
                assert key not in record.getMessage()
                # Also ensure the hex encoding is not leaked.
                assert key.encode().hex() not in record.getMessage()
        finally:
            self._reset_propagation()

    def test_encrypted_flag_off_without_driver(
        self,
        tmp_db_path: str,
    ) -> None:
        """Without pysqlcipher3, ``encrypted`` must report ``False`` even when a
        key is supplied — the caller relies on this flag to decide whether the
        data-at-rest promise is upheld.
        """
        registry = MetadataRegistry(
            db_path=tmp_db_path,
            encryption_key="unit-test-key",
        )
        assert registry.encrypted is False

    def test_encrypted_flag_off_without_key(
        self,
        tmp_db_path: str,
    ) -> None:
        registry = MetadataRegistry(db_path=tmp_db_path)
        assert registry.encrypted is False


# ---------------------------------------------------------------------------
# Settings fallbacks (default db_path)
# ---------------------------------------------------------------------------


class TestSettingsDefaults:
    def test_default_db_path_is_used_when_not_provided(
        self,
        tmp_path: Path,
    ) -> None:
        """When ``db_path`` is omitted, the registry falls through to
        ``settings.database.metadata_db_path``.

        Rather than fight pydantic-settings env-var caching, override the
        already-loaded singleton's ``metadata_db_path`` attribute and
        restore it after the test. This exercises the same fall-through
        branch in :meth:`MetadataRegistry.__init__`.

        The temporary path lives under the project root to keep
        :meth:`DatabaseSettings._validate_paths` happy when other tests
        subsequently reload settings.
        """
        from sec_generative_search.config import settings as settings_module

        settings = settings_module.get_settings()
        project_tmp = Path.cwd() / "tmp_test_registry.sqlite"
        original = settings.database.metadata_db_path
        settings.database.metadata_db_path = str(project_tmp)
        try:
            registry = MetadataRegistry()
            registry.register_filing(
                FilingIdentifier("AAPL", "10-K", date(2024, 1, 1), "ACC-DEFAULT"),
                chunk_count=1,
            )
            assert registry.count() == 1
            assert project_tmp.exists()
            registry.close()
        finally:
            settings.database.metadata_db_path = original
            cleanup = (
                project_tmp,
                project_tmp.with_suffix(".sqlite-wal"),
                project_tmp.with_suffix(".sqlite-shm"),
            )
            for extra in cleanup:
                if extra.exists():
                    extra.unlink()


@pytest.mark.security
class TestRegistryLogStreamIsIdentifierFree:
    """The **real** registry's log lines carry no ticker or accession.

    The static call-site lock in ``tests/core/test_logging.py`` proves
    every ticker expression is *wrapped*; it cannot prove the wrapper
    survives the value at runtime. This drives the production
    ``MetadataRegistry`` through register → remove with redaction on and
    captures through a handler that carries
    :class:`AccessionRedactionFilter`, which is what a Scenario-B/C
    deployment actually runs (``caplog`` attaches to root and would see
    unfiltered records).
    """

    def test_register_and_remove_emit_no_raw_identifier(
        self,
        registry: MetadataRegistry,
        sample_filing_id: FilingIdentifier,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.addFilter(AccessionRedactionFilter())
        package_logger = logging.getLogger(LOGGER_NAME)
        prior_level = package_logger.level
        package_logger.addHandler(handler)
        package_logger.setLevel(logging.DEBUG)
        try:
            registry.register_filing(sample_filing_id, chunk_count=12)
            # Atomic re-registration hits the "already registered" branch,
            # which logs the accession.
            registry.register_filing_if_new(sample_filing_id, chunk_count=12)
            registry.remove_filing(sample_filing_id.accession_number)
            # Second removal hits the "not found in registry" warning.
            registry.remove_filing(sample_filing_id.accession_number)
        finally:
            package_logger.removeHandler(handler)
            package_logger.setLevel(prior_level)

        emitted = stream.getvalue()
        assert emitted.strip(), "expected the registry to log something"
        assert sample_filing_id.ticker not in emitted
        assert sample_filing_id.accession_number not in emitted
        assert "<redacted:" in emitted
