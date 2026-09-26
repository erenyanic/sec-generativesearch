"""Tests for :class:`sec_generative_search.database.FilingStore`.

``FilingStore`` is the dual-store coordinator that enforces
"ChromaDB first, then SQLite" for the default write path and deletes,
and inverts the order for the atomic ``register_if_new=True`` path.
Tests drive real :class:`ChromaDBClient` and :class:`MetadataRegistry`
instances (under ``tmp_path``) through the coordinator to check the
ordering and rollback contract end-to-end.  Fault paths are exercised
with surgical monkey-patches on the heavier side (ChromaDB) or the
SQLite side depending on the path under test.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sec_generative_search.core.exceptions import DatabaseError
from sec_generative_search.core.types import (
    Chunk,
    ContentType,
    EmbedderStamp,
    EvictionReport,
    FilingIdentifier,
    IngestResult,
)
from sec_generative_search.database import (
    ChromaDBClient,
    FilingStore,
    MetadataRegistry,
)
from sec_generative_search.pipeline.orchestrator import ProcessedFiling

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chroma_path(tmp_path: Path) -> str:
    """Per-test ChromaDB persistence directory."""
    return str(tmp_path / "chroma")


@pytest.fixture
def stamp() -> EmbedderStamp:
    """A small-dimension stamp so fabricated vectors stay cheap."""
    return EmbedderStamp(
        provider="openai",
        model="text-embedding-3-small",
        dimension=4,
    )


@pytest.fixture
def chroma(stamp: EmbedderStamp, chroma_path: str) -> ChromaDBClient:
    """Stamped ChromaDB client over a temporary directory."""
    return ChromaDBClient(stamp, chroma_path=chroma_path)


@pytest.fixture
def registry(tmp_db_path: str) -> Iterator[MetadataRegistry]:
    """SQLite registry over a temporary file.

    Uses the shared ``tmp_db_path`` fixture from
    ``tests/database/conftest.py`` so the path is isolated per test.
    """
    reg = MetadataRegistry(db_path=tmp_db_path)
    yield reg
    reg.close()


@pytest.fixture
def store(chroma: ChromaDBClient, registry: MetadataRegistry) -> FilingStore:
    return FilingStore(chroma, registry)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_filing_id(
    *,
    ticker: str = "AAPL",
    form_type: str = "10-K",
    filing_date: date | None = None,
    accession_number: str = "0000320193-23-000077",
) -> FilingIdentifier:
    return FilingIdentifier(
        ticker=ticker,
        form_type=form_type,
        filing_date=filing_date or date(2023, 11, 3),
        accession_number=accession_number,
    )


def _make_chunks(filing_id: FilingIdentifier, n: int = 2) -> list[Chunk]:
    return [
        Chunk(
            content=f"Chunk {i} for {filing_id.ticker}",
            path="Part I > Item 1 > Business",
            content_type=ContentType.TEXT,
            filing_id=filing_id,
            chunk_index=i,
            token_count=5,
        )
        for i in range(n)
    ]


def _make_processed_filing(
    stamp: EmbedderStamp,
    *,
    filing_id: FilingIdentifier | None = None,
    n_chunks: int = 2,
    with_embeddings: bool = True,
) -> ProcessedFiling:
    filing_id = filing_id or _make_filing_id()
    chunks = _make_chunks(filing_id, n=n_chunks)
    if with_embeddings:
        embeddings: np.ndarray | None = np.arange(
            n_chunks * stamp.dimension,
            dtype=np.float32,
        ).reshape(n_chunks, stamp.dimension)
    else:
        embeddings = None
    return ProcessedFiling(
        filing_id=filing_id,
        chunks=chunks,
        embeddings=embeddings,
        ingest_result=IngestResult(
            filing_id=filing_id,
            segment_count=n_chunks,
            chunk_count=n_chunks,
            duration_seconds=0.0,
        ),
    )


def _filing_with_suffix(stamp: EmbedderStamp, suffix: str) -> ProcessedFiling:
    """Build a filing whose accession and ticker vary by *suffix*.

    The SQLite schema enforces ``UNIQUE(ticker, form_type, filing_date)``
    in addition to ``UNIQUE(accession_number)`` — tests that register
    multiple filings must therefore differ on the tuple, not only on
    the accession number.  Varying the ticker alone is enough.
    """
    return _make_processed_filing(
        stamp,
        filing_id=_make_filing_id(
            ticker=f"TCK{suffix}",
            accession_number=f"0000320193-23-000{suffix}",
        ),
    )


# ---------------------------------------------------------------------------
# Happy path — store
# ---------------------------------------------------------------------------


class TestStoreFiling:
    def test_store_filing_writes_to_both_stores(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
    ) -> None:
        pf = _make_processed_filing(stamp)

        stored = store.store_filing(pf)

        assert stored is True
        assert chroma.collection_count() == 2
        assert registry.is_duplicate(pf.filing_id.accession_number)
        record = registry.get_filing(pf.filing_id.accession_number)
        assert record is not None
        assert record.chunk_count == 2

    def test_store_filing_register_if_new_skip_on_duplicate(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
    ) -> None:
        """Atomic path must be a clean no-op on a duplicate accession.

        The SQLite-first order means the duplicate is detected before
        any ChromaDB write — so the vector store count stays at the
        first writer's contribution.
        """
        pf = _make_processed_filing(stamp)

        assert store.store_filing(pf, register_if_new=True) is True
        assert chroma.collection_count() == 2

        second = store.store_filing(pf, register_if_new=True)

        assert second is False
        assert chroma.collection_count() == 2  # no change — SQLite saw the dupe
        assert registry.count() == 1

    def test_store_filing_default_path_uses_chroma_first(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default path writes ChromaDB *before* SQLite.

        Verified by recording the invocation order of both ``store_*``
        methods — the test is the canonical check that the documented
        ordering is what runs on the wire.
        """
        pf = _make_processed_filing(stamp)
        order: list[str] = []

        real_chroma_store = chroma.store_filing
        real_registry_register = registry.register_filing

        def _chroma_spy(pf: ProcessedFiling) -> None:
            order.append("chroma")
            real_chroma_store(pf)

        def _registry_spy(*args: Any, **kwargs: Any) -> None:
            order.append("sqlite")
            real_registry_register(*args, **kwargs)

        monkeypatch.setattr(chroma, "store_filing", _chroma_spy)
        monkeypatch.setattr(registry, "register_filing", _registry_spy)

        store.store_filing(pf)

        assert order == ["chroma", "sqlite"]

    def test_store_filing_atomic_path_uses_sqlite_first(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Atomic path inverts the order (SQLite first).

        Documented in the module docstring: the inversion is what
        prevents the duplicate-ID no-op trap on ChromaDB from
        corrupting a winning writer during rollback.
        """
        pf = _make_processed_filing(stamp)
        order: list[str] = []

        real_chroma_store = chroma.store_filing
        real_register_if_new = registry.register_filing_if_new

        def _chroma_spy(pf: ProcessedFiling) -> None:
            order.append("chroma")
            real_chroma_store(pf)

        def _registry_spy(*args: Any, **kwargs: Any) -> bool:
            order.append("sqlite")
            return real_register_if_new(*args, **kwargs)

        monkeypatch.setattr(chroma, "store_filing", _chroma_spy)
        monkeypatch.setattr(registry, "register_filing_if_new", _registry_spy)

        store.store_filing(pf, register_if_new=True)

        assert order == ["sqlite", "chroma"]


# ---------------------------------------------------------------------------
# Fault injection — store
# ---------------------------------------------------------------------------


class TestStoreFilingFaults:
    @pytest.mark.security
    def test_store_rolls_back_chroma_when_sqlite_fails(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SQLite failure must not leave chunks behind in ChromaDB.

        Without rollback, a retry would double the ChromaDB write and
        defeat the dual-store consistency contract.
        """
        pf = _make_processed_filing(stamp)

        def _raise(*args: Any, **kwargs: Any) -> None:
            raise DatabaseError("simulated SQLite failure")

        monkeypatch.setattr(registry, "register_filing", _raise)

        with pytest.raises(DatabaseError, match="simulated SQLite failure"):
            store.store_filing(pf)

        # Rollback must have fired — ChromaDB is empty.
        assert chroma.collection_count() == 0
        assert registry.count() == 0

    @pytest.mark.security
    def test_atomic_path_rolls_back_sqlite_when_chroma_fails(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Atomic path rollback target is SQLite, not ChromaDB.

        Because the SQLite row was claimed first, a ChromaDB failure
        must undo that claim — otherwise the registry carries an
        orphan row and a retry sees a phantom duplicate.
        """
        pf = _make_processed_filing(stamp)

        def _raise(*args: Any, **kwargs: Any) -> None:
            raise DatabaseError("simulated ChromaDB failure")

        monkeypatch.setattr(chroma, "store_filing", _raise)

        with pytest.raises(DatabaseError, match="simulated ChromaDB failure"):
            store.store_filing(pf, register_if_new=True)

        # SQLite claim was rolled back — no orphan row.
        assert registry.count() == 0
        assert chroma.collection_count() == 0

    def test_store_does_not_touch_sqlite_when_chroma_fails(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pf = _make_processed_filing(stamp)

        def _raise(*args: Any, **kwargs: Any) -> None:
            raise DatabaseError("simulated ChromaDB failure")

        monkeypatch.setattr(chroma, "store_filing", _raise)

        with pytest.raises(DatabaseError, match="simulated ChromaDB failure"):
            store.store_filing(pf)

        # SQLite was never touched.
        assert registry.count() == 0

    @pytest.mark.security
    def test_store_surfaces_original_error_when_rollback_also_fails(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rollback failures must not shadow the root cause.

        The operator needs to diagnose *why* the dual store failed,
        not a secondary symptom from the cleanup.  The rollback error
        is logged internally (visible in stderr capture) while the
        original SQLite error propagates.
        """
        pf = _make_processed_filing(stamp)

        def _sqlite_fails(*args: Any, **kwargs: Any) -> None:
            raise DatabaseError("primary SQLite failure")

        def _rollback_fails(*args: Any, **kwargs: Any) -> None:
            raise DatabaseError("rollback failure (secondary)")

        monkeypatch.setattr(registry, "register_filing", _sqlite_fails)
        monkeypatch.setattr(chroma, "delete_filing", _rollback_fails)

        with pytest.raises(DatabaseError) as excinfo:
            store.store_filing(pf)

        # Original error wins; the secondary rollback error does not
        # reach the caller.
        assert "primary SQLite failure" in str(excinfo.value)
        assert "rollback failure" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Delete path
# ---------------------------------------------------------------------------


class _FailSecondAdd:
    """Delegate to the real collection but fail its second ``add``."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._adds = 0

    def add(self, **kwargs: Any) -> Any:
        self._adds += 1
        if self._adds == 2:
            raise RuntimeError("simulated chroma add failure")
        return self._inner.add(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class TestMultiSliceStoreFaults:
    """A ChromaDB failure after some slices landed leaves *both* stores empty (F23).

    Neither path's own rollback covers it — the atomic path only undoes the
    SQLite claim and the carry-over path undoes nothing on a ChromaDB
    failure — so ``ChromaDBClient.store_filing`` must clean up its slices.
    """

    @pytest.mark.security
    @pytest.mark.parametrize("register_if_new", [True, False])
    def test_partial_chroma_write_leaves_no_orphans(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
        monkeypatch: pytest.MonkeyPatch,
        register_if_new: bool,
    ) -> None:
        monkeypatch.setattr(chroma._client, "get_max_batch_size", lambda: 2)
        real_collection = chroma._collection
        chroma._collection = _FailSecondAdd(real_collection)
        pf = _make_processed_filing(stamp, n_chunks=5)

        with pytest.raises(DatabaseError, match="Failed to store filing"):
            store.store_filing(pf, register_if_new=register_if_new)

        assert chroma.collection_count() == 0
        assert registry.count() == 0

        # Nothing half-written blocks the retry.
        chroma._collection = real_collection
        assert store.store_filing(pf, register_if_new=register_if_new) is True
        assert chroma.collection_count() == 5
        assert registry.count() == 1


class TestDeleteFiling:
    def test_delete_filing_removes_both_stores(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
    ) -> None:
        pf = _make_processed_filing(stamp)
        store.store_filing(pf)

        assert store.delete_filing(pf.filing_id.accession_number) is True
        assert chroma.collection_count() == 0
        assert registry.count() == 0

    def test_delete_filing_returns_false_when_missing(
        self,
        store: FilingStore,
    ) -> None:
        # ChromaDB delete is idempotent; SQLite returns False for a
        # missing accession.  The FilingStore reports False.
        assert store.delete_filing("0000000000-00-000000") is False

    def test_delete_filing_does_not_touch_sqlite_on_chroma_failure(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pf = _make_processed_filing(stamp)
        store.store_filing(pf)

        def _raise(*args: Any, **kwargs: Any) -> None:
            raise DatabaseError("simulated ChromaDB failure")

        monkeypatch.setattr(chroma, "delete_filing", _raise)

        with pytest.raises(DatabaseError):
            store.delete_filing(pf.filing_id.accession_number)

        # SQLite row is still present — caller can retry.
        assert registry.count() == 1

    def test_delete_filing_uses_chroma_first(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pf = _make_processed_filing(stamp)
        store.store_filing(pf)

        order: list[str] = []
        real_chroma_delete = chroma.delete_filing
        real_registry_remove = registry.remove_filing

        def _chroma_spy(acc: str) -> None:
            order.append("chroma")
            real_chroma_delete(acc)

        def _registry_spy(acc: str) -> bool:
            order.append("sqlite")
            return real_registry_remove(acc)

        monkeypatch.setattr(chroma, "delete_filing", _chroma_spy)
        monkeypatch.setattr(registry, "remove_filing", _registry_spy)

        store.delete_filing(pf.filing_id.accession_number)

        assert order == ["chroma", "sqlite"]


class TestDeleteFilingsBatch:
    def test_delete_batch_removes_both_stores(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
    ) -> None:
        accessions: list[str] = []
        for suffix in ("077", "088", "099"):
            pf = _filing_with_suffix(stamp, suffix)
            store.store_filing(pf)
            accessions.append(pf.filing_id.accession_number)

        assert chroma.collection_count() == 6  # 3 filings, 2 chunks each
        assert registry.count() == 3

        removed = store.delete_filings_batch(accessions)

        assert removed == 3
        assert chroma.collection_count() == 0
        assert registry.count() == 0

    def test_delete_batch_empty_list_is_noop(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _fail(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("empty batch must short-circuit before any store call")

        monkeypatch.setattr(chroma, "delete_filings_batch", _fail)
        monkeypatch.setattr(registry, "remove_filings_batch", _fail)

        assert store.delete_filings_batch([]) == 0


class TestClearAll:
    def test_clear_all_clears_both_stores(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
    ) -> None:
        for suffix in ("077", "088"):
            pf = _filing_with_suffix(stamp, suffix)
            store.store_filing(pf)

        assert chroma.collection_count() == 4
        assert registry.count() == 2

        chunks_removed, filings_removed = store.clear_all()

        assert chunks_removed == 4
        assert filings_removed == 2
        assert chroma.collection_count() == 0
        assert registry.count() == 0

        # After clear, the collection must still be usable (re-seal
        # contract from ``ChromaDBClient.clear_collection``).
        pf = _filing_with_suffix(stamp, "111")
        store.store_filing(pf)
        assert chroma.collection_count() == 2


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


def _backdate_filing(registry: MetadataRegistry, accession: str, days: int) -> None:
    """Rewrite ``ingested_at`` so a registered filing reads as ``days`` old.

    Mirrors the helper used in :mod:`tests.database.test_metadata`; we
    duplicate it here rather than cross-importing because the test
    packages avoid sharing private helpers.
    """
    sql = (
        "UPDATE filings SET ingested_at = datetime('now', '-' || ? || ' days') "
        "WHERE accession_number = ?"
    )
    with registry._lock, registry._conn:
        registry._conn.execute(sql, (days, accession))


class TestEvictExpired:
    """``FilingStore.evict_expired`` composes the registry's expired-list
    reader with the existing batched delete; ChromaDB-first ordering
    and rollback semantics are inherited from ``delete_filings_batch``.
    """

    def test_empty_registry_returns_zero_report(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
    ) -> None:
        report = store.evict_expired(max_age_days=30)
        assert isinstance(report, EvictionReport)
        assert report.filings_evicted == 0
        assert report.chunks_evicted == 0
        assert report.max_age_days == 30
        # Empty-list short-circuit must not have touched either store.
        assert chroma.collection_count() == 0
        assert registry.count() == 0

    def test_nothing_expired_short_circuits(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
    ) -> None:
        pf = _filing_with_suffix(stamp, "077")
        store.store_filing(pf)

        report = store.evict_expired(max_age_days=30)
        assert report.filings_evicted == 0
        assert report.chunks_evicted == 0
        # The fresh filing is untouched.
        assert chroma.collection_count() == 2
        assert registry.count() == 1

    def test_expired_rows_removed_from_both_stores(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
    ) -> None:
        pf_old = _filing_with_suffix(stamp, "077")
        pf_fresh = _filing_with_suffix(stamp, "088")
        store.store_filing(pf_old)
        store.store_filing(pf_fresh)
        _backdate_filing(registry, pf_old.filing_id.accession_number, days=120)

        report = store.evict_expired(max_age_days=90)
        assert report.filings_evicted == 1
        # Each fixture filing has 2 chunks (the helper default).
        assert report.chunks_evicted == 2
        assert report.max_age_days == 90

        # Fresh filing survives both stores.
        assert chroma.collection_count() == 2
        assert registry.count() == 1
        assert registry.is_duplicate(pf_fresh.filing_id.accession_number) is True
        assert registry.is_duplicate(pf_old.filing_id.accession_number) is False

    def test_chroma_failure_leaves_sqlite_untouched(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Eviction inherits the ChromaDB-first delete contract — a
        Chroma failure must leave the SQLite rows intact so the caller
        can retry idempotently."""
        pf = _filing_with_suffix(stamp, "077")
        store.store_filing(pf)
        _backdate_filing(registry, pf.filing_id.accession_number, days=120)

        def _raise(*args: Any, **kwargs: Any) -> None:
            raise DatabaseError("simulated ChromaDB failure")

        monkeypatch.setattr(chroma, "delete_filings_batch", _raise)

        with pytest.raises(DatabaseError):
            store.evict_expired(max_age_days=90)

        assert registry.count() == 1

    def test_chroma_first_ordering(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pf = _filing_with_suffix(stamp, "077")
        store.store_filing(pf)
        _backdate_filing(registry, pf.filing_id.accession_number, days=120)

        order: list[str] = []
        real_chroma_delete = chroma.delete_filings_batch
        real_registry_remove = registry.remove_filings_batch

        def _chroma_spy(accs: list[str]) -> None:
            order.append("chroma")
            real_chroma_delete(accs)

        def _registry_spy(accs: list[str]) -> int:
            order.append("sqlite")
            return real_registry_remove(accs)

        monkeypatch.setattr(chroma, "delete_filings_batch", _chroma_spy)
        monkeypatch.setattr(registry, "remove_filings_batch", _registry_spy)

        store.evict_expired(max_age_days=90)
        assert order == ["chroma", "sqlite"]

    def test_chunks_summed_from_listed_records(
        self,
        store: FilingStore,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
    ) -> None:
        """``EvictionReport.chunks_evicted`` aggregates the registry's
        ``chunk_count`` column for every listed filing."""
        pf_a = _filing_with_suffix(stamp, "077")
        pf_b = _filing_with_suffix(stamp, "088")
        store.store_filing(pf_a)
        store.store_filing(pf_b)
        _backdate_filing(registry, pf_a.filing_id.accession_number, days=120)
        _backdate_filing(registry, pf_b.filing_id.accession_number, days=200)

        report = store.evict_expired(max_age_days=90)
        # Each fixture has 2 chunks.
        assert report.filings_evicted == 2
        assert report.chunks_evicted == 4

    @pytest.mark.security
    def test_zero_cutoff_rejected_via_registry(
        self,
        store: FilingStore,
    ) -> None:
        """The registry primitive rejects non-positive cutoffs;
        ``FilingStore`` must propagate that error verbatim so the
        defence-in-depth contract holds at every layer."""
        with pytest.raises(ValueError, match="must be positive"):
            store.evict_expired(max_age_days=0)

    @pytest.mark.security
    def test_negative_cutoff_rejected_via_registry(
        self,
        store: FilingStore,
    ) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            store.evict_expired(max_age_days=-30)


# ---------------------------------------------------------------------------
# Security / defence-in-depth
# ---------------------------------------------------------------------------


class TestSecurity:
    @pytest.mark.security
    def test_store_refuses_none_embeddings_default_path(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
    ) -> None:
        """Default path preserves ChromaDBClient's no-None refusal.

        The orchestrator contract says the storage layer refuses a
        ``ProcessedFiling`` whose embeddings are ``None``.  The
        default path must not touch SQLite when ChromaDB refuses.
        """
        pf = _make_processed_filing(stamp, with_embeddings=False)

        with pytest.raises(DatabaseError, match="embeddings is None"):
            store.store_filing(pf)

        assert registry.count() == 0
        assert chroma.collection_count() == 0

    @pytest.mark.security
    def test_store_refuses_none_embeddings_atomic_path(
        self,
        store: FilingStore,
        chroma: ChromaDBClient,
        registry: MetadataRegistry,
        stamp: EmbedderStamp,
    ) -> None:
        """Atomic path must also enforce the no-None refusal.

        The SQLite claim runs first but the ChromaDB step must still
        refuse a ``ProcessedFiling`` missing its embeddings.  When
        that happens the SQLite claim is rolled back so the atomic
        path leaves both stores in their pre-call state.
        """
        pf = _make_processed_filing(stamp, with_embeddings=False)

        with pytest.raises(DatabaseError, match="embeddings is None"):
            store.store_filing(pf, register_if_new=True)

        assert registry.count() == 0
        assert chroma.collection_count() == 0

    @pytest.mark.security
    def test_store_instance_holds_no_credentials(
        self,
        store: FilingStore,
    ) -> None:
        """FilingStore must not grow credential-bearing attributes.

        Mirrors the credential field-name checks used elsewhere.
        Guards against a future refactor that adds a secret-shaped
        attribute.  ``_chroma`` and ``_registry`` are intentionally
        excluded — they are collaborators, not credential slots.
        """
        forbidden = (
            "api_key",
            "apikey",
            "secret",
            "password",
            "bearer",
            "authorization",
            "credential",
            "encryption_key",
        )
        attrs = set(vars(store).keys())
        lower_attrs = {a.lower() for a in attrs}
        for hint in forbidden:
            assert hint not in lower_attrs, (
                f"FilingStore grew a credential-shaped attribute: {hint}"
            )
        # "token" is a separate check — allow common word fragments
        # (e.g. ``token_count`` on related dataclasses) but reject a
        # bare ``token`` / ``api_token`` style field.
        assert "token" not in lower_attrs
        assert "api_token" not in lower_attrs
