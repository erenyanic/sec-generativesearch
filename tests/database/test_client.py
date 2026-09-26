"""Tests for :class:`sec_generative_search.database.ChromaDBClient`.

Covers the embedder-stamp contract, the
store-refuse-on-None-embeddings guarantee (orchestrator contract), and
smoke coverage of the read/write surface.

Tests drive a real :class:`chromadb.PersistentClient` under
``tmp_path`` — Chroma's sqlite backend is fast and reliable enough that
mocking it would obscure real contract breakage without a meaningful
speed gain.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pytest

from sec_generative_search.config.constants import COLLECTION_NAME
from sec_generative_search.core.exceptions import (
    DatabaseError,
    EmbeddingCollectionMismatchError,
)
from sec_generative_search.core.logging import configure_logging
from sec_generative_search.core.types import (
    Chunk,
    ContentType,
    EmbedderStamp,
    FilingIdentifier,
    IngestResult,
    SearchResult,
)
from sec_generative_search.database import ChromaDBClient
from sec_generative_search.pipeline.orchestrator import ProcessedFiling

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chroma_path(tmp_path: Path) -> str:
    """A per-test Chroma persistence directory."""
    return str(tmp_path / "chroma")


@pytest.fixture
def openai_stamp() -> EmbedderStamp:
    return EmbedderStamp(
        provider="openai",
        model="text-embedding-3-small",
        dimension=4,
    )


@pytest.fixture
def local_stamp() -> EmbedderStamp:
    return EmbedderStamp(
        provider="local",
        model="google/embeddinggemma-300m",
        dimension=4,
    )


def _make_filing_id(
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
    # Match the stamp's declared dimension so ChromaDB accepts the
    # first insert and pins the collection dimension.
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


# ---------------------------------------------------------------------------
# Stamp seal + verification
# ---------------------------------------------------------------------------


class TestStampSealing:
    def test_fresh_collection_is_stamped(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)

        assert client.stamp == openai_stamp
        meta = client._collection.metadata or {}
        assert meta["embedding_provider"] == "openai"
        assert meta["embedding_model"] == "text-embedding-3-small"
        # Dimension is serialised as str in Chroma metadata.
        assert meta["embedding_dimension"] == "4"

    def test_reopen_with_matching_stamp_is_noop(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        # Second open must succeed against the same store.
        second = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        assert second.stamp == openai_stamp

    def test_reopen_with_different_provider_raises_mismatch(
        self,
        chroma_path: str,
        openai_stamp: EmbedderStamp,
        local_stamp: EmbedderStamp,
    ) -> None:
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)

        with pytest.raises(EmbeddingCollectionMismatchError) as excinfo:
            ChromaDBClient(local_stamp, chroma_path=chroma_path)

        err = excinfo.value
        assert err.expected == local_stamp
        assert err.actual == openai_stamp

    def test_reopen_with_different_model_raises_mismatch(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        other = EmbedderStamp(
            provider=openai_stamp.provider,
            model="text-embedding-3-large",
            dimension=openai_stamp.dimension,
        )
        with pytest.raises(EmbeddingCollectionMismatchError) as excinfo:
            ChromaDBClient(other, chroma_path=chroma_path)
        assert excinfo.value.actual.model == openai_stamp.model
        assert excinfo.value.expected.model == "text-embedding-3-large"

    def test_reopen_with_different_dimension_raises_mismatch(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        other = EmbedderStamp(
            provider=openai_stamp.provider,
            model=openai_stamp.model,
            dimension=openai_stamp.dimension + 1,
        )
        with pytest.raises(EmbeddingCollectionMismatchError) as excinfo:
            ChromaDBClient(other, chroma_path=chroma_path)
        assert excinfo.value.actual.dimension == openai_stamp.dimension

    def test_corrupt_stamp_raises_database_error(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)

        # Corrupt the dimension key directly on the collection metadata.
        raw = chromadb.PersistentClient(path=chroma_path)
        collection = raw.get_collection(name=COLLECTION_NAME)
        corrupt = {
            k: v for k, v in (collection.metadata or {}).items() if not k.startswith("hnsw:")
        }
        corrupt["embedding_dimension"] = "not-an-integer"
        collection.modify(metadata=corrupt)

        with pytest.raises(DatabaseError) as excinfo:
            ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        # Not a mismatch — category is distinct.
        assert not isinstance(excinfo.value, EmbeddingCollectionMismatchError)
        assert "corrupt" in str(excinfo.value).lower()

    def test_partial_stamp_raises_database_error(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)

        raw = chromadb.PersistentClient(path=chroma_path)
        collection = raw.get_collection(name=COLLECTION_NAME)
        # Remove the model key while keeping provider + dimension.
        partial = {
            k: v
            for k, v in (collection.metadata or {}).items()
            if not k.startswith("hnsw:") and k != "embedding_model"
        }
        collection.modify(metadata=partial)

        with pytest.raises(DatabaseError) as excinfo:
            ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        assert not isinstance(excinfo.value, EmbeddingCollectionMismatchError)

    def test_empty_unstamped_collection_is_stamped_on_open(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        # Pre-create an empty collection without a stamp (legacy path).
        raw = chromadb.PersistentClient(path=chroma_path)
        raw.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        meta = client._collection.metadata or {}
        assert meta["embedding_provider"] == "openai"
        assert meta["embedding_model"] == "text-embedding-3-small"
        assert meta["embedding_dimension"] == "4"

    def test_populated_unstamped_collection_is_refused(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        # Simulate a populated legacy collection with no stamp.
        raw = chromadb.PersistentClient(path=chroma_path)
        collection = raw.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            ids=["legacy-0"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            documents=["legacy chunk"],
            metadatas=[{"ticker": "AAPL"}],
        )

        with pytest.raises(DatabaseError) as excinfo:
            ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        assert not isinstance(excinfo.value, EmbeddingCollectionMismatchError)
        assert "reindex" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Write + read surface
# ---------------------------------------------------------------------------


class TestStoreAndQuery:
    def test_store_filing_persists_chunks(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        pf = _make_processed_filing(openai_stamp, n_chunks=3)

        client.store_filing(pf)

        assert client.collection_count() == 3

    def test_store_filing_refuses_none_embeddings(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        pf = _make_processed_filing(openai_stamp, with_embeddings=False)

        with pytest.raises(DatabaseError) as excinfo:
            client.store_filing(pf)
        assert "embeddings is none" in str(excinfo.value).lower()
        # No partial write — the collection stays empty.
        assert client.collection_count() == 0

    def test_query_returns_search_results(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        pf = _make_processed_filing(openai_stamp, n_chunks=2)
        client.store_filing(pf)

        # Use the first stored embedding as the query to guarantee a hit.
        query = pf.embeddings[0].tolist()  # type: ignore[union-attr]
        results = client.query([query], n_results=2)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].ticker == "AAPL"
        assert results[0].form_type == "10-K"
        # Highest similarity first (cosine sim = 1 - distance).
        assert results[0].similarity >= results[1].similarity

    def test_query_filters_by_ticker(self, chroma_path: str, openai_stamp: EmbedderStamp) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        aapl = _make_processed_filing(
            openai_stamp,
            filing_id=_make_filing_id(ticker="AAPL", accession_number="0000320193-23-000077"),
            n_chunks=2,
        )
        msft = _make_processed_filing(
            openai_stamp,
            filing_id=_make_filing_id(ticker="MSFT", accession_number="0000789019-23-000001"),
            n_chunks=2,
        )
        client.store_filing(aapl)
        client.store_filing(msft)

        results = client.query(
            [aapl.embeddings[0].tolist()],  # type: ignore[union-attr]
            n_results=10,
            ticker="MSFT",
        )
        assert all(r.ticker == "MSFT" for r in results)

    def test_query_filters_by_date_range(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        old = _make_processed_filing(
            openai_stamp,
            filing_id=_make_filing_id(
                filing_date=date(2020, 1, 15),
                accession_number="0000320193-20-000001",
            ),
            n_chunks=1,
        )
        new = _make_processed_filing(
            openai_stamp,
            filing_id=_make_filing_id(
                filing_date=date(2024, 6, 1),
                accession_number="0000320193-24-000002",
            ),
            n_chunks=1,
        )
        client.store_filing(old)
        client.store_filing(new)

        results = client.query(
            [old.embeddings[0].tolist()],  # type: ignore[union-attr]
            n_results=10,
            start_date="2024-01-01",
        )
        assert len(results) == 1
        assert results[0].filing_date == "2024-06-01"

    def test_delete_filing_removes_all_chunks(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        pf = _make_processed_filing(openai_stamp, n_chunks=3)
        client.store_filing(pf)

        assert client.collection_count() == 3
        client.delete_filing(pf.filing_id.accession_number)
        assert client.collection_count() == 0

    def test_delete_filings_batch_uses_single_call(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        a = _make_processed_filing(
            openai_stamp,
            filing_id=_make_filing_id(ticker="AAPL", accession_number="0000320193-23-000077"),
            n_chunks=2,
        )
        b = _make_processed_filing(
            openai_stamp,
            filing_id=_make_filing_id(ticker="MSFT", accession_number="0000789019-23-000001"),
            n_chunks=2,
        )
        client.store_filing(a)
        client.store_filing(b)
        assert client.collection_count() == 4

        client.delete_filings_batch([a.filing_id.accession_number, b.filing_id.accession_number])
        assert client.collection_count() == 0

    def test_delete_filings_batch_empty_list_is_noop(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        client.delete_filings_batch([])
        assert client.collection_count() == 0


class FaultyCollection:
    """Delegate to a real collection; fail the Nth ``add`` and optionally every ``delete``.

    A wrapper rather than a monkeypatch: the Chroma collection object is a
    model whose attributes are not reliably patchable.
    """

    def __init__(self, inner: Any, *, fail_add_call: int = 0, fail_delete: bool = False) -> None:
        self._inner = inner
        self._fail_add_call = fail_add_call
        self._fail_delete = fail_delete
        self.add_sizes: list[int] = []
        self.delete_calls = 0

    def add(self, **kwargs: Any) -> Any:
        self.add_sizes.append(len(kwargs["ids"]))
        if len(self.add_sizes) == self._fail_add_call:
            raise RuntimeError("simulated chroma add failure")
        return self._inner.add(**kwargs)

    def delete(self, **kwargs: Any) -> Any:
        self.delete_calls += 1
        if self._fail_delete:
            raise RuntimeError("simulated chroma delete failure")
        return self._inner.delete(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def cap_batch(client: ChromaDBClient, monkeypatch: pytest.MonkeyPatch, size: int) -> None:
    """Shrink the client's reported batch cap so slicing runs on a few chunks."""
    monkeypatch.setattr(client._client, "get_max_batch_size", lambda: size)


@pytest.fixture
def pkg_log_records(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Surface package records to ``caplog`` (the package logger does not propagate)."""
    configure_logging()
    pkg_logger = logging.getLogger("sec_generative_search")
    previous = pkg_logger.propagate
    pkg_logger.propagate = True
    caplog.set_level(logging.INFO, logger="sec_generative_search")
    try:
        yield caplog
    finally:
        pkg_logger.propagate = previous


class TestBatchedStore:
    """``store_filing`` slices by Chroma's batch cap and stays all-or-nothing (F23)."""

    def test_filing_above_the_chroma_batch_cap_is_stored_whole(
        self, openai_stamp: EmbedderStamp, chroma_path: str
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        cap = client._client.get_max_batch_size()
        n_chunks = cap + 1  # the smallest filing a single add refuses whole
        client.store_filing(_make_processed_filing(openai_stamp, n_chunks=n_chunks))
        assert client.collection_count() == n_chunks

    def test_slices_never_exceed_the_reported_cap(
        self,
        openai_stamp: EmbedderStamp,
        chroma_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        cap_batch(client, monkeypatch, 2)
        proxy = FaultyCollection(client._collection)
        client._collection = proxy
        client.store_filing(_make_processed_filing(openai_stamp, n_chunks=5))
        assert proxy.add_sizes == [2, 2, 1]
        assert client.collection_count() == 5

    @pytest.mark.security
    def test_mid_sequence_failure_rolls_back_the_written_slices(
        self,
        openai_stamp: EmbedderStamp,
        chroma_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Orphan chunks would be retrievable with no registry row — outside
        retention eviction and the ``DB_MAX_FILINGS`` ceiling."""
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        cap_batch(client, monkeypatch, 2)
        proxy = FaultyCollection(client._collection, fail_add_call=2)
        client._collection = proxy

        with pytest.raises(DatabaseError, match="Failed to store filing") as excinfo:
            client.store_filing(_make_processed_filing(openai_stamp, n_chunks=5))

        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert proxy.delete_calls == 1
        assert client.collection_count() == 0

    def test_first_slice_failure_needs_no_rollback(
        self,
        openai_stamp: EmbedderStamp,
        chroma_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        cap_batch(client, monkeypatch, 2)
        proxy = FaultyCollection(client._collection, fail_add_call=1)
        client._collection = proxy

        with pytest.raises(DatabaseError):
            client.store_filing(_make_processed_filing(openai_stamp, n_chunks=5))

        assert proxy.delete_calls == 0
        assert client.collection_count() == 0

    def test_rollback_failure_never_masks_the_original_error(
        self,
        openai_stamp: EmbedderStamp,
        chroma_path: str,
        monkeypatch: pytest.MonkeyPatch,
        pkg_log_records: pytest.LogCaptureFixture,
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        cap_batch(client, monkeypatch, 2)
        client._collection = FaultyCollection(client._collection, fail_add_call=2, fail_delete=True)

        with pytest.raises(DatabaseError) as excinfo:
            client.store_filing(_make_processed_filing(openai_stamp, n_chunks=5))

        assert str(excinfo.value.__cause__) == "simulated chroma add failure"
        failures = [r for r in pkg_log_records.records if r.levelno == logging.ERROR]
        assert len(failures) == 1
        rendered = failures[0].getMessage()
        assert "RuntimeError" in rendered
        # Exception type only — driver text never reaches the log line.
        assert "simulated chroma delete failure" not in rendered

    def test_duplicate_chunk_ids_are_refused_before_any_write(
        self,
        openai_stamp: EmbedderStamp,
        chroma_path: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One ``add`` refuses duplicate IDs; across slices a repeat would be a
        silent no-op and store fewer chunks than the registry records."""
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        cap_batch(client, monkeypatch, 2)
        pf = _make_processed_filing(openai_stamp, n_chunks=3)
        pf.chunks[2].chunk_index = 0  # same ID as chunk 0, in a later slice

        with pytest.raises(DatabaseError, match="duplicate chunk IDs"):
            client.store_filing(pf)
        assert client.collection_count() == 0


class TestClearCollection:
    def test_clear_empty_returns_zero(self, chroma_path: str, openai_stamp: EmbedderStamp) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        assert client.clear_collection() == 0

    def test_clear_populated_rebuilds_and_reseals(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        pf = _make_processed_filing(openai_stamp, n_chunks=3)
        client.store_filing(pf)
        assert client.collection_count() == 3

        removed = client.clear_collection()
        assert removed == 3
        assert client.collection_count() == 0

        meta = client._collection.metadata or {}
        assert meta["embedding_provider"] == openai_stamp.provider
        assert meta["embedding_model"] == openai_stamp.model
        assert meta["embedding_dimension"] == str(openai_stamp.dimension)

        # Reopening with the same stamp must still succeed.
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)

    def test_clear_then_open_with_different_stamp_still_raises(
        self,
        chroma_path: str,
        openai_stamp: EmbedderStamp,
        local_stamp: EmbedderStamp,
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        pf = _make_processed_filing(openai_stamp, n_chunks=1)
        client.store_filing(pf)
        client.clear_collection()

        with pytest.raises(EmbeddingCollectionMismatchError):
            ChromaDBClient(local_stamp, chroma_path=chroma_path)


# ---------------------------------------------------------------------------
# Migration flag preserved
# ---------------------------------------------------------------------------


class TestMigrationFlag:
    """``_MIGRATION_FLAG`` keeps the ``filing_date_int`` backfill O(1)
    after the first scan.  These tests are tagged ``@pytest.mark.security``
    because the flag is the load-bearing seal that prevents the migration
    from re-running on every startup; drift here means latent
    data-touching work on a hot path."""

    @pytest.mark.security
    def test_migration_flag_set_on_fresh_collection(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        meta = client._collection.metadata or {}
        assert meta.get(ChromaDBClient._MIGRATION_FLAG) is True

    @pytest.mark.security
    def test_migration_is_o1_after_first_run(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        # Second open must not re-scan — we can't easily observe that
        # from the outside, but we at least assert the flag survives.
        second = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        meta = second._collection.metadata or {}
        assert meta.get(ChromaDBClient._MIGRATION_FLAG) is True

    @pytest.mark.security
    def test_migration_short_circuits_on_stamped_open(
        self,
        chroma_path: str,
        openai_stamp: EmbedderStamp,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A second open with the flag already set must not execute the
        body of ``_migrate_filing_date_int``.

        We spy on Chroma's ``get`` (the first read inside the
        migration body) and assert it is never invoked on the second
        open.  The flag check happens before any data-touching
        operation, so this directly pins the O(1) startup contract.
        """
        # First open seals the flag.
        first = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        meta = first._collection.metadata or {}
        assert meta.get(ChromaDBClient._MIGRATION_FLAG) is True

        # Spy on Chroma's ``get`` — the migration body's first call.
        # Patching the class method makes the spy survive the new
        # ``ChromaDBClient`` constructing its own collection handle.
        from chromadb.api.models.Collection import Collection

        get_calls: list[tuple[Any, ...]] = []
        original_get = Collection.get

        def _spy(self: Any, *args: Any, **kwargs: Any) -> Any:
            get_calls.append((args, kwargs))
            return original_get(self, *args, **kwargs)

        monkeypatch.setattr(Collection, "get", _spy)

        # Second open — flag is set, so the migration body must skip.
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)

        assert get_calls == [], (
            "Migration body executed on a stamped collection; the O(1) startup invariant is broken."
        )


# ---------------------------------------------------------------------------
# Security-focused assertions
# ---------------------------------------------------------------------------


class TestSecurity:
    @pytest.mark.security
    def test_stamp_metadata_carries_no_credential_field_names(
        self, openai_stamp: EmbedderStamp
    ) -> None:
        """The collection-metadata keys must never resemble secrets.

        Defence in depth against a future refactor that tries to stash
        credentials alongside the stamp.
        """
        rendered = openai_stamp.to_metadata()
        credential_hints = {
            "api_key",
            "apikey",
            "secret",
            "password",
            "bearer",
            "token",
            "authorization",
            "auth",
        }
        for key in rendered:
            lowered = key.lower()
            assert all(hint not in lowered for hint in credential_hints), (
                f"Stamp metadata key {key!r} resembles a credential field name"
            )

    @pytest.mark.security
    def test_mismatch_error_surfaces_uniform_hint(
        self,
        chroma_path: str,
        openai_stamp: EmbedderStamp,
        local_stamp: EmbedderStamp,
    ) -> None:
        """Mismatch hint must be the uniform, deployment-unaware string.

        The API lifespan hook is the only scenario-aware translator of
        this error.  Branching the hint per deployment inside the
        storage layer would split the refusal message across N surfaces
        and regress operator clarity.
        """
        ChromaDBClient(openai_stamp, chroma_path=chroma_path)

        with pytest.raises(EmbeddingCollectionMismatchError) as excinfo:
            ChromaDBClient(local_stamp, chroma_path=chroma_path)

        hint = excinfo.value.hint
        assert "reindex" in hint.lower()
        assert "sec-rag manage reindex" in hint

    @pytest.mark.security
    def test_store_filing_refuses_none_embeddings(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        """Silent storage of chunks without vectors would corrupt retrieval.

        Honours the orchestrator docstring's contract that the storage
        layer refuses rather than quietly drops the embedding step.
        """
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        pf = _make_processed_filing(openai_stamp, with_embeddings=False)

        with pytest.raises(DatabaseError):
            client.store_filing(pf)
        assert client.collection_count() == 0

    @pytest.mark.security
    def test_populated_unstamped_collection_refuses_with_reindex_hint(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        """Legacy populated collections without a stamp must refuse traffic.

        We cannot prove the stored vectors were produced by the
        configured embedder, so retrieval would be silently wrong.
        """
        raw = chromadb.PersistentClient(path=chroma_path)
        collection = raw.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            ids=["legacy-0"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            documents=["legacy chunk"],
            metadatas=[{"ticker": "AAPL"}],
        )

        with pytest.raises(DatabaseError) as excinfo:
            ChromaDBClient(openai_stamp, chroma_path=chroma_path)
        assert "reindex" in str(excinfo.value).lower()

    @pytest.mark.security
    def test_client_constructor_does_not_retain_credentials(
        self, chroma_path: str, openai_stamp: EmbedderStamp
    ) -> None:
        """The client never stores an API key or EDGAR identity.

        Stamp fields are non-secret (provider/model/dimension); every
        other credential lives in the provider layer.  A future
        refactor must not sneak a key-shaped attribute onto the client.
        """
        client = ChromaDBClient(openai_stamp, chroma_path=chroma_path)

        credential_hints = {
            "api_key",
            "apikey",
            "secret",
            "password",
            "bearer",
            "token",
            "authorization",
        }
        for attr in vars(client):
            lowered = attr.lower()
            assert all(hint not in lowered for hint in credential_hints), (
                f"Client attribute {attr!r} resembles a credential field name"
            )
