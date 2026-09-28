"""Tests for ``sec-rag ingest add`` and ``sec-rag ingest batch``.

The two backing storage classes (:class:`ChromaDBClient`,
:class:`MetadataRegistry`), the :class:`FilingStore` coordinator, the
:class:`FilingFetcher`, the :class:`PipelineOrchestrator`, and the
:func:`build_embedder` factory are all stubbed at the ``cli.ingest``
import site — stubbing at the import site (rather than at the source
modules) proves the CLI wires the right seams.

Goals:

- Confirm writes flow through :class:`FilingStore`, never through
  :class:`ChromaDBClient` / :class:`MetadataRegistry` mutation methods.
- Confirm the embedder is built via :func:`build_embedder` (the sole
  construction seam) and the resulting stamp seals the ChromaDB client.
- Cover the operator-facing surface: success, duplicates, fetch
  failures, processing failures, storage failures, filing-limit
  exhaustion, mutually-exclusive flags, malformed dates.
- Validate the security stance: API keys / EDGAR identity must never
  appear in any output path.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from typing import Any, ClassVar

import pytest
import typer
from typer.testing import CliRunner

import sec_generative_search.cli.ingest as ingest_module
from sec_generative_search.cli.ingest import ingest_app
from sec_generative_search.config.settings import EmbeddingSettings
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    DatabaseError,
    FetchError,
    FilingLimitExceededError,
)
from sec_generative_search.core.types import (
    EmbedderStamp,
    FilingIdentifier,
    IngestResult,
)
from sec_generative_search.pipeline import FilingInfo, ProcessedFiling

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _make_filing_id(
    ticker: str = "AAPL",
    form_type: str = "10-K",
    filing_date: date = date(2024, 1, 15),
    accession: str = "0000320193-24-000001",
) -> FilingIdentifier:
    return FilingIdentifier(
        ticker=ticker,
        form_type=form_type,
        filing_date=filing_date,
        accession_number=accession,
    )


def _make_processed(filing_id: FilingIdentifier) -> ProcessedFiling:
    return ProcessedFiling(
        filing_id=filing_id,
        chunks=[],
        embeddings=None,
        ingest_result=IngestResult(
            filing_id=filing_id,
            segment_count=5,
            chunk_count=10,
            duration_seconds=1.0,
        ),
    )


class _FakeEmbedder:
    """Minimal duck-typed stand-in for a built embedder.

    The CLI never invokes the embedder directly — the orchestrator
    does — so the stub only needs to exist for the factory return
    contract.
    """


class _FakeFetcher:
    """Programmable :class:`FilingFetcher` stand-in.

    ``queued_filings`` is the listing: ``list_available`` returns one
    ``FilingInfo`` per entry and ``fetch_filing_content`` serves the
    entry's HTML.  Calls are recorded so a test can assert on what the CLI
    listed, and which filings it actually downloaded (F21: list first,
    then fetch only new filings, one ahead).
    """

    instances: ClassVar[list[_FakeFetcher]] = []

    def __init__(self) -> None:
        self.list_calls: list[tuple[str, str, dict]] = []
        self.list_across_calls: list[tuple[str, tuple, dict]] = []
        self.fetch_content_calls: list[FilingInfo] = []
        self.queued_filings: list[tuple[FilingIdentifier, str]] = []
        self.queued_list: list[FilingInfo] = []
        self.list_raises: BaseException | None = None
        self.fetch_content_raises: BaseException | None = None
        self.fetch_content_raises_for: dict[str, BaseException] = {}
        # ``company_cache()`` bookkeeping (F22): which tickers were listed
        # inside which scope.
        self.cache_scopes: list[list[str]] = []
        self.fetched_outside_scope: list[str] = []
        self._scope: list[str] | None = None
        _FakeFetcher.instances.append(self)

    @contextmanager
    def company_cache(self) -> Iterator[None]:
        self._scope = []
        self.cache_scopes.append(self._scope)
        try:
            yield
        finally:
            self._scope = None

    def _record_scope(self, ticker: str) -> None:
        if self._scope is None:
            self.fetched_outside_scope.append(ticker)
        else:
            self._scope.append(ticker)

    def list_available(self, ticker: str, form_type: str, **kwargs: Any) -> list[FilingInfo]:
        self._record_scope(ticker)
        self.list_calls.append((ticker, form_type, kwargs))
        if self.list_raises is not None:
            raise self.list_raises
        return [
            FilingInfo(
                ticker=fid.ticker,
                form_type=fid.form_type,
                filing_date=fid.filing_date,
                accession_number=fid.accession_number,
                company_name=f"{fid.ticker} Inc.",
            )
            for fid, _ in self.queued_filings
        ]

    def list_available_across_forms(
        self,
        ticker: str,
        form_types: tuple[str, ...],
        **kwargs: Any,
    ):
        self.list_across_calls.append((ticker, form_types, kwargs))
        if self.list_raises is not None:
            raise self.list_raises
        return list(self.queued_list)

    def fetch_filing_content(self, fi: FilingInfo):
        self.fetch_content_calls.append(fi)
        if self.fetch_content_raises is not None:
            raise self.fetch_content_raises
        if fi.accession_number in self.fetch_content_raises_for:
            raise self.fetch_content_raises_for[fi.accession_number]
        html = next(
            (h for fid, h in self.queued_filings if fid.accession_number == fi.accession_number),
            "<html>body</html>",
        )
        filing_id = FilingIdentifier(
            ticker=fi.ticker,
            form_type=fi.form_type,
            filing_date=fi.filing_date,
            accession_number=fi.accession_number,
        )
        return filing_id, html


class _FakeOrchestrator:
    """Programmable :class:`PipelineOrchestrator` stand-in.

    ``process_filing`` returns a canned :class:`ProcessedFiling` derived
    from the supplied ``filing_id``, or raises a queued exception.
    """

    instances: ClassVar[list[_FakeOrchestrator]] = []

    def __init__(self, *, fetcher: Any = None, embedder: Any = None) -> None:
        self.fetcher = fetcher
        self.embedder = embedder
        self.process_calls: list[FilingIdentifier] = []
        self.process_raises: BaseException | None = None
        _FakeOrchestrator.instances.append(self)

    def process_filing(
        self,
        filing_id: FilingIdentifier,
        html: str,
        progress_callback: Any = None,
    ) -> ProcessedFiling:
        self.process_calls.append(filing_id)
        if self.process_raises is not None:
            raise self.process_raises
        if progress_callback is not None:
            progress_callback("Parsing", 1, 4)
            progress_callback("Complete", 4, 4)
        return _make_processed(filing_id)


class _FakeRegistry:
    """In-memory :class:`MetadataRegistry` stand-in.

    Backs the duplicate-batch query and the filing-limit gate.  Mutation
    methods are deliberately absent — the CLI must never invoke them
    directly (writes flow through :class:`FilingStore`).
    """

    instances: ClassVar[list[_FakeRegistry]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.existing: set[str] = set()
        self.limit_raises: BaseException | None = None
        self.check_limit_calls = 0
        self.existing_queries: list[list[str]] = []
        _FakeRegistry.instances.append(self)

    def get_existing_accessions(self, accessions: list[str]) -> set[str]:
        self.existing_queries.append(list(accessions))
        return {a for a in accessions if a in self.existing}

    def check_filing_limit(self) -> None:
        self.check_limit_calls += 1
        if self.limit_raises is not None:
            raise self.limit_raises


class _FakeChroma:
    """Recording :class:`ChromaDBClient` stand-in.

    Only the constructor matters here — the CLI must seal the
    collection with a stamp built from settings + registry, and writes
    must never touch this object directly.
    """

    instances: ClassVar[list[_FakeChroma]] = []

    def __init__(self, stamp: EmbedderStamp, *, chroma_path: str | None = None) -> None:
        self.stamp = stamp
        self.chroma_path = chroma_path
        _FakeChroma.instances.append(self)


class _FakeStore:
    """:class:`FilingStore` stand-in that records every write.

    ``raises`` lets a test arm a per-write failure (mirrors the
    pattern in :mod:`tests.cli.test_evict`).
    """

    instances: ClassVar[list[_FakeStore]] = []

    def __init__(self, chroma: Any, registry: Any) -> None:
        self.chroma = chroma
        self.registry = registry
        self.calls: list[tuple[FilingIdentifier, bool]] = []
        self.raises: BaseException | None = None
        _FakeStore.instances.append(self)

    def store_filing(
        self,
        processed: ProcessedFiling,
        *,
        register_if_new: bool = False,
    ) -> bool:
        self.calls.append((processed.filing_id, register_if_new))
        if self.raises is not None:
            raise self.raises
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def app() -> typer.Typer:
    """Throw-away Typer app wrapping the ingest sub-Typer.

    The sub-Typer ships as ``ingest`` — keeping the test surface
    identical to the production registration in ``cli/main.py``.
    """
    test_app = typer.Typer()
    test_app.add_typer(ingest_app, name="ingest")
    return test_app


class _StubSettings:
    """Minimal settings stand-in for the CLI body.

    Only the fields the CLI actually reads are populated.  Anything
    else raising :class:`AttributeError` is the loud signal we want.
    """

    def __init__(self) -> None:
        self.embedding = EmbeddingSettings.model_construct(
            provider="openai",
            model_name="text-embedding-3-small",
            device="auto",
            batch_size=32,
            idle_timeout_minutes=0,
        )


@pytest.fixture
def patched_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace every seam ``cli.ingest`` reaches into with a recording stub."""
    _FakeFetcher.instances.clear()
    _FakeOrchestrator.instances.clear()
    _FakeRegistry.instances.clear()
    _FakeChroma.instances.clear()
    _FakeStore.instances.clear()

    monkeypatch.setattr(ingest_module, "get_settings", _StubSettings)
    monkeypatch.setattr(ingest_module, "FilingFetcher", _FakeFetcher)
    monkeypatch.setattr(ingest_module, "PipelineOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(ingest_module, "MetadataRegistry", _FakeRegistry)
    monkeypatch.setattr(ingest_module, "ChromaDBClient", _FakeChroma)
    monkeypatch.setattr(ingest_module, "FilingStore", _FakeStore)

    def _fake_build_embedder(settings: Any, **_kwargs: Any) -> _FakeEmbedder:
        return _FakeEmbedder()

    monkeypatch.setattr(ingest_module, "build_embedder", _fake_build_embedder)

    return {
        "fetchers": _FakeFetcher.instances,
        "orchestrators": _FakeOrchestrator.instances,
        "registries": _FakeRegistry.instances,
        "chromas": _FakeChroma.instances,
        "stores": _FakeStore.instances,
    }


# ---------------------------------------------------------------------------
# add — happy path
# ---------------------------------------------------------------------------


class TestAddHappyPath:
    def test_single_latest_end_to_end(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-to-end ``add AAPL -f 10-K``: list the newest filing, fetch it,
        single store call."""
        filing_id = _make_filing_id(ticker="AAPL", form_type="10-K")

        original_init = _FakeFetcher.__init__

        def _seeded_init(self: _FakeFetcher) -> None:
            original_init(self)
            self.queued_filings = [(filing_id, "<html>body</html>")]

        monkeypatch.setattr(_FakeFetcher, "__init__", _seeded_init)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K"])
        assert result.exit_code == 0, result.output

        fetcher = _FakeFetcher.instances[-1]
        store = _FakeStore.instances[-1]
        orchestrator = _FakeOrchestrator.instances[-1]
        chroma = _FakeChroma.instances[-1]

        assert fetcher.list_calls == [
            ("AAPL", "10-K", {"count": 1, "year": None, "start_date": None, "end_date": None})
        ]
        assert [fi.accession_number for fi in fetcher.fetch_content_calls] == [
            filing_id.accession_number
        ]
        assert orchestrator.process_calls == [filing_id]
        # Write goes through the store — not through chroma directly.
        assert store.calls == [(filing_id, False)]
        # Stamp seals the chroma client with the registry-resolved dim.
        assert chroma.stamp.provider == "openai"
        assert chroma.stamp.model == "text-embedding-3-small"
        assert chroma.stamp.dimension == 1536
        assert "Ingested" in result.output

    def test_with_year_filter_lists_all_matching(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``add AAPL -f 10-K -y 2023`` (a filter, no count) lists every
        matching filing (``count=None``, capped by ``max_filings`` in the
        fetcher)."""
        filing_id = _make_filing_id(filing_date=date(2023, 5, 1))
        original_init = _FakeFetcher.__init__

        def _seeded_init(self: _FakeFetcher) -> None:
            original_init(self)
            self.queued_filings = [(filing_id, "<html>body</html>")]

        monkeypatch.setattr(_FakeFetcher, "__init__", _seeded_init)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K", "-y", "2023"])
        assert result.exit_code == 0, result.output
        fetcher = _FakeFetcher.instances[-1]
        # Filter present without explicit count → "all matching".
        assert fetcher.list_calls and fetcher.list_calls[0][0] == "AAPL"
        assert fetcher.list_calls[0][2]["year"] == 2023
        assert fetcher.list_calls[0][2]["count"] is None

    def test_explicit_number_lists_that_many(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``-n 2`` lists two filings and streams both through the pipeline."""
        fid_a = _make_filing_id(accession="0000320193-24-000001")
        fid_b = _make_filing_id(
            filing_date=date(2023, 1, 15),
            accession="0000320193-23-000077",
        )

        original_init = _FakeFetcher.__init__

        def _seeded_init(self: _FakeFetcher) -> None:
            original_init(self)
            self.queued_filings = [
                (fid_a, "<html>a</html>"),
                (fid_b, "<html>b</html>"),
            ]

        monkeypatch.setattr(_FakeFetcher, "__init__", _seeded_init)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K", "-n", "2"])
        assert result.exit_code == 0, result.output
        fetcher = _FakeFetcher.instances[-1]
        assert fetcher.list_calls and fetcher.list_calls[0][2]["count"] == 2
        assert len(fetcher.fetch_content_calls) == 2

        store = _FakeStore.instances[-1]
        assert len(store.calls) == 2
        assert all(register_if_new is False for _, register_if_new in store.calls)

    def test_total_uses_cross_form_listing(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``-t N`` enters the cross-form path via ``list_available_across_forms``."""
        fi = FilingInfo(
            ticker="AAPL",
            form_type="10-K",
            filing_date=date(2024, 1, 15),
            accession_number="0000320193-24-000001",
            company_name="Apple Inc.",
        )

        original_init = _FakeFetcher.__init__

        def _seeded_init(self: _FakeFetcher) -> None:
            original_init(self)
            self.queued_list = [fi]

        monkeypatch.setattr(_FakeFetcher, "__init__", _seeded_init)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-t", "1"])
        assert result.exit_code == 0, result.output
        fetcher = _FakeFetcher.instances[-1]
        assert len(fetcher.list_across_calls) == 1
        assert fetcher.list_across_calls[0][1] == ("10-K", "10-Q")
        # Cross-form path uses fetch_filing_content (cached _filing_obj path).
        assert len(fetcher.fetch_content_calls) == 1

        store = _FakeStore.instances[-1]
        assert len(store.calls) == 1


# ---------------------------------------------------------------------------
# add — duplicates, failures, edge cases
# ---------------------------------------------------------------------------


class TestAddEdgeCases:
    def test_duplicate_skips_store(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An already-registered accession short-circuits before storage."""
        filing_id = _make_filing_id()
        original_fetcher_init = _FakeFetcher.__init__
        original_registry_init = _FakeRegistry.__init__

        def _seeded_fetcher(self: _FakeFetcher) -> None:
            original_fetcher_init(self)
            self.queued_filings = [(filing_id, "<html>body</html>")]

        def _seeded_registry(self: _FakeRegistry, *args: Any, **kwargs: Any) -> None:
            original_registry_init(self, *args, **kwargs)
            self.existing = {filing_id.accession_number}

        monkeypatch.setattr(_FakeFetcher, "__init__", _seeded_fetcher)
        monkeypatch.setattr(_FakeRegistry, "__init__", _seeded_registry)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K"])
        assert result.exit_code == 0, result.output
        assert "Already ingested" in result.output

        store = _FakeStore.instances[-1]
        assert store.calls == []  # No write attempted for a duplicate.
        # F21: listed first, so a duplicate's HTML is never downloaded.
        assert _FakeFetcher.instances[-1].fetch_content_calls == []

    def test_fetch_error_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A listing failure exits non-zero with a hint."""
        original_init = _FakeFetcher.__init__

        def _seeded_init(self: _FakeFetcher) -> None:
            original_init(self)
            self.list_raises = FetchError("network down", details="EDGAR unreachable")

        monkeypatch.setattr(_FakeFetcher, "__init__", _seeded_init)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K"])
        assert result.exit_code == 1
        # The fetcher's message no longer names the ticker (it is logged by
        # the API worker); the operator's console still does.
        assert "Fetch failed for AAPL 10-K" in result.output
        assert "network down" in result.output

    def test_storage_error_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A :class:`DatabaseError` from the store surfaces as exit 1."""
        filing_id = _make_filing_id()
        original_fetcher_init = _FakeFetcher.__init__
        original_store_init = _FakeStore.__init__

        def _seeded_fetcher(self: _FakeFetcher) -> None:
            original_fetcher_init(self)
            self.queued_filings = [(filing_id, "<html>body</html>")]

        def _seeded_store(self: _FakeStore, *args: Any, **kwargs: Any) -> None:
            original_store_init(self, *args, **kwargs)
            self.raises = DatabaseError("disk full", details="ENOSPC")

        monkeypatch.setattr(_FakeFetcher, "__init__", _seeded_fetcher)
        monkeypatch.setattr(_FakeStore, "__init__", _seeded_store)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K"])
        assert result.exit_code == 1
        assert "Storage failed" in result.output

    def test_mutually_exclusive_flags(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
    ) -> None:
        """``-t`` and ``-n`` cannot coexist — exit 1 with a clear message."""
        result = runner.invoke(app, ["ingest", "add", "AAPL", "-t", "2", "-n", "1"])
        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

    def test_invalid_form_type(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
    ) -> None:
        """An unsupported form rejects at the boundary before any I/O."""
        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "S-1"])
        assert result.exit_code == 1
        assert "Invalid form type" in result.output
        # No pipeline construction occurred.
        assert _FakeStore.instances == []

    def test_invalid_date_format(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
    ) -> None:
        """A malformed ``--start-date`` bounces with ``typer.BadParameter`` shape."""
        result = runner.invoke(
            app, ["ingest", "add", "AAPL", "-f", "10-K", "--start-date", "01-2024"]
        )
        assert result.exit_code != 0
        assert "Invalid date format" in result.output or "01-2024" in result.output

    def test_filing_limit_exit_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``FilingLimitExceededError`` aborts before any fetch happens."""
        original_init = _FakeRegistry.__init__

        def _seeded_init(self: _FakeRegistry, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            self.limit_raises = FilingLimitExceededError(10000, 10000)

        monkeypatch.setattr(_FakeRegistry, "__init__", _seeded_init)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K"])
        assert result.exit_code == 1
        assert "Filing limit reached" in result.output


class TestCompanyCacheScope:
    """F22: all of a ticker's forms are listed inside one ``company_cache()``
    scope — one edgartools ``Company`` per ticker, never one per form."""

    def test_add_lists_every_form_in_one_scope(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        filing_id = _make_filing_id(ticker="AAPL", form_type="10-K")
        original_init = _FakeFetcher.__init__

        def _seeded_init(self: _FakeFetcher) -> None:
            original_init(self)
            self.queued_filings = [(filing_id, "<html>body</html>")]

        monkeypatch.setattr(_FakeFetcher, "__init__", _seeded_init)
        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K,10-Q,8-K"])
        assert result.exit_code == 0, result.output

        fetcher = _FakeFetcher.instances[-1]
        assert fetcher.cache_scopes == [["AAPL", "AAPL", "AAPL"]]
        assert fetcher.fetched_outside_scope == []

    def test_batch_opens_one_scope_per_ticker(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        filing_id = _make_filing_id(ticker="AAPL", form_type="10-K")
        original_init = _FakeFetcher.__init__

        def _seeded_init(self: _FakeFetcher) -> None:
            original_init(self)
            self.queued_filings = [(filing_id, "<html>body</html>")]

        monkeypatch.setattr(_FakeFetcher, "__init__", _seeded_init)
        result = runner.invoke(app, ["ingest", "batch", "AAPL", "MSFT", "-f", "10-K,10-Q"])
        assert result.exit_code == 0, result.output

        fetcher = _FakeFetcher.instances[-1]
        # Per ticker, so one ticker's index is never held across the batch.
        assert fetcher.cache_scopes == [["AAPL", "AAPL"], ["MSFT", "MSFT"]]
        assert fetcher.fetched_outside_scope == []


class TestStreamingIngest:
    """F21: the CLI lists metadata, then fetches only new filings — one ahead
    of the filing being processed — instead of downloading every filing's
    HTML up front."""

    @staticmethod
    def _three_filings() -> list[FilingIdentifier]:
        return [
            _make_filing_id(accession=f"0000320193-24-00000{i}", filing_date=date(2024, 1, i))
            for i in (1, 2, 3)
        ]

    def _seed(self, monkeypatch: pytest.MonkeyPatch, fids: list[FilingIdentifier]) -> None:
        original_init = _FakeFetcher.__init__

        def _seeded_init(self: _FakeFetcher) -> None:
            original_init(self)
            self.queued_filings = [(fid, f"<html>{fid.accession_number}</html>") for fid in fids]
            self.queued_list = [
                FilingInfo(
                    ticker=fid.ticker,
                    form_type=fid.form_type,
                    filing_date=fid.filing_date,
                    accession_number=fid.accession_number,
                    company_name="Apple Inc.",
                )
                for fid in fids
            ]

        monkeypatch.setattr(_FakeFetcher, "__init__", _seeded_init)

    def _assert_overlap(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fids: list[FilingIdentifier],
    ) -> list[bool]:
        """While filing 1 is processed, filing 2 must already be downloading."""
        overlap: list[bool] = []
        original_process = _FakeOrchestrator.process_filing

        def _process(self, filing_id, html, progress_callback=None):
            if filing_id == fids[0]:
                fetcher = _FakeFetcher.instances[-1]
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    if any(
                        fi.accession_number == fids[1].accession_number
                        for fi in fetcher.fetch_content_calls
                    ):
                        break
                    time.sleep(0.005)
                overlap.append(
                    any(
                        fi.accession_number == fids[1].accession_number
                        for fi in fetcher.fetch_content_calls
                    )
                )
            return original_process(self, filing_id, html, progress_callback)

        monkeypatch.setattr(_FakeOrchestrator, "process_filing", _process)
        return overlap

    def test_per_form_path_fetches_the_next_filing_during_processing(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fids = self._three_filings()
        self._seed(monkeypatch, fids)
        overlap = self._assert_overlap(monkeypatch, fids)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K", "-n", "3"])
        assert result.exit_code == 0, result.output
        assert overlap == [True]
        assert [fid for fid, _ in _FakeStore.instances[-1].calls] == fids

    def test_cross_form_path_fetches_the_next_filing_during_processing(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fids = self._three_filings()
        self._seed(monkeypatch, fids)
        overlap = self._assert_overlap(monkeypatch, fids)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-t", "3"])
        assert result.exit_code == 0, result.output
        assert overlap == [True]
        assert [fid for fid, _ in _FakeStore.instances[-1].calls] == fids

    @pytest.mark.parametrize(
        "flags", [["-f", "10-K", "-n", "3"], ["-t", "3"]], ids=["per-form", "cross-form"]
    )
    def test_a_duplicate_after_a_new_filing_is_never_prefetched(
        self,
        flags: list[str],
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fids = self._three_filings()
        self._seed(monkeypatch, fids)
        original_registry_init = _FakeRegistry.__init__

        def _second_is_known(self: _FakeRegistry, *args: Any, **kwargs: Any) -> None:
            original_registry_init(self, *args, **kwargs)
            self.existing = {fids[1].accession_number}

        monkeypatch.setattr(_FakeRegistry, "__init__", _second_is_known)

        result = runner.invoke(app, ["ingest", "add", "AAPL", *flags])
        assert result.exit_code == 0, result.output
        fetched = [fi.accession_number for fi in _FakeFetcher.instances[-1].fetch_content_calls]
        assert fetched == [fids[0].accession_number, fids[2].accession_number]

    def test_a_failed_filing_fetch_is_reported_and_the_run_continues(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Before F21 ``fetch()`` logged and silently dropped such a filing;
        now it is a visible, counted failure, like the cross-form path."""
        fids = self._three_filings()
        self._seed(monkeypatch, fids)
        original_init = _FakeFetcher.__init__

        def _failing_second(self: _FakeFetcher) -> None:
            original_init(self)
            self.fetch_content_raises_for = {
                fids[1].accession_number: FetchError("Empty HTML content received")
            }

        monkeypatch.setattr(_FakeFetcher, "__init__", _failing_second)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K", "-n", "3"])
        assert result.exit_code == 0, result.output
        assert "Fetch failed [2/3]" in result.output
        assert [fid for fid, _ in _FakeStore.instances[-1].calls] == [fids[0], fids[2]]
        assert "1 failed" in result.output


# ---------------------------------------------------------------------------
# batch — happy path
# ---------------------------------------------------------------------------


class TestBatchHappyPath:
    def test_two_tickers_each_one_filing(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``batch AAPL MSFT`` issues one ``store_filing`` per ticker."""
        fid_aapl = _make_filing_id(ticker="AAPL", accession="0000320193-24-000001")
        fid_msft = _make_filing_id(ticker="MSFT", accession="0000789019-24-000002")

        original_init = _FakeFetcher.__init__
        calls = {"count": 0}

        def _seeded_init(self: _FakeFetcher) -> None:
            original_init(self)
            # Both tickers share the same fetcher; the CLI lists once per
            # ticker, so serve each ticker its own filing.
            calls["count"] = 0
            original_list = self.list_available

            def _list_for(ticker: str, form: str, **kwargs: Any) -> list[FilingInfo]:
                fid = fid_aapl if ticker == "AAPL" else fid_msft
                self.queued_filings = [(fid, f"<html>{ticker.lower()}</html>")]
                return original_list(ticker, form, **kwargs)

            self.list_available = _list_for  # type: ignore[method-assign]

        monkeypatch.setattr(_FakeFetcher, "__init__", _seeded_init)

        result = runner.invoke(app, ["ingest", "batch", "AAPL", "MSFT", "-f", "10-K"])
        assert result.exit_code == 0, result.output

        store = _FakeStore.instances[-1]
        assert {fid for fid, _ in store.calls} == {fid_aapl, fid_msft}
        assert "Batch complete" in result.output


# ---------------------------------------------------------------------------
# Construction-failure surface
# ---------------------------------------------------------------------------


class TestBuildPipelineFailures:
    def test_missing_api_key_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``build_embedder`` raising :class:`ConfigurationError` surfaces
        as a single operator-facing error, never as a stack trace."""

        def _raise(*_args: Any, **_kwargs: Any) -> None:
            raise ConfigurationError("No API key resolved for embedding provider 'openai'.")

        monkeypatch.setattr(ingest_module, "get_settings", _StubSettings)
        monkeypatch.setattr(ingest_module, "build_embedder", _raise)
        # Stub registry / chroma / store so we never reach a real driver.
        _FakeStore.instances.clear()
        monkeypatch.setattr(ingest_module, "MetadataRegistry", _FakeRegistry)
        monkeypatch.setattr(ingest_module, "ChromaDBClient", _FakeChroma)
        monkeypatch.setattr(ingest_module, "FilingStore", _FakeStore)
        monkeypatch.setattr(ingest_module, "FilingFetcher", _FakeFetcher)
        monkeypatch.setattr(ingest_module, "PipelineOrchestrator", _FakeOrchestrator)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K"])
        assert result.exit_code == 1
        assert "Embedder construction failed" in result.output
        # No store ever constructed.
        assert _FakeStore.instances == []

    def test_storage_init_failure_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A :class:`DatabaseError` during ChromaDB / registry init exits 1
        with a single error envelope."""

        class _BoomChroma:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise DatabaseError("collection stamp mismatch")

        monkeypatch.setattr(ingest_module, "get_settings", _StubSettings)
        monkeypatch.setattr(ingest_module, "build_embedder", lambda *a, **k: _FakeEmbedder())
        monkeypatch.setattr(ingest_module, "ChromaDBClient", _BoomChroma)
        monkeypatch.setattr(ingest_module, "MetadataRegistry", _FakeRegistry)
        monkeypatch.setattr(ingest_module, "FilingStore", _FakeStore)
        monkeypatch.setattr(ingest_module, "FilingFetcher", _FakeFetcher)
        monkeypatch.setattr(ingest_module, "PipelineOrchestrator", _FakeOrchestrator)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K"])
        assert result.exit_code == 1
        assert "Storage initialisation failed" in result.output


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSecurity:
    def test_writes_never_touch_chroma_or_registry_mutation(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The CLI MUST route every write through :class:`FilingStore` —
        a forgotten ``ChromaDBClient.store_filing`` or
        ``MetadataRegistry.register_filing`` would collapse the dual-store
        invariant.  We enforce this by leaving both classes' mutation
        surface unimplemented in the stubs and asserting the call lands
        on the store stub instead."""
        filing_id = _make_filing_id()
        original_init = _FakeFetcher.__init__

        def _seeded_init(self: _FakeFetcher) -> None:
            original_init(self)
            self.queued_filings = [(filing_id, "<html>body</html>")]

        monkeypatch.setattr(_FakeFetcher, "__init__", _seeded_init)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K"])
        assert result.exit_code == 0, result.output

        store = _FakeStore.instances[-1]
        assert store.calls == [(filing_id, False)]
        # The chroma + registry stubs are write-free by construction —
        # if the CLI tried to mutate them, AttributeError would surface
        # as a non-zero exit above.

    def test_api_key_never_appears_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An OPENAI_API_KEY in the environment must never leak into
        stdout — including error paths.  We arm a storage failure to
        exercise the loudest path."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-canary-token-must-not-appear-in-output")
        original_fetcher_init = _FakeFetcher.__init__
        original_store_init = _FakeStore.__init__
        filing_id = _make_filing_id()

        def _seeded_fetcher(self: _FakeFetcher) -> None:
            original_fetcher_init(self)
            self.queued_filings = [(filing_id, "<html>body</html>")]

        def _seeded_store(self: _FakeStore, *args: Any, **kwargs: Any) -> None:
            original_store_init(self, *args, **kwargs)
            self.raises = DatabaseError("storage exploded")

        monkeypatch.setattr(_FakeFetcher, "__init__", _seeded_fetcher)
        monkeypatch.setattr(_FakeStore, "__init__", _seeded_store)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K"])
        assert "sk-canary-token" not in result.output

    def test_edgar_identity_never_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched_pipeline: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """EDGAR identity is PII — it must not appear in any CLI output."""
        monkeypatch.setenv("EDGAR_IDENTITY_NAME", "Test User")
        monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "canary-edgar@example.test")
        filing_id = _make_filing_id()
        original_init = _FakeFetcher.__init__

        def _seeded_init(self: _FakeFetcher) -> None:
            original_init(self)
            self.queued_filings = [(filing_id, "<html>body</html>")]

        monkeypatch.setattr(_FakeFetcher, "__init__", _seeded_init)

        result = runner.invoke(app, ["ingest", "add", "AAPL", "-f", "10-K"])
        assert "canary-edgar@example.test" not in result.output
        assert "Test User" not in result.output
