"""Tests for :mod:`sec_generative_search.pipeline.orchestrator`.

The orchestrator is a coordination layer — it wires together
:class:`FilingFetcher`, :class:`FilingParser`, :class:`TextChunker`, and
an optional embedder.  These tests substitute stubs for each collaborator
so the suite exercises orchestration logic (ordering, progress
reporting, optional-embedder contract, skip-on-failure) without needing
EDGAR access or a real model.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from datetime import date

import numpy as np
import pytest

from sec_generative_search.core.logging import LOGGER_NAME, AccessionRedactionFilter
from sec_generative_search.core.types import (
    Chunk,
    ContentType,
    FilingIdentifier,
    Segment,
)
from sec_generative_search.pipeline.orchestrator import (
    ChunkEmbedder,
    PipelineOrchestrator,
    ProcessedFiling,
)


@pytest.fixture
def filing_id() -> FilingIdentifier:
    return FilingIdentifier(
        ticker="AAPL",
        form_type="10-K",
        filing_date=date(2023, 11, 3),
        accession_number="0000320193-23-000077",
    )


def _make_segment(filing_id: FilingIdentifier, content: str = "text") -> Segment:
    return Segment(
        path="Part I > Item 1A",
        content_type=ContentType.TEXT,
        content=content,
        filing_id=filing_id,
    )


def _make_chunk(filing_id: FilingIdentifier, idx: int = 0) -> Chunk:
    return Chunk(
        content=f"chunk {idx}",
        path="Part I > Item 1A",
        content_type=ContentType.TEXT,
        filing_id=filing_id,
        chunk_index=idx,
        token_count=2,
    )


# ---------------------------------------------------------------------------
# Stubs for pipeline collaborators
# ---------------------------------------------------------------------------


class StubParser:
    def __init__(self, segments: list[Segment]) -> None:
        self._segments = segments
        self.calls: list[tuple[str, FilingIdentifier]] = []

    def parse(self, html: str, filing_id: FilingIdentifier) -> list[Segment]:
        self.calls.append((html, filing_id))
        return list(self._segments)


class StubChunker:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.calls: list[list[Segment]] = []

    def chunk_segments(self, segments: list[Segment]) -> list[Chunk]:
        self.calls.append(list(segments))
        return list(self._chunks)


class StubEmbedder:
    """Concrete :class:`ChunkEmbedder` stub with call recording."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[tuple[list[Chunk], bool]] = []

    def embed_chunks(
        self,
        chunks: list[Chunk],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        self.calls.append((list(chunks), show_progress))
        return np.zeros((len(chunks), self.dim), dtype=np.float32)


class StubFetcher:
    def __init__(
        self,
        results: list[tuple[FilingIdentifier, str]] | None = None,
    ) -> None:
        self.results = results or []

    def fetch_latest(self, ticker: str, form_type: str) -> tuple[FilingIdentifier, str]:
        return self.results[0]

    def fetch_one(
        self,
        ticker: str,
        form_type: str,
        *,
        index: int = 0,
        year: int | list[int] | range | None = None,
    ) -> tuple[FilingIdentifier, str]:
        return self.results[index]

    def fetch(
        self,
        ticker: str,
        form_type: str,
        *,
        count: int | None = None,
        year: int | list[int] | range | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Iterator[tuple[FilingIdentifier, str]]:
        yield from self.results

    def fetch_batch(
        self,
        tickers: list[str],
        form_type: str,
        *,
        count_per_ticker: int | None = None,
        year: int | list[int] | range | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Iterator[tuple[FilingIdentifier, str]]:
        yield from self.results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProcessFiling:
    def test_runs_parse_chunk_embed_in_order(self, filing_id: FilingIdentifier) -> None:
        segments = [_make_segment(filing_id)]
        chunks = [_make_chunk(filing_id, 0), _make_chunk(filing_id, 1)]

        parser = StubParser(segments)
        chunker = StubChunker(chunks)
        embedder = StubEmbedder(dim=3)

        orchestrator = PipelineOrchestrator(
            fetcher=StubFetcher(),
            parser=parser,
            chunker=chunker,
            embedder=embedder,
        )

        result = orchestrator.process_filing(filing_id, "<html/>")

        assert parser.calls == [("<html/>", filing_id)]
        assert chunker.calls == [segments]
        assert embedder.calls == [(chunks, False)]

        assert isinstance(result, ProcessedFiling)
        assert result.filing_id is filing_id
        assert result.chunks == chunks
        assert result.embeddings is not None
        assert result.embeddings.shape == (2, 3)
        assert result.ingest_result.segment_count == 1
        assert result.ingest_result.chunk_count == 2

    def test_embedder_none_yields_none_embeddings(self, filing_id: FilingIdentifier) -> None:
        # No embedder wired. The orchestrator must
        # still return a ProcessedFiling with embeddings=None rather
        # than failing or silently embedding via a default.
        orchestrator = PipelineOrchestrator(
            fetcher=StubFetcher(),
            parser=StubParser([_make_segment(filing_id)]),
            chunker=StubChunker([_make_chunk(filing_id)]),
            embedder=None,
        )

        result = orchestrator.process_filing(filing_id, "<html/>")

        assert result.embeddings is None
        assert result.chunks  # chunks are still produced

    def test_progress_callback_invoked_for_each_step(self, filing_id: FilingIdentifier) -> None:
        calls: list[tuple[str, int, int]] = []

        orchestrator = PipelineOrchestrator(
            fetcher=StubFetcher(),
            parser=StubParser([_make_segment(filing_id)]),
            chunker=StubChunker([_make_chunk(filing_id)]),
            embedder=None,
        )

        orchestrator.process_filing(
            filing_id, "<html/>", progress_callback=lambda s, c, t: calls.append((s, c, t))
        )

        # Four steps: Parsing, Chunking, Embedding, Complete — all share
        # total=4 so a progress bar renders coherently.
        step_names = [c[0] for c in calls]
        assert step_names == ["Parsing", "Chunking", "Embedding", "Complete"]
        assert {c[2] for c in calls} == {4}
        assert [c[1] for c in calls] == [1, 2, 3, 4]


class TestChunkEmbedderProtocol:
    def test_duck_typed_embedder_accepted(self, filing_id: FilingIdentifier) -> None:
        # An object that merely *quacks* like ChunkEmbedder — no
        # inheritance — must be accepted (runtime-checkable Protocol).

        class DuckEmbedder:
            def embed_chunks(
                self,
                chunks: list[Chunk],
                *,
                show_progress: bool = False,
            ) -> np.ndarray:
                return np.ones((len(chunks), 2), dtype=np.float32)

        embedder: ChunkEmbedder = DuckEmbedder()  # type: ignore[assignment]

        orchestrator = PipelineOrchestrator(
            fetcher=StubFetcher(),
            parser=StubParser([_make_segment(filing_id)]),
            chunker=StubChunker([_make_chunk(filing_id)]),
            embedder=embedder,
        )

        result = orchestrator.process_filing(filing_id, "<html/>")

        assert result.embeddings is not None
        assert result.embeddings.shape == (1, 2)
        assert np.all(result.embeddings == 1.0)


class TestBatchMethods:
    def test_ingest_multiple_skips_failing_filings(self, filing_id: FilingIdentifier) -> None:
        other_id = FilingIdentifier(
            ticker="MSFT",
            form_type="10-K",
            filing_date=date(2023, 7, 27),
            accession_number="0000789019-23-000014",
        )
        fetcher = StubFetcher(results=[(filing_id, "<ok/>"), (other_id, "<boom/>")])

        # Parser blows up on the second filing — the orchestrator must
        # log-and-continue so one bad filing doesn't abort the batch.
        class PickyParser:
            def parse(self, html: str, fid: FilingIdentifier) -> list[Segment]:
                if html == "<boom/>":
                    raise RuntimeError("parser died")
                return [_make_segment(fid)]

        orchestrator = PipelineOrchestrator(
            fetcher=fetcher,
            parser=PickyParser(),
            chunker=StubChunker([_make_chunk(filing_id)]),
            embedder=None,
        )

        results = list(orchestrator.ingest_multiple("AAPL", "10-K"))

        assert len(results) == 1
        assert results[0].filing_id is filing_id

    def test_ingest_batch_skips_failures_across_tickers(self, filing_id: FilingIdentifier) -> None:
        fetcher = StubFetcher(results=[(filing_id, "<ok/>"), (filing_id, "<boom/>")])

        class PickyParser:
            def parse(self, html: str, fid: FilingIdentifier) -> list[Segment]:
                if html == "<boom/>":
                    raise RuntimeError("parser died")
                return [_make_segment(fid)]

        orchestrator = PipelineOrchestrator(
            fetcher=fetcher,
            parser=PickyParser(),
            chunker=StubChunker([_make_chunk(filing_id)]),
            embedder=None,
        )

        results = list(orchestrator.ingest_batch(["AAPL", "MSFT"], "10-K"))

        assert len(results) == 1


@pytest.mark.security
class TestOrchestratorSecurity:
    def test_embedder_not_called_when_none(self, filing_id: FilingIdentifier) -> None:
        # Security-adjacent: when embedder is None, nothing in the
        # orchestrator should attempt to hit an embedding provider —
        # we assert this by passing an embedder whose method would
        # raise if ever invoked.

        class ForbiddenEmbedder:
            def embed_chunks(self, chunks: list[Chunk], *, show_progress: bool = False):
                raise AssertionError("embed_chunks must not be called when embedder=None")

        # Pass embedder=None explicitly; ForbiddenEmbedder is a safety
        # net — it must never execute.
        orchestrator = PipelineOrchestrator(
            fetcher=StubFetcher(),
            parser=StubParser([_make_segment(filing_id)]),
            chunker=StubChunker([_make_chunk(filing_id)]),
            embedder=None,
        )
        orchestrator.embedder = None  # belt-and-braces

        result = orchestrator.process_filing(filing_id, "<html/>")

        assert result.embeddings is None
        _ = ForbiddenEmbedder  # keep the type alive for readers


@pytest.mark.security
class TestOrchestratorLogStreamIsIdentifierFree:
    """The **real** orchestrator's progress lines carry no identifier.

    Runtime companion to the static call-site lock in
    ``tests/core/test_logging.py`` — ``pipeline/orchestrator.py`` is one
    of the two files with the most wrapped ticker sites, and a shape
    lock proves only that an expression is *wrapped*, never that the
    wrapper survives the value.
    """

    def test_ingest_latest_emits_no_raw_ticker_or_accession(
        self,
        filing_id: FilingIdentifier,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")

        orchestrator = PipelineOrchestrator(
            fetcher=StubFetcher(results=[(filing_id, "<html/>")]),
            parser=StubParser([_make_segment(filing_id)]),
            chunker=StubChunker([_make_chunk(filing_id, 0)]),
            embedder=StubEmbedder(dim=3),
        )

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.addFilter(AccessionRedactionFilter())
        package_logger = logging.getLogger(LOGGER_NAME)
        prior_level = package_logger.level
        package_logger.addHandler(handler)
        package_logger.setLevel(logging.DEBUG)
        try:
            assert orchestrator.ingest_latest(filing_id.ticker, filing_id.form_type)
        finally:
            package_logger.removeHandler(handler)
            package_logger.setLevel(prior_level)

        emitted = stream.getvalue()
        assert emitted.strip(), "expected the orchestrator to log something"
        assert filing_id.ticker not in emitted
        assert filing_id.accession_number not in emitted
        assert "<redacted:" in emitted
