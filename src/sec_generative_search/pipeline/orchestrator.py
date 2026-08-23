"""
Pipeline orchestrator for SEC filing ingestion.

This module coordinates the filing-processing pipeline:

    Fetch → Parse → Chunk → [Embed]

Embedding is optional — the orchestrator accepts any callable
conforming to :class:`ChunkEmbedder` and produces chunk-only
:class:`ProcessedFiling` output when no embedder is supplied. The
storage layer is responsible for refusing to persist a
:class:`ProcessedFiling` whose embeddings are ``None``.

Usage:
    from sec_generative_search.pipeline import PipelineOrchestrator

    orchestrator = PipelineOrchestrator()

    # Process a single filing without embeddings
    result = orchestrator.process_filing(filing_id, html_content)

    # Ingest latest filing for a company
    result = orchestrator.ingest_latest("AAPL", "10-K")

    # Batch ingest multiple companies
    for result in orchestrator.ingest_batch(["AAPL", "MSFT"], "10-K"):
        print(f"Ingested {result.filing_id.ticker}")
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sec_generative_search.core.logging import get_logger, redact_for_log
from sec_generative_search.core.types import Chunk, FilingIdentifier, IngestResult
from sec_generative_search.pipeline.chunk import TextChunker
from sec_generative_search.pipeline.fetch import FilingFetcher
from sec_generative_search.pipeline.parse import FilingParser

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)


# Type alias for progress callback — (step_name, current, total)
ProgressCallback = Callable[[str, int, int], None]


class ChunkEmbedder(Protocol):
    """Structural interface the orchestrator expects of an embedder.

    Kept deliberately narrow — :class:`BaseEmbeddingProvider` satisfies
    this protocol via its ``embed_chunks`` method, and no further
    coupling is imposed here. The ``show_progress`` flag is passed
    through so CLI callers can drive a progress bar without plumbing
    new arguments through this layer.
    """

    def embed_chunks(
        self,
        chunks: list[Chunk],
        *,
        show_progress: bool = False,
    ) -> np.ndarray: ...


@dataclass
class ProcessedFiling:
    """
    Result of processing a single filing through the pipeline.

    Segment count is captured in ``ingest_result``; the raw segments
    are not retained because no consumer reads them after chunking.

    Attributes:
        filing_id: Identifier for the filing.
        chunks: Chunked text ready for embedding/storage.
        embeddings: Vector embeddings for each chunk.  ``None`` when
            the orchestrator was run without an embedder — the storage
            layer must reject a ``ProcessedFiling`` whose ``embeddings``
            are ``None`` rather than silently writing
            chunks without vectors.
        ingest_result: Statistics about the ingestion.
    """

    filing_id: FilingIdentifier
    chunks: list[Chunk]
    embeddings: np.ndarray | None
    ingest_result: IngestResult


class PipelineOrchestrator:
    """
    Coordinates the SEC filing ingestion pipeline.

    The orchestrator handles:
        - Single filing processing (when HTML is already available)
        - Single company ingestion (fetch + process)
        - Batch ingestion (multiple companies/filings)
        - Progress reporting via callbacks

    Note:
        The orchestrator does NOT handle database storage or duplicate
        detection — both live in the metadata registry. It
        returns :class:`ProcessedFiling` objects that the database
        layer can persist.

    Example:
        >>> orchestrator = PipelineOrchestrator()
        >>> result = orchestrator.ingest_latest("AAPL", "10-K")
        >>> print(f"Processed {result.ingest_result.chunk_count} chunks")
    """

    def __init__(
        self,
        fetcher: FilingFetcher | None = None,
        parser: FilingParser | None = None,
        chunker: TextChunker | None = None,
        embedder: ChunkEmbedder | None = None,
    ) -> None:
        """
        Initialise the orchestrator with pipeline components.

        Components are created with defaults where possible, allowing
        dependency injection for testing. The embedder has **no
        default**; when omitted, the orchestrator returns
        ``ProcessedFiling`` objects with ``embeddings=None``.

        Args:
            fetcher: FilingFetcher instance (optional).
            parser: FilingParser instance (optional).
            chunker: TextChunker instance (optional).
            embedder: Any object conforming to :class:`ChunkEmbedder`
                (optional).  When ``None``, the embedding step is
                skipped.
        """
        self.fetcher = fetcher or FilingFetcher()
        self.parser = parser or FilingParser()
        self.chunker = chunker or TextChunker()
        self.embedder = embedder

        logger.debug(
            "PipelineOrchestrator initialised (embedder=%s)",
            "enabled" if self.embedder is not None else "disabled",
        )

    def process_filing(
        self,
        filing_id: FilingIdentifier,
        html_content: str,
        progress_callback: ProgressCallback | None = None,
    ) -> ProcessedFiling:
        """
        Process a single filing through the pipeline.

        This method runs the full pipeline on HTML content that has
        already been fetched.  Use this when you have the HTML content
        available (e.g. from a previous fetch or cache).

        Pipeline steps:
            1. Parse HTML → Segments
            2. Chunk segments → Chunks
            3. Generate embeddings (skipped when ``embedder`` is None)

        Args:
            filing_id: Identifier for the filing.
            html_content: Raw HTML content.
            progress_callback: Optional callback
                ``(step_name, current, total)``.

        Returns:
            ProcessedFiling containing all processed data.

        Example:
            >>> result = orchestrator.process_filing(filing_id, html)
            >>> print(f"Created {len(result.chunks)} chunks")
        """
        start_time = time.time()
        total_steps = 4

        def report_progress(step: str, current: int, total: int) -> None:
            if progress_callback:
                progress_callback(step, current, total)

        logger.info(
            "Processing %s %s (%s)",
            redact_for_log(filing_id.ticker),
            filing_id.form_type,
            filing_id.date_str,
        )

        # Step 1: Parse
        report_progress("Parsing", 1, total_steps)
        segments = self.parser.parse(html_content, filing_id)

        # Step 2: Chunk
        report_progress("Chunking", 2, total_steps)
        chunks = self.chunker.chunk_segments(segments)

        # Step 3: Embed (optional)
        report_progress("Embedding", 3, total_steps)
        embeddings: np.ndarray | None
        if self.embedder is not None:
            embeddings = self.embedder.embed_chunks(chunks, show_progress=False)
        else:
            embeddings = None
            logger.debug("Skipping embedding step — no embedder wired.")

        # Complete
        report_progress("Complete", 4, total_steps)
        duration = time.time() - start_time

        ingest_result = IngestResult(
            filing_id=filing_id,
            segment_count=len(segments),
            chunk_count=len(chunks),
            duration_seconds=duration,
        )

        logger.info(
            "Processed %s %s: %d segments → %d chunks in %.1fs",
            redact_for_log(filing_id.ticker),
            filing_id.form_type,
            len(segments),
            len(chunks),
            duration,
        )

        return ProcessedFiling(
            filing_id=filing_id,
            chunks=chunks,
            embeddings=embeddings,
            ingest_result=ingest_result,
        )

    def ingest_latest(
        self,
        ticker: str,
        form_type: str = "10-K",
        progress_callback: ProgressCallback | None = None,
    ) -> ProcessedFiling:
        """
        Fetch and process the latest filing for a company.

        Args:
            ticker: Stock ticker symbol.
            form_type: SEC form type ("8-K", "10-K", or "10-Q").
            progress_callback: Optional callback
                ``(step_name, current, total)``.

        Returns:
            ProcessedFiling containing all processed data.
        """
        logger.info("Ingesting latest %s for %s", form_type, redact_for_log(ticker))

        if progress_callback:
            progress_callback("Fetching", 0, 4)

        filing_id, html_content = self.fetcher.fetch_latest(ticker, form_type)
        return self.process_filing(filing_id, html_content, progress_callback)

    def ingest_one(
        self,
        ticker: str,
        form_type: str = "10-K",
        *,
        index: int = 0,
        year: int | list[int] | range | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ProcessedFiling:
        """
        Fetch and process a specific filing by index.

        Args:
            ticker: Stock ticker symbol.
            form_type: SEC form type.
            index: Position in results (0=most recent).
            year: Optional year filter.
            progress_callback: Optional callback.

        Returns:
            ProcessedFiling containing all processed data.
        """
        logger.info(
            "Ingesting %s %s at index %d",
            redact_for_log(ticker),
            form_type,
            index,
        )

        if progress_callback:
            progress_callback("Fetching", 0, 4)

        filing_id, html_content = self.fetcher.fetch_one(ticker, form_type, index=index, year=year)
        return self.process_filing(filing_id, html_content, progress_callback)

    def ingest_multiple(
        self,
        ticker: str,
        form_type: str = "10-K",
        *,
        count: int | None = None,
        year: int | list[int] | range | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Iterator[ProcessedFiling]:
        """
        Fetch and process multiple filings for a company.

        Yields ProcessedFiling objects one at a time, allowing
        incremental processing and storage.  Failed filings are logged
        and skipped so a single bad filing does not abort the batch.

        Args:
            ticker: Stock ticker symbol.
            form_type: SEC form type.
            count: Maximum number of filings.
            year: Year filter.
            start_date: Date range start.
            end_date: Date range end.

        Yields:
            ProcessedFiling for each successfully processed filing.
        """
        logger.info(
            "Ingesting multiple %s filings for %s",
            form_type,
            redact_for_log(ticker),
        )

        for filing_id, html_content in self.fetcher.fetch(
            ticker,
            form_type,
            count=count,
            year=year,
            start_date=start_date,
            end_date=end_date,
        ):
            try:
                yield self.process_filing(filing_id, html_content)
            except Exception as e:
                logger.warning(
                    "Failed to process %s: %s",
                    filing_id.accession_number,
                    str(e),
                )
                continue

    def ingest_batch(
        self,
        tickers: list[str],
        form_type: str = "10-K",
        *,
        count_per_ticker: int | None = None,
        year: int | list[int] | range | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Iterator[ProcessedFiling]:
        """
        Fetch and process filings for multiple companies.

        Yields ProcessedFiling objects for each successfully processed
        filing across all specified tickers.  Failed filings are
        logged and skipped.

        Args:
            tickers: List of stock ticker symbols.
            form_type: SEC form type.
            count_per_ticker: Max filings per company.
            year: Year filter.
            start_date: Date range start.
            end_date: Date range end.

        Yields:
            ProcessedFiling for each successfully processed filing.
        """
        logger.info(
            "Batch ingesting %s filings for %d companies",
            form_type,
            len(tickers),
        )

        for filing_id, html_content in self.fetcher.fetch_batch(
            tickers,
            form_type,
            count_per_ticker=count_per_ticker,
            year=year,
            start_date=start_date,
            end_date=end_date,
        ):
            try:
                yield self.process_filing(filing_id, html_content)
            except Exception as e:
                logger.warning(
                    "Failed to process %s %s: %s",
                    redact_for_log(filing_id.ticker),
                    filing_id.accession_number,
                    str(e),
                )
                continue
