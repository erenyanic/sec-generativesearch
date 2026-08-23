"""ChromaDB client wrapper for vector storage operations.

The client owns a single collection (``sec_filings``) that stores chunk
embeddings, their source text, and filing metadata for cosine-similarity
retrieval.  It is intentionally unaware of which embedder produced the
vectors — callers provide an :class:`EmbedderStamp` at construction time
and the client seals that stamp onto the collection.

Stamp contract:

    - On first creation the stamp's ``(provider, model, dimension)`` is
      written into the collection's metadata alongside the HNSW space
      configuration.
    - On every subsequent open the client reads the stored stamp and
      refuses to serve traffic when it disagrees with the configured
      stamp.  A retrieval against a mismatched collection would silently
      return garbage — this refuse-with-hint is the failure mode the
      stamp exists to prevent.

Usage:
    from sec_generative_search.core.types import EmbedderStamp
    from sec_generative_search.database import ChromaDBClient

    stamp = EmbedderStamp(provider="openai", model="text-embedding-3-small", dimension=1536)
    client = ChromaDBClient(stamp)
    client.store_filing(processed_filing)
    results = client.query(query_embeddings, n_results=5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb

from sec_generative_search.config.constants import COLLECTION_NAME
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import (
    DatabaseError,
    EmbeddingCollectionMismatchError,
)
from sec_generative_search.core.logging import get_logger, redact_for_log
from sec_generative_search.core.types import EmbedderStamp, SearchResult

if TYPE_CHECKING:
    from sec_generative_search.pipeline import ProcessedFiling

logger = get_logger(__name__)


class ChromaDBClient:
    """Wrapper around ChromaDB for SEC filing vector storage.

    One collection, cosine similarity, embedder-stamped on creation.
    See module docstring for the stamp contract.

    Example:
        >>> stamp = EmbedderStamp(provider="local",
        ...                       model="google/embeddinggemma-300m",
        ...                       dimension=768)
        >>> client = ChromaDBClient(stamp)
        >>> client.store_filing(processed_filing)
        >>> results = client.query(query_embeddings, n_results=5)
    """

    _MIGRATION_FLAG = "migration_filing_date_int_done"

    def __init__(
        self,
        stamp: EmbedderStamp,
        *,
        chroma_path: str | None = None,
    ) -> None:
        """Initialise the ChromaDB client and seal the collection.

        Args:
            stamp: The configured embedder's ``(provider, model,
                dimension)`` triple.  Required — storage never assumes a
                default embedder.
            chroma_path: Path to ChromaDB storage directory.  Falls
                through to ``settings.database.chroma_path`` when
                ``None``.

        Raises:
            EmbeddingCollectionMismatchError: Collection exists and its
                stamp disagrees with ``stamp``.
            DatabaseError: ChromaDB failed to open, the collection's
                stamp is corrupt, or the collection is populated but
                unstamped (legacy import that predates sealing).
        """
        settings = get_settings()
        self._chroma_path = chroma_path or settings.database.chroma_path
        self._stamp = stamp

        try:
            self._client = chromadb.PersistentClient(path=self._chroma_path)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise DatabaseError(
                "Failed to initialise ChromaDB",
                details=str(e),
            ) from e

        # ``_verify_stamp`` is the single seal seam — it stamps a fresh or
        # legacy-empty collection, compares against an existing stamp,
        # and refuses mismatched / corrupt / populated-but-unstamped
        # collections.  Keeping the stamp logic out of
        # ``get_or_create_collection`` sidesteps the ambiguity about how
        # Chroma handles the ``metadata=`` argument on existing
        # collections (ignored vs. silently updated varies by version).
        self._verify_stamp()

        logger.debug(
            "ChromaDBClient initialised: %s (collection: %s, count: %d, stamp: %s/%s dim=%d)",
            self._chroma_path,
            COLLECTION_NAME,
            self._collection.count(),
            stamp.provider,
            stamp.model,
            stamp.dimension,
        )

        self._migrate_filing_date_int()

    @property
    def stamp(self) -> EmbedderStamp:
        """Return the embedder stamp this client enforces on the collection."""
        return self._stamp

    # ------------------------------------------------------------------
    # Stamp verification
    # ------------------------------------------------------------------

    def _verify_stamp(self) -> None:
        """Compare the configured stamp against the collection's metadata.

        Outcomes:

        - Stamp present and equal → no-op.
        - Stamp present and different → ``EmbeddingCollectionMismatchError``.
        - Stamp metadata partially present or non-parseable →
          ``DatabaseError`` flagged as corrupt.  The operator reaction is
          the same as for a mismatch (reindex), but the category stays
          distinct so logs and tests can tell them apart.
        - Stamp absent entirely:
            - Empty collection → stamp it now (fresh collection or
              legacy unstamped collection with no data at risk).
            - Non-empty collection → ``DatabaseError`` refusing to
              serve traffic.  We cannot know which embedder produced
              the existing vectors, so any retrieval would be unsafe.
        """
        current = dict(self._collection.metadata or {})
        stamp_keys = (
            EmbedderStamp._METADATA_PROVIDER_KEY,
            EmbedderStamp._METADATA_MODEL_KEY,
            EmbedderStamp._METADATA_DIMENSION_KEY,
        )
        has_any_stamp_key = any(k in current for k in stamp_keys)

        if has_any_stamp_key:
            try:
                actual = EmbedderStamp.from_metadata(current)
            except ValueError as exc:
                raise DatabaseError(
                    "ChromaDB collection has a corrupt embedder stamp — "
                    "rebuild with 'sec-rag manage reindex'.",
                    details=str(exc),
                ) from exc
            if actual != self._stamp:
                raise EmbeddingCollectionMismatchError(
                    expected=self._stamp,
                    actual=actual,
                )
            return

        if self._collection.count() == 0:
            # Fresh or pre-stamp legacy collection with no vectors yet —
            # stamp it now and proceed.  ``get_or_create_collection``
            # already applied the stamp on actual create; this branch
            # only triggers when we opened an empty pre-existing
            # collection imported from an older code path.
            self._set_collection_metadata(self._stamp.to_metadata())
            logger.info(
                "Stamped previously unstamped empty collection with embedder %s/%s dim=%d",
                self._stamp.provider,
                self._stamp.model,
                self._stamp.dimension,
            )
            return

        raise DatabaseError(
            "ChromaDB collection is populated but has no embedder stamp — "
            "cannot verify compatibility with the configured embedder. "
            "Rebuild with 'sec-rag manage reindex' to seal the collection."
        )

    # ------------------------------------------------------------------
    # Collection-metadata helpers
    # ------------------------------------------------------------------

    def _set_collection_metadata(self, updates: dict[str, str | bool]) -> None:
        """Merge ``updates`` into the collection metadata.

        Filters out HNSW configuration keys (e.g. ``hnsw:space``) before
        calling ``modify()`` — ChromaDB rejects metadata updates that
        include distance-function settings, even if the value is
        unchanged.  Existing non-HNSW keys (the stamp, the migration
        flag) are preserved.
        """
        current = self._collection.metadata or {}
        merged: dict[str, str | bool] = {
            k: v for k, v in current.items() if not k.startswith("hnsw:")
        }
        merged.update(updates)
        self._collection.modify(metadata=merged)

    def _set_collection_flag(self, flag: str, value: bool = True) -> None:
        """Set a boolean collection-metadata flag without clobbering other keys."""
        self._set_collection_metadata({flag: value})

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    def _migrate_filing_date_int(self) -> None:
        """Backfill ``filing_date_int`` on chunks ingested before BF-012.

        Scans all documents in the collection and adds the integer
        ``YYYYMMDD`` field to any chunk that has ``filing_date`` but is
        missing ``filing_date_int``.  A metadata flag on the collection
        tracks completion so subsequent startups are O(1).
        """
        collection_meta = self._collection.metadata or {}
        if collection_meta.get(self._MIGRATION_FLAG):
            logger.debug("filing_date_int migration already complete — skipping")
            return

        total = self._collection.count()
        if total == 0:
            self._set_collection_flag(self._MIGRATION_FLAG)
            return

        batch_size = 1000
        migrated = 0

        for offset in range(0, total, batch_size):
            batch = self._collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas"],
            )

            ids_to_update: list[str] = []
            metas_to_update: list[dict] = []

            for doc_id, meta in zip(batch["ids"], batch["metadatas"], strict=True):
                if "filing_date_int" not in meta and "filing_date" in meta:
                    meta["filing_date_int"] = int(meta["filing_date"].replace("-", ""))
                    ids_to_update.append(doc_id)
                    metas_to_update.append(meta)

            if ids_to_update:
                self._collection.update(ids=ids_to_update, metadatas=metas_to_update)
                migrated += len(ids_to_update)

        if migrated > 0:
            logger.info("Migrated %d chunk(s): added filing_date_int field", migrated)

        self._set_collection_flag(self._MIGRATION_FLAG)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def store_filing(self, processed_filing: ProcessedFiling) -> None:
        """Store all chunks and embeddings from a processed filing.

        Refuses a :class:`ProcessedFiling` whose embeddings are ``None``
        — the orchestrator's docstring contract says the storage layer
        must reject rather than silently writing chunks without vectors.

        Args:
            processed_filing: Output from :class:`PipelineOrchestrator`.

        Raises:
            DatabaseError: Storage failed or ``processed_filing`` has no
                embeddings.
        """
        if processed_filing.embeddings is None:
            raise DatabaseError(
                f"Refusing to store filing {processed_filing.filing_id.accession_number}: "
                "ProcessedFiling.embeddings is None (pipeline was run without an embedder). "
                "Wire an embedder into the orchestrator and retry."
            )

        chunks = processed_filing.chunks
        embeddings = processed_filing.embeddings
        filing_id = processed_filing.filing_id

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]

        try:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info(
                "Stored %d chunks for %s %s (%s)",
                len(chunks),
                redact_for_log(filing_id.ticker),
                filing_id.form_type,
                filing_id.date_str,
            )
        except Exception as e:
            raise DatabaseError(
                f"Failed to store filing {filing_id.accession_number}",
                details=str(e),
            ) from e

    def delete_filing(self, accession_number: str) -> None:
        """Delete all chunks belonging to a filing by accession number."""
        try:
            self._collection.delete(where={"accession_number": accession_number})
            logger.info("Deleted chunks from ChromaDB for accession: %s", accession_number)
        except Exception as e:
            raise DatabaseError(
                f"Failed to delete filing {accession_number} from ChromaDB",
                details=str(e),
            ) from e

    def delete_filings_batch(self, accession_numbers: list[str]) -> None:
        """Delete all chunks belonging to multiple filings in a single call.

        Uses ChromaDB's ``$in`` operator so N filings collapse to one
        round-trip.
        """
        if not accession_numbers:
            return

        try:
            self._collection.delete(
                where={"accession_number": {"$in": accession_numbers}},
            )
            logger.info(
                "Batch-deleted chunks from ChromaDB for %d filing(s)",
                len(accession_numbers),
            )
        except Exception as e:
            raise DatabaseError(
                f"Failed to batch-delete {len(accession_numbers)} filing(s) from ChromaDB",
                details=str(e),
            ) from e

    def clear_collection(self) -> int:
        """Delete all documents and re-seal the collection.

        More efficient than iterating accession numbers — drops the
        whole collection and recreates it with the same HNSW space,
        migration flag, and embedder stamp.  Re-sealing is load-bearing:
        without it the new collection would be empty and unstamped, and
        ``_verify_stamp`` would wrongly treat a subsequent open as a
        first-run scenario.

        Returns:
            Number of chunks removed.
        """
        try:
            count = self._collection.count()
            if count == 0:
                return 0

            self._client.delete_collection(name=COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",
                    self._MIGRATION_FLAG: True,
                    **self._stamp.to_metadata(),
                },
            )
            logger.info("Cleared ChromaDB collection: %d chunk(s) removed", count)
            return count
        except Exception as e:
            raise DatabaseError(
                "Failed to clear ChromaDB collection",
                details=str(e),
            ) from e

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        ticker: str | list[str] | None = None,
        form_type: str | list[str] | None = None,
        accession_number: str | list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[SearchResult]:
        """Query the collection for similar chunks.

        Args:
            query_embeddings: Query embedding in ChromaDB format
                (``list[list[float]]``).
            n_results: Maximum number of results to return.
            ticker: Optional filter — single ticker or list of tickers
                (matched via ``$in``).
            form_type: Optional filter — single form type or list.
            accession_number: Optional filter — single or list of
                accession numbers.
            start_date: Optional inclusive lower bound
                (``YYYY-MM-DD``).
            end_date: Optional inclusive upper bound.

        Returns:
            List of :class:`SearchResult`, ordered by similarity
            (highest first).
        """
        where_filter = self._build_where_filter(
            ticker, form_type, accession_number, start_date, end_date
        )

        try:
            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )

            search_results: list[SearchResult] = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    search_results.append(
                        SearchResult.from_chromadb_result(
                            document=results["documents"][0][i],
                            metadata=results["metadatas"][0][i],
                            distance=results["distances"][0][i],
                            chunk_id=results["ids"][0][i],
                        )
                    )

            logger.debug("Query returned %d results", len(search_results))
            return search_results

        except Exception as e:
            raise DatabaseError(
                "ChromaDB query failed",
                details=str(e),
            ) from e

    def collection_count(self) -> int:
        """Return the total number of chunks in the collection."""
        return self._collection.count()

    # ------------------------------------------------------------------
    # Internal filter builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_field_condition(
        field: str,
        value: str | list[str],
        upper: bool = False,
    ) -> dict:
        """Build a single ChromaDB field condition (single or ``$in``)."""
        if isinstance(value, list):
            values = [v.upper() for v in value] if upper else list(value)
            if len(values) == 1:
                return {field: values[0]}
            return {field: {"$in": values}}
        return {field: value.upper() if upper else value}

    @staticmethod
    def _date_str_to_int(date_str: str) -> int:
        """Convert ``YYYY-MM-DD`` to the ``YYYYMMDD`` integer form."""
        return int(date_str.replace("-", ""))

    @staticmethod
    def _build_where_filter(
        ticker: str | list[str] | None = None,
        form_type: str | list[str] | None = None,
        accession_number: str | list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict | None:
        """Assemble a ChromaDB ``where`` filter from optional parameters.

        Date-range filters hit ``filing_date_int`` (the integer
        ``YYYYMMDD`` mirror of ``filing_date``) because ChromaDB's
        comparison operators only accept numeric operands.
        """
        conditions = []
        if ticker:
            conditions.append(ChromaDBClient._build_field_condition("ticker", ticker, upper=True))
        if form_type:
            conditions.append(
                ChromaDBClient._build_field_condition("form_type", form_type, upper=True)
            )
        if accession_number:
            conditions.append(
                ChromaDBClient._build_field_condition("accession_number", accession_number)
            )
        if start_date:
            conditions.append(
                {"filing_date_int": {"$gte": ChromaDBClient._date_str_to_int(start_date)}}
            )
        if end_date:
            conditions.append(
                {"filing_date_int": {"$lte": ChromaDBClient._date_str_to_int(end_date)}}
            )

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
