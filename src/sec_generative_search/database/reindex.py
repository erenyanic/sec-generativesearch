"""Surface-agnostic reindex of the ``sec_filings`` ChromaDB collection.

:class:`ReindexService` re-embeds the stored chunk text through a new
embedder without re-fetching anything from EDGAR, writes the new vectors
into a staging collection stamped with the target
:class:`EmbedderStamp`, and on success atomic-swaps the staging
collection into the live ``sec_filings`` name.  On any failure before
the swap, the partial staging collection is dropped and the live
collection is left untouched.

Rebuild cost scales with the number of chunks (read ``documents`` +
``metadatas`` → embed through the new provider → write back) and never
re-touches SEC EDGAR.  That matters because EDGAR identity and network
access are a per-deployment concern the storage layer deliberately does
not carry; reindex is purely a storage-layer operation.

Surface boundaries:

- The service is deliberately **surface-agnostic**: the CLI wrapper
    the admin API handler, and the web UI trigger all drive the same
    :meth:`run`.  Progress is reported
  through an injected ``Callable[[str, int, int], None]`` callback so
  this module pulls no ``rich`` / ``typer`` dependency — the CLI wraps
  the callback around a Rich progress bar; the API handler wraps it
  around a WebSocket; the UI wraps it around a React state setter.
- The service **does not construct an embedder**.  The caller brings a
  pre-built :class:`BaseEmbeddingProvider` (from
  :func:`providers.factory.build_embedder`) and the matching target
  :class:`EmbedderStamp`.  That keeps credential resolution on the
    factory seam rather than re-exposing it here.
- Destructive operations are a deliberate operator action.  The service
  refuses no-op runs (source stamp equals target stamp) and
  dimension-mismatched runs (target embedder's dimension disagrees with
  the target stamp) before touching any state.  Operators who need a
  rollback path are expected to take a filesystem-level backup of the
  Chroma directory before invoking the service — this stays out of the
  service's scope because Chroma exposes no atomic-swap primitive and a
  rollback mechanism built on top would add error paths without adding
  real safety guarantees.

Usage:

    from sec_generative_search.core.types import EmbedderStamp
    from sec_generative_search.database import ReindexService
    from sec_generative_search.providers import build_embedder

    target_stamp = EmbedderStamp(
        provider="openai",
        model="text-embedding-3-small",
        dimension=1536,
    )
    embedder = build_embedder(settings.embedding)
    service = ReindexService()
    report = service.run(target_stamp, embedder)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from sec_generative_search.config.constants import COLLECTION_NAME
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import DatabaseError
from sec_generative_search.core.logging import get_logger
from sec_generative_search.core.types import EmbedderStamp, ReindexReport

if TYPE_CHECKING:
    from collections.abc import Callable

    import chromadb

    from sec_generative_search.providers.base import BaseEmbeddingProvider


__all__ = ["ReindexService"]


logger = get_logger(__name__)


# Progress-callback signature — (step_name, current, total).  Mirrors
# ``pipeline.orchestrator.ProgressCallback`` so CLI / API / UI wrappers
# reuse the same shape they already know from ingestion.
ProgressCallback = "Callable[[str, int, int], None]"


class ReindexService:
    """Re-embed the ``sec_filings`` collection through a new embedder.

    The service is a thin coordinator: it opens a raw
    :class:`chromadb.PersistentClient`, pages through the live
    collection, drives the caller-supplied embedder, writes to a staging
    collection, and atomically swaps.

    Opening the live collection is done through the raw client rather
    than through :class:`ChromaDBClient` because the whole point of
    reindex is that the *currently configured* embedder disagrees with
    the collection's stamp — ``ChromaDBClient`` would refuse to open it.
    The raw-client detour is therefore load-bearing, not an abstraction
    leak.

    The service itself is stateless beyond its Chroma client reference;
    ``run`` is safe to call multiple times on the same instance.

    Example:
        >>> service = ReindexService()
        >>> report = service.run(target_stamp, target_embedder)
        >>> print(f"Re-embedded {report.chunks_copied} chunks")
    """

    # Staging collection name is fixed (not timestamped) so that a
    # crashed prior run leaves a recognisable residue the next ``run``
    # can drop on sight.  Timestamping would require the cleanup step to
    # enumerate collections by prefix, which is more fragile without
    # providing a real concurrency benefit — reindex is a single-writer
    # admin operation by design.
    _STAGING_COLLECTION_NAME = "sec_filings_reindex_staging"

    # Reuse the migration-flag key from ``ChromaDBClient`` so the new
    # collection is sealed with the same ``_MIGRATION_FLAG=True`` marker
    # the rest of the storage layer expects on populated collections.
    # A literal re-declaration (rather than an import) keeps this module
    # dependency-light; the two constants are covered by an equality
    # test to stop them drifting silently.
    _MIGRATION_FLAG = "migration_filing_date_int_done"

    def __init__(
        self,
        *,
        chroma_path: str | None = None,
        batch_size: int = 256,
    ) -> None:
        """Construct a reindex service over a Chroma persistence directory.

        Args:
            chroma_path: ChromaDB persistence directory.  Falls through
                to ``settings.database.chroma_path`` when ``None``.
            batch_size: Number of chunks to embed and write per round
                trip.  The default balances provider call overhead
                against Chroma insert latency on the scale of tens of
                thousands of chunks typical for a portfolio-scope SEC
                deployment.  Callers may drop it to match rate-limit
                budgets on hosted embedders.

        Raises:
            ValueError: ``batch_size`` is not strictly positive.
            DatabaseError: ChromaDB failed to open its persistence
                directory.
        """
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be > 0, got {batch_size}",
            )

        settings = get_settings()
        self._chroma_path = chroma_path or settings.database.chroma_path
        self._batch_size = batch_size

        try:
            # Lazy (F28): keeps chromadb (~0.6 s) off every database import.
            import chromadb

            self._client = chromadb.PersistentClient(path=self._chroma_path)
        except Exception as exc:
            raise DatabaseError(
                "Failed to open ChromaDB for reindex",
                details=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        target_stamp: EmbedderStamp,
        target_embedder: BaseEmbeddingProvider,
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> ReindexReport:
        """Re-embed the live collection against *target_embedder*.

        Args:
            target_stamp: The ``(provider, model, dimension)`` triple
                that will seal the rebuilt collection.  Must match the
                embedder's own reported dimension — the service checks
                this up front and refuses on drift.
            target_embedder: Pre-built embedding provider from
                :func:`providers.factory.build_embedder`.  The service
                never constructs its own embedder; credential
                resolution stays on the factory seam.
                progress_callback: Optional ``(step, current, total)``
                callback.  Emitted as ``"reindex"`` throughout the
                embedding work so CLI / API / UI wrappers can show a
                single unified progress surface.

        Returns:
            A frozen :class:`ReindexReport` with source/target stamps,
            chunk count, and wall-clock duration.

        Raises:
            DatabaseError: No-op run (source == target), dimension
                mismatch between *target_embedder* and *target_stamp*,
                missing live collection, corrupt or unstamped live
                collection, or any ChromaDB / embedder failure during
                the reindex.  On any failure the staging collection is
                dropped (best-effort) and the live collection is left
                untouched.
        """
        started_at = time.monotonic()

        target_dim = target_embedder.get_dimension()
        if target_dim != target_stamp.dimension:
            # Dimension drift between the caller's stamp and the
            # caller's embedder means *somebody* has the wrong idea of
            # what they are building.  Surface it before any collection
            # is created — the wrong stamp baked into a staging
            # collection would make the failure harder to diagnose
            # after the fact.
            raise DatabaseError(
                "Target embedder dimension does not match target stamp: "
                f"embedder reports {target_dim}, stamp declares "
                f"{target_stamp.dimension}.",
            )

        source_collection, source_stamp = self._open_source_collection()

        if source_stamp == target_stamp:
            # Refuse no-op loudly.  A silent early-return would mask a
            # CLI typo (wrong --provider / --model) and a reindex-to-
            # self would otherwise tear down and rebuild the live
            # collection for no benefit.
            raise DatabaseError(
                "Source and target stamps are identical — nothing to "
                f"reindex ({source_stamp.provider}/{source_stamp.model} "
                f"dim={source_stamp.dimension}).",
            )

        total_chunks = source_collection.count()
        if total_chunks == 0:
            # An empty source collection is a misconfiguration more
            # than a success case.  Refuse so the operator can decide
            # whether to ``sec-rag manage clear`` + ingest-from-scratch
            # under the new embedder instead.
            raise DatabaseError(
                "Source collection is empty — nothing to reindex. "
                "Run ingestion under the new embedder instead.",
            )

        logger.info(
            "Reindex starting: %s/%s dim=%d → %s/%s dim=%d (%d chunks)",
            source_stamp.provider,
            source_stamp.model,
            source_stamp.dimension,
            target_stamp.provider,
            target_stamp.model,
            target_stamp.dimension,
            total_chunks,
        )

        # Drop any residue from a crashed prior run.  Staging is
        # internal and always partial if it survived a crash, so this
        # is safe.
        self._drop_collection_if_exists(self._STAGING_COLLECTION_NAME)

        staging = self._client.create_collection(
            name=self._STAGING_COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                **target_stamp.to_metadata(),
            },
        )

        try:
            chunks_copied = self._embed_into_staging(
                source_collection=source_collection,
                staging=staging,
                embedder=target_embedder,
                total=total_chunks,
                progress_callback=progress_callback,
            )
            self._swap_collections(
                staging=staging,
                target_stamp=target_stamp,
                progress_callback=progress_callback,
                total=total_chunks,
            )
        except Exception:
            # Best-effort cleanup — ``_drop_collection_if_exists`` logs
            # and swallows its own DatabaseError so the original error
            # always reaches the caller.  Same rule FilingStore uses.
            self._drop_collection_if_exists(self._STAGING_COLLECTION_NAME)
            raise

        duration = time.monotonic() - started_at

        logger.info(
            "Reindex complete: %d chunks in %.2fs (%s/%s → %s/%s)",
            chunks_copied,
            duration,
            source_stamp.provider,
            source_stamp.model,
            target_stamp.provider,
            target_stamp.model,
        )

        return ReindexReport(
            source_stamp=source_stamp,
            target_stamp=target_stamp,
            chunks_copied=chunks_copied,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Source-collection probing
    # ------------------------------------------------------------------

    def _open_source_collection(
        self,
    ) -> tuple[chromadb.api.models.Collection.Collection, EmbedderStamp]:
        """Return the live collection and its parsed source stamp.

        Raises :class:`DatabaseError` when the collection is missing,
        unstamped, or carries corrupt stamp metadata.  We apply the
        same rules ``ChromaDBClient._verify_stamp`` applies, minus the
        compare-against-target step — the whole point of reindex is
        that the configured embedder disagrees with the stored stamp,
        so we must not route this open through ``ChromaDBClient``.
        """
        try:
            collection = self._client.get_collection(name=COLLECTION_NAME)
        except Exception as exc:
            raise DatabaseError(
                f"Source collection {COLLECTION_NAME!r} not found — nothing to reindex.",
                details=str(exc),
            ) from exc

        current = dict(collection.metadata or {})
        stamp_keys = (
            EmbedderStamp._METADATA_PROVIDER_KEY,
            EmbedderStamp._METADATA_MODEL_KEY,
            EmbedderStamp._METADATA_DIMENSION_KEY,
        )
        has_any_stamp_key = any(k in current for k in stamp_keys)

        if not has_any_stamp_key:
            raise DatabaseError(
                "Source collection is unstamped — cannot determine which "
                "embedder produced its vectors, so reindex cannot safely "
                "re-embed from it.  Wipe with 'sec-rag manage clear' and "
                "ingest afresh under the new embedder.",
            )

        try:
            source_stamp = EmbedderStamp.from_metadata(current)
        except ValueError as exc:
            raise DatabaseError(
                "Source collection has a corrupt embedder stamp — wipe "
                "with 'sec-rag manage clear' and ingest afresh under "
                "the new embedder.",
                details=str(exc),
            ) from exc

        return collection, source_stamp

    # ------------------------------------------------------------------
    # Embedding phase
    # ------------------------------------------------------------------

    def _embed_into_staging(
        self,
        *,
        source_collection: chromadb.api.models.Collection.Collection,
        staging: chromadb.api.models.Collection.Collection,
        embedder: BaseEmbeddingProvider,
        total: int,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> int:
        """Paginate source → embed → write staging.

        Embeddings are converted via ``.tolist()`` because ChromaDB's
        ``add()`` expects nested Python lists; the cost is small
        against the embedder wall-clock time and avoids dragging a
        NumPy dependency into this module's import graph beyond the
        embedder's own.
        """
        processed = 0
        batch_size = self._batch_size

        for offset in range(0, total, batch_size):
            batch = source_collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = batch["ids"]
            documents = batch["documents"]
            metadatas = batch["metadatas"]

            if not ids:
                # Defensive: a concurrent writer shrunk the collection
                # between the ``count()`` and this page.  Stop cleanly
                # rather than looping forever.
                break

            try:
                embeddings = embedder.embed_texts(documents)
            except Exception as exc:
                raise DatabaseError(
                    f"Embedder failed on batch offset={offset} size={len(ids)}",
                    details=str(exc),
                ) from exc

            try:
                staging.add(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    documents=documents,
                    metadatas=metadatas,
                )
            except Exception as exc:
                raise DatabaseError(
                    f"ChromaDB staging write failed at offset={offset} size={len(ids)}",
                    details=str(exc),
                ) from exc

            processed += len(ids)
            if progress_callback is not None:
                progress_callback("reindex", processed, total)

        return processed

    # ------------------------------------------------------------------
    # Swap phase
    # ------------------------------------------------------------------

    def _swap_collections(
        self,
        *,
        staging: chromadb.api.models.Collection.Collection,
        target_stamp: EmbedderStamp,
        progress_callback: Callable[[str, int, int], None] | None,
        total: int,
    ) -> None:
        """Replace the live collection with the staging contents.

        ChromaDB exposes no atomic rename or alias, so the swap is:

            drop live → create live (stamped target) → copy staging →
            drop staging

        The window between the drop and the end of the copy is
        non-atomic; a process crash here leaves ``sec_filings`` either
        missing (crash before create) or partial (crash during copy).
        A subsequent :class:`ChromaDBClient` open would surface the
        former as a fresh-create and the latter as a stamp-matched
        collection with missing chunks.  Both are operator problems,
        not failure modes the service tries to self-heal — a Chroma-
        level rollback would itself be fallible and is explicitly out
        of scope.  Take a filesystem-level backup of the Chroma
        directory before running reindex.

        On any exception raised from ``staging.get()`` /
        ``live.add()`` inside this method, the outer ``run`` handler
        drops staging and re-raises; it does **not** attempt to
        reconstitute the live collection.
        """
        try:
            self._client.delete_collection(name=COLLECTION_NAME)
        except Exception as exc:
            raise DatabaseError(
                "Failed to drop live collection during reindex swap",
                details=str(exc),
            ) from exc

        try:
            live = self._client.create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",
                    self._MIGRATION_FLAG: True,
                    **target_stamp.to_metadata(),
                },
            )
        except Exception as exc:
            raise DatabaseError(
                "Failed to recreate live collection during reindex swap",
                details=str(exc),
            ) from exc

        # Paginate staging → live.  Reuse the same paging pattern for
        # symmetry and to avoid loading the whole staging collection
        # into memory on large SEC deployments.
        copied = 0
        batch_size = self._batch_size
        while True:
            try:
                batch = staging.get(
                    limit=batch_size,
                    offset=copied,
                    include=["documents", "metadatas", "embeddings"],
                )
            except Exception as exc:
                raise DatabaseError(
                    f"Failed to read staging at offset={copied} during swap",
                    details=str(exc),
                ) from exc

            ids = batch["ids"]
            if not ids:
                break

            embeddings = batch["embeddings"]
            # ChromaDB returns a NumPy array for ``embeddings`` when
            # included.  ``add()`` is equally happy with either array or
            # list-of-lists, but ``.tolist()`` normalises the contract
            # and sidesteps any dtype surprises across Chroma versions.
            if hasattr(embeddings, "tolist"):
                embeddings_payload = embeddings.tolist()
            else:
                embeddings_payload = embeddings

            try:
                live.add(
                    ids=ids,
                    embeddings=embeddings_payload,
                    documents=batch["documents"],
                    metadatas=batch["metadatas"],
                )
            except Exception as exc:
                raise DatabaseError(
                    f"Failed to write live collection at offset={copied} during swap",
                    details=str(exc),
                ) from exc

            copied += len(ids)
            if progress_callback is not None:
                # Reuse the same step name so the CLI progress bar
                # carries straight through; the total is the chunk
                # count we already embedded, so the swap copy reads as
                # a second pass of the same work unit.
                progress_callback("reindex-swap", copied, total)

        try:
            self._client.delete_collection(name=self._STAGING_COLLECTION_NAME)
        except Exception as exc:
            # Staging leftover after a successful swap is cosmetic, not
            # a correctness issue — the live collection is already
            # sealed with the target stamp.  Log at ``warning`` and
            # move on rather than erroring the whole reindex.
            logger.warning(
                "Reindex succeeded but failed to drop staging collection "
                "%s (manual cleanup required): %s",
                self._STAGING_COLLECTION_NAME,
                exc,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drop_collection_if_exists(self, name: str) -> None:
        """Drop *name* if present; swallow errors.

        Used for staging cleanup on both the happy path and the
        failure path.  Never shadows the original exception — callers
        that need the error surfaced should call
        ``self._client.delete_collection`` directly.
        """
        try:
            self._client.delete_collection(name=name)
        except Exception as exc:
            # Non-existent collection is the normal case on a fresh
            # run; log at debug so it does not clutter operator output.
            logger.debug(
                "delete_collection(%s) swallowed during cleanup: %s",
                name,
                exc,
            )
