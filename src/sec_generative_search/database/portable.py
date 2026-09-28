"""Portable, embedder-decoupled JSONL export and import for the storage layer.

The portable artefact is **not** a backup — it carries chunk text and
filing metadata but no embeddings, because vectors produced under one
embedder are useless on a host with a different one.  Imports re-embed
through the host's configured embedder via the
:func:`providers.factory.build_embedder` seam, which keeps credential
resolution on the factory boundary and allows operators to move chunk
data between hosts that run different providers.

Output layout (and import input layout):

.. code-block:: text

    export_dir/
    ├── manifest.json   # format_version, created_at_utc, source_embedder_stamp
    └── chunks.jsonl    # one JSON object per chunk

The import is **idempotent at the filing grain** — every accession is
written through :meth:`FilingStore.store_filing` with
``register_if_new=True``, so duplicate accessions on the host are
skipped without overwrite.  The atomic SQLite-first claim path is the
only safe shape against concurrent writers because ChromaDB's ``add()``
silently no-ops duplicate IDs (see
:class:`FilingStore` module docstring for the rationale).

Surface-agnostic and credential-free; the CLI / admin API / web UI all
drive the same :meth:`PortableExportService.export` and
:meth:`PortableImportService.import_`.  This module pulls no ``rich``,
``typer``, ``edgartools``, or ingestion-pipeline imports — a
source-level ``@pytest.mark.security`` test pins that.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sec_generative_search.config.constants import COLLECTION_NAME
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import DatabaseError
from sec_generative_search.core.logging import get_logger
from sec_generative_search.core.types import (
    Chunk,
    ContentType,
    EmbedderStamp,
    ExportReport,
    FilingIdentifier,
    ImportReport,
    IngestResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sec_generative_search.database.metadata import MetadataRegistry
    from sec_generative_search.database.store import FilingStore
    from sec_generative_search.providers.base import BaseEmbeddingProvider


__all__ = ["PortableExportService", "PortableImportService"]


logger = get_logger(__name__)


_FORMAT_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_CHUNKS_NAME = "chunks.jsonl"


def _refuse_symlink_lexical_parents(path: Path, *, label: str) -> None:
    """Refuse if any existing lexical parent of *path* is a symlink.

    Mirrors :meth:`DatabaseSettings._validate_paths` and the matching
    helper in :mod:`database.backup`.  The walk is over the *lexical*
    parents — calling :meth:`Path.resolve` first would have already
    followed every symlink.
    """
    check = path.absolute()
    while check != check.parent:
        if check.exists() and check.is_symlink():
            raise DatabaseError(
                f"{label} contains a symlink at '{check}'. "
                f"Symlinks are not permitted in portable-export paths "
                f"for security.",
            )
        check = check.parent


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class PortableExportService:
    """Export filings as JSONL chunks plus a manifest.

    Constructed with paths only; opens the registry and the raw
    ChromaDB client lazily inside :meth:`export`.  No embedder is
    needed — export only reads.

    Example:
        >>> service = PortableExportService()
        >>> report = service.export(
        ...     "/tmp/export-2026-04-28",
        ...     tickers=["AAPL"],
        ...     form_types=["10-K"],
        ... )
    """

    _FORMAT_VERSION = _FORMAT_VERSION
    _MANIFEST_NAME = _MANIFEST_NAME
    _CHUNKS_NAME = _CHUNKS_NAME

    def __init__(
        self,
        *,
        chroma_path: str | None = None,
        metadata_db_path: str | None = None,
    ) -> None:
        """Construct the exporter over the configured storage paths.

        Args:
            chroma_path: ChromaDB persistence directory.  Falls
                through to ``settings.database.chroma_path`` when
                ``None``.
            metadata_db_path: SQLite file path.  Falls through to
                ``settings.database.metadata_db_path`` when ``None``.
        """
        settings = get_settings()
        self._chroma_path = Path(
            chroma_path if chroma_path is not None else settings.database.chroma_path
        )
        self._metadata_db_path = Path(
            metadata_db_path if metadata_db_path is not None else settings.database.metadata_db_path
        )

    def export(
        self,
        output_dir: str | Path,
        *,
        force: bool = False,
        tickers: list[str] | None = None,
        form_types: list[str] | None = None,
        accessions: list[str] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> ExportReport:
        """Write a portable JSONL export of the configured storage.

        Filtering is **AND**-composed: a chunk is included only when
        its filing matches *every* supplied filter.  All three filters
        are optional; passing none exports everything in the storage.

        Args:
            output_dir: Destination directory.  Created if missing.
                Refuses an existing non-empty directory unless
                ``force=True`` so an operator never silently merges a
                new export into the residue of an old one.
            force: Allow overwriting an existing non-empty directory.
            tickers: Filter to filings whose ticker is in this list
                (case-insensitive; uppercased for the SQL match).
            form_types: Filter to filings whose form type is in this
                list (case-insensitive; uppercased for the SQL match).
            accessions: Filter to this exact accession-number list.
            progress_callback: ``(step, current, total)`` callback;
                steps emitted: ``"export-list-filings"``,
                ``"export-chunks"``, ``"export-manifest"``.

        Returns:
            :class:`ExportReport` with the source stamp, counts, and
            wall-clock duration.

        Raises:
            DatabaseError: Source storage missing or unreadable, output
                path refused, or the ChromaDB scan failed.
        """
        # Local imports to keep the module's import graph small at
        # process start — these collaborators pull SQLite and Chroma
        # respectively, neither of which the type hints alone need.
        from sec_generative_search.database.metadata import MetadataRegistry

        started_at = time.monotonic()
        output_dir = Path(output_dir)

        _refuse_symlink_lexical_parents(output_dir, label="Export output path")

        if output_dir.exists():
            if not output_dir.is_dir():
                raise DatabaseError(
                    f"Export output path is not a directory: '{output_dir}'.",
                )
            if any(output_dir.iterdir()) and not force:
                raise DatabaseError(
                    f"Export output directory '{output_dir}' is not empty. "
                    "Pass force=True (CLI: --force) to overwrite, or pick "
                    "an empty directory.",
                )

        if not self._chroma_path.exists():
            raise DatabaseError(
                f"ChromaDB path not found at '{self._chroma_path}'. "
                "Cannot export from an empty deployment.",
            )
        if not self._metadata_db_path.exists():
            raise DatabaseError(
                f"Metadata SQLite file not found at '{self._metadata_db_path}'. "
                "Cannot export from an empty deployment.",
            )

        source_stamp = self._read_embedder_stamp()

        # Resolve the working accession set.  We always go through the
        # registry first — that gives us authoritative filing-level
        # filtering (ticker/form-type) and per-row metadata we need
        # for the JSONL.  ChromaDB filtering by accession is cheaper
        # than per-chunk Python filtering and matches the existing
        # delete-batch pattern in :class:`ChromaDBClient`.
        registry = MetadataRegistry(db_path=str(self._metadata_db_path))
        try:
            filings_by_accession = self._resolve_filings(
                registry=registry,
                tickers=tickers,
                form_types=form_types,
                accessions=accessions,
            )
        finally:
            registry.close()

        if progress_callback is not None:
            progress_callback(
                "export-list-filings",
                len(filings_by_accession),
                len(filings_by_accession),
            )

        output_dir.mkdir(parents=True, exist_ok=True)

        chunks_path = output_dir / self._CHUNKS_NAME
        manifest_path = output_dir / self._MANIFEST_NAME

        chunk_count = self._write_chunks_jsonl(
            chunks_path=chunks_path,
            filings_by_accession=filings_by_accession,
            progress_callback=progress_callback,
        )

        manifest = {
            "format_version": self._FORMAT_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_embedder_stamp": {
                "provider": source_stamp.provider,
                "model": source_stamp.model,
                "dimension": source_stamp.dimension,
            },
            "filing_count": len(filings_by_accession),
            "chunk_count": chunk_count,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        if progress_callback is not None:
            progress_callback("export-manifest", 1, 1)

        # Restrict file permissions on both artefacts.  The directory
        # itself is not chmod'd — operators may want shared-read
        # access on the directory while keeping the contents tight.
        os.chmod(chunks_path, 0o600)
        os.chmod(manifest_path, 0o600)

        duration = time.monotonic() - started_at
        logger.info(
            "Export complete: %d filings, %d chunks, embedder=%s/%s in %.2fs (%s)",
            len(filings_by_accession),
            chunk_count,
            source_stamp.provider,
            source_stamp.model,
            duration,
            output_dir,
        )

        return ExportReport(
            output_dir=str(output_dir),
            source_embedder_stamp=source_stamp,
            filing_count=len(filings_by_accession),
            chunk_count=chunk_count,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Internal — read primitives
    # ------------------------------------------------------------------

    def _read_embedder_stamp(self) -> EmbedderStamp:
        """Read the live ChromaDB collection's stamp via raw client."""
        try:
            # Lazy (F28): keeps chromadb (~0.6 s) off every database import.
            import chromadb

            client = chromadb.PersistentClient(path=str(self._chroma_path))
            collection = client.get_collection(name=COLLECTION_NAME)
        except Exception as exc:
            raise DatabaseError(
                f"Failed to open ChromaDB collection '{COLLECTION_NAME}' for export",
                details=str(exc),
            ) from exc

        metadata = dict(collection.metadata or {})
        try:
            return EmbedderStamp.from_metadata(metadata)
        except ValueError as exc:
            raise DatabaseError(
                "ChromaDB collection has no valid embedder stamp; "
                "refusing to export an unstamped collection.",
                details=str(exc),
            ) from exc

    def _resolve_filings(
        self,
        *,
        registry: MetadataRegistry,
        tickers: list[str] | None,
        form_types: list[str] | None,
        accessions: list[str] | None,
    ) -> dict[str, Any]:
        """Return a dict of ``accession -> FilingRecord`` for the filter set.

        The registry's ``list_filings`` accepts a single ticker / single
        form_type filter; for multi-value filters we union and
        post-filter.  ``accessions`` always wins as the most specific
        filter — it is intersected with the ticker/form-type result so
        passing all three together still narrows correctly.
        """
        # Empty filter lists are treated as "no filter" (matches the
        # CLI surface where an unset flag is equivalent to omitting it).
        ticker_set = {t.upper() for t in tickers or []}
        form_set = {f.upper() for f in form_types or []}
        accession_set = set(accessions or [])

        if accession_set:
            records = registry.get_filings_by_accessions(list(accession_set))
        else:
            records = registry.list_filings()

        result: dict[str, Any] = {}
        for record in records:
            if ticker_set and record.ticker.upper() not in ticker_set:
                continue
            if form_set and record.form_type.upper() not in form_set:
                continue
            result[record.accession_number] = record
        return result

    def _write_chunks_jsonl(
        self,
        *,
        chunks_path: Path,
        filings_by_accession: dict[str, Any],
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> int:
        """Stream ChromaDB chunks for *filings_by_accession* into JSONL.

        Pages through ChromaDB with a ``where={"accession_number":
        {"$in": [...]}}`` filter so the wire request stays bounded.
        ChromaDB's ``$in`` is itself bounded by SQLite's 999-parameter
        cap on the underlying filter — we mirror the registry's
        chunking behaviour by splitting large accession sets into
        sub-batches.
        """
        if not filings_by_accession:
            chunks_path.write_bytes(b"")
            return 0

        try:
            # Lazy (F28): keeps chromadb (~0.6 s) off every database import.
            import chromadb

            client = chromadb.PersistentClient(path=str(self._chroma_path))
            collection = client.get_collection(name=COLLECTION_NAME)
        except Exception as exc:
            raise DatabaseError(
                f"Failed to open ChromaDB collection '{COLLECTION_NAME}' for export",
                details=str(exc),
            ) from exc

        accessions = list(filings_by_accession.keys())
        chunk_count = 0
        # 999 mirrors the SQLite parameter cap that the registry uses;
        # ChromaDB's ``$in`` operator goes through the same SQLite
        # backend in the persistent driver, so the same bound applies.
        sub_batch = 999

        try:
            with chunks_path.open("w", encoding="utf-8") as fh:
                for offset in range(0, len(accessions), sub_batch):
                    batch_accessions = accessions[offset : offset + sub_batch]
                    try:
                        result = collection.get(
                            where={
                                "accession_number": {"$in": batch_accessions},
                            },
                            include=["documents", "metadatas"],
                        )
                    except Exception as exc:
                        raise DatabaseError(
                            "ChromaDB scan failed during portable export",
                            details=str(exc),
                        ) from exc

                    ids = result.get("ids") or []
                    documents = result.get("documents") or []
                    metadatas = result.get("metadatas") or []

                    for chunk_id, document, metadata in zip(
                        ids, documents, metadatas, strict=False
                    ):
                        accession = metadata.get("accession_number")
                        record = filings_by_accession.get(accession)
                        if record is None:
                            # Defensive: a chunk whose accession is
                            # not in the resolved filing set means the
                            # registry / Chroma drifted.  Skip rather
                            # than fabricate a row.
                            continue
                        row = self._build_jsonl_row(
                            chunk_id=chunk_id,
                            document=document,
                            metadata=metadata,
                        )
                        fh.write(json.dumps(row, ensure_ascii=False))
                        fh.write("\n")
                        chunk_count += 1
                        if progress_callback is not None:
                            progress_callback(
                                "export-chunks",
                                chunk_count,
                                # Total is unknown until the scan
                                # finishes; pass ``chunk_count`` as
                                # both fields so the bar advances
                                # without claiming a total.
                                chunk_count,
                            )
        except OSError as exc:
            raise DatabaseError(
                f"Failed to write '{chunks_path}'",
                details=str(exc),
            ) from exc

        return chunk_count

    @staticmethod
    def _build_jsonl_row(
        *,
        chunk_id: str,
        document: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Render a JSONL row from ChromaDB output.

        ``chunk_index`` is recovered from the chunk id (the ``-NNN``
        trailing component) so a future chunker that diverges from
        the existing pattern still produces a parseable export.
        """
        accession = metadata.get("accession_number")
        ticker = metadata.get("ticker")
        form_type = metadata.get("form_type")
        filing_date = metadata.get("filing_date")
        section_path = metadata.get("path")
        content_type = metadata.get("content_type")

        # Recover chunk_index from the trailing zero-padded segment of
        # the chunk_id (``{TICKER}_{FORM}_{DATE}_{INDEX}``).
        chunk_index: int | None = None
        try:
            chunk_index = int(chunk_id.rsplit("_", 1)[-1])
        except (ValueError, AttributeError):
            chunk_index = None

        return {
            "accession": accession,
            "ticker": ticker,
            "form_type": form_type,
            "filing_date": filing_date,
            "chunk_index": chunk_index,
            "section_path": section_path,
            "content_type": content_type,
            "text": document,
        }


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


class PortableImportService:
    """Re-import a JSONL export under the host's configured embedder.

    The service composes :class:`FilingStore` (caller-built over the
    host's ``ChromaDBClient`` + ``MetadataRegistry``) with a pre-built
    embedding provider from :func:`providers.factory.build_embedder`.
    Routing through :class:`FilingStore` preserves the dual-store
    ordering rules the rest of the project relies on; routing through
    the factory keeps credential resolution on the factory seam.

    Duplicate accessions on the host are skipped (not overwritten) via
    :meth:`FilingStore.store_filing` ``register_if_new=True``.

    Example:
        >>> store = FilingStore(chroma, registry)
        >>> embedder = build_embedder(settings.embedding)
        >>> service = PortableImportService(store, embedder)
        >>> report = service.import_("/tmp/export-2026-04-28")
    """

    _FORMAT_VERSION = _FORMAT_VERSION
    _MANIFEST_NAME = _MANIFEST_NAME
    _CHUNKS_NAME = _CHUNKS_NAME

    def __init__(
        self,
        store: FilingStore,
        embedder: BaseEmbeddingProvider,
    ) -> None:
        """Construct the importer over a pre-built store + embedder.

        Args:
            store: A configured :class:`FilingStore` over the host's
                live ChromaDB client and metadata registry.  The store
                already enforces duplicate-detection ordering and
                rollback semantics; the importer never speaks to the
                two backing stores directly.
            embedder: A pre-built :class:`BaseEmbeddingProvider`
                whose ``get_dimension`` matches the live ChromaDB
                collection's stamped dimension.  Construct it via
                :func:`providers.factory.build_embedder` to keep
                credential resolution on the factory seam.
        """
        self._store = store
        self._embedder = embedder

    def import_(
        self,
        input_dir: str | Path,
        *,
        tickers: list[str] | None = None,
        form_types: list[str] | None = None,
        accessions: list[str] | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> ImportReport:
        """Read *input_dir* and re-embed every filing into the live storage.

        Args:
            input_dir: Directory containing ``manifest.json`` and
                ``chunks.jsonl``.
            tickers: Optional filter — only filings whose ticker is in
                this list are imported (case-insensitive).
            form_types: Optional filter on form type.
            accessions: Optional filter on accession number.
            progress_callback: ``(step, current, total)`` callback;
                steps emitted: ``"import-read"``, ``"import-embed"``,
                ``"import-write"``.

        Returns:
            :class:`ImportReport` with imported / skipped / chunk counts.

        Raises:
            DatabaseError: Input path refused, manifest malformed,
                JSONL malformed, or the embedder/store failed.
        """
        started_at = time.monotonic()
        input_dir = Path(input_dir)

        _refuse_symlink_lexical_parents(input_dir, label="Import input path")

        if not input_dir.is_dir():
            raise DatabaseError(
                f"Import input is not a directory: '{input_dir}'.",
            )

        manifest_path = input_dir / self._MANIFEST_NAME
        chunks_path = input_dir / self._CHUNKS_NAME
        if not manifest_path.is_file():
            raise DatabaseError(
                f"Import directory is missing '{self._MANIFEST_NAME}' at '{manifest_path}'.",
            )
        if not chunks_path.is_file():
            raise DatabaseError(
                f"Import directory is missing '{self._CHUNKS_NAME}' at '{chunks_path}'.",
            )

        manifest = self._read_manifest(manifest_path)
        self._validate_format_version(manifest)
        source_stamp = self._parse_manifest_stamp(manifest)

        # Group rows by accession.  We cannot stream filing-by-filing
        # without buffering the whole JSONL because rows for one
        # accession may interleave with rows for another, and we need
        # the full chunk set per accession to drive a single
        # :meth:`FilingStore.store_filing` call.
        grouped = self._group_by_accession(
            chunks_path=chunks_path,
            tickers=tickers,
            form_types=form_types,
            accessions=accessions,
            progress_callback=progress_callback,
        )

        filings_imported = 0
        filings_skipped = 0
        chunks_imported = 0

        for idx, (accession, rows) in enumerate(grouped.items(), start=1):
            processed = self._build_processed_filing(accession=accession, rows=rows)

            if progress_callback is not None:
                progress_callback("import-embed", idx, len(grouped))

            try:
                texts = [chunk.content for chunk in processed.chunks]
                processed.embeddings = self._embedder.embed_texts(texts)
            except Exception as exc:
                raise DatabaseError(
                    f"Embedder failed on accession '{accession}'",
                    details=str(exc),
                ) from exc

            # ``register_if_new=True`` is the atomic SQLite-first
            # path — the loser of a same-accession race gets ``False``
            # back without any ChromaDB write.  See
            # :class:`FilingStore` for the full rationale.
            stored = self._store.store_filing(
                processed,
                register_if_new=True,
            )

            if stored:
                filings_imported += 1
                chunks_imported += len(processed.chunks)
            else:
                filings_skipped += 1
                logger.info(
                    "Skipped already-registered accession '%s' during import",
                    accession,
                )

            if progress_callback is not None:
                progress_callback("import-write", idx, len(grouped))

        duration = time.monotonic() - started_at
        logger.info(
            "Import complete: %d filings imported, %d skipped, %d chunks in %.2fs",
            filings_imported,
            filings_skipped,
            chunks_imported,
            duration,
        )

        return ImportReport(
            input_dir=str(input_dir),
            source_embedder_stamp=source_stamp,
            filings_imported=filings_imported,
            filings_skipped=filings_skipped,
            chunks_imported=chunks_imported,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Internal — manifest parsing
    # ------------------------------------------------------------------

    def _read_manifest(self, manifest_path: Path) -> dict[str, Any]:
        try:
            payload = manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DatabaseError(
                f"Failed to read '{manifest_path}'",
                details=str(exc),
            ) from exc
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DatabaseError(
                f"'{manifest_path}' is not valid JSON; export is corrupt.",
                details=str(exc),
            ) from exc
        if not isinstance(decoded, dict):
            raise DatabaseError(
                f"'{manifest_path}' must be a JSON object; export is corrupt.",
            )
        return decoded

    def _validate_format_version(self, manifest: dict[str, Any]) -> None:
        version = manifest.get("format_version")
        if version != self._FORMAT_VERSION:
            raise DatabaseError(
                f"Export format_version is {version!r}; this build supports "
                f"{self._FORMAT_VERSION}. Upgrade or downgrade to match.",
            )

    def _parse_manifest_stamp(self, manifest: dict[str, Any]) -> EmbedderStamp:
        raw = manifest.get("source_embedder_stamp")
        if not isinstance(raw, dict):
            raise DatabaseError(
                "Export manifest is missing or has malformed "
                "source_embedder_stamp; export is corrupt.",
            )
        try:
            return EmbedderStamp(
                provider=raw["provider"],
                model=raw["model"],
                dimension=int(raw["dimension"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatabaseError(
                "Export manifest source_embedder_stamp is malformed.",
                details=str(exc),
            ) from exc

    # ------------------------------------------------------------------
    # Internal — JSONL grouping and ProcessedFiling reconstruction
    # ------------------------------------------------------------------

    def _group_by_accession(
        self,
        *,
        chunks_path: Path,
        tickers: list[str] | None,
        form_types: list[str] | None,
        accessions: list[str] | None,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> dict[str, list[dict[str, Any]]]:
        ticker_set = {t.upper() for t in tickers or []}
        form_set = {f.upper() for f in form_types or []}
        accession_set = set(accessions or [])

        grouped: dict[str, list[dict[str, Any]]] = {}
        rows_seen = 0
        try:
            with chunks_path.open(encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    rows_seen += 1
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise DatabaseError(
                            f"Line {line_no} of '{chunks_path}' is not "
                            f"valid JSON; export is corrupt.",
                            details=str(exc),
                        ) from exc
                    if not isinstance(row, dict):
                        raise DatabaseError(
                            f"Line {line_no} of '{chunks_path}' is not a "
                            f"JSON object; export is corrupt.",
                        )

                    accession = row.get("accession")
                    if not isinstance(accession, str) or not accession:
                        raise DatabaseError(
                            f"Line {line_no} of '{chunks_path}' is missing "
                            "'accession'; export is corrupt.",
                        )
                    ticker = row.get("ticker")
                    form_type = row.get("form_type")

                    if accession_set and accession not in accession_set:
                        continue
                    if ticker_set and isinstance(ticker, str) and ticker.upper() not in ticker_set:
                        continue
                    if (
                        form_set
                        and isinstance(form_type, str)
                        and form_type.upper() not in form_set
                    ):
                        continue

                    grouped.setdefault(accession, []).append(row)
        except OSError as exc:
            raise DatabaseError(
                f"Failed to read '{chunks_path}'",
                details=str(exc),
            ) from exc

        if progress_callback is not None:
            progress_callback("import-read", rows_seen, rows_seen)

        return grouped

    def _build_processed_filing(
        self,
        *,
        accession: str,
        rows: list[dict[str, Any]],
    ) -> ProcessedFilingShim:
        """Reconstruct a :class:`ProcessedFiling` from JSONL rows.

        The first row supplies the filing-level metadata; all rows
        must agree on ``ticker`` / ``form_type`` / ``filing_date``.
        Rows are sorted by ``chunk_index`` so the rebuild emits the
        chunks in their original order (chunk_index is also fed back
        to :class:`Chunk` so the chunk_id round-trips).
        """
        first = rows[0]
        ticker = first.get("ticker")
        form_type = first.get("form_type")
        filing_date_str = first.get("filing_date")
        if not isinstance(ticker, str) or not ticker:
            raise DatabaseError(
                f"Accession '{accession}' rows are missing 'ticker'.",
            )
        if not isinstance(form_type, str) or not form_type:
            raise DatabaseError(
                f"Accession '{accession}' rows are missing 'form_type'.",
            )
        if not isinstance(filing_date_str, str):
            raise DatabaseError(
                f"Accession '{accession}' rows are missing 'filing_date'.",
            )
        try:
            filing_date = date.fromisoformat(filing_date_str)
        except ValueError as exc:
            raise DatabaseError(
                f"Accession '{accession}' has invalid filing_date '{filing_date_str}'.",
                details=str(exc),
            ) from exc

        filing_id = FilingIdentifier(
            ticker=ticker,
            form_type=form_type,
            filing_date=filing_date,
            accession_number=accession,
        )

        # Sort by chunk_index so the export's order survives round-trip.
        rows_sorted = sorted(
            rows,
            key=lambda r: r.get("chunk_index") if isinstance(r.get("chunk_index"), int) else 0,
        )

        chunks: list[Chunk] = []
        for idx, row in enumerate(rows_sorted):
            text = row.get("text")
            section_path = row.get("section_path") or ""
            content_type_raw = row.get("content_type") or "text"
            chunk_index = row.get("chunk_index")
            if not isinstance(text, str):
                raise DatabaseError(
                    f"Accession '{accession}' row {idx} is missing 'text'.",
                )
            try:
                content_type = ContentType(content_type_raw)
            except ValueError as exc:
                raise DatabaseError(
                    f"Accession '{accession}' row {idx} has invalid "
                    f"content_type '{content_type_raw}'.",
                    details=str(exc),
                ) from exc
            chunks.append(
                Chunk(
                    content=text,
                    path=section_path,
                    content_type=content_type,
                    filing_id=filing_id,
                    chunk_index=(chunk_index if isinstance(chunk_index, int) else idx),
                    token_count=0,
                )
            )

        return ProcessedFilingShim(
            filing_id=filing_id,
            chunks=chunks,
            embeddings=None,
            ingest_result=IngestResult(
                filing_id=filing_id,
                segment_count=len(chunks),
                chunk_count=len(chunks),
                duration_seconds=0.0,
            ),
        )


# ---------------------------------------------------------------------------
# ProcessedFilingShim
# ---------------------------------------------------------------------------
# A minimal stand-in for ``pipeline.orchestrator.ProcessedFiling``
# kept inside this module so the import service never pulls the
# pipeline package — that keeps the source-level import-hygiene test
# clean and lets the importer ship in a build that has the pipeline
# extras pruned out (cloud-only deployments).  The structural fields
# match what :meth:`FilingStore.store_filing` actually consumes:
# ``filing_id``, ``chunks``, ``embeddings``, ``ingest_result``.


class ProcessedFilingShim:
    """Lightweight ProcessedFiling stand-in for the import path.

    Mirrors the exact attribute surface
    :meth:`FilingStore.store_filing` reads (see
    :class:`ChromaDBClient.store_filing` for the underlying contract):

    - ``filing_id`` — :class:`FilingIdentifier`
    - ``chunks`` — list of :class:`Chunk`
    - ``embeddings`` — :class:`numpy.ndarray` or ``None``
    - ``ingest_result`` — :class:`IngestResult`

    Defined here rather than reusing the pipeline class to keep the
    portable module's import graph credential-free and pipeline-free.
    """

    __slots__ = ("chunks", "embeddings", "filing_id", "ingest_result")

    def __init__(
        self,
        *,
        filing_id: FilingIdentifier,
        chunks: list[Chunk],
        embeddings: np.ndarray | None,
        ingest_result: IngestResult,
    ) -> None:
        self.filing_id = filing_id
        self.chunks = chunks
        self.embeddings = embeddings
        self.ingest_result = ingest_result
