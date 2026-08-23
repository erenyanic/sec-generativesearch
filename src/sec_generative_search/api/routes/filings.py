"""Filing management routes.

Surface for inspecting and removing filings already ingested into the
dual store.  The routes are written against the current package seams
— :class:`FilingStore` for every write, the ``admin_route_dependencies()``
helper for destructive routes, and the structured
``{error, message, details, hint}`` envelope.

Read tier (``Depends(verify_api_key)``):

    - ``GET /api/filings/``               — list with optional filters
    - ``GET /api/filings/{accession}``    — single-record lookup

Destructive tier (``admin_route_dependencies()`` —
``verify_api_key`` THEN ``verify_admin_key``):

    - ``DELETE /api/filings/{accession}``    — remove one filing
    - ``POST   /api/filings/delete-by-ids``  — remove by accession list
    - ``POST   /api/filings/bulk-delete``    — remove by ticker/form filter
    - ``DELETE /api/filings/?confirm=true``  — wipe every filing

Why route through :class:`FilingStore` and never the underlying
collaborators directly:

    - The store is the single seam that owns the ChromaDB-first delete
      ordering and the rollback semantics.  Calling
      ``ChromaDBClient.delete_filing`` and ``MetadataRegistry.remove_filing``
      individually here would silently re-implement that ordering with
      drift potential — and existing storage tests pin the behaviour at
      the store layer.
    - It mirrors the ingestion path (which also writes through
      :class:`FilingStore`), keeping the operator's mental model of "all
      mutations of the dual store happen in one place" intact.

Audit-log discipline:

    - Every destructive call emits a ``SECURITY_AUDIT:`` line via
      :func:`audit_log` carrying the action, client IP, endpoint, and a
      compact effect summary.  Tickers and accession numbers in the
      detail string are pre-scrubbed by the existing privacy contract
      for task-history rows.
    - Lookup misses (404) and validation failures (400 / 422) are not
      audit-logged here — they are noise on a normal browser session and
      are already covered by the access log.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Path, Query, Request

from sec_generative_search.api.dependencies import (
    admin_route_dependencies,
    get_filing_store,
    get_registry,
    verify_api_key,
)
from sec_generative_search.api.errors import http_error
from sec_generative_search.api.schemas import (
    BulkDeleteRequest,
    BulkDeleteResponse,
    ClearAllResponse,
    DeleteByIdsRequest,
    DeleteByIdsResponse,
    DeleteResponse,
    FilingListResponse,
    FilingSchema,
)
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import DatabaseError
from sec_generative_search.core.logging import audit_log, get_logger, redact_for_log
from sec_generative_search.database import FilingRecord, FilingStore, MetadataRegistry

__all__ = ["router"]


logger = get_logger(__name__)


# Accession numbers are pinned at the path level too so a shape mismatch
# bounces with a 422 before any route logic runs.  The pattern matches
# the schema's ``_ACCESSION_PATTERN``; do not relax either site without
# updating the other.
_ACCESSION_PATH_PATTERN = r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$"


# Generous per-page ceiling for the opt-in ``limit`` query param.  The
# ``filings`` table is itself bounded by ``DB_MAX_FILINGS`` (≤ 10 000 in
# B/C), so this cap only guards against an absurd single-page request;
# omitting ``limit`` still returns the full (bounded) list unchanged.
_MAX_PAGE_SIZE = 10_000


# Read-tier router: list / detail are gated by ``verify_api_key`` only.
router = APIRouter(dependencies=[Depends(verify_api_key)])


def _record_to_schema(record: FilingRecord) -> FilingSchema:
    """Lift the SQLite-backed dataclass to the wire shape.

    The auto-increment ``id`` is intentionally dropped — it is an
    internal sequence number that has no meaning to API consumers.
    """
    return FilingSchema(
        ticker=record.ticker,
        form_type=record.form_type,
        filing_date=record.filing_date,
        accession_number=record.accession_number,
        chunk_count=record.chunk_count,
        ingested_at=record.ingested_at,
    )


def _client_ip(request: Request) -> str:
    """Best-effort client IP for audit-log lines.

    ``request.client`` is ``None`` for ASGI scopes lacking a peer
    address (older test clients, certain proxies).  Returning ``"unknown"``
    keeps the audit-line shape stable so downstream parsers do not need
    to handle a missing field.
    """
    return request.client.host if request.client else "unknown"


def _database_error(exc: DatabaseError) -> Exception:
    """Wrap a raw :class:`DatabaseError` in the API's structured envelope.

    The driver-level ``details`` field is intentionally NOT echoed back
    to the client — SQLite / ChromaDB error strings routinely include
    file paths and SQL fragments unsuitable for a public response.  The
    exception is logged in full at the call site.
    """
    return http_error(
        status_code=500,
        error="database_error",
        message="Database operation failed. Check server logs.",
        hint="Check that the data directory is writable and the database is intact.",
    )


# ---------------------------------------------------------------------------
# Read routes
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=FilingListResponse,
    summary="List ingested filings",
)
def list_filings(
    registry: MetadataRegistry = Depends(get_registry),
    ticker: str | None = Query(
        default=None,
        max_length=16,
        pattern=r"^[A-Za-z][A-Za-z0-9.\-]{0,15}$",
        description="Filter by ticker symbol (case-insensitive).",
    ),
    form_type: str | None = Query(
        default=None,
        max_length=16,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9/\-]{0,15}$",
        description="Filter by form type (e.g. 10-K, 10-K/A).",
    ),
    sort_by: Literal[
        "filing_date",
        "ticker",
        "form_type",
        "chunk_count",
        "ingested_at",
    ] = Query(
        default="filing_date",
        description="Column to sort by.",
    ),
    order: Literal["asc", "desc"] = Query(
        default="desc",
        description="Sort order.",
    ),
    limit: int | None = Query(
        default=None,
        ge=1,
        le=_MAX_PAGE_SIZE,
        description=(
            "Maximum filings to return (one page). Omit for the full "
            "bounded list. Pair with 'offset' to page through results."
        ),
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of filings to skip before the page starts.",
    ),
) -> FilingListResponse:
    """List filings with optional filters, operator-chosen sort, and paging.

    Sorting and pagination are resolved entirely in SQL — the registry
    applies ``ORDER BY … LIMIT … OFFSET`` (backed by
    ``idx_filings_filing_date`` for the default sort) with a deterministic
    ``id`` tie-breaker, so a page is stable across requests.

    Pagination is **opt-in and additive**: omitting ``limit`` returns the
    full ``DB_MAX_FILINGS``-bounded list exactly as before, so existing
    callers are unaffected. ``total`` remains the count of *returned* rows
    (a page size when paginating), not the registry-wide cardinality —
    operators needing the global count use ``GET /api/status/``. A client
    walking pages stops when a page returns fewer than ``limit`` rows.
    """
    try:
        records = registry.list_filings(
            ticker=ticker.upper() if ticker else None,
            form_type=form_type.upper() if form_type else None,
            sort_by=sort_by,
            order=order,
            limit=limit,
            offset=offset,
        )
    except DatabaseError as exc:
        logger.error("list_filings failed: %s", exc.details)
        raise _database_error(exc) from exc

    schemas = [_record_to_schema(r) for r in records]
    return FilingListResponse(filings=schemas, total=len(schemas))


@router.get(
    "/{accession}",
    response_model=FilingSchema,
    summary="Get a single filing by accession number",
)
def get_filing(
    accession: str = Path(
        ...,
        max_length=20,
        pattern=_ACCESSION_PATH_PATTERN,
        description="SEC accession number (NNNNNNNNNN-NN-NNNNNN).",
    ),
    registry: MetadataRegistry = Depends(get_registry),
) -> FilingSchema:
    """Return a single filing record or surface a 404 envelope.

    The 404 hint deliberately steers the caller back to ``GET /api/filings/``
    rather than echoing ticker / form-type guesses; we do not enumerate
    candidate matches here because that would let an unauthenticated
    discovery tool probe the corpus shape.
    """
    try:
        record = registry.get_filing(accession)
    except DatabaseError as exc:
        logger.error("get_filing(%s) failed: %s", accession, exc.details)
        raise _database_error(exc) from exc

    if record is None:
        raise http_error(
            status_code=404,
            error="not_found",
            message=f"Filing not found: {accession}",
            hint="Use GET /api/filings/ to list available accession numbers.",
        )
    return _record_to_schema(record)


# ---------------------------------------------------------------------------
# Destructive routes (admin-gated)
# ---------------------------------------------------------------------------


@router.delete(
    "/{accession}",
    response_model=DeleteResponse,
    summary="Delete a single filing",
    dependencies=admin_route_dependencies(),
)
def delete_filing(
    request: Request,
    accession: str = Path(
        ...,
        max_length=20,
        pattern=_ACCESSION_PATH_PATTERN,
    ),
    registry: MetadataRegistry = Depends(get_registry),
    store: FilingStore = Depends(get_filing_store),
) -> DeleteResponse:
    """Remove a single filing from both stores.

    The pre-check via :meth:`MetadataRegistry.get_filing` lets us return
    a clean 404 without paying the ChromaDB round-trip when the
    accession is not registered.  After the delete, the audit-log line
    carries the chunk count so the operator can correlate the row count
    with the ChromaDB segment delete reported in the storage logs.
    """
    try:
        record = registry.get_filing(accession)
    except DatabaseError as exc:
        logger.error("delete_filing get(%s) failed: %s", accession, exc.details)
        raise _database_error(exc) from exc

    if record is None:
        raise http_error(
            status_code=404,
            error="not_found",
            message=f"Filing not found: {accession}",
            hint="Use GET /api/filings/ to list available accession numbers.",
        )

    try:
        store.delete_filing(accession)
    except DatabaseError as exc:
        logger.error("delete_filing(%s) failed: %s", accession, exc.details)
        raise _database_error(exc) from exc

    audit_log(
        "delete_filing",
        client_ip=_client_ip(request),
        endpoint="DELETE /api/filings/{accession}",
        detail=(
            f"accession={accession} ticker={redact_for_log(record.ticker)} "
            f"form={record.form_type} chunks={record.chunk_count}"
        ),
    )
    return DeleteResponse(
        accession_number=accession,
        chunks_deleted=record.chunk_count,
    )


@router.post(
    "/delete-by-ids",
    response_model=DeleteByIdsResponse,
    summary="Delete filings by accession-number list",
    dependencies=admin_route_dependencies(),
)
def delete_by_ids(
    request: Request,
    body: DeleteByIdsRequest,
    registry: MetadataRegistry = Depends(get_registry),
    store: FilingStore = Depends(get_filing_store),
) -> DeleteByIdsResponse:
    """Delete a bounded list of accession numbers in one round-trip.

    Looks up the registry once (single ``IN (?, ?, …)`` query) so the
    response can report the missing accessions without paying N
    individual ``get_filing`` calls.  Pre-existence is required only for
    *reporting* — the underlying batch delete is idempotent on missing
    accessions because both stores treat them as no-ops.
    """
    try:
        found = registry.get_filings_by_accessions(body.accession_numbers)
    except DatabaseError as exc:
        logger.error("delete_by_ids lookup failed: %s", exc.details)
        raise _database_error(exc) from exc

    found_accessions = {r.accession_number for r in found}
    not_found = [a for a in body.accession_numbers if a not in found_accessions]

    if not found:
        return DeleteByIdsResponse(
            filings_deleted=0,
            chunks_deleted=0,
            not_found=not_found,
        )

    accessions = [record.accession_number for record in found]
    chunks_total = sum(record.chunk_count for record in found)

    try:
        store.delete_filings_batch(accessions)
    except DatabaseError as exc:
        logger.error("delete_by_ids batch failed: %s", exc.details)
        raise _database_error(exc) from exc

    audit_log(
        "delete_filings_batch",
        client_ip=_client_ip(request),
        endpoint="POST /api/filings/delete-by-ids",
        detail=(f"deleted={len(found)} chunks={chunks_total} not_found={len(not_found)}"),
    )
    return DeleteByIdsResponse(
        filings_deleted=len(found),
        chunks_deleted=chunks_total,
        not_found=not_found,
    )


@router.post(
    "/bulk-delete",
    response_model=BulkDeleteResponse,
    summary="Bulk delete filings by ticker / form_type filter",
    dependencies=admin_route_dependencies(),
)
def bulk_delete(
    request: Request,
    body: BulkDeleteRequest,
    registry: MetadataRegistry = Depends(get_registry),
    store: FilingStore = Depends(get_filing_store),
) -> BulkDeleteResponse:
    """Delete every filing matching the given filter.

    The filter must narrow the result set — a fully-empty body is
    rejected at 400 so an unguarded "delete everything" is not
    reachable via this surface.  Wiping the entire registry has its own
    confirm-gated route below.
    """
    if body.ticker is None and body.form_type is None:
        raise http_error(
            status_code=400,
            error="validation_error",
            message="At least one filter is required.",
            hint=(
                "Provide 'ticker' and/or 'form_type'. "
                "To delete everything, use DELETE /api/filings/?confirm=true."
            ),
        )

    try:
        filings = registry.list_filings(
            ticker=body.ticker.upper() if body.ticker else None,
            form_type=body.form_type.upper() if body.form_type else None,
        )
    except DatabaseError as exc:
        logger.error("bulk_delete list failed: %s", exc.details)
        raise _database_error(exc) from exc

    if not filings:
        return BulkDeleteResponse(
            filings_deleted=0,
            chunks_deleted=0,
            tickers_affected=[],
        )

    accessions = [f.accession_number for f in filings]
    chunks_total = sum(f.chunk_count for f in filings)

    try:
        store.delete_filings_batch(accessions)
    except DatabaseError as exc:
        logger.error("bulk_delete batch failed: %s", exc.details)
        raise _database_error(exc) from exc

    tickers_affected = sorted({f.ticker for f in filings})

    audit_log(
        "bulk_delete",
        client_ip=_client_ip(request),
        endpoint="POST /api/filings/bulk-delete",
        detail=(f"filings={len(filings)} chunks={chunks_total} tickers={len(tickers_affected)}"),
    )
    return BulkDeleteResponse(
        filings_deleted=len(filings),
        chunks_deleted=chunks_total,
        tickers_affected=tickers_affected,
    )


@router.delete(
    "/",
    response_model=ClearAllResponse,
    summary="Clear every filing from both stores",
    dependencies=admin_route_dependencies(),
)
def clear_all(
    request: Request,
    confirm: bool = Query(
        default=False,
        description="Safety flag — must be true to proceed.",
    ),
    store: FilingStore = Depends(get_filing_store),
) -> ClearAllResponse:
    """Wipe every filing across both stores.

    Two safety gates:

    1. ``confirm=true`` MUST be present.  Without it the route returns
       400 — preventing a stray ``curl -X DELETE`` from emptying the
       corpus.
    2. ``API_DEMO_MODE=true`` returns 403 unconditionally — the demo
       deployment relies on the nightly reset job for wipes and a
       human-driven clear-all bypassing that schedule has historically
       caused contention.
    """
    if get_settings().api.demo_mode:
        raise http_error(
            status_code=403,
            error="demo_mode",
            message="Clear-all is disabled while the deployment is in demo mode.",
            hint="Demo data resets automatically on the nightly schedule.",
        )

    if not confirm:
        raise http_error(
            status_code=400,
            error="confirmation_required",
            message="This will delete ALL filings. Pass ?confirm=true to proceed.",
            hint="Add '?confirm=true' to the request URL to acknowledge the wipe.",
        )

    try:
        chunks_deleted, filings_deleted = store.clear_all()
    except DatabaseError as exc:
        logger.error("clear_all failed: %s", exc.details)
        raise _database_error(exc) from exc

    audit_log(
        "clear_all",
        client_ip=_client_ip(request),
        endpoint="DELETE /api/filings/?confirm=true",
        detail=f"filings={filings_deleted} chunks={chunks_deleted}",
    )
    return ClearAllResponse(
        filings_deleted=filings_deleted,
        chunks_deleted=chunks_deleted,
    )
