"""Retrieval-only search route.

Single endpoint: ``POST /api/search``. Wraps the pre-built
:class:`RetrievalService` attached to ``app.state`` and returns the
ranked list of :class:`RetrievalResult` as :class:`SearchHit` rows.

The query stays in the request body, the response never echoes it back,
and the route only depends on the read-tier API key gate.

Error mapping:

        - :class:`SearchError` → 400 ``invalid_query`` envelope.
        - :class:`DatabaseError` → 500 ``database_error``.
        - :class:`ProviderError` → 502 ``provider_error``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from sec_generative_search.api.dependencies import (
    get_retrieval_service,
    verify_api_key,
)
from sec_generative_search.api.errors import http_error
from sec_generative_search.api.schemas import (
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from sec_generative_search.core.exceptions import (
    DatabaseError,
    ProviderError,
    SearchError,
)
from sec_generative_search.core.logging import audit_log, get_logger, redact_for_log
from sec_generative_search.core.types import RetrievalResult
from sec_generative_search.search import RetrievalService

__all__ = ["router"]


logger = get_logger(__name__)


router = APIRouter(dependencies=[Depends(verify_api_key)])


def _client_ip(request: Request) -> str:
    """Best-effort client IP for audit-log lines.

    ``request.client`` is ``None`` for ASGI scopes lacking a peer
    address (older test clients, certain proxies).  Returning
    ``"unknown"`` keeps the audit-line shape stable so downstream
    parsers do not need to handle a missing field.
    """
    return request.client.host if request.client else "unknown"


def _hit_from_result(result: RetrievalResult) -> SearchHit:
    """Lift the dataclass result onto the wire schema.

    Only fields explicitly declared on :class:`SearchHit` are surfaced —
    no ``**asdict()`` splat, so a future field addition on
    :class:`RetrievalResult` does not silently leak onto the API.
    """
    return SearchHit(
        chunk_id=result.chunk_id,
        content=result.content,
        path=result.path,
        content_type=result.content_type.value,
        ticker=result.ticker,
        form_type=result.form_type,
        filing_date=result.filing_date,
        accession_number=result.accession_number,
        similarity=result.similarity,
        rerank_score=result.rerank_score,
        token_count=result.token_count,
        truncated=result.truncated,
        section_boundaries=list(result.section_boundaries),
    )


@router.post(
    "",
    response_model=SearchResponse,
    tags=["search"],
    summary="Retrieve filing chunks for a natural-language query",
)
def search_filings(
    request: Request,
    body: SearchRequest,
    service: RetrievalService = Depends(get_retrieval_service),
) -> SearchResponse:
    """Run a single-query retrieval and return the ranked hits.

    The route is a thin adapter around
    :meth:`RetrievalService.retrieve` — every business rule (token
    budgeting, diversity caps, date validation, dedup) lives in the
    service.  Keeping the route this thin makes the resolver-chain
    seam explicit: a future change to credential resolution does not
    need to touch routes.
    """
    try:
        results = service.retrieve(
            body.query,
            top_k=body.top_k,
            ticker=body.ticker,
            form_type=body.form_type,
            accession_number=body.accession_number,
            start_date=body.start_date,
            end_date=body.end_date,
            min_similarity=body.min_similarity,
            max_per_section=body.max_per_section,
            max_per_filing=body.max_per_filing,
            rerank_over_fetch_factor=body.rerank_over_fetch_factor,
            context_token_budget=body.context_token_budget,
        )
    except SearchError as exc:
        # Caller-input fault (empty query, malformed date, bad top_k) or
        # an untyped downstream failure the service could not classify.
        # ``str(exc)`` renders ``"{message} — {details}"``, and
        # ``details`` may carry driver text or a file path, so the body
        # gets a FIXED message and the detail stays in the operator log
        # only (``redact_for_log`` honours ``LOG_REDACT_QUERIES`` there).
        logger.warning(
            "search rejected: query=%r details=%s",
            redact_for_log(body.query[:80]),
            exc.details,
        )
        raise http_error(
            status_code=400,
            error="invalid_query",
            message="The search query could not be processed.",
            hint=(
                "Check the query (must be non-empty), date filters "
                "(YYYY-MM-DD), and top_k (positive integer)."
            ),
        ) from exc
    except ProviderError as exc:
        # Embedding-side failure — the corpus and storage layer are fine,
        # the embedder upstream is not.  Surface as 502 so the client
        # does not retry into the same broken path indefinitely.
        logger.error(
            "search embedder failure: %s",
            type(exc).__name__,
        )
        raise http_error(
            status_code=502,
            error="provider_error",
            message="The embedding provider failed while processing the query.",
            hint=(
                "Retry after a short backoff; if the failure persists, "
                "check the embedder admin-env credential."
            ),
        ) from exc
    except DatabaseError as exc:
        # Storage-layer failure — same redaction pattern as filings.
        logger.error("search database failure: %s", exc.details)
        raise http_error(
            status_code=500,
            error="database_error",
            message="Database operation failed. Check server logs.",
            hint="Check that the data directory is readable and the database is intact.",
        ) from exc

    audit_log(
        "search_executed",
        client_ip=_client_ip(request),
        endpoint="POST /api/search",
        detail=(
            f"hits={len(results)} top_k={body.top_k or 'default'} "
            f"ticker={redact_for_log(body.ticker) if body.ticker else 'any'} "
            f"form_type={body.form_type or 'any'}"
        ),
    )
    return SearchResponse(
        hits=[_hit_from_result(r) for r in results],
        total=len(results),
    )
