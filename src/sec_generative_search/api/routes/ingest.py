"""Ingestion routes for task creation, lookup, and cancellation.

Rewritten against the current package seams:

        - :class:`~sec_generative_search.api.tasks.TaskManager` is the only
            construction site for an ingest task; the route layer is a thin
            validation + lifting shim.
        - EDGAR identity is resolved through the canonical resolver chain
            (``header → session → admin-env``) by reusing
            :func:`~sec_generative_search.api.dependencies.get_edgar_identity`.
            A request without a per-session identity fails the resolver tier
            gating BEFORE any work-list build happens.
        - Rate limiting flows through ``api/policies.py::ROUTE_POLICIES``
            via the ``ingest`` category. The bespoke per-IP cooldown dict and
            ``INGEST_COOLDOWN_SECONDS`` are gone.
        - Session-scoped ownership: ``GET /api/ingest/tasks`` only returns
            tasks owned by the current ``session_id``; lookups / cancels for
            tasks owned by another session surface as ``404 not_found`` rather
            than ``403`` so the route never confirms that a foreign task
            exists (information-leak prophylaxis).

Routes (every one read-tier — ``verify_api_key`` only):

    - ``POST   /api/ingest/add``           — single-ticker ingest
    - ``POST   /api/ingest/batch``         — multi-ticker ingest
    - ``GET    /api/ingest/tasks``         — list this session's tasks
    - ``GET    /api/ingest/tasks/{id}``    — task status (own only)
    - ``DELETE /api/ingest/tasks/{id}``    — cancel a task (own only)

Why two endpoints for what is structurally the same body? The
``/add`` variant pins ``len(tickers) == 1`` at the handler so a UI
implementing a per-ticker "Add" button cannot accidentally start a
batch ingest by passing two tickers in one request.

Audit-log discipline:

        - Every create / cancel emits a ``SECURITY_AUDIT:`` line via
            :func:`audit_log` carrying the action, client IP, masked task id
            tail, the ticker **count**, and the form types. The symbols
            themselves are research-pattern data and never reach the line
            (finding M3); names / emails / provider keys never reach this
            surface either, so there is nothing further to redact here that
            is not already covered by the access-log layer.
        - Read paths (``GET /api/ingest/tasks/{id}``, list) do not audit-log
            — they would flood under WebSocket-less polling clients and they
            already appear in the access log.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Path, Request

from sec_generative_search.api.dependencies import (
    extract_session_id,
    get_edgar_identity,
    get_task_manager,
    verify_api_key,
)
from sec_generative_search.api.errors import http_error
from sec_generative_search.api.schemas import (
    IngestCancelResponse,
    IngestRequest,
    IngestResultSchema,
    IngestTaskResponse,
    TaskListResponse,
    TaskProgressSchema,
    TaskStatusResponse,
)
from sec_generative_search.api.tasks import TaskInfo, TaskManager, TaskQueueFullError
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.edgar_identity import EdgarIdentity
from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.core.security import mask_secret

if TYPE_CHECKING:
    pass

__all__ = ["router"]


logger = get_logger(__name__)


# Task ids are 32-char hex UUIDs; pin them at the path level so an
# obviously forged path bounces with 422 before any state is read.
_TASK_ID_PATH_PATTERN = r"^[0-9a-f]{32}$"


# Read-tier router: every ingest endpoint is gated by ``verify_api_key`` only.
router = APIRouter(dependencies=[Depends(verify_api_key)])


def _client_ip(request: Request) -> str:
    """Best-effort client IP for audit-log lines."""
    return request.client.host if request.client else "unknown"


def _enforce_request_caps(body: IngestRequest) -> None:
    """Reject payloads that exceed the operator-configured caps.

    ``max_tickers_per_request`` and ``max_filings_per_request`` sit on
    top of the schema-layer bounds (``tickers`` 1..50, ``count`` 1..500).
    Both default to 0 (disabled); when set, they trump the schema bounds.
    """
    settings = get_settings()
    api = settings.api

    if api.max_tickers_per_request > 0 and len(body.tickers) > api.max_tickers_per_request:
        raise http_error(
            status_code=400,
            error="request_cap_exceeded",
            message=(
                f"Too many tickers: {len(body.tickers)} "
                f"(maximum {api.max_tickers_per_request} per request)."
            ),
            hint=f"Submit at most {api.max_tickers_per_request} tickers per request.",
        )

    if (
        api.max_filings_per_request > 0
        and body.count is not None
        and body.count > api.max_filings_per_request
    ):
        raise http_error(
            status_code=400,
            error="request_cap_exceeded",
            message=(
                f"Too many filings requested: {body.count} "
                f"(maximum {api.max_filings_per_request} per request)."
            ),
            hint=f"Request at most {api.max_filings_per_request} filings per request.",
        )


def _create_task(
    request: Request,
    body: IngestRequest,
    manager: TaskManager,
    identity: EdgarIdentity,
) -> IngestTaskResponse:
    """Shared create-path for ``/add`` and ``/batch``.

    Captures the per-request EDGAR identity into a closure held by the
    manager off ``TaskInfo`` so the dataclass stays free of identity data.
    """
    _enforce_request_caps(body)

    # Snapshot the identity into a closure so the resolver returns the
    # value captured at create time, not whatever the request scope held
    # when the worker thread eventually fired. ``EdgarIdentity`` is
    # frozen, so the closure capture is safe.
    captured_identity = identity

    def _identity_resolver() -> EdgarIdentity:
        return captured_identity

    session_id = extract_session_id(request)

    try:
        task_id = manager.create_task(
            tickers=list(body.tickers),
            form_types=list(body.form_types),
            count_mode=body.count_mode,
            count=body.count,
            year=body.year,
            start_date=body.start_date,
            end_date=body.end_date,
            session_id=session_id,
            edgar_identity_resolver=_identity_resolver,
        )
    except TaskQueueFullError as exc:
        details = exc.details
        raise http_error(
            status_code=429,
            error="queue_full",
            message=str(exc),
            details=details,
            hint="Wait for existing tasks to complete before submitting new ones.",
        ) from exc

    audit_log(
        "ingest_task_created",
        detail=(
            f"client_ip={_client_ip(request)} "
            f"task_id_tail={mask_secret(task_id)} "
            f"tickers={len(body.tickers)} "
            f"form_types={list(body.form_types)} "
            f"count_mode={body.count_mode}"
        ),
    )

    return IngestTaskResponse(
        task_id=task_id,
        status="pending",
        websocket_url=f"/ws/ingest/{task_id}",
    )


def _task_info_to_schema(info: TaskInfo) -> TaskStatusResponse:
    """Lift the worker's ``TaskInfo`` to the wire schema.

    ``session_id`` is intentionally not surfaced; the wire schema mirrors
    the visible-progress fields only.
    """
    return TaskStatusResponse(
        task_id=info.task_id,
        status=info.state.value,
        tickers=list(info.tickers),
        form_types=list(info.form_types),
        progress=TaskProgressSchema(
            current_ticker=info.progress.current_ticker,
            current_form_type=info.progress.current_form_type,
            step_label=info.progress.step_label,
            step_index=info.progress.step_index,
            step_total=info.progress.step_total,
            filings_done=info.progress.filings_done,
            filings_total=info.progress.filings_total,
            filings_skipped=info.progress.filings_skipped,
            filings_failed=info.progress.filings_failed,
        ),
        results=[
            IngestResultSchema(
                ticker=r.ticker,
                form_type=r.form_type,
                filing_date=r.filing_date,
                accession_number=r.accession_number,
                segment_count=r.segment_count,
                chunk_count=r.chunk_count,
                duration_seconds=r.duration_seconds,
            )
            for r in info.results
        ],
        error=info.error,
        started_at=info.started_at.isoformat() if info.started_at else None,
        completed_at=info.completed_at.isoformat() if info.completed_at else None,
    )


def _resolve_owned_task(
    request: Request,
    task_id: str,
    manager: TaskManager,
) -> TaskInfo:
    """Return the task only when the caller owns it.

    Foreign tasks (different ``session_id``) surface as ``404 not_found``
    rather than ``403`` so the route never confirms a task's existence
    to a non-owner. Single-tenant deployments (no session cookie at all)
    match tasks whose ``session_id`` is also ``None``.
    """
    info = manager.get_task(task_id)
    if info is None:
        raise http_error(
            status_code=404,
            error="not_found",
            message=f"Task '{task_id}' not found.",
            hint="The task may have been evicted from memory after completion.",
        )

    caller_session = extract_session_id(request)
    if info.session_id != caller_session:
        raise http_error(
            status_code=404,
            error="not_found",
            message=f"Task '{task_id}' not found.",
            hint="The task may have been evicted from memory after completion.",
        )
    return info


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/add",
    response_model=IngestTaskResponse,
    status_code=202,
    summary="Ingest filings for a single ticker",
)
async def ingest_add(
    request: Request,
    body: IngestRequest,
    manager: TaskManager = Depends(get_task_manager),
    identity: EdgarIdentity = Depends(get_edgar_identity),
) -> IngestTaskResponse:
    """Start a single-ticker ingestion task.

    The schema accepts ``tickers: list[str]`` (so the body shape is
    identical to ``/batch``); the handler pins ``len == 1`` so a
    multi-ticker payload routed to ``/add`` bounces with ``400`` rather
    than silently consuming all of them.
    """
    if len(body.tickers) != 1:
        raise http_error(
            status_code=400,
            error="validation_error",
            message="The /add endpoint accepts exactly one ticker.",
            details={"received": len(body.tickers)},
            hint="Use POST /api/ingest/batch for multi-ticker ingestion.",
        )
    return _create_task(request, body, manager, identity)


@router.post(
    "/batch",
    response_model=IngestTaskResponse,
    status_code=202,
    summary="Ingest filings for one or more tickers",
)
async def ingest_batch(
    request: Request,
    body: IngestRequest,
    manager: TaskManager = Depends(get_task_manager),
    identity: EdgarIdentity = Depends(get_edgar_identity),
) -> IngestTaskResponse:
    """Start a multi-ticker ingestion task."""
    return _create_task(request, body, manager, identity)


@router.get(
    "/tasks",
    response_model=TaskListResponse,
    summary="List ingestion tasks owned by the current session",
)
async def list_tasks(
    request: Request,
    manager: TaskManager = Depends(get_task_manager),
) -> TaskListResponse:
    """List the current session's ingestion tasks.

    Cross-session enumeration is intentionally not exposed. Callers with
    no session cookie see tasks whose ``session_id`` is ``None``.
    """
    session_id = extract_session_id(request)
    tasks = manager.list_tasks_for_session(session_id)
    statuses = [_task_info_to_schema(t) for t in tasks]
    return TaskListResponse(tasks=statuses, total=len(statuses))


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get a single task's status",
)
async def get_task(
    request: Request,
    task_id: str = Path(pattern=_TASK_ID_PATH_PATTERN),
    manager: TaskManager = Depends(get_task_manager),
) -> TaskStatusResponse:
    """Return the live progress + result set for a task the caller owns."""
    info = _resolve_owned_task(request, task_id, manager)
    return _task_info_to_schema(info)


@router.delete(
    "/tasks/{task_id}",
    response_model=IngestCancelResponse,
    summary="Cancel a running task the caller owns",
)
async def cancel_task(
    request: Request,
    task_id: str = Path(pattern=_TASK_ID_PATH_PATTERN),
    manager: TaskManager = Depends(get_task_manager),
) -> IngestCancelResponse:
    """Cancel a pending or running task.

    Returns ``404 not_found`` for tasks the caller does not own, the
    same shape as a genuinely missing task — that keeps the route from
    confirming the existence of someone else's work. Already-terminal
    tasks surface as ``409 conflict`` so the UI can distinguish "you
    were too late" from "it never existed".
    """
    info = _resolve_owned_task(request, task_id, manager)
    cancelled = manager.cancel_task(task_id)
    if not cancelled:
        raise http_error(
            status_code=409,
            error="conflict",
            message=f"Task '{task_id}' has already finished ({info.state.value}).",
            hint="Only pending or running tasks can be cancelled.",
        )

    audit_log(
        "ingest_task_cancelled",
        detail=(
            f"client_ip={_client_ip(request)} "
            f"task_id_tail={mask_secret(task_id)} "
            f"state_before={info.state.value}"
        ),
    )
    return IngestCancelResponse(task_id=task_id, status="cancelling")
