"""``POST /api/admin/demo-reset`` — the scheduled demo-corpus reset, in-process.

Why this route exists (OPTIMIZATIONS F27): the public demo (Scenario C) keeps
SQLite + ChromaDB on a Cloud Storage FUSE volume, which "does not provide
concurrency control for multiple writes (file locking) to the same file".
The nightly reset used to be a Cloud Run *Job* running ``sec-rag manage
clear -y`` against that volume **while the API served from it** — two
writer processes on unlockable files.  The reset now runs inside the API,
the single writer, under :meth:`TaskManager.exclusive_maintenance` so no
ingest writes beside it.

Access model — a deliberate, reviewed exception to rule **A**
(``admin_route_dependencies``):

- The caller (Cloud Scheduler, over internal ingress) presents
  ``X-Demo-Reset-Token`` = ``API_DEMO_RESET_TOKEN``.  **Not** ``API_KEY``
  / ``API_ADMIN_KEY``: the API key alone can spend the admin-env provider
  keys, so the scheduler holds a secret that can do nothing but this.
- Missing token, wrong token and no configured token are indistinguishable
  (``401``); settings refuse a token outside ``API_DEMO_MODE``.
- Kept **off** the SPA admin proxy's allow-list — the browser cannot reach
  it even with an admin session.
- The token is never logged (the access-log layer suppresses the header)
  and never echoed.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response, Security
from fastapi.security import APIKeyHeader

from sec_generative_search.api.dependencies import client_ip, get_filing_store, get_task_manager
from sec_generative_search.api.errors import database_error, http_error
from sec_generative_search.api.schemas import DemoResetResponse
from sec_generative_search.api.tasks import IngestActiveError, TaskManager
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import DatabaseError
from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.core.security import secure_compare
from sec_generative_search.database import FilingStore

__all__ = ["DEMO_RESET_TOKEN_HEADER", "router", "verify_demo_reset_token"]

logger = get_logger(__name__)

DEMO_RESET_TOKEN_HEADER = "X-Demo-Reset-Token"
_ENDPOINT = "POST /api/admin/demo-reset"

_token_header = APIKeyHeader(name=DEMO_RESET_TOKEN_HEADER, auto_error=False)

router = APIRouter()


async def verify_demo_reset_token(
    request: Request,
    supplied: str | None = Security(_token_header),
) -> None:
    """Require ``X-Demo-Reset-Token`` to equal the configured token.

    ``401`` for a missing or wrong token **and** when no token is
    configured, so a probe learns nothing about the deployment.  Constant
    time (:func:`secure_compare`); denials are audit-logged with the peer
    address only.
    """
    configured = get_settings().api.demo_reset_token
    if configured is None or not secure_compare(supplied, configured):
        audit_log(
            "demo_reset_denied",
            client_ip=client_ip(request),
            endpoint=_ENDPOINT,
        )
        raise http_error(
            status_code=401,
            error="unauthorised",
            message="Invalid or missing demo-reset token.",
            hint=f"Provide the configured token via the {DEMO_RESET_TOKEN_HEADER} header.",
        )


@router.post(
    "/demo-reset",
    response_model=DemoResetResponse,
    summary="Reset the demo corpus (scheduler-triggered, token-gated)",
    dependencies=[Depends(verify_demo_reset_token)],
)
def demo_reset(
    request: Request,
    response: Response,
    store: FilingStore = Depends(get_filing_store),
    manager: TaskManager = Depends(get_task_manager),
) -> DemoResetResponse:
    """Clear every filing from both stores, with ingest held off.

    A sync ``def`` (F1): the clear is blocking storage work and runs in the
    threadpool.  Refuses with ``409`` while an ingest task is pending or
    running (the scheduler retries later) rather than clearing under it.
    """
    response.headers["Cache-Control"] = "no-store"
    try:
        with manager.exclusive_maintenance():
            chunks_removed, filings_removed = store.clear_all()
    except IngestActiveError as exc:
        raise http_error(
            status_code=409,
            error="ingest_in_progress",
            message="An ingest task is in progress; the demo was not reset.",
            details=exc.details,
            hint="Retry once the ingest finishes (configure a scheduler retry).",
        ) from exc
    except DatabaseError as exc:
        logger.error("demo reset failed: %s", exc.details)
        raise database_error() from exc

    audit_log(
        "demo_reset",
        client_ip=client_ip(request),
        endpoint=_ENDPOINT,
        detail=f"filings={filings_removed} chunks={chunks_removed}",
    )
    return DemoResetResponse(
        reset=True,
        filings_removed=filings_removed,
        chunks_removed=chunks_removed,
    )
