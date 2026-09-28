"""Structured error envelope for the FastAPI surface.

Every route that needs to raise a controlled error MUST go through one
of the helpers in this module — never a bare ``HTTPException`` with an
ad-hoc dict.  The shape is fixed::

    {
        "error":   <stable machine-readable code, e.g. "rate_limited">,
        "message": <human-readable English sentence>,
        "details": <None or a small JSON-serialisable mapping>,
        "hint":    <None or a one-line operator hint>,
    }

Why the explicit envelope rather than FastAPI's default ``{"detail": ...}``?

    - The envelope is the public API contract that the frontend codes
      against; ``detail`` shape varies across FastAPI / Starlette
      releases and across validation vs handler-raised errors.
    - The fields are deliberately small and stable so log scrubbers can
      key on ``error`` without parsing prose.

Security rules:

    - ``message`` and ``hint`` MUST never echo a credential, an EDGAR
      identity, raw retrieved-context text, or any value taken straight
      from request headers without redaction.  Pass user-supplied
      identifiers through :func:`mask_secret` first.
    - ``details`` is a place for *small* structured context (e.g.
      ``{"limit": 60}`` for a rate-limit response).  Do not stuff
      free-form data here; that surface invariably grows leaks.

The exception handler installed by :func:`install_error_handlers` rewrites
both :class:`HTTPException` (whose ``detail`` may already be an
``ErrorEnvelope`` from one of the helpers) and validation failures
(``RequestValidationError``) into the same shape.  A bare ``Exception``
escaping a handler still surfaces as a 500 with a generic envelope —
the actual exception is logged but never returned in the body.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException

from sec_generative_search.core.logging import get_logger

__all__ = [
    "ErrorEnvelope",
    "database_error",
    "envelope",
    "http_error",
    "install_error_handlers",
]


logger = get_logger(__name__)


class ErrorEnvelope(BaseModel):
    """The single response shape for every error path."""

    model_config = ConfigDict(extra="forbid")

    error: str
    message: str
    details: dict[str, Any] | None = None
    hint: str | None = None


def envelope(
    *,
    error: str,
    message: str,
    details: dict[str, Any] | None = None,
    hint: str | None = None,
) -> dict[str, Any]:
    """Return a serialised :class:`ErrorEnvelope` mapping.

    Helpers and route code use this rather than constructing the dict
    inline so a future schema migration only touches one site.
    """
    return ErrorEnvelope(
        error=error,
        message=message,
        details=details,
        hint=hint,
    ).model_dump()


def http_error(
    *,
    status_code: int,
    error: str,
    message: str,
    details: dict[str, Any] | None = None,
    hint: str | None = None,
    headers: dict[str, str] | None = None,
) -> HTTPException:
    """Build an :class:`HTTPException` whose ``detail`` is an envelope.

    Routes raise the return value directly; the installed exception
    handler unwraps the envelope and emits it as the JSON body.
    """
    return HTTPException(
        status_code=status_code,
        detail=envelope(error=error, message=message, details=details, hint=hint),
        headers=headers,
    )


def database_error() -> HTTPException:
    """The one ``500 database_error`` envelope for a storage failure.

    Fixed text: never echo ``DatabaseError.details`` (SQLite / ChromaDB
    driver strings carry file paths and SQL fragments) — the caller logs
    it.  Shared by every route so the wording cannot drift (F33).
    """
    return http_error(
        status_code=500,
        error="database_error",
        message="Database operation failed. Check server logs.",
        hint="Check that the data directory is accessible and the database is intact.",
    )


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


async def _http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """Emit ``exc.detail`` as-is when it is already an envelope dict.

    A handler raised via :func:`http_error` carries an envelope dict in
    ``detail``.  A handler raised with a plain string still receives a
    well-formed envelope so the response shape is uniform.
    """
    detail = exc.detail
    if isinstance(detail, dict) and {"error", "message"} <= detail.keys():
        body = detail
    else:
        body = envelope(
            error="http_error",
            message=str(detail) if detail else "Request rejected.",
        )
    return JSONResponse(
        status_code=exc.status_code,
        content=body,
        headers=exc.headers,
    )


async def _validation_exception_handler(
    _: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Map FastAPI validation errors into the unified envelope."""
    # ``exc.errors()`` is already a list of small dicts safe to surface;
    # we trim the ``input`` field on each entry to avoid echoing arbitrary
    # request bodies back at the caller.  ``ctx`` may carry a raw
    # exception instance (e.g. ``ValueError`` from a custom validator)
    # which is not JSON-serialisable — coerce it to its string form so
    # the envelope still renders.
    errors = []
    for raw in exc.errors():
        cleaned = {k: v for k, v in raw.items() if k != "input"}
        ctx = cleaned.get("ctx")
        if isinstance(ctx, dict) and "error" in ctx:
            ctx_safe = dict(ctx)
            ctx_safe["error"] = str(ctx_safe["error"])
            cleaned["ctx"] = ctx_safe
        errors.append(cleaned)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=envelope(
            error="validation_failed",
            message="Request payload failed validation.",
            details={"errors": errors},
            hint="Check the request body shape and types against the API schema.",
        ),
    )


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: log the exception, return a generic 500 envelope.

    The exception text is intentionally NOT surfaced — internal error
    messages routinely include file paths, SQL fragments, and other
    artefacts unsuitable for a public response body.
    """
    logger.exception(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=envelope(
            error="internal_error",
            message="The server encountered an unexpected error.",
            hint="Retry the request; if the failure persists, contact the operator.",
        ),
    )


def install_error_handlers(app: FastAPI) -> None:
    """Register the unified envelope on FastAPI's exception machinery.

    Both the FastAPI ``HTTPException`` and Starlette's
    ``HTTPException`` (raised by routing for 404 / 405 on unmatched
    paths) flow through :func:`_http_exception_handler` so the response
    shape stays uniform regardless of which layer originated the error.
    """
    app.add_exception_handler(HTTPException, _http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
