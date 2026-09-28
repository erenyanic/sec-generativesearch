"""WebSocket endpoint for live ingestion progress.

The route is the only sanctioned forwarder of the per-task message
queue owned by :class:`~sec_generative_search.api.tasks.TaskManager`;
the manager keeps every wire-shape rule (key omission, masked tails)
single-sourced via :func:`TaskManager._push`.

Security model
--------------

CORS does **not** protect WebSocket upgrades — the same-origin policy
applies to the *browser's choice to initiate the upgrade*, not to the
upgrade traffic itself.  We treat ``Origin`` as the canonical
authorisation surface for the handshake and reject anything not on the
configured allow-list (``API_CORS_ORIGINS``).  Non-browser clients that
need to connect (CLI tools, server-side proxies) must therefore supply
an explicit allowed origin.

Auth ordering on the upgrade (when ``API_KEY`` is set):

1.  ``X-API-Key`` header on the upgrade request — the standard path.
2.  First-message fallback ``{"type": "auth", "api_key": "..."}`` —
    needed because browser ``WebSocket`` constructors cannot attach
    custom headers.  The window is bounded by ``_AUTH_TIMEOUT_SECONDS``;
    silence rounds to a 4001 close.

After the API-key gate, **session ownership is enforced** against the
server-minted ``session_id`` cookie via the same shape check used by
HTTP routes (:func:`is_valid_session_id_shape`).  A foreign-session task
surfaces with the same 4404 ``not_found`` envelope as a genuinely
missing one, so the route never confirms a task's existence to a
non-owner — the same information-leak discipline the ingest HTTP routes
already follow.

The route never reads the API key or cookie value into a log record;
the access-log layer already redacts both header families, and audit-log
lines carry only the masked task-id tail and the client IP.

Message protocol
----------------

Server → client (all framed as JSON text frames):

- ``snapshot``       — initial state on connect (also used to catch up
                       reconnecting clients).
- ``step``           — pipeline progress callback fired by the worker.
- ``filing_done``    — one successful filing ingest.
- ``filing_skipped`` — duplicate or late-duplicate.
- ``filing_failed`` — per-filing fetch / parse / store error
                       (worker continues).
- ``eviction``       — demo-mode FIFO eviction notice.
- ``heartbeat``      — emitted every ``_HEARTBEAT_SECONDS`` of inter-
                       event silence so a proxy idle-timeout does not
                       half-close the connection.
- ``completed`` / ``failed`` / ``cancelled`` — terminal frames; the
   route closes the WebSocket after sending one of these.
- ``error``          — pre-stream rejection envelope (auth / ownership);
                       always followed by a close frame.

Client → server: only the optional ``auth`` payload during the
handshake.  Everything else is dropped — the worker is the sole source
of events for an in-flight task and a client cannot mutate it through
this surface (cancellation goes through ``DELETE /api/ingest/tasks/{id}``).

Close codes
-----------

We use the application-layer range (4000-4999) per RFC 6455 §7.4.2:

- ``4400`` — invalid task-id shape (rejected pre-accept).
- ``4003`` — origin not on the allow-list.
- ``4001`` — unauthenticated (missing / invalid API key).
- ``4404`` — task not found, or owned by a different session.

Disconnect handling: the worker thread is decoupled from the consumer
via :class:`asyncio.Queue`; a client disconnect during streaming raises
:class:`WebSocketDisconnect`, the route releases its resources, and the
worker continues to completion.  Reconnecting picks up the current
state via the snapshot frame plus the queued events that have not yet
been drained — the most recent ``_MESSAGE_QUEUE_MAXSIZE`` of them, since
the per-task queue drops its oldest frame at capacity (the terminal
frame is always the newest, so it survives).
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import re

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sec_generative_search.api.dependencies import (
    SESSION_COOKIE_NAME,
    client_ip,
    is_valid_session_id_shape,
)
from sec_generative_search.api.tasks import TaskInfo, TaskManager, TaskState, new_message_queue
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.core.security import mask_secret

__all__ = ["router"]


logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Task ids are 32-char hex UUIDs (same shape as the HTTP ingest routes).
# Pre-accept rejection on the upgrade saves one round trip for obviously
# forged URLs and keeps audit-log noise off the ``accept`` path.
_TASK_ID_RE = re.compile(r"^[0-9a-f]{32}$")


_TERMINAL_TYPES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})
_TERMINAL_STATES: frozenset[TaskState] = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}
)


# Heartbeat cadence on inter-event silence.  Mirrors the SSE route's
# ``_SSE_HEARTBEAT_SECONDS`` — every operator wants the same semantics,
# so we deliberately do NOT expose this as a settings knob.
_HEARTBEAT_SECONDS = 15.0


# First-message auth timeout when the browser cannot attach the
# ``X-API-Key`` header.  Short enough that an unauthenticated probe
# cannot hold a slot open for long.
_AUTH_TIMEOUT_SECONDS = 5.0


# Application-layer close codes (RFC 6455 §7.4.2 private-use range).
_CLOSE_INVALID_TASK_ID = 4400
_CLOSE_ORIGIN_NOT_ALLOWED = 4003
_CLOSE_UNAUTHENTICATED = 4001
_CLOSE_NOT_FOUND = 4404


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.websocket("/ws/ingest/{task_id}")
async def ingest_progress(websocket: WebSocket, task_id: str) -> None:
    """Stream live ingestion progress for a task the caller owns.

    Lifecycle: validate task-id shape → origin allow-list → ``accept``
    → API-key handshake → session-ownership check → snapshot frame →
    queue forward / heartbeat loop until terminal or disconnect.

    Errors before ``accept`` close with a code in the 4000-4999 range and
    no body.  Errors after ``accept`` send an ``error`` frame first so a
    JS client can render a structured message before the close fires.
    """
    if not _TASK_ID_RE.match(task_id):
        await websocket.close(code=_CLOSE_INVALID_TASK_ID, reason="Invalid task id")
        return

    settings = get_settings()
    origin = websocket.headers.get("origin")
    if not _is_origin_allowed(origin, settings.api.cors_origins):
        await websocket.close(code=_CLOSE_ORIGIN_NOT_ALLOWED, reason="Origin not allowed")
        return

    await websocket.accept()

    if not await _authenticate_websocket(websocket):
        # ``_authenticate_websocket`` has already sent the close frame.
        return

    session_id = _extract_ws_session_id(websocket)
    manager: TaskManager = websocket.app.state.task_manager
    info = manager.get_task(task_id)

    if info is None or info.session_id != session_id:
        # Same shape as ``GET /api/ingest/tasks/{id}`` for a foreign or
        # missing task — never confirm a task's existence to a non-owner.
        await _send_safe(
            websocket,
            {
                "type": "error",
                "error": "not_found",
                "message": f"Task '{task_id}' not found.",
            },
        )
        await websocket.close(code=_CLOSE_NOT_FOUND, reason="Task not found")
        return

    audit_log(
        "ws_ingest_connected",
        client_ip=client_ip(websocket),
        endpoint=f"WS /ws/ingest/{mask_secret(task_id)}",
        detail=f"task_id_tail={mask_secret(task_id)} state={info.state.value}",
    )

    completion = "ok"
    try:
        await _send_safe(websocket, _build_snapshot(info))

        if info.state in _TERMINAL_STATES:
            # The task already finished; deliver the terminal frame
            # (preferring a queued one so the count fields match the
            # final ``filing_*`` events the worker pushed) and close.
            terminal_msg = _drain_terminal_message(info) or _build_terminal_from_state(info)
            await _send_safe(websocket, terminal_msg)
            return

        await _stream_loop(websocket, info)

    except WebSocketDisconnect:
        completion = "client_disconnected"
    except Exception:
        completion = "error"
        logger.exception("WebSocket error for task %s", mask_secret(task_id))
    finally:
        audit_log(
            "ws_ingest_disconnected",
            client_ip=client_ip(websocket),
            endpoint=f"WS /ws/ingest/{mask_secret(task_id)}",
            detail=f"task_id_tail={mask_secret(task_id)} completion={completion}",
        )
        # Best-effort close: the client may have torn the connection
        # down already, which surfaces as a RuntimeError/ConnectionError;
        # the audit-log line above is the canonical disconnect signal.
        with contextlib.suppress(Exception):
            await websocket.close()


# ---------------------------------------------------------------------------
# Origin allow-list
# ---------------------------------------------------------------------------


def _is_origin_allowed(origin: str | None, allowed: list[str]) -> bool:
    """Reject missing / unknown origins.

    A missing ``Origin`` header is treated identically to an unknown
    one: every modern browser sends ``Origin`` on a WebSocket upgrade,
    so the absence is a strong signal of a non-browser client that has
    not been provisioned with an allow-listed value.  Non-browser clients
    that legitimately need to connect must therefore send an explicit
    origin matching ``API_CORS_ORIGINS``.
    """
    if not origin:
        return False
    return origin in allowed


# ---------------------------------------------------------------------------
# Session-id extraction
# ---------------------------------------------------------------------------


def _extract_ws_session_id(websocket: WebSocket) -> str | None:
    """Return the validated ``session_id`` cookie value, or ``None``.

    Mirrors :func:`api.dependencies.extract_session_id` but reads from
    :attr:`WebSocket.cookies` instead of the HTTP ``Request``.  The
    shape check is the single seam — both call sites share it so the
    accepted alphabet / length stay in lockstep.
    """
    raw = websocket.cookies.get(SESSION_COOKIE_NAME)
    if not is_valid_session_id_shape(raw):
        return None
    return raw


# ---------------------------------------------------------------------------
# API-key handshake
# ---------------------------------------------------------------------------


async def _authenticate_websocket(websocket: WebSocket) -> bool:
    """Validate ``API_KEY`` via header (preferred) or first message.

    Returns ``True`` on success or when auth is disabled.  Returns
    ``False`` after sending a 4001 close frame; callers MUST NOT send
    any further frames in that case.

    The header path is preferred because every non-browser client can
    populate ``X-API-Key`` on the upgrade.  The browser fallback exists
    because the standard ``WebSocket`` constructor exposes no
    application-supplied header surface; we accept a single ``auth``
    message within ``_AUTH_TIMEOUT_SECONDS``.

    Header comparison uses :func:`hmac.compare_digest` to keep the
    timing characteristic the same as the HTTP auth path.
    """
    expected = get_settings().api.key
    if expected is None:
        return True

    header_key = websocket.headers.get("x-api-key")
    if header_key is not None and hmac.compare_digest(header_key, expected):
        return True

    try:
        message = await asyncio.wait_for(
            websocket.receive_json(),
            timeout=_AUTH_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        await _close_unauth(websocket, "Authentication timed out")
        return False
    except WebSocketDisconnect:
        return False
    except (ValueError, TypeError):
        # Malformed JSON / non-text frame surfaces here; collapse to the
        # same 4001 close as a missing key so the route never confirms
        # *why* the handshake failed (which would leak the auth model).
        await _close_unauth(websocket, "Invalid auth payload")
        return False

    if not isinstance(message, dict) or message.get("type") != "auth":
        await _close_unauth(websocket, "Invalid auth payload")
        return False

    provided = message.get("api_key")
    if not isinstance(provided, str) or not hmac.compare_digest(provided, expected):
        await _close_unauth(websocket, "Invalid or missing API key")
        return False

    return True


async def _close_unauth(websocket: WebSocket, reason: str) -> None:
    """Send a 4001 close frame, swallowing already-closed errors."""
    # Suppress: a peer that has already half-closed the channel raises
    # ``RuntimeError`` ("WebSocket is not connected") from Starlette;
    # there is nothing to do beyond returning to the caller.
    with contextlib.suppress(Exception):
        await websocket.close(code=_CLOSE_UNAUTHENTICATED, reason=reason)


# ---------------------------------------------------------------------------
# Snapshot + terminal-frame helpers
# ---------------------------------------------------------------------------


def _build_snapshot(info: TaskInfo) -> dict:
    """Render the current task state as a ``snapshot`` frame.

    Sent immediately on every connect so a reconnecting client can pick
    up where it left off without polling the HTTP status route.
    """
    return {
        "type": "snapshot",
        "task_id": info.task_id,
        "status": info.state.value,
        "progress": {
            "current_ticker": info.progress.current_ticker,
            "current_form_type": info.progress.current_form_type,
            "step_label": info.progress.step_label,
            "step_index": info.progress.step_index,
            "step_total": info.progress.step_total,
            "filings_done": info.progress.filings_done,
            "filings_total": info.progress.filings_total,
            "filings_skipped": info.progress.filings_skipped,
            "filings_failed": info.progress.filings_failed,
        },
        "results": [r.to_dict() for r in info.results],
    }


def _build_terminal_from_state(info: TaskInfo) -> dict:
    """Reconstruct the terminal frame from authoritative task state.

    Used as a fallback when the queued terminal message has already
    been consumed by a previous connection or has not yet arrived via
    ``call_soon_threadsafe``.  ``info.state`` is the source of truth.
    """
    if info.state == TaskState.COMPLETED:
        return {
            "type": "completed",
            "results": [r.to_dict() for r in info.results],
            "summary": {
                "total": (
                    len(info.results) + info.progress.filings_skipped + info.progress.filings_failed
                ),
                "succeeded": len(info.results),
                "skipped": info.progress.filings_skipped,
                "failed": info.progress.filings_failed,
            },
        }
    if info.state == TaskState.FAILED:
        return {
            "type": "failed",
            "error": info.error or "Unknown error",
            "details": None,
        }
    return {"type": "cancelled"}


def _drain_terminal_message(info: TaskInfo) -> dict | None:
    """Pop the terminal frame off the queue if one is still buffered.

    Non-terminal frames encountered along the way are dropped — the
    snapshot we just sent already covers the cumulative state.
    """
    queue = info._message_queue
    if queue is None:
        return None
    while not queue.empty():
        try:
            msg = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if msg.get("type") in _TERMINAL_TYPES:
            return msg
    return None


# ---------------------------------------------------------------------------
# Stream loop
# ---------------------------------------------------------------------------


async def _stream_loop(websocket: WebSocket, info: TaskInfo) -> None:
    """Forward queue events until a terminal frame fires.

    Idle timeout emits a ``heartbeat`` frame so proxy layers
    (nginx, Cloudflare) do not half-close the connection while a slow
    embed step runs.  If the task transitions to a terminal state while
    the queue is idle (the terminal message was already consumed by a
    prior connection), the loop synthesises a terminal frame from
    authoritative state and returns.
    """
    # ``TaskManager._push`` builds the queue lazily — when no message has
    # been pushed yet (a freshly created task waiting for the GPU slot),
    # we need a queue to ``get()`` from.  Constructing here is safe
    # because we are already on the async loop.
    if info._message_queue is None:
        info._message_queue = new_message_queue()

    queue = info._message_queue

    while True:
        try:
            message = await asyncio.wait_for(queue.get(), timeout=_HEARTBEAT_SECONDS)
        except TimeoutError:
            if info.state in _TERMINAL_STATES:
                # The terminal frame was already drained by a prior
                # connection — synthesise from state so the reconnecting
                # client still sees a clean close.
                await _send_safe(websocket, _build_terminal_from_state(info))
                return
            await _send_safe(websocket, {"type": "heartbeat"})
            continue

        await _send_safe(websocket, message)
        if message.get("type") in _TERMINAL_TYPES:
            return


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


async def _send_safe(websocket: WebSocket, payload: dict) -> None:
    """Send JSON; lets :class:`WebSocketDisconnect` propagate.

    A disconnected client raises here, which is exactly what the
    outer ``try`` is structured to catch.
    """
    await websocket.send_json(payload)
