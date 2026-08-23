"""Background task manager for the ingestion pipeline.

The earlier task runner shipped as a stand-alone in-memory module that
wrote to `ChromaDBClient` and `MetadataRegistry` directly. This module
is the rewrite against the current package surfaces, with three
load-bearing differences.

1. **All dual-store writes go through :class:`FilingStore`.** ChromaDB-first
    ordering, best-effort rollbacks, and the atomic ``register_if_new=True``
    path are the contract on that single seam. The worker never calls
    ``self._chroma.store_filing`` /
   ``registry.register_filing_if_new`` /
   ``self._chroma.delete_filing`` / ``registry.remove_filing`` directly
   — those routes are bug magnets for orphan ChromaDB chunks because
   Chroma's ``add()`` silently no-ops duplicate IDs.

2. **Zero-key contract on :class:`TaskInfo`**
   Ingestion runs on the admin-env embedder; there is *no* provider key
   in flight, so :class:`TaskInfo` carries no credential-shaped attribute name.
   The EDGAR identity (PII, not a credential) is also kept off
   :class:`TaskInfo` — the per-task identity resolver lives on the
   manager in :attr:`TaskManager._task_resolvers`, a parallel dict
   cleared in the worker's ``finally`` block. A
   ``@pytest.mark.security`` test mirrors ``tests/core/test_types.py``
   to assert no key-shaped name on :class:`TaskInfo`.

3. **Session-scoped ownership**
   :class:`TaskInfo` carries the server-minted ``session_id`` from the
   cookie at create time so the route layer can isolate ``GET /api/ingest/tasks`` /
   ``GET /api/ingest/tasks/{id}`` / ``DELETE /api/ingest/tasks/{id}``
   per tenant.  The manager itself does *not* gate on session — it
   exposes :meth:`list_tasks_for_session` as a convenience and lets the
   route choose whether to surface a foreign task as ``404`` or hide it
   entirely.

4. **No recurring cleanup timer**
   Terminal-state transitions persist the row to ``task_history``
   immediately, and lazy eviction sweeps memory on every
   :meth:`create_task` / :meth:`get_task` / :meth:`list_tasks` call.
    Mirrors the in-memory credential / EDGAR identity stores and the
    embedding-provider idle unload — every other in-memory store in
    this codebase uses caller-driven eviction; one stray background
    timer makes interpreter shutdown ordering noisy and offers no
    measurable win.

5. **Per-filing failure isolation.**
    Fetch / process / store stages each catch the project's
    :class:`SECGenerativeSearchError` family **and** a broader
    :class:`Exception` net so a single malformed filing emits
    ``filing_failed`` and the worker continues with the next item. The
    bare-exception net logs ``logger.exception`` with the full traceback
    for the operator and pushes a generic ``"<stage> failed"`` envelope
    onto the WebSocket so internal class names / file paths never reach
    the wire. ``cancel_event`` is checked between every stage (top of
    loop, after fetch, between processing and storing, and inside the
    orchestrator via the progress callback) so a long-running fetch /
    embed step cannot defer cancellation.

6. **Caller-driven embedder idle-unload.**
    The manager optionally holds a reference to the bound embedder
    (passed by the lifespan; absent in unit tests) and fires
    :meth:`maybe_unload` at task lifecycle boundaries — in the worker's
    ``finally`` block immediately after the GPU semaphore release, and
    on every lazy-eviction sweep so a steady-state operator UI poll
    triggers VRAM release after the configured idle window.  Hosted
    embedders have no idle state to release; the helper duck-types on
    the public ``maybe_unload`` method so an OpenAI-only deployment

    never imports :mod:`torch`.

The module intentionally does **not** ship the API routes / WebSocket /
cooldown plumbing. This file is the worker substrate they wire onto.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.correlation import bind_correlation_id, get_correlation_id
from sec_generative_search.core.exceptions import (
    DatabaseError,
    FetchError,
    FilingLimitExceededError,
    SECGenerativeSearchError,
)
from sec_generative_search.core.logging import (
    get_logger,
    redact_all_for_log,
    redact_for_log,
)
from sec_generative_search.core.metrics import get_metrics
from sec_generative_search.pipeline.fetch import FilingFetcher, FilingInfo
from sec_generative_search.pipeline.orchestrator import PipelineOrchestrator

if TYPE_CHECKING:
    from sec_generative_search.core.edgar_identity import EdgarIdentity
    from sec_generative_search.database.metadata import MetadataRegistry
    from sec_generative_search.database.store import FilingStore
    from sec_generative_search.providers.base import BaseEmbeddingProvider

__all__ = [
    "FilingResult",
    "TaskInfo",
    "TaskManager",
    "TaskProgress",
    "TaskQueueFullError",
    "TaskState",
    "run_retention_eviction_safe",
]


logger = get_logger(__name__)


# In-memory TTL for terminal tasks. Rows are persisted to ``task_history``
# at the terminal-state transition; the in-memory entry is kept around
# for a day so a polling client can still see the result without a SQLite
# round-trip. Lazy eviction only — no background thread.
_TASK_TTL_SECONDS = 86_400  # 24 hours


# Type alias for the per-task EDGAR identity resolver. Returning ``None``
# means "use the admin-env fallback in the fetcher". The route layer
# fills this in at create time; tests pass ``None`` to exercise the
# admin-env path.
EdgarIdentityResolver = Callable[[], "EdgarIdentity | None"]


# ---------------------------------------------------------------------------
# Retention eviction helper
# ---------------------------------------------------------------------------


def run_retention_eviction_safe(
    filing_store: FilingStore,
    max_age_days: int,
    *,
    context_label: str,
) -> None:
    """Run a best-effort time-based eviction sweep.

    Called from two seams: the API lifespan startup (one-shot on boot)
    and the worker's post-successful-ingest hook (per completed task).
    Disabled at ``max_age_days <= 0`` — the operator-facing toggle is
    ``DB_RETENTION_MAX_AGE_DAYS=0``.

    The sweep is defence-in-depth, never load-bearing — operator cron
    and ``sec-rag manage evict`` remain the canonical paths for
    scheduled / on-demand eviction.  Any :class:`DatabaseError` or
    :class:`ValueError` raised by the store is logged at ``warning``
    and swallowed so a transient backend hiccup does not crash startup
    or shadow the worker's user-visible terminal-state transition.

    Args:
        filing_store: The dual-store seam to delegate to.
        max_age_days: Cutoff age in days; ``<= 0`` short-circuits to a
            no-op.  Positive values are forwarded verbatim to
            :meth:`FilingStore.evict_expired`.
        context_label: Tag distinguishing the two call sites
            (``"startup"`` / ``"post_ingest"``) in log lines.  Audit-log
            discipline forbids tickers / accession numbers here; only
            the count and the cutoff are emitted.
    """
    if max_age_days <= 0:
        return

    try:
        report = filing_store.evict_expired(max_age_days)
    except (DatabaseError, ValueError) as exc:
        message = getattr(exc, "message", str(exc))
        logger.warning(
            "Retention sweep (%s) failed; continuing best-effort: %s",
            context_label,
            message,
        )
        return

    if report.filings_evicted > 0:
        logger.info(
            "Retention sweep (%s): evicted %d filing(s) (%d chunk(s)) older than %d day(s)",
            context_label,
            report.filings_evicted,
            report.chunks_evicted,
            max_age_days,
        )


# ---------------------------------------------------------------------------
# Task state
# ---------------------------------------------------------------------------


class TaskState(StrEnum):
    """Lifecycle states for an ingestion task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_STATES: frozenset[TaskState] = frozenset(
    {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}
)


@dataclass
class TaskProgress:
    """Mutable progress snapshot updated by the worker thread.

    Access to individual scalar fields is inherently thread-safe in
    CPython (GIL); we never mutate the structure outside the worker
    thread that owns the task, so no lock is needed at this layer.
    """

    current_ticker: str | None = None
    current_form_type: str | None = None
    step_label: str = ""
    step_index: int = 0
    step_total: int = 5
    filings_done: int = 0
    filings_total: int = 0
    filings_skipped: int = 0
    filings_failed: int = 0


@dataclass
class FilingResult:
    """Per-filing outcome stored after a successful ingest.

    The two serialisation helpers project onto different consumers:
    :meth:`to_dict` is the wire shape used by the WebSocket
    ``filing_done`` event (and, later, the ``GET /api/ingest/tasks/{id}``
    response); :meth:`to_history_dict` is the SQLite ``task_history``
    shape, which keeps the explicit field names for forward compatibility.
    """

    ticker: str
    form_type: str
    filing_date: str
    accession_number: str
    segment_count: int
    chunk_count: int
    duration_seconds: float

    def to_dict(self) -> dict:
        """Serialise to a dict for WebSocket messages."""
        return {
            "ticker": self.ticker,
            "form_type": self.form_type,
            "filing_date": self.filing_date,
            "accession_number": self.accession_number,
            "segments": self.segment_count,
            "chunks": self.chunk_count,
            "time": round(self.duration_seconds, 1),
        }

    def to_history_dict(self) -> dict:
        """Serialise to a dict for task-history persistence."""
        return {
            "ticker": self.ticker,
            "form_type": self.form_type,
            "filing_date": self.filing_date,
            "accession_number": self.accession_number,
            "segment_count": self.segment_count,
            "chunk_count": self.chunk_count,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class TaskInfo:
    """Full state for a single ingestion task.

    Mutated by the worker thread; read by route handlers and the
    WebSocket forwarder. The dataclass deliberately omits any
    credential-shaped attribute name and any EDGAR identity field — the
    per-task EDGAR identity resolver lives on the manager in
    :attr:`TaskManager._task_resolvers` and is cleared in the worker's
    ``finally`` block.

    ``session_id`` carries the server-minted ownership key so routes can
    filter tasks per tenant. The ``_message_queue`` field carries
    WebSocket messages from the worker thread to the async consumer.
    """

    task_id: str
    tickers: list[str]
    form_types: list[str]
    count_mode: str = "latest"
    count: int | None = None
    year: int | None = None
    start_date: str | None = None
    end_date: str | None = None

    # Server-minted session_id from the cookie. ``None`` when no session
    # is available — the route layer collapses ownership checks in that
    # case.
    session_id: str | None = None

    state: TaskState = TaskState.PENDING
    progress: TaskProgress = field(default_factory=TaskProgress)
    results: list[FilingResult] = field(default_factory=list)
    error: str | None = None

    cancel_event: threading.Event = field(default_factory=threading.Event)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # GPU time-limit timer; cancelled on terminal-state transition.
    # Underscored so a future field walker (eg. JSON serialiser) can
    # filter privately-prefixed attributes out by convention.
    _duration_timer: threading.Timer | None = field(default=None, repr=False)

    # Accession numbers stored in the *current* task — used for partial
    # rollback on cancellation. Cleared after a successful rollback so
    # repeated cancels do not redo a no-op delete pass.
    _stored_accessions: list[str] = field(default_factory=list)

    # WebSocket message queue. Worker thread pushes typed dicts via
    # ``call_soon_threadsafe`` when an event loop is bound (the
    # production path); falls back to a direct ``put_nowait`` when not
    # (the unit-test path). The queue is built lazily so creating a
    # task outside an async context does not bind it to a closed loop.
    _message_queue: asyncio.Queue | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TaskQueueFullError(SECGenerativeSearchError):
    """Raised when the active task queue is at capacity.

    Inherits from :class:`SECGenerativeSearchError` so a route handler
    can catch the base exception and surface the unified envelope; the
    constructor populates ``details`` with the active / max counts so a
    UI can render an informative message without re-parsing the
    ``message`` field.
    """

    def __init__(self, active: int, maximum: int) -> None:
        super().__init__(
            f"Task queue is full ({active}/{maximum} active).",
            details={"active": active, "maximum": maximum},
        )


class _CancelledError(Exception):
    """Internal sentinel raised from a progress callback to abort the pipeline.

    Caught only inside :meth:`TaskManager._execute` and translated into
    the ``CANCELLED`` terminal state. Not exported because callers
    should signal cancellation through ``info.cancel_event``, never by
    raising this directly.
    """


# ---------------------------------------------------------------------------
# Task manager
# ---------------------------------------------------------------------------


class TaskManager:
    """In-memory manager for background ingestion tasks.

    The manager is a process-local singleton attached to ``app.state``
    by the lifespan in :mod:`api.app`. Its public surface is:

        - :meth:`create_task` — enqueue a new ingest; spawns a daemon
          worker thread immediately. Returns the task id.
        - :meth:`get_task` / :meth:`list_tasks` / :meth:`list_tasks_for_session`
          — read-side queries; cheap, lazy-evict stale terminal tasks
          on entry.
        - :meth:`cancel_task` — sets the task's ``cancel_event``; the
          worker observes it between pipeline steps and rolls back any
          ChromaDB writes through :meth:`FilingStore.delete_filings_batch`.
        - :meth:`has_active_task` — for the ``/api/status`` endpoint.
        - :meth:`set_event_loop` — late-bound during lifespan so
          ``_push`` can bridge the sync worker thread → async queue.
        - :meth:`shutdown` — drains any duration timers and prevents
          new tasks; called from the lifespan teardown.

    Construction takes a :class:`FilingStore` (the dual-store seam) plus
    a :class:`MetadataRegistry` (for batch duplicate checks, FIFO
    eviction reads, and ``task_history`` persistence). The registry
    *must not* be used for filing writes from this module — those go
    through ``filing_store``. The split keeps the write path isolated
    while letting the worker do read-side queries
    (``count``, ``list_oldest_filings``, ``get_existing_accessions``)
    that don't belong on :class:`FilingStore`.
    """

    def __init__(
        self,
        *,
        filing_store: FilingStore,
        registry: MetadataRegistry,
        fetcher: FilingFetcher,
        orchestrator: PipelineOrchestrator,
        embedder: BaseEmbeddingProvider | None = None,
    ) -> None:
        self._filing_store = filing_store
        self._registry = registry
        self._fetcher = fetcher
        self._orchestrator = orchestrator
        # Optional embedder reference. Only used to fire the idle-unload
        # hook on local providers — hosted embedders have no idle state
        # to release. ``None`` is the documented test path; production
        # wiring lives in the app lifespan.
        self._embedder = embedder

        self._tasks: dict[str, TaskInfo] = {}
        # Per-task EDGAR identity resolver, kept off ``TaskInfo`` so
        # downstream JSON serialisers cannot accidentally splat identity
        # onto a wire payload.
        self._task_resolvers: dict[str, EdgarIdentityResolver] = {}

        # ``edgar.set_identity`` mutates process-global state, so every
        # EDGAR-bound operation re-applies the intended identity under
        # this lock. Two concurrent ingest tasks would otherwise race
        # on whose identity wins.
        self._edgar_lock = threading.Lock()
        # GPU semaphore: one task at a time, FIFO queueing. Captured in
        # a sentinel attribute name so a future ``threading.BoundedSemaphore``
        # swap does not change every call site.
        self._gpu_semaphore = threading.Semaphore(1)
        # Guards ``_tasks`` / ``_task_resolvers`` dict mutations.
        self._lock = threading.Lock()
        # Late-bound event loop reference for ``_push``.
        self._loop: asyncio.AbstractEventLoop | None = None
        # Set during ``shutdown``; observed by ``create_task`` so a
        # late-arriving request after lifespan teardown is refused.
        self._shutdown_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_task(
        self,
        *,
        tickers: list[str],
        form_types: list[str],
        count_mode: str = "latest",
        count: int | None = None,
        year: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        session_id: str | None = None,
        edgar_identity_resolver: EdgarIdentityResolver | None = None,
    ) -> str:
        """Enqueue a new ingestion task and start its worker thread.

        ``session_id`` carries the server-minted cookie value when one
        is available; the route layer uses it later for per-tenant
        ownership filtering. Passing ``None`` is the documented
        Scenario-A behaviour (no API_KEY, single tenant).

        ``edgar_identity_resolver`` is called by the worker before each
        EDGAR-bound operation. The callable is held off
        :class:`TaskInfo` in :attr:`TaskManager._task_resolvers` to keep
        the zero-key / no-PII contract on the dataclass; the manager
        clears the entry in the worker's ``finally`` block. Passing
        ``None`` resolves to the fetcher's admin-env default.

        Raises:
            TaskQueueFullError: When the active queue is at capacity.
        """
        if self._shutdown_event.is_set():
            raise TaskQueueFullError(active=0, maximum=0)

        # Lazy eviction first so a long-quiescent server does not refuse
        # a new task on the back of stale terminal entries.
        self._evict_stale_locked()

        max_active = get_settings().api.max_task_queue_size
        with self._lock:
            active_count = sum(
                1 for t in self._tasks.values() if t.state in (TaskState.PENDING, TaskState.RUNNING)
            )
            if active_count >= max_active:
                raise TaskQueueFullError(active=active_count, maximum=max_active)

        task_id = uuid.uuid4().hex
        info = TaskInfo(
            task_id=task_id,
            tickers=tickers,
            form_types=form_types,
            count_mode=count_mode,
            count=count,
            year=year,
            start_date=start_date,
            end_date=end_date,
            session_id=session_id,
        )

        with self._lock:
            self._tasks[task_id] = info
            if edgar_identity_resolver is not None:
                self._task_resolvers[task_id] = edgar_identity_resolver

        # Capture the originating request's correlation ID so the worker
        # thread — which does not inherit the request's ContextVar — can
        # re-bind it and stitch its log records back to the request that
        # enqueued the task. ``None`` when enqueued outside a request
        # scope (e.g. a unit test).
        correlation_id = get_correlation_id()

        thread = threading.Thread(
            target=self._run_task,
            args=(info, correlation_id),
            name=f"ingest-{task_id[:8]}",
            daemon=True,
        )
        thread.start()

        logger.info(
            "Created task %s: tickers=%s, forms=%s, mode=%s",
            task_id[:8],
            redact_all_for_log(tickers),
            form_types,
            count_mode,
        )
        return task_id

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Return the in-memory or persisted task, or ``None``.

        Falls back to :meth:`MetadataRegistry.get_task_history` for tasks
        that have already been evicted from memory. The reconstructed
        :class:`TaskInfo` is read-only by convention — the persisted
        shape has no live ``cancel_event`` / ``_message_queue``, so
        mutating it would raise downstream.
        """
        self._evict_stale_locked()
        info = self._tasks.get(task_id)
        if info is not None:
            return info

        try:
            history = self._registry.get_task_history(task_id)
        except Exception:
            logger.debug("Task history lookup failed for %s", task_id[:8])
            return None

        if history is None:
            return None

        return self._reconstruct_task_info(history)

    def list_tasks(self) -> list[TaskInfo]:
        """Return every in-memory task (snapshot)."""
        self._evict_stale_locked()
        with self._lock:
            return list(self._tasks.values())

    def list_tasks_for_session(self, session_id: str | None) -> list[TaskInfo]:
        """Return tasks owned by ``session_id``.

        ``session_id=None`` returns tasks created without a session
        cookie (the Scenario-A single-tenant case). Filtering happens
        here rather than at the route so the manager owns the
        ownership-key shape — a future change (eg. ``user_id`` post-SSO)
        is a one-method swap.
        """
        self._evict_stale_locked()
        with self._lock:
            return [t for t in self._tasks.values() if t.session_id == session_id]

    def cancel_task(self, task_id: str) -> bool:
        """Request cancellation of a pending or running task.

        Sets ``cancel_event``; the worker checks it between pipeline
        steps and triggers a ChromaDB-first rollback through
        :meth:`FilingStore.delete_filings_batch`. Returns ``True`` when
        the signal was sent, ``False`` for unknown / terminal tasks.
        Cancelling a task that has not yet acquired the GPU semaphore
        still works — the worker observes the event before doing any
        SEC fetch.
        """
        with self._lock:
            info = self._tasks.get(task_id)
        if info is None:
            return False
        if info.state in _TERMINAL_STATES:
            return False
        info.cancel_event.set()
        logger.info("Cancel requested for task %s", task_id[:8])
        return True

    def has_active_task(self) -> bool:
        """``True`` if any task is pending or running."""
        with self._lock:
            return any(
                t.state in (TaskState.PENDING, TaskState.RUNNING) for t in self._tasks.values()
            )

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the running asyncio loop for cross-thread message pushes."""
        self._loop = loop

    def shutdown(self) -> None:
        """Block new tasks and cancel any in-flight duration timers.

        Existing worker threads run to completion — they are daemon
        threads, so the interpreter exit cleans them up if a hard
        shutdown is required. Cancelling a long-running embed call
        cleanly is out of scope here; the route layer covers user
        cancellation via :meth:`cancel_task`.
        """
        self._shutdown_event.set()
        with self._lock:
            for info in self._tasks.values():
                timer = info._duration_timer
                if timer is not None:
                    timer.cancel()
                    info._duration_timer = None
        logger.info("TaskManager shut down")

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run_task(self, info: TaskInfo, correlation_id: str | None = None) -> None:
        """Top-level worker entry. Acquires the GPU slot then runs ``_execute``.

        Wraps every error path with the terminal-state persistence
        hook so a worker crash still leaves ``task_history`` with the
        right ``status``. The duration timer is cancelled in
        ``finally`` so a worker that finishes early does not page the
        operator with a spurious auto-cancel a few minutes later.

        ``correlation_id`` is the ID captured from the enqueuing request;
        it is re-bound for the lifetime of the worker so every record
        the worker emits carries the originating request's ID.
        """
        with bind_correlation_id(correlation_id):
            self._run_task_body(info)

    def _run_task_body(self, info: TaskInfo) -> None:
        logger.info("Task %s waiting for GPU slot...", info.task_id[:8])
        self._gpu_semaphore.acquire()

        try:
            if info.cancel_event.is_set():
                self._mark_terminal(info, TaskState.CANCELLED)
                self._push(info, {"type": "cancelled"})
                return

            info.state = TaskState.RUNNING
            info.started_at = datetime.now(UTC)

            max_minutes = get_settings().api.max_task_duration_minutes
            if max_minutes > 0:
                timer = threading.Timer(
                    max_minutes * 60,
                    self._timeout_task,
                    args=(info,),
                )
                timer.daemon = True
                timer.start()
                info._duration_timer = timer

            self._execute(info)

        except Exception as exc:
            info.error = str(exc)
            self._mark_terminal(info, TaskState.FAILED)
            self._push(
                info,
                {
                    "type": "failed",
                    "error": str(exc),
                    "details": None,
                },
            )
            logger.exception("Task %s failed unexpectedly", info.task_id[:8])
        finally:
            if info._duration_timer is not None:
                info._duration_timer.cancel()
                info._duration_timer = None
            with self._lock:
                self._task_resolvers.pop(info.task_id, None)
            self._gpu_semaphore.release()
            # Fire idle-unload after releasing the GPU gate so a queued
            # task can grab the slot without waiting on cleanup;
            # ``maybe_unload`` is a cheap threshold check when the model
            # is still warm, so the lifecycle boundary is the right seam.
            self._maybe_unload_embedder()

    def _execute(self, info: TaskInfo) -> None:
        """Ingest one filing at a time, fetching HTML on demand.

        Fetching HTML per filing (rather than batching) keeps memory
        bounded at one filing's worth of HTML regardless of work-list
        size; on a 4 GB-VRAM single-user setup the alternative would
        OOM on a wide ticker selection.
        """
        # Build the work list under the correct effective identity.
        work = self._run_with_edgar_identity(info, self._build_work_list, info)
        info.progress.filings_total = len(work)

        # Batch dup check — one SQL ``IN (?, ?, …)`` instead of N
        # round-trips through ``is_duplicate``.
        all_accessions = [fi.accession_number for fi in work]
        existing = self._registry.get_existing_accessions(all_accessions)

        settings = get_settings()
        if settings.api.demo_mode:
            new_count = sum(1 for fi in work if fi.accession_number not in existing)
            self._maybe_evict(info, new_count)

        cached_count = self._registry.count()
        max_filings = settings.database.max_filings

        for filing_info in work:
            filing_id = filing_info.to_identifier()

            if info.cancel_event.is_set():
                self._rollback(info)
                self._mark_terminal(info, TaskState.CANCELLED)
                self._push(info, {"type": "cancelled"})
                logger.info("Task %s cancelled", info.task_id[:8])
                return

            ticker = filing_id.ticker
            form_type = filing_id.form_type

            info.progress.current_ticker = ticker
            info.progress.current_form_type = form_type
            info.progress.step_label = "Checking duplicate"
            info.progress.step_index = 1

            if filing_id.accession_number in existing:
                info.progress.filings_skipped += 1
                info.progress.filings_done += 1
                self._push(
                    info,
                    {
                        "type": "filing_skipped",
                        "ticker": ticker,
                        "form_type": form_type,
                        "accession_number": filing_id.accession_number,
                        "reason": "duplicate",
                    },
                )
                logger.info(
                    "Task %s: skipped duplicate %s",
                    info.task_id[:8],
                    filing_id.accession_number,
                )
                continue

            # Filing-count ceiling. Demo mode tries one more eviction
            # round before failing so a steady-state demo never blows
            # the limit; non-demo deployments rely on operator policy
            # and surface ``FilingLimitExceededError`` instead.
            if cached_count >= max_filings:
                if settings.api.demo_mode:
                    self._maybe_evict(info, 1)
                    cached_count = self._registry.count()
                    if cached_count >= max_filings:
                        self._fail_with_limit(info, cached_count, max_filings)
                        return
                else:
                    self._fail_with_limit(info, cached_count, max_filings)
                    return

            info.progress.step_label = "Fetching"
            info.progress.step_index = 0

            try:
                _, html_content = self._run_with_edgar_identity(
                    info,
                    self._fetcher.fetch_filing_content,
                    filing_info,
                )
            except FetchError as exc:
                info.progress.filings_failed += 1
                info.progress.filings_done += 1
                self._push(
                    info,
                    {
                        "type": "filing_failed",
                        "ticker": ticker,
                        "form_type": form_type,
                        "accession_number": filing_id.accession_number,
                        "error": exc.message,
                    },
                )
                logger.warning(
                    "Task %s: fetch failed for %s — %s",
                    info.task_id[:8],
                    filing_id.accession_number,
                    exc.message,
                )
                continue
            except Exception as exc:
                # Defence-in-depth: edgartools occasionally surfaces a
                # bare RuntimeError / ValueError on EDGAR responses that
                # do not match its parsed shape (truncated HTML, server
                # rendering glitches, undocumented form variants).
                # Per-filing failure isolation keeps one malformed
                # response from collapsing the whole task.
                info.progress.filings_failed += 1
                info.progress.filings_done += 1
                self._push(
                    info,
                    {
                        "type": "filing_failed",
                        "ticker": ticker,
                        "form_type": form_type,
                        "accession_number": filing_id.accession_number,
                        "error": "fetch failed",
                    },
                )
                logger.exception(
                    "Task %s: unexpected fetch failure for %s — %s",
                    info.task_id[:8],
                    filing_id.accession_number,
                    type(exc).__name__,
                )
                continue

            # Stage boundary cancel check: fetch can run for many
            # seconds, so checking before kicking off parse / chunk /
            # embed avoids burning work that will be rolled back. The
            # progress callback covers the later stages.
            if info.cancel_event.is_set():
                self._rollback(info)
                self._mark_terminal(info, TaskState.CANCELLED)
                self._push(info, {"type": "cancelled"})
                logger.info("Task %s cancelled after fetch", info.task_id[:8])
                return

            def _progress_cb(
                step: str,
                current: int,
                total: int,
                _self: TaskManager = self,
                _info: TaskInfo = info,
                _ticker: str = ticker,
                _form: str = form_type,
            ) -> None:
                """Pipeline progress callback — feeds task state + WebSocket.

                Default-argument capture (``_info``/``_ticker``/``_form``)
                is the closure-friendly idiom for late-binding the
                current loop iteration's values without leaking them
                into the outer scope.
                """
                _info.progress.current_ticker = _ticker
                _info.progress.current_form_type = _form
                _info.progress.step_label = step
                _info.progress.step_index = current
                _info.progress.step_total = 5

                _self._push(
                    _info,
                    {
                        "type": "step",
                        "ticker": _ticker,
                        "form_type": _form,
                        "step": step,
                        "step_number": current,
                        "total_steps": 5,
                    },
                )

                if _info.cancel_event.is_set():
                    raise _CancelledError

            info.progress.step_label = "Processing"
            info.progress.step_index = 1

            try:
                result = self._orchestrator.process_filing(
                    filing_id,
                    html_content,
                    progress_callback=_progress_cb,
                )
            except _CancelledError:
                self._rollback(info)
                self._mark_terminal(info, TaskState.CANCELLED)
                self._push(info, {"type": "cancelled"})
                logger.info("Task %s cancelled during processing", info.task_id[:8])
                return
            except SECGenerativeSearchError as exc:
                info.progress.filings_failed += 1
                info.progress.filings_done += 1
                self._push(
                    info,
                    {
                        "type": "filing_failed",
                        "ticker": ticker,
                        "form_type": form_type,
                        "accession_number": filing_id.accession_number,
                        "error": exc.message,
                    },
                )
                logger.warning(
                    "Task %s: processing failed for %s — %s",
                    info.task_id[:8],
                    filing_id.accession_number,
                    exc.message,
                )
                continue
            except Exception as exc:
                # Defence-in-depth: parse / chunk / embed wrap their
                # known failure modes in :class:`SECGenerativeSearchError`,
                # but a malformed filing can still surface a bare
                # ``KeyError`` / ``ValueError`` from the doc2dict tree
                # walker on undocumented HTML shapes. Per-filing failure
                # isolation means one bad filing must not collapse the
                # whole task. ``logger.exception`` records the full
                # traceback for the operator; the WebSocket envelope
                # carries a generic ``"processing failed"`` to avoid
                # leaking implementation detail onto the wire.
                info.progress.filings_failed += 1
                info.progress.filings_done += 1
                self._push(
                    info,
                    {
                        "type": "filing_failed",
                        "ticker": ticker,
                        "form_type": form_type,
                        "accession_number": filing_id.accession_number,
                        "error": "processing failed",
                    },
                )
                logger.exception(
                    "Task %s: unexpected processing failure for %s — %s",
                    info.task_id[:8],
                    filing_id.accession_number,
                    type(exc).__name__,
                )
                continue

            info.progress.step_label = "Storing"
            info.progress.step_index = 4

            if info.cancel_event.is_set():
                self._rollback(info)
                self._mark_terminal(info, TaskState.CANCELLED)
                self._push(info, {"type": "cancelled"})
                return

            try:
                # ``register_if_new=True`` is the atomic check-and-claim
                # path: SQLite-first, then ChromaDB. A losing caller
                # sees ``False`` and we treat it as a late duplicate —
                # exactly the race window the carry-over path falls
                # into when two workers run concurrently. See
                # ``FilingStore`` module docstring for the full
                # rationale.
                registered = self._filing_store.store_filing(
                    result,
                    register_if_new=True,
                )
                if not registered:
                    info.progress.filings_skipped += 1
                    info.progress.filings_done += 1
                    self._push(
                        info,
                        {
                            "type": "filing_skipped",
                            "ticker": ticker,
                            "form_type": form_type,
                            "accession_number": filing_id.accession_number,
                            "reason": "duplicate",
                        },
                    )
                    logger.info(
                        "Task %s: skipped late duplicate %s",
                        info.task_id[:8],
                        filing_id.accession_number,
                    )
                    continue
            except DatabaseError as exc:
                info.progress.filings_failed += 1
                info.progress.filings_done += 1
                self._push(
                    info,
                    {
                        "type": "filing_failed",
                        "ticker": ticker,
                        "form_type": form_type,
                        "accession_number": filing_id.accession_number,
                        "error": exc.message,
                    },
                )
                logger.warning(
                    "Task %s: storage failed for %s — %s",
                    info.task_id[:8],
                    filing_id.accession_number,
                    exc.message,
                )
                continue
            except Exception as exc:
                # Defence-in-depth: ChromaDB / SQLite normally surface
                # backend issues as :class:`DatabaseError`, but a
                # corrupted index or transient driver glitch can leak a
                # bare exception. One store failure must not collapse
                # the task.
                info.progress.filings_failed += 1
                info.progress.filings_done += 1
                self._push(
                    info,
                    {
                        "type": "filing_failed",
                        "ticker": ticker,
                        "form_type": form_type,
                        "accession_number": filing_id.accession_number,
                        "error": "storage failed",
                    },
                )
                logger.exception(
                    "Task %s: unexpected storage failure for %s — %s",
                    info.task_id[:8],
                    filing_id.accession_number,
                    type(exc).__name__,
                )
                continue

            cached_count += 1
            info._stored_accessions.append(filing_id.accession_number)
            info.results.append(
                FilingResult(
                    ticker=filing_id.ticker,
                    form_type=filing_id.form_type,
                    filing_date=filing_id.date_str,
                    accession_number=filing_id.accession_number,
                    segment_count=result.ingest_result.segment_count,
                    chunk_count=result.ingest_result.chunk_count,
                    duration_seconds=result.ingest_result.duration_seconds,
                )
            )
            info.progress.filings_done += 1

            # Ingestion latency histogram. The orchestrator already
            # measured the per-filing fetch→parse→chunk→embed→store
            # wall-clock; reuse it rather than starting a second timer.
            # No labels — ticker / form type would be both Tier-2/3 and
            # high-cardinality.
            get_metrics().observe_ingestion(result.ingest_result.duration_seconds)

            self._push(
                info,
                {
                    "type": "filing_done",
                    "ticker": filing_id.ticker,
                    "form_type": filing_id.form_type,
                    "filing_date": filing_id.date_str,
                    "accession_number": filing_id.accession_number,
                    "segments": result.ingest_result.segment_count,
                    "chunks": result.ingest_result.chunk_count,
                    "time": round(result.ingest_result.duration_seconds, 1),
                },
            )

            logger.info(
                "Task %s: ingested %s %s (%s) — %d chunks in %.1fs",
                info.task_id[:8],
                redact_for_log(filing_id.ticker),
                filing_id.form_type,
                filing_id.date_str,
                result.ingest_result.chunk_count,
                result.ingest_result.duration_seconds,
            )

        if info.state == TaskState.RUNNING:
            info.progress.step_label = "Complete"
            self._mark_terminal(info, TaskState.COMPLETED)
            self._push(
                info,
                {
                    "type": "completed",
                    "results": [r.to_dict() for r in info.results],
                    "summary": {
                        "total": len(info.results)
                        + info.progress.filings_skipped
                        + info.progress.filings_failed,
                        "succeeded": len(info.results),
                        "skipped": info.progress.filings_skipped,
                        "failed": info.progress.filings_failed,
                    },
                },
            )
            logger.info(
                "Task %s completed: %d ingested, %d skipped, %d failed",
                info.task_id[:8],
                len(info.results),
                info.progress.filings_skipped,
                info.progress.filings_failed,
            )

            # Post-ingest retention sweep. Best-effort: a failure here
            # MUST NOT shadow the user-visible COMPLETED transition above.
            # Disabled at ``DB_RETENTION_MAX_AGE_DAYS=0``; the helper is
            # also the lifespan-startup call site so the discipline
            # (audit-log, swallowed errors) stays single-sourced.
            run_retention_eviction_safe(
                self._filing_store,
                get_settings().database.retention_max_age_days,
                context_label="post_ingest",
            )

    # ------------------------------------------------------------------
    # EDGAR identity application
    # ------------------------------------------------------------------

    def _run_with_edgar_identity(
        self,
        info: TaskInfo,
        operation: Callable,
        *args,
        **kwargs,
    ):
        """Re-apply the task's EDGAR identity, then run ``operation``.

        ``edgar.set_identity`` mutates process-global state, so calling
        this immediately before every EDGAR-bound operation is
        load-bearing — without it, a concurrent task could fetch under
        another tenant's identity. The lock serialises the
        set-then-call sequence end-to-end; releasing the lock between
        set and call would re-open the race.

        The resolver is fetched from :attr:`TaskManager._task_resolvers`
        (kept off :class:`TaskInfo` so it never leaks onto the wire).
        A ``None`` resolver or a resolver that returns ``None`` defers
        to the fetcher's admin-env default.
        """
        with self._lock:
            resolver = self._task_resolvers.get(info.task_id)
        identity = resolver() if resolver is not None else None

        with self._edgar_lock:
            if identity is not None:
                self._fetcher.apply_identity(identity.name, identity.email)
            else:
                self._fetcher.apply_identity(None, None)
            return operation(*args, **kwargs)

    # ------------------------------------------------------------------
    # GPU time limit
    # ------------------------------------------------------------------

    @staticmethod
    def _timeout_task(info: TaskInfo) -> None:
        """Auto-cancel a task that has exceeded the GPU time limit.

        Routes through the same ``cancel_event`` path as a user
        cancellation so the worker performs a clean rollback. The
        timer is set when ``API_MAX_TASK_DURATION_MINUTES > 0``;
        Scenario A (``0`` default) skips the timer entirely.
        """
        if info.state == TaskState.RUNNING:
            info.cancel_event.set()
            logger.warning(
                "Task %s auto-cancelled: exceeded GPU time limit",
                info.task_id[:8],
            )

    # ------------------------------------------------------------------
    # WebSocket queue plumbing
    # ------------------------------------------------------------------

    def _push(self, info: TaskInfo, message: dict) -> None:
        """Push a WebSocket message onto the task's async queue.

        Two transports:

            - When an event loop is bound (production), use
              :meth:`asyncio.AbstractEventLoop.call_soon_threadsafe` so
              the synchronous worker thread can hand off to the
              asyncio-owned queue without violating its thread-safety
              contract.
            - Without an event loop (unit tests), the message lands on
              the queue directly via ``put_nowait``. Tests can then
              drain the queue with ``info._message_queue.get_nowait()``.

        The queue is built lazily because :class:`asyncio.Queue` binds
        to the running loop at construction time — building it in
        ``TaskInfo.__init__`` (off the worker thread) would either
        require ``asyncio.run`` to be active or fail with "no current
        event loop". Lazy construction sidesteps both.
        """
        if info._message_queue is None:
            info._message_queue = asyncio.Queue()

        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(info._message_queue.put_nowait, message)
        else:
            info._message_queue.put_nowait(message)

    # ------------------------------------------------------------------
    # Work list builder
    # ------------------------------------------------------------------

    def _build_work_list(self, info: TaskInfo) -> list[FilingInfo]:
        """Flatten the (ticker, form_type) matrix into a per-filing work list.

        Only the lightweight metadata is fetched here — HTML is pulled
        per filing inside :meth:`_execute`. The two modes
        (``count_mode='total'`` cross-form vs. per-form) live here
        because the choice is purely about how the matrix is unrolled;
        once unrolled, the rest of the pipeline is identical.

        Per-form fetch errors are logged and **swallowed** — a single
        ticker x form failure should not nuke the whole batch. The
        per-filing fetch errors inside :meth:`_execute` get the same
        treatment.
        """
        work: list[FilingInfo] = []

        for ticker in info.tickers:
            if info.cancel_event.is_set():
                break

            info.progress.current_ticker = ticker
            info.progress.step_label = "Fetching"
            info.progress.step_index = 0

            if info.count_mode == "total" and info.count is not None:
                filings = self._fetcher.list_available_across_forms(
                    ticker,
                    tuple(info.form_types),
                    count=info.count,
                    year=info.year,
                    start_date=info.start_date,
                    end_date=info.end_date,
                )
                work.extend(filings)
            else:
                for form_type in info.form_types:
                    if info.cancel_event.is_set():
                        break

                    info.progress.current_form_type = form_type
                    effective_count = self._effective_count(info)

                    try:
                        available = self._fetcher.list_available(
                            ticker,
                            form_type,
                            count=effective_count,
                            year=info.year,
                            start_date=info.start_date,
                            end_date=info.end_date,
                        )
                        work.extend(available)
                    except FetchError as exc:
                        logger.warning(
                            "Task %s: fetch failed for %s %s — %s",
                            info.task_id[:8],
                            redact_for_log(ticker),
                            form_type,
                            exc.message,
                        )

        return work

    @staticmethod
    def _effective_count(info: TaskInfo) -> int | None:
        """Mirror the CLI's filter-aware default count logic."""
        if info.count_mode == "per_form" and info.count is not None:
            return info.count
        has_filters = (
            info.year is not None or info.start_date is not None or info.end_date is not None
        )
        if has_filters and info.count is None:
            return None
        if info.count is not None:
            return info.count
        return 1

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def _rollback(self, info: TaskInfo) -> None:
        """Roll back filings stored during this task.

        Delegates to :meth:`FilingStore.delete_filings_batch` so the
        ChromaDB-first ordering and best-effort rollback semantics stay
        single-sourced. A failed rollback is logged but **never**
        raised — the worker's terminal-state transition is the user-
        facing signal and we do not want a Chroma blip to mask a clean
        cancel.
        """
        if not info._stored_accessions:
            return

        logger.info(
            "Task %s: rolling back %d filing(s)",
            info.task_id[:8],
            len(info._stored_accessions),
        )

        try:
            self._filing_store.delete_filings_batch(list(info._stored_accessions))
        except DatabaseError as exc:
            logger.error(
                "Task %s: rollback failed — %s",
                info.task_id[:8],
                exc.message,
            )

        info._stored_accessions.clear()

    # ------------------------------------------------------------------
    # FIFO eviction (demo mode)
    # ------------------------------------------------------------------

    def _maybe_evict(self, info: TaskInfo, new_filings: int) -> None:
        """Drop the oldest filings when demo mode would overflow.

        No-op when there is room. The eviction count adds a small
        buffer so a steady-state demo does not churn on every ingest —
        the buffer comes from ``API_DEMO_EVICTION_BUFFER``.
        """
        settings = get_settings()
        max_filings = settings.database.max_filings
        current_count = self._registry.count()
        available = max_filings - current_count

        if new_filings <= available:
            return

        slots_needed = new_filings - available
        eviction_count = slots_needed + settings.api.demo_eviction_buffer
        eviction_count = min(eviction_count, current_count)
        if eviction_count <= 0:
            return

        oldest = self._registry.list_oldest_filings(eviction_count)
        if not oldest:
            return

        evicted_tickers = sorted({f.ticker for f in oldest})
        accessions = [f.accession_number for f in oldest]
        chunks_deleted = self._filing_store.delete_filings_batch(accessions)

        logger.info(
            "Task %s: FIFO eviction — deleted %d filing(s) (%d chunks) "
            "to make room for %d new filing(s)",
            info.task_id[:8],
            len(oldest),
            chunks_deleted,
            new_filings,
        )

        self._push(
            info,
            {
                "type": "eviction",
                "filings_evicted": len(oldest),
                "chunks_evicted": chunks_deleted,
                "tickers_affected": evicted_tickers,
            },
        )

    def _fail_with_limit(self, info: TaskInfo, current: int, maximum: int) -> None:
        """Centralise the filing-limit-exceeded terminal path.

        Both the pre-loop and inside-loop cases route through here so
        the audit-log line and the WebSocket envelope shape stay
        consistent — without this helper, a future ``details`` field
        addition would have to be applied in two places.
        """
        exc = FilingLimitExceededError(current, maximum)
        info.error = exc.message
        self._mark_terminal(info, TaskState.FAILED)
        self._push(
            info,
            {
                "type": "failed",
                "error": exc.message,
                "details": exc.details,
            },
        )

    # ------------------------------------------------------------------
    # Terminal-state hook (persist + record completion timestamp)
    # ------------------------------------------------------------------

    def _mark_terminal(self, info: TaskInfo, state: TaskState) -> None:
        """Flip a task to a terminal state and persist it to SQLite.

        Persisting here (rather than via a recurring sweeper as the
        legacy did) means the ``task_history`` row exists the moment
        the user receives the terminal WebSocket event — no eventual-
        consistency window where ``GET /api/ingest/tasks/{id}`` would
        return a stale state or 404 during the eviction lag.

        ``save_task_history`` enforces the privacy controls
        (ticker stripping, error scrubbing); the manager never passes a
        key or PII here in the first place — the parameter list is the
        contract surface.
        """
        info.state = state
        info.completed_at = datetime.now(UTC)

        try:
            self._registry.save_task_history(
                info.task_id,
                status=info.state.value,
                tickers=info.tickers,
                form_types=info.form_types,
                results=[r.to_history_dict() for r in info.results],
                error=info.error,
                started_at=(info.started_at.isoformat() if info.started_at else None),
                completed_at=info.completed_at.isoformat(),
                filings_done=info.progress.filings_done,
                filings_skipped=info.progress.filings_skipped,
                filings_failed=info.progress.filings_failed,
            )
        except Exception:
            logger.exception(
                "Failed to persist task %s to history",
                info.task_id[:8],
            )

    # ------------------------------------------------------------------
    # Caller-driven embedder idle-unload
    # ------------------------------------------------------------------

    def _maybe_unload_embedder(self) -> None:
        """Best-effort idle-unload check on the bound embedder.

        Fires at task lifecycle boundaries (worker ``finally`` after
        the GPU semaphore release) and on every lazy eviction sweep.
        The provider's :meth:`maybe_unload` honours its configured
        ``EMBEDDING_IDLE_TIMEOUT_MINUTES`` threshold; back-to-back
        tasks therefore do not churn (``_last_used`` was just
        refreshed inside the worker's ``encode`` call).

        Hosted embedders have no idle state to release, so the helper
        short-circuits when ``maybe_unload`` is absent — duck-typing on
        the public method keeps the manager free of an import-time
        dependency on :class:`LocalEmbeddingProvider` (and its optional
        ``[local-embeddings]`` extra).  Errors are logged at ``warning``
        and swallowed; an unload failure must not shadow the worker's
        terminal-state transition or block a polling read.

        The lifecycle boundary + lazy seam is the sanctioned pattern
        here; a background timer would fight CUDA teardown ordering.
        """
        if self._embedder is None:
            return
        unload = getattr(self._embedder, "maybe_unload", None)
        if unload is None:
            return
        try:
            unload()
        except Exception:
            logger.warning(
                "Embedder idle-unload check failed; continuing",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Lazy eviction (no background thread)
    # ------------------------------------------------------------------

    def _evict_stale_locked(self) -> None:
        """Drop in-memory entries for terminal tasks older than the TTL.

        Called on every read / create. The persisted ``task_history``
        row is the long-term store; the in-memory entry is a cache for
        polling clients. The pattern mirrors
        :class:`InMemorySessionCredentialStore` and
        :class:`InMemorySessionEdgarIdentityStore` — every short-lived
        in-memory store in this codebase evicts lazily rather than
        running a background timer.

        Also fires the idle-unload check so a steady-state operator UI
        poll can release VRAM after the configured idle window even
        when no new ingestion task arrives.  Hosted embedders no-op via
        duck-typing in :meth:`_maybe_unload_embedder`.
        """
        now = time.time()
        to_remove: list[str] = []

        with self._lock:
            for task_id, info in self._tasks.items():
                if info.state not in _TERMINAL_STATES:
                    continue
                if info.completed_at is None:
                    continue
                if now - info.completed_at.timestamp() > _TASK_TTL_SECONDS:
                    to_remove.append(task_id)

            for task_id in to_remove:
                self._tasks.pop(task_id, None)
                self._task_resolvers.pop(task_id, None)

        # Lazy embedder unload runs outside the manager lock — the
        # provider has its own load lock and unload can take a moment
        # when the CUDA cache flush is non-trivial.
        self._maybe_unload_embedder()

        if to_remove:
            logger.debug("Evicted %d stale task(s) from memory", len(to_remove))
            # Prune the history table on the same cadence — settings-
            # driven retention is what bounds the on-disk row count.
            try:
                self._registry.prune_task_history()
            except Exception:
                logger.exception("Failed to prune task history")

    # ------------------------------------------------------------------
    # History rehydration
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_task_info(history: dict) -> TaskInfo:
        """Rebuild a read-only :class:`TaskInfo` from a persisted row.

        The reconstructed object has no live ``cancel_event`` /
        ``_message_queue``; downstream callers expect the in-memory
        path to own those. Treating the result as read-only is a
        convention enforced by code review, not by the dataclass.
        """
        info = TaskInfo(
            task_id=history["task_id"],
            tickers=history["tickers"] or [],
            form_types=history["form_types"] or [],
        )
        info.state = TaskState(history["status"])
        info.error = history["error"]
        info.progress = TaskProgress(
            filings_done=history["filings_done"],
            filings_skipped=history["filings_skipped"],
            filings_failed=history["filings_failed"],
        )
        info.results = [
            FilingResult(
                ticker=r["ticker"],
                form_type=r["form_type"],
                filing_date=r["filing_date"],
                accession_number=r["accession_number"],
                segment_count=r["segment_count"],
                chunk_count=r["chunk_count"],
                duration_seconds=r["duration_seconds"],
            )
            for r in history["results"]
        ]
        if history["started_at"]:
            info.started_at = datetime.fromisoformat(history["started_at"])
        if history["completed_at"]:
            info.completed_at = datetime.fromisoformat(history["completed_at"])
        return info
