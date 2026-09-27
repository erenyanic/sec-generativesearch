"""Tests for the background ``TaskManager``.

Strategy
--------

The task manager composes a :class:`FilingStore`, a :class:`MetadataRegistry`,
a :class:`FilingFetcher`, and a :class:`PipelineOrchestrator`.  Wiring
real instances would pull in ChromaDB, SQLite, edgartools, and the
sentence-transformers embedder — none of which we need to exercise the
worker's *coordination* contract.  Each collaborator is replaced by a
small in-process stub that records calls and lets the test shape the
relevant happy / sad path.

Coverage areas:

    - Lifecycle: create → terminal-state transition → history persistence.
    - Skip / duplicate / late-duplicate paths.
    - Cancellation: pre-fetch (queued), mid-pipeline (via progress
      callback), and rollback through ``FilingStore.delete_filings_batch``.
    - Filing-limit ceiling.
    - GPU semaphore serialises concurrent tasks.
    - Lazy eviction of stale terminal entries (no background thread).
    - Per-session listing.
    - **Security:** :class:`TaskInfo` carries no credential-shaped attribute
      name (mirrors :mod:`tests.core.test_types`); ``task_history`` write
      sites never receive a key argument.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, fields
from datetime import date
from typing import Any

import numpy as np
import pytest

from sec_generative_search.api.tasks import (
    FilingResult,
    TaskInfo,
    TaskManager,
    TaskProgress,
    TaskQueueFullError,
    TaskState,
    run_retention_eviction_safe,
)
from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.exceptions import (
    DatabaseError,
    FetchError,
    FilingLimitExceededError,
)
from sec_generative_search.core.types import (
    Chunk,
    ContentType,
    EvictionReport,
    FilingIdentifier,
    IngestResult,
)
from sec_generative_search.pipeline.fetch import FilingInfo
from sec_generative_search.pipeline.orchestrator import ProcessedFiling

# ---------------------------------------------------------------------------
# Stub collaborators
# ---------------------------------------------------------------------------


def _make_filing_info(
    ticker: str,
    accession_number: str,
    form_type: str = "10-K",
    filing_date: date = date(2023, 9, 30),
) -> FilingInfo:
    """Build a minimal ``FilingInfo`` (no cached ``_filing_obj``)."""
    return FilingInfo(
        ticker=ticker,
        form_type=form_type,
        filing_date=filing_date,
        accession_number=accession_number,
        company_name=f"{ticker} Inc.",
    )


def _make_processed_filing(filing_info: FilingInfo) -> ProcessedFiling:
    """Build a ProcessedFiling with one chunk + tiny embedding."""
    filing_id = filing_info.to_identifier()
    chunk = Chunk(
        content="Sample chunk text.",
        path="Part I > Item 1",
        content_type=ContentType.TEXT,
        filing_id=filing_id,
        chunk_index=0,
    )
    return ProcessedFiling(
        filing_id=filing_id,
        chunks=[chunk],
        embeddings=np.zeros((1, 4), dtype=np.float32),
        ingest_result=IngestResult(
            filing_id=filing_id,
            segment_count=1,
            chunk_count=1,
            duration_seconds=0.05,
        ),
    )


@dataclass
class _StubRegistry:
    """In-memory stand-in for :class:`MetadataRegistry`.

    Implements only the read paths and ``save_task_history`` /
    ``prune_task_history`` calls the manager actually invokes — enough
    to drive every code path without booting SQLite.
    """

    existing_accessions: set[str] = field(default_factory=set)
    filing_count: int = 0
    oldest: list[Any] = field(default_factory=list)
    history: dict[str, dict] = field(default_factory=dict)
    save_calls: list[dict] = field(default_factory=list)
    prune_calls: int = 0

    def get_existing_accessions(self, accessions: list[str]) -> set[str]:
        return {a for a in accessions if a in self.existing_accessions}

    def count(self) -> int:
        return self.filing_count

    def list_oldest_filings(self, limit: int) -> list[Any]:
        return self.oldest[:limit]

    def save_task_history(self, task_id: str, **kwargs: Any) -> None:
        # The manager never passes a key argument — assert here so a
        # future regression surfaces in *every* test, not just the
        # explicit security one.
        for forbidden in ("api_key", "secret", "credential", "auth_token"):
            assert forbidden not in kwargs, (
                f"save_task_history received forbidden kwarg {forbidden}"
            )
        self.save_calls.append({"task_id": task_id, **kwargs})
        self.history[task_id] = {"task_id": task_id, **kwargs}

    def get_task_history(self, task_id: str) -> dict | None:
        row = self.history.get(task_id)
        if row is None:
            return None
        # Shape mirrors ``MetadataRegistry.get_task_history``.
        return {
            "task_id": row["task_id"],
            "status": row["status"],
            "tickers": row.get("tickers") or [],
            "form_types": row.get("form_types") or [],
            "results": row.get("results") or [],
            "error": row.get("error"),
            "started_at": row.get("started_at"),
            "completed_at": row.get("completed_at"),
            "filings_done": row.get("filings_done", 0),
            "filings_skipped": row.get("filings_skipped", 0),
            "filings_failed": row.get("filings_failed", 0),
        }

    def prune_task_history(self) -> int:
        self.prune_calls += 1
        return 0


@dataclass
class _StubFetcher:
    """Records ``apply_identity`` calls and serves canned work lists."""

    work_lists: dict[str, list[FilingInfo]] = field(default_factory=dict)
    cross_form: dict[tuple, list[FilingInfo]] = field(default_factory=dict)
    content_map: dict[str, str] = field(default_factory=dict)
    identity_calls: list[tuple[str | None, str | None]] = field(default_factory=list)
    raise_on_fetch: dict[str, Exception] = field(default_factory=dict)

    def apply_identity(self, name: str | None = None, email: str | None = None) -> None:
        self.identity_calls.append((name, email))

    def list_available(
        self,
        ticker: str,
        form_type: str,
        *,
        count: int | None = None,
        year: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[FilingInfo]:
        return list(self.work_lists.get((ticker, form_type), []))

    def list_available_across_forms(
        self,
        ticker: str,
        form_types: tuple,
        *,
        count: int | None = None,
        year: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[FilingInfo]:
        return list(self.cross_form.get((ticker, form_types), []))

    def fetch_filing_content(
        self,
        filing_info: FilingInfo,
    ) -> tuple[FilingIdentifier, str]:
        if filing_info.accession_number in self.raise_on_fetch:
            raise self.raise_on_fetch[filing_info.accession_number]
        return (
            filing_info.to_identifier(),
            self.content_map.get(filing_info.accession_number, "<html></html>"),
        )


@dataclass
class _StubOrchestrator:
    """Replays a canned ``ProcessedFiling`` per filing id.

    Optionally invokes the ``progress_callback`` between each step so
    the cancel-mid-pipeline test can fire ``_CancelledError`` from a
    plausible code path.
    """

    by_accession: dict[str, ProcessedFiling] = field(default_factory=dict)
    invoke_progress: bool = False
    raise_on: dict[str, Exception] = field(default_factory=dict)

    def process_filing(
        self,
        filing_id: FilingIdentifier,
        html_content: str,
        progress_callback=None,
    ) -> ProcessedFiling:
        if filing_id.accession_number in self.raise_on:
            raise self.raise_on[filing_id.accession_number]
        if self.invoke_progress and progress_callback is not None:
            # Three steps mirroring the real orchestrator's 4-step
            # pipeline (parse / chunk / embed / complete).
            for idx, step in enumerate(("Parsing", "Chunking", "Embedding"), start=1):
                progress_callback(step, idx, 4)
        return self.by_accession[filing_id.accession_number]


@dataclass
class _StubFilingStore:
    """Records writes + deletes; the manager's single dual-store seam."""

    duplicate_accessions: set[str] = field(default_factory=set)
    stored: list[str] = field(default_factory=list)
    deleted: list[list[str]] = field(default_factory=list)
    raise_on_store: dict[str, Exception] = field(default_factory=dict)
    # Per-call recording for the post-ingest retention sweep.
    evict_calls: list[int] = field(default_factory=list)
    evict_report: EvictionReport = field(
        default_factory=lambda: EvictionReport(filings_evicted=0, chunks_evicted=0, max_age_days=0)
    )
    raise_on_evict: Exception | None = None

    def store_filing(
        self,
        processed: ProcessedFiling,
        *,
        register_if_new: bool = False,
    ) -> bool:
        accession = processed.filing_id.accession_number
        if accession in self.raise_on_store:
            raise self.raise_on_store[accession]
        if register_if_new and accession in self.duplicate_accessions:
            return False
        self.stored.append(accession)
        return True

    def delete_filings_batch(self, accessions: list[str]) -> int:
        # Manager passes a list copy; capture it verbatim.
        self.deleted.append(list(accessions))
        return len(accessions)

    def evict_expired(self, max_age_days: int) -> EvictionReport:
        """Stub matching :meth:`FilingStore.evict_expired`'s contract.

        Records the cutoff for each call so the post-ingest hook tests
        can assert it lands with the configured retention value, and
        optionally raises a pre-canned exception to drive the
        best-effort error path.
        """
        self.evict_calls.append(max_age_days)
        if self.raise_on_evict is not None:
            raise self.raise_on_evict
        return EvictionReport(
            filings_evicted=self.evict_report.filings_evicted,
            chunks_evicted=self.evict_report.chunks_evicted,
            max_age_days=max_age_days,
        )


def _build_manager(
    *,
    registry: _StubRegistry | None = None,
    fetcher: _StubFetcher | None = None,
    orchestrator: _StubOrchestrator | None = None,
    filing_store: _StubFilingStore | None = None,
) -> tuple[TaskManager, _StubFilingStore, _StubRegistry, _StubFetcher, _StubOrchestrator]:
    registry = registry or _StubRegistry()
    fetcher = fetcher or _StubFetcher()
    orchestrator = orchestrator or _StubOrchestrator()
    filing_store = filing_store or _StubFilingStore()
    manager = TaskManager(
        filing_store=filing_store,  # type: ignore[arg-type]
        registry=registry,  # type: ignore[arg-type]
        fetcher=fetcher,  # type: ignore[arg-type]
        orchestrator=orchestrator,  # type: ignore[arg-type]
    )
    return manager, filing_store, registry, fetcher, orchestrator


_TERMINAL_STATES = frozenset({TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED})


def _join_worker(task_id: str, *, timeout: float) -> None:
    """Join the task's worker thread (named ``ingest-<id[:8]>``) if it is alive."""
    for thread in threading.enumerate():
        if thread.name == f"ingest-{task_id[:8]}":
            thread.join(timeout)
            if thread.is_alive():
                raise AssertionError(f"worker for task {task_id[:8]} did not exit")


def _wait_for_state(
    manager: TaskManager,
    task_id: str,
    *,
    target: TaskState | set[TaskState],
    timeout: float = 3.0,
) -> TaskInfo:
    """Poll until the task reaches the target state or the deadline.

    A terminal target also waits for the worker thread to exit.  The worker
    publishes the terminal state *before* its post-terminal work — history
    persistence, the terminal event, the post-ingest retention sweep and,
    in ``finally``, dropping the EDGAR resolver and idle-unload — so a test
    asserting on any of those right after the state flips races the worker.
    Serially that window is microseconds; under ``pytest -n auto`` CPU
    contention it failed ~1 run in 10 (OPTIMIZATIONS.md F31).
    """
    targets = target if isinstance(target, set) else {target}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = manager.get_task(task_id)
        if info is not None and info.state in targets:
            if targets <= _TERMINAL_STATES:
                _join_worker(task_id, timeout=max(0.0, deadline - time.monotonic()))
            return info
        time.sleep(0.01)
    raise AssertionError(
        f"task {task_id[:8]} did not reach {targets} within {timeout}s "
        f"(last state: {info.state if info else 'unknown'})"
    )


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch: pytest.MonkeyPatch):
    """Clear API_ env vars between tests so settings stay deterministic."""
    import os

    for key in list(os.environ.keys()):
        if key.startswith("API_"):
            monkeypatch.delenv(key, raising=False)
    reload_settings()
    yield
    reload_settings()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestTaskLifecycle:
    """A single-filing task progresses to COMPLETED and persists history."""

    def test_completed_task_persists_history(self) -> None:
        filing_info = _make_filing_info("AAPL", "0000320193-23-000077")
        processed = _make_processed_filing(filing_info)

        fetcher = _StubFetcher(
            work_lists={("AAPL", "10-K"): [filing_info]},
        )
        orchestrator = _StubOrchestrator(by_accession={filing_info.accession_number: processed})

        manager, store, registry, fetcher, _ = _build_manager(
            fetcher=fetcher,
            orchestrator=orchestrator,
        )

        task_id = manager.create_task(
            tickers=["AAPL"],
            form_types=["10-K"],
            session_id="sess-A",
        )
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        # Storage went through FilingStore.store_filing.
        assert store.stored == [filing_info.accession_number]
        # Single FilingResult recorded.
        assert len(info.results) == 1
        assert info.results[0].accession_number == filing_info.accession_number
        # History row written at terminal transition.
        assert task_id in registry.history
        assert registry.history[task_id]["status"] == "completed"

    def test_get_task_falls_back_to_history(self) -> None:
        """Tasks evicted from memory still resolve via ``task_history``."""
        manager, _, registry, _, _ = _build_manager()
        # Seed the history table directly so we can simulate an evicted
        # task without driving a full worker run.
        registry.history["evicted-id"] = {
            "task_id": "evicted-id",
            "status": "completed",
            "tickers": ["AAPL"],
            "form_types": ["10-K"],
            "results": [],
            "error": None,
            "started_at": None,
            "completed_at": None,
            "filings_done": 0,
            "filings_skipped": 0,
            "filings_failed": 0,
        }
        info = manager.get_task("evicted-id")
        assert info is not None
        assert info.state == TaskState.COMPLETED


# ---------------------------------------------------------------------------
# Duplicate / skip handling
# ---------------------------------------------------------------------------


class TestDuplicateHandling:
    """The two skip paths: known-up-front and late (race-window)."""

    def test_batch_dup_check_skips_known_accession(self) -> None:
        filing_info = _make_filing_info("AAPL", "0000320193-23-000077")
        registry = _StubRegistry(existing_accessions={filing_info.accession_number})
        manager, store, _, _, _ = _build_manager(
            registry=registry,
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [filing_info]}),
        )
        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)
        # No write went through the store — the dup short-circuit fired.
        assert store.stored == []
        assert info.progress.filings_skipped == 1

    def test_late_duplicate_via_register_if_new_false(self) -> None:
        """``FilingStore.store_filing(register_if_new=True)`` may return False."""
        filing_info = _make_filing_info("AAPL", "0000320193-23-000077")
        processed = _make_processed_filing(filing_info)
        store = _StubFilingStore(duplicate_accessions={filing_info.accession_number})
        manager, store, _, _, _ = _build_manager(
            filing_store=store,
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [filing_info]}),
            orchestrator=_StubOrchestrator(
                by_accession={filing_info.accession_number: processed},
            ),
        )
        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)
        assert info.progress.filings_skipped == 1
        assert info.progress.filings_done == 1
        # No stored accession — the atomic claim path lost the race.
        assert store.stored == []


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    def test_cancel_before_acquire_marks_cancelled(self, monkeypatch) -> None:
        """A cancel before the GPU acquire still terminates cleanly."""
        # Hold the GPU slot so the worker blocks on ``acquire``.
        manager, store, _, _, _ = _build_manager()
        manager._gpu_semaphore.acquire()
        try:
            task_id = manager.create_task(
                tickers=["AAPL"],
                form_types=["10-K"],
            )
            # Cancel while the worker is queued behind the semaphore.
            assert manager.cancel_task(task_id) is True
        finally:
            # Let the worker proceed.
            manager._gpu_semaphore.release()

        _wait_for_state(manager, task_id, target=TaskState.CANCELLED)
        # No filings were even attempted, so no rollback delete fired.
        assert store.deleted == []
        # Cancel signal returns False on a terminal task.
        assert manager.cancel_task(task_id) is False

    def test_cancel_during_pipeline_rolls_back_stored_filings(self) -> None:
        """The progress callback's cancel check triggers rollback.

        Two-filing race-free design: the orchestrator stub for f2 looks
        up the running task via ``manager.list_tasks()`` rather than
        capturing the task info ahead of time. The manager reference is
        bound before the worker thread starts, so there is no window
        where the stub could execute before the holder is populated.
        """
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        f2 = _make_filing_info("AAPL", "0000320193-23-000002")
        p1 = _make_processed_filing(f1)
        p2 = _make_processed_filing(f2)

        manager_holder: dict[str, TaskManager] = {}

        class _CancelOnSecond(_StubOrchestrator):
            def process_filing(self, filing_id, html_content, progress_callback=None):
                if filing_id.accession_number == f2.accession_number:
                    mgr = manager_holder["mgr"]
                    for task in mgr.list_tasks():
                        if task.state == TaskState.RUNNING:
                            task.cancel_event.set()
                    if progress_callback is not None:
                        progress_callback("Embedding", 3, 4)
                return self.by_accession[filing_id.accession_number]

        manager, store, _, _, _ = _build_manager(
            fetcher=_StubFetcher(
                work_lists={("AAPL", "10-K"): [f1, f2]},
            ),
            orchestrator=_CancelOnSecond(
                by_accession={f1.accession_number: p1, f2.accession_number: p2},
            ),
        )
        # Bind the manager *before* spawning the worker so the stub
        # cannot race ahead of holder population.
        manager_holder["mgr"] = manager

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.CANCELLED)
        # f1 was stored, then rolled back via FilingStore.
        assert store.stored == [f1.accession_number]
        assert store.deleted == [[f1.accession_number]]
        # The rollback path cleared ``_stored_accessions``.
        assert info._stored_accessions == []

    def test_rollback_swallows_database_error(self, caplog) -> None:
        """A failing rollback logs but does not raise.

        Two-filing scenario: f1 stores successfully, then the
        progress-callback cancel fires inside f2's orchestrator call,
        forcing the rollback path through ``delete_filings_batch``
        which is configured to raise. The worker must still reach the
        ``CANCELLED`` terminal state.
        """
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        f2 = _make_filing_info("AAPL", "0000320193-23-000002")
        p1 = _make_processed_filing(f1)
        p2 = _make_processed_filing(f2)

        class _FailingStore(_StubFilingStore):
            def delete_filings_batch(self, accessions: list[str]) -> int:
                raise DatabaseError("chroma down")

        manager_holder: dict[str, TaskManager] = {}

        class _CancelOnSecond(_StubOrchestrator):
            def process_filing(self, filing_id, html_content, progress_callback=None):
                if filing_id.accession_number == f2.accession_number:
                    mgr = manager_holder["mgr"]
                    for task in mgr.list_tasks():
                        if task.state == TaskState.RUNNING:
                            task.cancel_event.set()
                    if progress_callback is not None:
                        progress_callback("Embedding", 3, 4)
                return self.by_accession[filing_id.accession_number]

        manager, _, _, _, _ = _build_manager(
            filing_store=_FailingStore(),
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1, f2]}),
            orchestrator=_CancelOnSecond(
                by_accession={f1.accession_number: p1, f2.accession_number: p2},
            ),
        )
        manager_holder["mgr"] = manager

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.CANCELLED)
        # Worker terminated cleanly even though rollback raised
        # ``DatabaseError`` inside ``delete_filings_batch``.
        assert info.state == TaskState.CANCELLED


# ---------------------------------------------------------------------------
# Filing-limit ceiling
# ---------------------------------------------------------------------------


class TestFilingLimit:
    def test_exceeded_limit_fails_task(self, monkeypatch) -> None:
        """When the registry is at capacity the worker surfaces a failure."""
        monkeypatch.setenv("DB_MAX_FILINGS", "1")
        reload_settings()

        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        registry = _StubRegistry(filing_count=1)
        manager, _, _, _, _ = _build_manager(
            registry=registry,
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1]}),
            orchestrator=_StubOrchestrator(
                by_accession={f1.accession_number: _make_processed_filing(f1)},
            ),
        )
        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.FAILED)
        assert info.error is not None
        assert info.error == FilingLimitExceededError(1, 1).message


# ---------------------------------------------------------------------------
# Demo-mode FIFO eviction (Scenario C)
# ---------------------------------------------------------------------------


@dataclass
class _EvictingRegistry(_StubRegistry):
    """A registry whose ``count`` shrinks as filings are evicted.

    The base :class:`_StubRegistry` returns a static ``filing_count``;
    the real ``MetadataRegistry.count()`` falls as the FIFO sweep
    deletes rows.  Modelling that decrement is load-bearing: without it
    the in-loop ceiling check (``cached_count >= max_filings``) would
    never clear and a demo ingest could never make progress.
    """

    def drop(self, accessions: list[str]) -> None:
        keep = [f for f in self.oldest if f.accession_number not in accessions]
        self.filing_count -= len(self.oldest) - len(keep)
        self.oldest = keep


@dataclass
class _EvictingStore(_StubFilingStore):
    """A filing store that reflects FIFO deletes back onto its registry."""

    linked_registry: _EvictingRegistry | None = None

    def delete_filings_batch(self, accessions: list[str]) -> int:
        deleted = super().delete_filings_batch(accessions)
        if self.linked_registry is not None:
            self.linked_registry.drop(accessions)
        return deleted


@pytest.mark.security
class TestDemoModeFifoEviction:
    """FIFO eviction is the Scenario C resource-exhaustion control.

    Demo mode caps the corpus at ``DB_MAX_FILINGS`` by silently dropping
    the *oldest* filings to admit new ones.  Two invariants are
    security-load-bearing:

    1. The eviction count is bounded by the live filing count — the
       ``API_DEMO_EVICTION_BUFFER`` can never drive a delete larger than
       the corpus (a self-inflicted data-loss / churn amplifier).
    2. FIFO eviction fires **only** in demo mode.  A team / cloud
       deployment (``API_DEMO_MODE`` unset) must surface
       ``FilingLimitExceededError`` and NEVER silently delete a user's
       filings.
    """

    @staticmethod
    def _drain(info: TaskInfo) -> list[dict]:
        assert info._message_queue is not None
        out: list[dict] = []
        while True:
            try:
                out.append(info._message_queue.get_nowait())
            except Exception:
                break
        return out

    def _run_demo_ingest(
        self,
        monkeypatch,
        *,
        max_filings: int,
        buffer: int,
        start_count: int,
        oldest: list[FilingInfo],
        new: FilingInfo,
        target: TaskState = TaskState.COMPLETED,
    ) -> tuple[TaskInfo, _EvictingStore, _EvictingRegistry]:
        monkeypatch.setenv("API_DEMO_MODE", "true")
        monkeypatch.setenv("DB_MAX_FILINGS", str(max_filings))
        monkeypatch.setenv("API_DEMO_EVICTION_BUFFER", str(buffer))
        reload_settings()

        registry = _EvictingRegistry(filing_count=start_count, oldest=list(oldest))
        store = _EvictingStore(linked_registry=registry)
        manager, _, _, _, _ = _build_manager(
            registry=registry,
            filing_store=store,
            fetcher=_StubFetcher(work_lists={(new.ticker, new.form_type): [new]}),
            orchestrator=_StubOrchestrator(
                by_accession={new.accession_number: _make_processed_filing(new)},
            ),
        )
        task_id = manager.create_task(tickers=[new.ticker], form_types=[new.form_type])
        info = _wait_for_state(manager, task_id, target=target)
        return info, store, registry

    def test_overflow_evicts_oldest_filing(self, monkeypatch) -> None:
        """At the ceiling, the oldest filing is dropped to admit the new one."""
        old1 = _make_filing_info("OLD1", "0000000001-23-000001")
        old2 = _make_filing_info("OLD2", "0000000002-23-000002")
        new = _make_filing_info("AAPL", "0000320193-23-000077")

        info, store, _ = self._run_demo_ingest(
            monkeypatch,
            max_filings=2,
            buffer=0,
            start_count=2,
            oldest=[old1, old2],
            new=new,
        )

        # Exactly the single oldest filing was evicted, and the new one stored.
        assert store.deleted == [[old1.accession_number]]
        assert store.stored == [new.accession_number]
        assert info.state == TaskState.COMPLETED

    def test_eviction_buffer_widens_the_delete(self, monkeypatch) -> None:
        """``API_DEMO_EVICTION_BUFFER`` is added on top of the slots needed."""
        oldest = [
            _make_filing_info(f"OLD{i}", f"00000000{i:02d}-23-0000{i:02d}") for i in range(10)
        ]
        new = _make_filing_info("AAPL", "0000320193-23-000077")

        # slots_needed = 1; buffer = 3 → 4 filings evicted in one sweep.
        _, store, _ = self._run_demo_ingest(
            monkeypatch,
            max_filings=10,
            buffer=3,
            start_count=10,
            oldest=oldest,
            new=new,
        )

        assert len(store.deleted) == 1
        assert len(store.deleted[0]) == 4

    def test_eviction_count_bounded_by_corpus_size(self, monkeypatch) -> None:
        """A huge buffer can never delete more filings than exist (DoS guard)."""
        old1 = _make_filing_info("OLD1", "0000000001-23-000001")
        old2 = _make_filing_info("OLD2", "0000000002-23-000002")
        new = _make_filing_info("AAPL", "0000320193-23-000077")

        # buffer 1000 ≫ corpus (2): the sweep is clamped to current_count.
        _, store, _ = self._run_demo_ingest(
            monkeypatch,
            max_filings=2,
            buffer=1000,
            start_count=2,
            oldest=[old1, old2],
            new=new,
        )

        assert len(store.deleted[0]) == 2

    def test_no_eviction_when_room_available(self, monkeypatch) -> None:
        """Demo mode with spare capacity ingests without deleting anything."""
        oldest = [_make_filing_info("OLD1", "0000000001-23-000001")]
        new = _make_filing_info("AAPL", "0000320193-23-000077")

        _, store, _ = self._run_demo_ingest(
            monkeypatch,
            max_filings=10,
            buffer=0,
            start_count=5,
            oldest=oldest,
            new=new,
        )

        assert store.deleted == []
        assert store.stored == [new.accession_number]

    def test_eviction_emits_event_with_affected_tickers(self, monkeypatch) -> None:
        """The FIFO sweep pushes an ``eviction`` envelope for the UI."""
        old1 = _make_filing_info("OLD1", "0000000001-23-000001")
        old2 = _make_filing_info("OLD2", "0000000002-23-000002")
        new = _make_filing_info("AAPL", "0000320193-23-000077")

        info, _, _ = self._run_demo_ingest(
            monkeypatch,
            max_filings=2,
            buffer=0,
            start_count=2,
            oldest=[old1, old2],
            new=new,
        )

        eviction_events = [e for e in self._drain(info) if e.get("type") == "eviction"]
        assert len(eviction_events) == 1
        event = eviction_events[0]
        assert event["filings_evicted"] == 1
        assert event["tickers_affected"] == ["OLD1"]

    def test_non_demo_overflow_fails_without_eviction(self, monkeypatch) -> None:
        """Outside demo mode, a full corpus FAILS — it is never silently pruned.

        This is the load-bearing gate: silent FIFO data-loss is a
        demo-only behaviour.  A team / cloud deployment must refuse the
        ingest, not delete a user's existing filings to make room.
        """
        monkeypatch.delenv("API_DEMO_MODE", raising=False)
        monkeypatch.setenv("DB_MAX_FILINGS", "1")
        reload_settings()

        old1 = _make_filing_info("OLD1", "0000000001-23-000001")
        new = _make_filing_info("AAPL", "0000320193-23-000077")
        registry = _EvictingRegistry(filing_count=1, oldest=[old1])
        store = _EvictingStore(linked_registry=registry)
        manager, _, _, _, _ = _build_manager(
            registry=registry,
            filing_store=store,
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [new]}),
            orchestrator=_StubOrchestrator(
                by_accession={new.accession_number: _make_processed_filing(new)},
            ),
        )
        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.FAILED)

        assert info.error == FilingLimitExceededError(1, 1).message
        # Nothing was evicted and nothing was stored.
        assert store.deleted == []
        assert store.stored == []


# ---------------------------------------------------------------------------
# Queue cap
# ---------------------------------------------------------------------------


class TestQueueCap:
    def test_queue_cap_rejects_new_task(self, monkeypatch) -> None:
        monkeypatch.setenv("API_MAX_TASK_QUEUE_SIZE", "1")
        reload_settings()

        manager, _, _, _, _ = _build_manager()
        # Hold the GPU slot so the first task stays PENDING and counts
        # against the active cap.
        manager._gpu_semaphore.acquire()
        try:
            first = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
            with pytest.raises(TaskQueueFullError) as excinfo:
                manager.create_task(tickers=["MSFT"], form_types=["10-K"])
            assert excinfo.value.details == {"active": 1, "maximum": 1}
        finally:
            manager._gpu_semaphore.release()
        # Drain the first task so the fixture's reload_settings doesn't
        # race with a daemon thread still iterating.
        info = manager.get_task(first)
        assert info is not None
        info.cancel_event.set()
        _wait_for_state(manager, first, target={TaskState.CANCELLED, TaskState.COMPLETED})

    def test_shutdown_blocks_new_tasks(self) -> None:
        manager, _, _, _, _ = _build_manager()
        manager.shutdown()
        with pytest.raises(TaskQueueFullError):
            manager.create_task(tickers=["AAPL"], form_types=["10-K"])


# ---------------------------------------------------------------------------
# GPU semaphore serialisation
# ---------------------------------------------------------------------------


class TestGpuSerialisation:
    def test_two_tasks_run_sequentially(self) -> None:
        """The ``Semaphore(1)`` gate forbids overlapping pipeline runs."""
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        f2 = _make_filing_info("MSFT", "0000789019-23-000007")
        p1 = _make_processed_filing(f1)
        p2 = _make_processed_filing(f2)

        # Track concurrent entries into the orchestrator. A correct
        # implementation gates with the GPU semaphore, so this counter
        # must never exceed 1 across the run.
        active_lock = threading.Lock()
        active = {"count": 0, "max": 0}

        class _CountingOrch(_StubOrchestrator):
            def process_filing(self, filing_id, html_content, progress_callback=None):
                with active_lock:
                    active["count"] += 1
                    active["max"] = max(active["max"], active["count"])
                try:
                    # Sleep so concurrent tasks would actually overlap
                    # if the semaphore were broken.
                    time.sleep(0.05)
                    return self.by_accession[filing_id.accession_number]
                finally:
                    with active_lock:
                        active["count"] -= 1

        manager, store, _, _, _ = _build_manager(
            fetcher=_StubFetcher(
                work_lists={
                    ("AAPL", "10-K"): [f1],
                    ("MSFT", "10-K"): [f2],
                }
            ),
            orchestrator=_CountingOrch(
                by_accession={f1.accession_number: p1, f2.accession_number: p2},
            ),
        )

        task_a = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        task_b = manager.create_task(tickers=["MSFT"], form_types=["10-K"])
        _wait_for_state(manager, task_a, target=TaskState.COMPLETED)
        _wait_for_state(manager, task_b, target=TaskState.COMPLETED)

        # Both tasks stored their filing; neither overlapped in
        # ``process_filing`` (the GPU semaphore is the gate).
        assert set(store.stored) == {f1.accession_number, f2.accession_number}
        assert active["max"] == 1


# ---------------------------------------------------------------------------
# Session ownership
# ---------------------------------------------------------------------------


class TestSessionOwnership:
    def test_list_tasks_for_session_filters(self) -> None:
        manager, _, _, _, _ = _build_manager()
        # Suspend the GPU so the tasks stay in flight and we can list
        # them without racing the COMPLETED transition.
        manager._gpu_semaphore.acquire()
        try:
            a = manager.create_task(
                tickers=["AAPL"],
                form_types=["10-K"],
                session_id="sess-A",
            )
            b = manager.create_task(
                tickers=["MSFT"],
                form_types=["10-K"],
                session_id="sess-B",
            )
            c = manager.create_task(
                tickers=["NVDA"],
                form_types=["10-K"],
                session_id=None,
            )

            for_a = {t.task_id for t in manager.list_tasks_for_session("sess-A")}
            for_b = {t.task_id for t in manager.list_tasks_for_session("sess-B")}
            for_none = {t.task_id for t in manager.list_tasks_for_session(None)}

            assert for_a == {a}
            assert for_b == {b}
            assert for_none == {c}
        finally:
            manager._gpu_semaphore.release()
            for task_id in (a, b, c):
                info = manager.get_task(task_id)
                if info is not None:
                    info.cancel_event.set()


# ---------------------------------------------------------------------------
# Lazy eviction
# ---------------------------------------------------------------------------


class TestLazyEviction:
    def test_stale_terminal_task_evicted_on_read(self, monkeypatch) -> None:
        """Terminal tasks older than the TTL are dropped from memory on the
        next ``list_tasks`` / ``get_task`` / ``create_task`` call."""
        from datetime import UTC, datetime, timedelta

        manager, _, registry, _, _ = _build_manager()
        # Insert a synthetic terminal task with an old ``completed_at``.
        info = TaskInfo(
            task_id="ancient",
            tickers=["AAPL"],
            form_types=["10-K"],
        )
        info.state = TaskState.COMPLETED
        info.completed_at = datetime.now(UTC) - timedelta(days=2)
        manager._tasks["ancient"] = info

        # Force a low TTL so the eviction sweep would fire even on a
        # younger task — keeps the test robust against clock skew.
        monkeypatch.setattr(
            "sec_generative_search.api.tasks._TASK_TTL_SECONDS",
            1,
        )

        # First read sweeps. Even though the task is gone from memory,
        # ``get_task`` does NOT need a history fallback for this case
        # (we never wrote one) so ``None`` is the expected result.
        result = manager.get_task("ancient")
        assert result is None
        assert "ancient" not in manager._tasks
        # Eviction also called ``prune_task_history`` so the on-disk
        # row count stays bounded by the operator-configured retention.
        assert registry.prune_calls >= 1


# ---------------------------------------------------------------------------
# EDGAR identity resolver
# ---------------------------------------------------------------------------


class TestEdgarIdentityResolver:
    def test_resolver_value_never_lands_on_task_info(self) -> None:
        """The resolver lives on the manager dict, not on TaskInfo."""
        from sec_generative_search.core.edgar_identity import EdgarIdentity

        captured: list[tuple] = []

        def _resolver() -> EdgarIdentity:
            return EdgarIdentity(
                name="Test User",
                email="test@example.invalid",
            )

        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        p1 = _make_processed_filing(f1)
        fetcher = _StubFetcher(
            work_lists={("AAPL", "10-K"): [f1]},
        )
        # Capture identity arguments rather than asserting in the stub
        # so we can inspect the call ordering.
        original_apply = fetcher.apply_identity

        def _apply(name=None, email=None):
            captured.append((name, email))
            return original_apply(name, email)

        fetcher.apply_identity = _apply  # type: ignore[method-assign]

        manager, _, _, _, _ = _build_manager(
            fetcher=fetcher,
            orchestrator=_StubOrchestrator(by_accession={f1.accession_number: p1}),
        )

        task_id = manager.create_task(
            tickers=["AAPL"],
            form_types=["10-K"],
            session_id="sess",
            edgar_identity_resolver=_resolver,
        )
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        # apply_identity received the resolver's identity at least once.
        assert ("Test User", "test@example.invalid") in captured

        # TaskInfo has no field carrying the identity value.
        for f in fields(info):
            if isinstance(getattr(info, f.name), str):
                assert getattr(info, f.name) != "test@example.invalid"
                assert getattr(info, f.name) != "Test User"

        # Resolver is cleared from the parallel dict in the worker's
        # ``finally``.
        assert task_id not in manager._task_resolvers


# ---------------------------------------------------------------------------
# Fetcher errors
# ---------------------------------------------------------------------------


class TestFetcherErrors:
    def test_fetch_error_marks_filing_failed_continues(self) -> None:
        """A single fetch failure should not nuke the whole batch."""
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        f2 = _make_filing_info("AAPL", "0000320193-23-000002")
        p2 = _make_processed_filing(f2)

        fetcher = _StubFetcher(
            work_lists={("AAPL", "10-K"): [f1, f2]},
            raise_on_fetch={f1.accession_number: FetchError("EDGAR 502")},
        )
        manager, store, _, _, _ = _build_manager(
            fetcher=fetcher,
            orchestrator=_StubOrchestrator(by_accession={f2.accession_number: p2}),
        )
        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)
        # f1 marked failed, f2 succeeded.
        assert info.progress.filings_failed == 1
        assert store.stored == [f2.accession_number]


# ---------------------------------------------------------------------------
# Worker policy — per-filing failure isolation
# ---------------------------------------------------------------------------


class TestPerFilingFailureIsolation:
    """A bare ``Exception`` from fetch / process / store stays local.

    A single malformed filing emits ``filing_failed`` and the loop
    continues with the next item. Without the broader ``except
    Exception`` net, a ``KeyError`` from doc2dict on undocumented HTML
    would collapse the whole task to ``FAILED`` and lose unrelated
    work.
    """

    def test_bare_exception_from_fetch_marks_filing_failed_and_continues(self) -> None:
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        f2 = _make_filing_info("AAPL", "0000320193-23-000002")
        p2 = _make_processed_filing(f2)

        fetcher = _StubFetcher(
            work_lists={("AAPL", "10-K"): [f1, f2]},
            # ``RuntimeError`` mimics the bare exception edgartools can
            # surface from undocumented EDGAR response shapes.
            raise_on_fetch={f1.accession_number: RuntimeError("edgar payload mangled")},
        )
        manager, store, _, _, _ = _build_manager(
            fetcher=fetcher,
            orchestrator=_StubOrchestrator(by_accession={f2.accession_number: p2}),
        )

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        # f1 marked failed via the bare-exception net, f2 succeeded.
        assert info.progress.filings_failed == 1
        assert store.stored == [f2.accession_number]
        # Whole task did NOT collapse to FAILED.
        assert info.state == TaskState.COMPLETED

    def test_bare_exception_from_process_marks_filing_failed_and_continues(self) -> None:
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        f2 = _make_filing_info("AAPL", "0000320193-23-000002")
        p2 = _make_processed_filing(f2)

        orchestrator = _StubOrchestrator(
            by_accession={f2.accession_number: p2},
            # Bare ``KeyError`` mimics doc2dict's tree-walker on a
            # filing whose section path map is missing an expected key.
            raise_on={f1.accession_number: KeyError("Part I > Item 1A")},
        )
        manager, store, _, _, _ = _build_manager(
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1, f2]}),
            orchestrator=orchestrator,
        )

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        assert info.progress.filings_failed == 1
        assert store.stored == [f2.accession_number]
        assert info.state == TaskState.COMPLETED

    def test_bare_exception_from_store_marks_filing_failed_and_continues(self) -> None:
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        f2 = _make_filing_info("AAPL", "0000320193-23-000002")
        p1 = _make_processed_filing(f1)
        p2 = _make_processed_filing(f2)

        store = _StubFilingStore(
            # Bare ``RuntimeError`` mimics a transient ChromaDB driver
            # glitch — the project normally surfaces it as
            # :class:`DatabaseError` but defence-in-depth catches the
            # untyped path.
            raise_on_store={f1.accession_number: RuntimeError("chroma backend glitch")},
        )
        manager, store, _, _, _ = _build_manager(
            filing_store=store,
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1, f2]}),
            orchestrator=_StubOrchestrator(
                by_accession={f1.accession_number: p1, f2.accession_number: p2},
            ),
        )

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        assert info.progress.filings_failed == 1
        # f2 still stored — the bare-exception net did not abort the loop.
        assert store.stored == [f2.accession_number]
        assert info.state == TaskState.COMPLETED


@pytest.mark.security
class TestPerFilingFailureWireDiscipline:
    """The bare-exception net must never leak class names / paths.

    The wire envelope on the WebSocket is a generic ``"<stage> failed"``
    string; the full traceback lands in the operator's log via
    ``logger.exception``. Mirrors the broader API discipline that
    internal exception text never reaches a user-visible payload.
    """

    def test_bare_exception_envelope_does_not_carry_internal_message(self) -> None:
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")

        # Sentinel string we never want to see on the wire.
        sentinel = "/internal/path/to/parser.py line 42 KeyError"
        orchestrator = _StubOrchestrator(
            by_accession={},
            raise_on={f1.accession_number: KeyError(sentinel)},
        )
        manager, _, _, _, _ = _build_manager(
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1]}),
            orchestrator=orchestrator,
        )

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        # Drain the WebSocket queue and assert the sentinel never
        # surfaces on any envelope.
        assert info._message_queue is not None
        envelopes: list[dict] = []
        while True:
            try:
                envelopes.append(info._message_queue.get_nowait())
            except Exception:
                break
        for envelope in envelopes:
            for value in envelope.values():
                assert sentinel not in str(value), (
                    "bare-exception net leaked internal exception text onto the wire"
                )


# ---------------------------------------------------------------------------
# Cancellation — auto-cancel by ``_duration_timer``
# ---------------------------------------------------------------------------


class TestDurationTimerAutoCancel:
    """``_duration_timer`` auto-cancel must roll back through ``FilingStore``.

    The auto-cancel routes through the same ``cancel_event`` path as a
    user cancellation, so the rollback contract (ChromaDB-first via
    :meth:`FilingStore.delete_filings_batch`) must apply identically.
    The test forces the policy by invoking ``_timeout_task`` directly
    rather than waiting on a real ``threading.Timer`` — the contract is
    on the rollback, not on the wall-clock.
    """

    def test_timer_auto_cancel_rolls_back_via_filing_store(self) -> None:
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        f2 = _make_filing_info("AAPL", "0000320193-23-000002")
        p1 = _make_processed_filing(f1)
        p2 = _make_processed_filing(f2)

        manager_holder: dict[str, TaskManager] = {}

        class _TimeoutOnSecond(_StubOrchestrator):
            """Trip ``_timeout_task`` mid-second-filing.

            Mirrors the production path: the duration timer fires from
            its own daemon thread, sets ``cancel_event`` via
            :meth:`TaskManager._timeout_task`, and the worker observes
            it on the next stage boundary.
            """

            def process_filing(self, filing_id, html_content, progress_callback=None):
                if filing_id.accession_number == f2.accession_number:
                    mgr = manager_holder["mgr"]
                    for task in mgr.list_tasks():
                        if task.state == TaskState.RUNNING:
                            # Drive the auto-cancel path through the
                            # public timer entry point — proves rollback
                            # consistency between user cancel and
                            # timer-driven cancel.
                            mgr._timeout_task(task)
                    if progress_callback is not None:
                        progress_callback("Embedding", 3, 4)
                return self.by_accession[filing_id.accession_number]

        manager, store, _, _, _ = _build_manager(
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1, f2]}),
            orchestrator=_TimeoutOnSecond(
                by_accession={f1.accession_number: p1, f2.accession_number: p2},
            ),
        )
        manager_holder["mgr"] = manager

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.CANCELLED)

        # f1 was stored, then rolled back via FilingStore — same path
        # as user cancel.
        assert store.stored == [f1.accession_number]
        assert store.deleted == [[f1.accession_number]]
        assert info._stored_accessions == []


# ---------------------------------------------------------------------------
# Cancellation — between-stage check
# ---------------------------------------------------------------------------


class TestCancelBetweenStages:
    """``cancel_event`` is observed at every stage boundary.

    A long-running fetch (EDGAR rate-limited at 9 req/s + network
    latency) can defer cancellation by many seconds without a check
    immediately after fetch. This test pins the post-fetch check.
    """

    def test_cancel_after_fetch_skips_processing_and_rolls_back(self) -> None:
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        f2 = _make_filing_info("AAPL", "0000320193-23-000002")
        p1 = _make_processed_filing(f1)
        p2 = _make_processed_filing(f2)

        manager_holder: dict[str, TaskManager] = {}

        # Cancel inside the fetch stub for f2 — by the time the worker
        # exits ``fetch_filing_content`` the cancel flag is set, and
        # the post-fetch check must catch it before processing /
        # storing f2.
        class _CancelInsideFetch(_StubFetcher):
            def fetch_filing_content(self, filing_info):
                if filing_info.accession_number == f2.accession_number:
                    mgr = manager_holder["mgr"]
                    for task in mgr.list_tasks():
                        if task.state == TaskState.RUNNING:
                            task.cancel_event.set()
                return super().fetch_filing_content(filing_info)

        manager, store, _, _, _ = _build_manager(
            fetcher=_CancelInsideFetch(work_lists={("AAPL", "10-K"): [f1, f2]}),
            orchestrator=_StubOrchestrator(
                by_accession={f1.accession_number: p1, f2.accession_number: p2},
            ),
        )
        manager_holder["mgr"] = manager

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.CANCELLED)

        # f1 stored then rolled back; f2 fetched but never stored
        # because the post-fetch cancel check fired first.
        assert store.stored == [f1.accession_number]
        assert store.deleted == [[f1.accession_number]]
        assert f2.accession_number not in store.stored
        assert info._stored_accessions == []


# ---------------------------------------------------------------------------
# Security — zero-key contract
# ---------------------------------------------------------------------------


_SECRET_FIELD_HINTS = (
    "api_key",
    "api-key",
    "apikey",
    "secret",
    "password",
    "credential",
    "private_key",
    "auth_token",
    "bearer",
)


@pytest.mark.security
class TestTaskInfoZeroKeyContract:
    """``TaskInfo`` carries no credential-shaped attribute name.

    Mirrors the domain-type security check pattern used elsewhere in
    the codebase. ``TaskInfo`` lives in :mod:`api.tasks` rather than
    :mod:`core.types`, so it needs an explicit test here.
    """

    @pytest.mark.parametrize("cls", [TaskInfo, TaskProgress, FilingResult])
    def test_no_secret_looking_fields(self, cls: type) -> None:
        for f in fields(cls):
            lowered = f.name.lower()
            for hint in _SECRET_FIELD_HINTS:
                assert hint not in lowered, (
                    f"{cls.__name__}.{f.name} looks credential-bearing; "
                    "TaskManager dataclasses must not carry secrets."
                )

    def test_task_info_does_not_expose_edgar_identity(self) -> None:
        """Legacy ``TaskInfo`` carried ``edgar_name`` / ``edgar_email``.

        Identity now lives on the manager-level resolver dict. Verify
        the legacy field names are gone so a careless re-introduction
        trips the test.
        """
        field_names = {f.name for f in fields(TaskInfo)}
        assert "edgar_name" not in field_names
        assert "edgar_email" not in field_names


@pytest.mark.security
class TestTaskHistoryNoKeys:
    """``save_task_history`` never receives a key argument from the worker.

    The ``_StubRegistry.save_task_history`` asserts this inline for
    every test that triggers a terminal transition. The explicit case
    here proves the inline check is actually enforcing something.
    """

    def test_completed_task_save_call_has_no_secret_kwargs(self) -> None:
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        p1 = _make_processed_filing(f1)
        manager, _, registry, _, _ = _build_manager(
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1]}),
            orchestrator=_StubOrchestrator(by_accession={f1.accession_number: p1}),
        )
        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        _wait_for_state(manager, task_id, target=TaskState.COMPLETED)
        assert registry.save_calls, "expected a save_task_history call on terminal transition"
        call = registry.save_calls[0]
        for forbidden in _SECRET_FIELD_HINTS:
            assert all(forbidden not in k.lower() for k in call)


# ---------------------------------------------------------------------------
# Post-ingest retention sweep
# ---------------------------------------------------------------------------


class TestPostIngestEviction:
    """``_execute`` triggers ``FilingStore.evict_expired`` after COMPLETED.

    The sweep is best-effort: failures must NOT shadow the worker's
    terminal-state transition.  ``DB_RETENTION_MAX_AGE_DAYS=0``
    short-circuits the call so Scenario A keeps the historic
    no-eviction behaviour.
    """

    def test_completed_task_triggers_eviction(self, monkeypatch) -> None:
        monkeypatch.setenv("DB_RETENTION_MAX_AGE_DAYS", "30")
        reload_settings()

        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        p1 = _make_processed_filing(f1)
        manager, store, _, _, _ = _build_manager(
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1]}),
            orchestrator=_StubOrchestrator(by_accession={f1.accession_number: p1}),
        )

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        # Hook fired exactly once, with the configured cutoff.
        assert store.evict_calls == [30]

    def test_retention_zero_skips_eviction(self, monkeypatch) -> None:
        """``DB_RETENTION_MAX_AGE_DAYS=0`` disables the hook entirely."""
        monkeypatch.setenv("DB_RETENTION_MAX_AGE_DAYS", "0")
        reload_settings()

        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        p1 = _make_processed_filing(f1)
        manager, store, _, _, _ = _build_manager(
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1]}),
            orchestrator=_StubOrchestrator(by_accession={f1.accession_number: p1}),
        )

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        assert store.evict_calls == []

    def test_failed_task_does_not_trigger_eviction(self, monkeypatch) -> None:
        """``FAILED`` short-circuits before the COMPLETED branch."""
        monkeypatch.setenv("DB_RETENTION_MAX_AGE_DAYS", "30")
        monkeypatch.setenv("DB_MAX_FILINGS", "1")
        reload_settings()

        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        registry = _StubRegistry(filing_count=1)  # already at the ceiling
        manager, store, _, _, _ = _build_manager(
            registry=registry,
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1]}),
            orchestrator=_StubOrchestrator(
                by_accession={f1.accession_number: _make_processed_filing(f1)},
            ),
        )

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        _wait_for_state(manager, task_id, target=TaskState.FAILED)

        # FAILED tasks return early from ``_execute`` — no post-ingest hook.
        assert store.evict_calls == []

    def test_cancelled_task_does_not_trigger_eviction(self, monkeypatch) -> None:
        """``CANCELLED`` short-circuits before the COMPLETED branch."""
        monkeypatch.setenv("DB_RETENTION_MAX_AGE_DAYS", "30")
        reload_settings()

        manager, store, _, _, _ = _build_manager()
        # Hold the GPU so the worker sees the cancel before fetching.
        manager._gpu_semaphore.acquire()
        try:
            task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
            assert manager.cancel_task(task_id) is True
        finally:
            manager._gpu_semaphore.release()

        _wait_for_state(manager, task_id, target=TaskState.CANCELLED)

        assert store.evict_calls == []

    def test_eviction_failure_does_not_shadow_completion(self, monkeypatch, caplog) -> None:
        """A ``DatabaseError`` from ``evict_expired`` is logged and swallowed."""
        monkeypatch.setenv("DB_RETENTION_MAX_AGE_DAYS", "30")
        reload_settings()

        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        p1 = _make_processed_filing(f1)
        failing_store = _StubFilingStore(
            raise_on_evict=DatabaseError("chroma down"),
        )
        manager, _, _, _, _ = _build_manager(
            filing_store=failing_store,
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1]}),
            orchestrator=_StubOrchestrator(by_accession={f1.accession_number: p1}),
        )

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        # Worker reached COMPLETED in spite of the eviction failure.
        assert info.state == TaskState.COMPLETED
        # Hook was attempted exactly once.
        assert failing_store.evict_calls == [30]


# ---------------------------------------------------------------------------
# Retention helper (audit-log + best-effort discipline)
# ---------------------------------------------------------------------------


class _EvictRecordingStore:
    """Minimal store stub for the standalone helper.

    Tests only need the ``evict_expired`` method — the helper does not
    touch the rest of the dual-store surface.
    """

    def __init__(
        self,
        *,
        report: EvictionReport | None = None,
        raise_with: Exception | None = None,
    ) -> None:
        self.calls: list[int] = []
        self.report = report or EvictionReport(filings_evicted=0, chunks_evicted=0, max_age_days=0)
        self.raise_with = raise_with

    def evict_expired(self, max_age_days: int) -> EvictionReport:
        self.calls.append(max_age_days)
        if self.raise_with is not None:
            raise self.raise_with
        return EvictionReport(
            filings_evicted=self.report.filings_evicted,
            chunks_evicted=self.report.chunks_evicted,
            max_age_days=max_age_days,
        )


class TestRunRetentionEvictionSafe:
    """Behavioural contract of :func:`run_retention_eviction_safe`.

    The helper is the single seam shared between the API lifespan
    startup and the post-ingest hook, so its discipline (disabled at
    ``<= 0``, swallows :class:`DatabaseError` / :class:`ValueError`,
    audit-log metadata-only) is asserted here once.
    """

    def test_zero_cutoff_skips_call(self) -> None:
        store = _EvictRecordingStore()
        run_retention_eviction_safe(store, 0, context_label="startup")
        assert store.calls == []

    def test_negative_cutoff_skips_call(self) -> None:
        store = _EvictRecordingStore()
        run_retention_eviction_safe(store, -7, context_label="startup")
        assert store.calls == []

    def test_positive_cutoff_calls_store(self) -> None:
        store = _EvictRecordingStore()
        run_retention_eviction_safe(store, 30, context_label="post_ingest")
        assert store.calls == [30]

    def test_database_error_is_swallowed(self, caplog) -> None:
        store = _EvictRecordingStore(
            raise_with=DatabaseError("chroma down"),
        )
        # Must not raise.
        run_retention_eviction_safe(store, 30, context_label="startup")
        assert store.calls == [30]

    def test_value_error_is_swallowed(self) -> None:
        """``FilingStore.evict_expired`` raises ``ValueError`` on ``<= 0``.

        The helper already guards the cutoff but defends against a
        future store-level invariant by catching ``ValueError`` too —
        the post-ingest hook must never propagate it into the worker.
        """
        store = _EvictRecordingStore(raise_with=ValueError("nope"))
        run_retention_eviction_safe(store, 30, context_label="post_ingest")
        assert store.calls == [30]


@pytest.mark.security
class TestRunRetentionEvictionAuditLog:
    """Audit-log discipline: never echo a ticker or accession.

    The helper logs metadata only — the count, chunk count, cutoff, and
    a non-PII context label.  Mirrors the ``ingest_task_created`` /
    ``ingest_task_cancelled`` audit lines and the broader logging policy
    used across the API.

    These tests attach a handler directly to ``sec_generative_search.api.tasks``
    rather than relying on ``caplog`` because :func:`configure_logging`
    sets ``propagate=False`` on the package logger; ``caplog`` (which
    attaches to root) would never see the records.  Pattern documented
    in the local logging-test setup.
    """

    @staticmethod
    def _capture_records():
        import logging as _logging

        records: list[_logging.LogRecord] = []

        class _CollectingHandler(_logging.Handler):
            def emit(self, record: _logging.LogRecord) -> None:
                records.append(record)

        handler = _CollectingHandler(level=_logging.DEBUG)
        pkg_logger = _logging.getLogger("sec_generative_search.api.tasks")
        previous_level = pkg_logger.level
        pkg_logger.addHandler(handler)
        pkg_logger.setLevel(_logging.DEBUG)
        return records, handler, pkg_logger, previous_level

    def test_success_log_carries_only_metadata(self) -> None:
        store = _EvictRecordingStore(
            report=EvictionReport(filings_evicted=3, chunks_evicted=42, max_age_days=30),
        )
        records, handler, pkg_logger, prev_level = self._capture_records()
        try:
            run_retention_eviction_safe(store, 30, context_label="startup")
        finally:
            pkg_logger.removeHandler(handler)
            pkg_logger.setLevel(prev_level)

        emitted = " ".join(rec.getMessage() for rec in records)
        # The helper never sees a ticker or an accession; defence-in-depth
        # check covers both shapes in case a future change leaks the
        # store's diagnostic detail into the log line.
        assert "AAPL" not in emitted
        assert "0000320193" not in emitted
        # The context label, count, and cutoff DO appear.
        assert "startup" in emitted
        assert "3 filing" in emitted

    def test_failure_log_carries_no_ticker(self) -> None:
        # An exception whose ``message`` could contain a ticker would
        # leak if we let it through verbatim. The helper logs
        # ``exc.message`` — the test is a tripwire: if a future change
        # drops the metadata discipline (e.g. logs ``repr(exc)``) it
        # will catch the regression.
        store = _EvictRecordingStore(
            raise_with=DatabaseError("backend transient failure"),
        )
        records, handler, pkg_logger, prev_level = self._capture_records()
        try:
            run_retention_eviction_safe(store, 30, context_label="post_ingest")
        finally:
            pkg_logger.removeHandler(handler)
            pkg_logger.setLevel(prev_level)

        emitted = " ".join(rec.getMessage() for rec in records)
        assert "AAPL" not in emitted
        assert "0000320193" not in emitted
        assert "post_ingest" in emitted


class TestLifespanWiresStartupEviction:
    """Tripwire: ``api.app.lifespan`` references the eviction helper.

    The production lifespan boots ChromaDB / SQLite / the embedder, so
    exercising it in a unit test is expensive and out of scope for the
    API test surface (see ``tests/api/conftest.py``).  This test
    inspects the lifespan's referenced names instead so a future
    regression that removes the startup hook is caught at unit-test
    time, not in a Scenario-B/C smoke run.
    """

    def test_lifespan_references_run_retention_eviction_safe(self) -> None:
        # The package's ``__init__`` re-exports a FastAPI instance named
        # ``app`` so ``from sec_generative_search.api import app`` would
        # shadow the module. Use ``importlib`` for a deterministic
        # module reference.
        import importlib
        import inspect

        app_module = importlib.import_module("sec_generative_search.api.app")
        tasks_module = importlib.import_module("sec_generative_search.api.tasks")

        # The import path is the load-bearing seam — both the helper
        # name and its module of origin must match so a future rename
        # cannot quietly drop the wiring.
        assert hasattr(app_module, "run_retention_eviction_safe")
        assert app_module.run_retention_eviction_safe is tasks_module.run_retention_eviction_safe

        # The lifespan source must call the helper with the
        # ``"startup"`` context label so its log line is grep-able and
        # distinct from the post-ingest line.
        source = inspect.getsource(app_module.lifespan)
        assert "run_retention_eviction_safe(" in source
        assert 'context_label="startup"' in source


# ---------------------------------------------------------------------------
# Caller-driven embedder idle-unload
# ---------------------------------------------------------------------------


class _StubEmbedder:
    """Minimal duck-type for the manager's ``maybe_unload`` hook.

    Records every invocation so the lifecycle-boundary tests can
    assert on call ordering. Mirrors the public surface of
    :class:`LocalEmbeddingProvider` that the manager actually touches —
    ``maybe_unload(now=None)`` only, no ``encode`` / ``unload`` because
    the manager never calls them directly.
    """

    def __init__(self, *, raise_on_unload: Exception | None = None) -> None:
        self.calls: int = 0
        self.raise_on_unload = raise_on_unload

    def maybe_unload(self, now: float | None = None) -> bool:
        self.calls += 1
        if self.raise_on_unload is not None:
            raise self.raise_on_unload
        return False


class _HostedEmbedderStub:
    """Stand-in for a hosted embedder (no idle state to release).

    Has no ``maybe_unload`` attribute — the manager's helper must
    duck-type around its absence rather than crash with
    ``AttributeError``.
    """

    def embed_query(self, text: str):
        raise NotImplementedError


class TestEmbedderIdleUnloadHook:
    """The manager fires ``maybe_unload`` at task lifecycle boundaries.

    Lifecycle boundaries:

        - Worker ``finally`` after the GPU semaphore release — every
          task transition (COMPLETED / FAILED / CANCELLED) reaches it.
        - Lazy-eviction sweep on every ``get_task`` / ``list_tasks`` /
          ``create_task`` — picks up the steady-state operator-poll
          path even when no new ingestion task arrives.

    The provider's :meth:`maybe_unload` honours its configured
    ``EMBEDDING_IDLE_TIMEOUT_MINUTES`` threshold internally; the
    manager's responsibility is solely to *call* the hook at the
    right boundary.
    """

    def test_completed_task_triggers_unload_hook(self) -> None:
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        p1 = _make_processed_filing(f1)
        embedder = _StubEmbedder()
        manager, _, _, _, _ = _build_manager(
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1]}),
            orchestrator=_StubOrchestrator(by_accession={f1.accession_number: p1}),
        )
        # Inject the embedder after construction so the existing
        # ``_build_manager`` helper signature stays minimal — the
        # manager-level attribute is the same seam the lifespan binds.
        manager._embedder = embedder

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        # At least one call from the worker ``finally``; lazy reads
        # may have added more (the get_task poll inside _wait_for_state
        # also fires the hook), so the lower bound is what we pin.
        assert embedder.calls >= 1

    def test_failed_task_still_triggers_unload_hook(self, monkeypatch) -> None:
        """``FAILED`` paths must release VRAM too — the worker exits via
        ``finally`` regardless of which terminal state it reached."""
        monkeypatch.setenv("DB_MAX_FILINGS", "1")
        reload_settings()

        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        registry = _StubRegistry(filing_count=1)  # at the ceiling
        embedder = _StubEmbedder()
        manager, _, _, _, _ = _build_manager(
            registry=registry,
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1]}),
            orchestrator=_StubOrchestrator(
                by_accession={f1.accession_number: _make_processed_filing(f1)},
            ),
        )
        manager._embedder = embedder

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        _wait_for_state(manager, task_id, target=TaskState.FAILED)

        assert embedder.calls >= 1

    def test_cancelled_task_still_triggers_unload_hook(self) -> None:
        """Cancellation in-flight reaches the same ``finally``."""
        embedder = _StubEmbedder()
        manager, _, _, _, _ = _build_manager()
        manager._embedder = embedder

        manager._gpu_semaphore.acquire()
        try:
            task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
            assert manager.cancel_task(task_id) is True
        finally:
            manager._gpu_semaphore.release()

        _wait_for_state(manager, task_id, target=TaskState.CANCELLED)

        assert embedder.calls >= 1

    def test_lazy_eviction_sweep_fires_unload_hook(self) -> None:
        """``_evict_stale_locked`` runs the hook on every read.

        Drives the operator-poll seam: in steady state, no new task is
        created, but a periodic ``GET /api/ingest/tasks`` poll still
        triggers the idle-unload check.  Pin the contract by calling
        ``list_tasks`` directly.
        """
        embedder = _StubEmbedder()
        manager, _, _, _, _ = _build_manager()
        manager._embedder = embedder

        # No task created — ``list_tasks`` exercises the lazy seam alone.
        manager.list_tasks()
        manager.get_task("nonexistent")
        manager.list_tasks_for_session(None)

        # All three reads run the eviction sweep, which now fires the
        # unload hook on each pass.
        assert embedder.calls >= 3

    def test_hosted_embedder_without_maybe_unload_is_noop(self) -> None:
        """Duck-typed: hosted embedder lacks ``maybe_unload`` and must not crash.

        OpenAI / Gemini / Anthropic embedders have no idle state to
        release — the helper must skip them silently instead of
        crashing with ``AttributeError`` on a hot read path.
        """
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        p1 = _make_processed_filing(f1)
        manager, _, _, _, _ = _build_manager(
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1]}),
            orchestrator=_StubOrchestrator(by_accession={f1.accession_number: p1}),
        )
        manager._embedder = _HostedEmbedderStub()

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        # Worker reached COMPLETED — the hook silently no-op'd on the
        # hosted embedder rather than collapsing the task.
        assert info.state == TaskState.COMPLETED

    def test_unload_failure_does_not_shadow_terminal_state(self, caplog) -> None:
        """A raising ``maybe_unload`` is logged and swallowed.

        Best-effort discipline mirrors the retention sweep: a transient
        cleanup glitch must NEVER shadow the worker's terminal-state
        transition or block a polling read.
        """
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        p1 = _make_processed_filing(f1)
        embedder = _StubEmbedder(raise_on_unload=RuntimeError("cuda blip"))
        manager, _, _, _, _ = _build_manager(
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1]}),
            orchestrator=_StubOrchestrator(by_accession={f1.accession_number: p1}),
        )
        manager._embedder = embedder

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        # COMPLETED reached even though the unload hook raised.
        assert info.state == TaskState.COMPLETED
        # The hook was attempted (the exception came from inside it).
        assert embedder.calls >= 1

    def test_no_embedder_bound_is_safe(self) -> None:
        """Manager constructed without an ``embedder`` is the test path.

        Existing tests use ``_build_manager`` which omits the embedder
        argument — the helper must not break that pattern.
        """
        f1 = _make_filing_info("AAPL", "0000320193-23-000001")
        p1 = _make_processed_filing(f1)
        manager, _, _, _, _ = _build_manager(
            fetcher=_StubFetcher(work_lists={("AAPL", "10-K"): [f1]}),
            orchestrator=_StubOrchestrator(by_accession={f1.accession_number: p1}),
        )
        # No ``manager._embedder = ...`` — explicit None on construction.

        task_id = manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        info = _wait_for_state(manager, task_id, target=TaskState.COMPLETED)

        assert info.state == TaskState.COMPLETED


class TestLifespanWiresEmbedderHook:
    """Tripwire: ``api.app.lifespan`` passes the embedder to TaskManager.

    The helper hook is only useful if the lifespan actually wires the
    embedder reference through. Inspect the lifespan source rather
    than booting it (the production path pulls in ChromaDB / SQLite /
    sentence-transformers) — same pattern as the retention-eviction
    tripwire above.
    """

    def test_lifespan_passes_embedder_kwarg_to_task_manager(self) -> None:
        import importlib
        import inspect

        app_module = importlib.import_module("sec_generative_search.api.app")
        source = inspect.getsource(app_module.lifespan)
        # The keyword binding must be present so a future refactor
        # that drops it surfaces here, not in a Scenario-A smoke run.
        assert "TaskManager(" in source
        assert "embedder=embedder" in source
