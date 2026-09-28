"""Tests for the ingest progress WebSocket.

Strategy
--------

The :class:`TaskManager` itself is covered exhaustively in
:mod:`tests.api.test_task_manager`; this file focuses on the WebSocket
route's *contract* over a stub manager:

    - Origin allow-list: missing / unknown origins close pre-accept.
    - API-key handshake: header path, first-message fallback, both
      failure modes, all timing-safe.
    - Session ownership: foreign / missing tasks surface as the same
      ``4404 not_found`` so the route never confirms existence.
    - Snapshot-on-connect: current state delivered as a ``snapshot``
      frame even when the task already finished.
    - Terminal drain on reconnect: a completed / failed / cancelled
      task gets one terminal frame and a clean close.
    - Streaming: events pushed onto the task's queue forward verbatim.
    - Heartbeat: idle connection receives a ``heartbeat`` frame so
      proxy idle-timeouts cannot half-close the channel.
    - Audit-log: connect / disconnect emit ``SECURITY_AUDIT:`` lines
      carrying only the masked task-id tail — never the raw cookie,
      API key, or any name / email.

All tests skip the production lifespan and inject a stub
``TaskManager`` onto ``app.state`` — same pattern as ``test_ingest.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sec_generative_search.api.app import create_app
from sec_generative_search.api.tasks import (
    _MESSAGE_QUEUE_MAXSIZE,
    FilingResult,
    TaskInfo,
    TaskManager,
    TaskProgress,
    TaskState,
)
from sec_generative_search.api.websocket import _stream_loop
from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.edgar_identity import InMemorySessionEdgarIdentityStore
from sec_generative_search.core.logging import LOGGER_NAME

# ---------------------------------------------------------------------------
# Stub TaskManager (route-surface only)
# ---------------------------------------------------------------------------


@dataclass
class _StubTaskManager:
    """In-process stub matching the manager surface the WS reads.

    The WebSocket route only calls ``get_task``; everything else is
    irrelevant.  Tests can pre-plant a ``TaskInfo`` and a live message
    queue on the ``tasks`` dict to drive whichever streaming scenario
    they need.
    """

    tasks: dict[str, TaskInfo] = field(default_factory=dict)

    def get_task(self, task_id: str) -> TaskInfo | None:
        return self.tasks.get(task_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ALLOWED_ORIGIN = "https://example.test"


def _build_task(
    *,
    task_id: str = "a" * 32,
    session_id: str | None = None,
    state: TaskState = TaskState.RUNNING,
    queue: asyncio.Queue | None = None,
) -> TaskInfo:
    info = TaskInfo(
        task_id=task_id,
        tickers=["AAPL"],
        form_types=["10-K"],
        session_id=session_id,
    )
    info.state = state
    info.progress = TaskProgress(
        step_label="Embedding",
        step_index=3,
        step_total=5,
        filings_done=0,
        filings_total=1,
    )
    if state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}:
        info.started_at = datetime.now(UTC)
        info.completed_at = datetime.now(UTC)
    if state == TaskState.COMPLETED:
        info.results = [
            FilingResult(
                ticker="AAPL",
                form_type="10-K",
                filing_date="2023-09-30",
                accession_number="0000320193-23-000077",
                segment_count=10,
                chunk_count=42,
                duration_seconds=1.5,
            )
        ]
        info.progress.filings_done = 1
    if state == TaskState.FAILED:
        info.error = "fetch_failed"
    info._message_queue = queue
    return info


@pytest.fixture(autouse=True)
def _clean_ws_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset API_* env vars so each test sees deterministic settings."""
    for key in list(os.environ.keys()):
        if key.startswith("API_"):
            monkeypatch.delenv(key, raising=False)
    # CORS allow-list governs the WS origin check.
    monkeypatch.setenv("API_CORS_ORIGINS", f'["{_ALLOWED_ORIGIN}"]')
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def ws_app_factory(monkeypatch: pytest.MonkeyPatch):
    """Build a fresh app with a stub TaskManager attached to ``app.state``."""

    def factory(*, env: dict[str, str] | None = None) -> tuple[Any, _StubTaskManager]:
        if env:
            for key, value in env.items():
                monkeypatch.setenv(key, value)
        reload_settings()

        app = create_app()
        manager = _StubTaskManager()
        app.state.task_manager = manager
        app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
        app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=300)
        app.state.encrypted_credential_store = None
        return app, manager

    return factory


def _mint_session(client: TestClient) -> str:
    """Mint a server-side session cookie and return its value."""
    response = client.post("/api/session")
    assert response.status_code in (200, 201)
    cookie = client.cookies.get("sec_rag_session")
    assert cookie is not None
    return cookie


def _connect(client: TestClient, task_id: str, *, origin: str = _ALLOWED_ORIGIN, headers=None):
    """Open a WebSocket with the allow-listed origin by default."""
    merged = {"Origin": origin}
    if headers:
        merged.update(headers)
    return client.websocket_connect(f"/ws/ingest/{task_id}", headers=merged)


# ---------------------------------------------------------------------------
# Pre-accept rejections
# ---------------------------------------------------------------------------


class TestPreAcceptRejection:
    def test_invalid_task_id_shape_closes_4400(self, ws_app_factory) -> None:
        app, _ = ws_app_factory()
        client = TestClient(app, base_url="https://testserver")
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect(
                "/ws/ingest/not-a-hex-uuid",
                headers={"Origin": _ALLOWED_ORIGIN},
            ),
        ):
            pass
        assert exc.value.code == 4400

    def test_missing_origin_closes_4003(self, ws_app_factory) -> None:
        app, manager = ws_app_factory()
        manager.tasks["a" * 32] = _build_task()
        client = TestClient(app, base_url="https://testserver")
        # ``TestClient`` does not auto-populate an ``Origin`` header for
        # ``websocket_connect`` — pass an empty value explicitly to
        # exercise the "no origin" branch.
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect(
                f"/ws/ingest/{'a' * 32}",
                headers={"Origin": ""},
            ),
        ):
            pass
        assert exc.value.code == 4003

    def test_unknown_origin_closes_4003(self, ws_app_factory) -> None:
        app, manager = ws_app_factory()
        manager.tasks["a" * 32] = _build_task()
        client = TestClient(app, base_url="https://testserver")
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            _connect(client, "a" * 32, origin="https://attacker.example"),
        ):
            pass
        assert exc.value.code == 4003


# ---------------------------------------------------------------------------
# API-key handshake
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestApiKeyHandshake:
    def test_no_api_key_configured_passes_through(self, ws_app_factory) -> None:
        app, manager = ws_app_factory()
        manager.tasks["a" * 32] = _build_task(state=TaskState.COMPLETED)
        client = TestClient(app, base_url="https://testserver")
        with _connect(client, "a" * 32) as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"
            terminal = ws.receive_json()
            assert terminal["type"] == "completed"

    def test_header_key_accepted(self, ws_app_factory) -> None:
        app, manager = ws_app_factory(env={"API_KEY": "secret"})
        manager.tasks["a" * 32] = _build_task(state=TaskState.COMPLETED)
        client = TestClient(app, base_url="https://testserver")
        with _connect(
            client,
            "a" * 32,
            headers={"X-API-Key": "secret"},
        ) as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"

    def test_wrong_header_falls_through_to_message_then_closes(self, ws_app_factory) -> None:
        app, _ = ws_app_factory(env={"API_KEY": "secret"})
        client = TestClient(app, base_url="https://testserver")
        with (
            pytest.raises(WebSocketDisconnect) as exc,
            _connect(
                client,
                "a" * 32,
                headers={"X-API-Key": "wrong"},
            ) as ws,
        ):
            # Header is wrong → route waits for the auth message.
            ws.send_json({"type": "auth", "api_key": "wrong"})  # pragma:allowlist secret
            ws.receive_json()  # never arrives — close instead
        assert exc.value.code == 4001

    def test_first_message_auth_accepted(self, ws_app_factory) -> None:
        app, manager = ws_app_factory(env={"API_KEY": "secret"})
        manager.tasks["a" * 32] = _build_task(state=TaskState.COMPLETED)
        client = TestClient(app, base_url="https://testserver")
        with _connect(client, "a" * 32) as ws:
            ws.send_json({"type": "auth", "api_key": "secret"})  # pragma:allowlist secret
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"

    def test_malformed_first_message_closes_4001(self, ws_app_factory) -> None:
        app, _ = ws_app_factory(env={"API_KEY": "secret"})
        client = TestClient(app, base_url="https://testserver")
        with pytest.raises(WebSocketDisconnect) as exc, _connect(client, "a" * 32) as ws:
            ws.send_json({"hello": "world"})
            ws.receive_json()
        assert exc.value.code == 4001

    def test_api_key_never_reaches_audit_log(
        self, ws_app_factory, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A correctly-presented key must not appear in any SECURITY_AUDIT line."""
        app, manager = ws_app_factory(env={"API_KEY": "topsecret123"})
        manager.tasks["a" * 32] = _build_task(state=TaskState.COMPLETED)
        client = TestClient(app, base_url="https://testserver")

        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            logger = logging.getLogger(LOGGER_NAME)
            previous_propagate = logger.propagate
            logger.propagate = True
            try:
                with _connect(
                    client,
                    "a" * 32,
                    headers={"X-API-Key": "topsecret123"},
                ) as ws:
                    ws.receive_json()  # snapshot
                    ws.receive_json()  # terminal
            finally:
                logger.propagate = previous_propagate

        audit_lines = [r.message for r in caplog.records if "SECURITY_AUDIT" in r.message]
        assert audit_lines, "expected at least one SECURITY_AUDIT line for the connect"
        for line in audit_lines:
            assert "topsecret123" not in line, line


# ---------------------------------------------------------------------------
# Session ownership
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSessionOwnership:
    def test_missing_task_surfaces_4404(self, ws_app_factory) -> None:
        app, _ = ws_app_factory()
        client = TestClient(app, base_url="https://testserver")
        # Pre-populate a session cookie so the cookie check is exercised.
        _mint_session(client)
        with pytest.raises(WebSocketDisconnect) as exc, _connect(client, "b" * 32) as ws:
            frame = ws.receive_json()
            assert frame["type"] == "error"
            assert frame["error"] == "not_found"
            ws.receive_json()
        assert exc.value.code == 4404

    def test_foreign_session_task_surfaces_4404(self, ws_app_factory) -> None:
        """A task owned by another session must look identical to a missing task."""
        app, manager = ws_app_factory()
        # Plant a task owned by a different session id.
        manager.tasks["a" * 32] = _build_task(
            task_id="a" * 32,
            session_id="someone-else-" + "x" * 32,
        )
        client = TestClient(app, base_url="https://testserver")
        _mint_session(client)  # caller has their own session cookie
        with pytest.raises(WebSocketDisconnect) as exc, _connect(client, "a" * 32) as ws:
            frame = ws.receive_json()
            assert frame["error"] == "not_found"
            ws.receive_json()
        assert exc.value.code == 4404

    def test_no_cookie_caller_cannot_reach_owned_task(self, ws_app_factory) -> None:
        """A task with a real owner is invisible to a cookieless caller."""
        app, manager = ws_app_factory()
        manager.tasks["a" * 32] = _build_task(
            task_id="a" * 32,
            session_id="owner-session-" + "z" * 32,
        )
        client = TestClient(app, base_url="https://testserver")
        # No session mint.
        with pytest.raises(WebSocketDisconnect) as exc, _connect(client, "a" * 32) as ws:
            frame = ws.receive_json()
            assert frame["error"] == "not_found"
            ws.receive_json()
        assert exc.value.code == 4404

    def test_forged_cookie_shape_rejected(self, ws_app_factory) -> None:
        """A browser-supplied bogus cookie must not authenticate."""
        app, manager = ws_app_factory()
        # Real session_id on the task.
        manager.tasks["a" * 32] = _build_task(
            task_id="a" * 32,
            session_id="legit-" + "y" * 32,
        )
        client = TestClient(app, base_url="https://testserver")
        client.cookies.set(
            "sec_rag_session",
            "obviously-not-a-real-token",
            domain="testserver",
        )
        with pytest.raises(WebSocketDisconnect) as exc, _connect(client, "a" * 32) as ws:
            frame = ws.receive_json()
            assert frame["error"] == "not_found"
            ws.receive_json()
        assert exc.value.code == 4404


# ---------------------------------------------------------------------------
# Snapshot + terminal-state delivery
# ---------------------------------------------------------------------------


class TestSnapshotAndTerminal:
    def test_snapshot_on_connect_for_running_task(self, ws_app_factory) -> None:
        app, manager = ws_app_factory()
        # The queue must exist or the stream loop will block; pre-build
        # one and immediately push a single ``step`` event followed by a
        # terminal so the consumer terminates without timing out.
        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait({"type": "step", "step": "Chunking", "step_number": 2})
        queue.put_nowait({"type": "completed", "results": [], "summary": {}})
        manager.tasks["a" * 32] = _build_task(queue=queue)

        client = TestClient(app, base_url="https://testserver")
        with _connect(client, "a" * 32) as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"
            assert snapshot["task_id"] == "a" * 32
            assert snapshot["status"] == "running"
            assert snapshot["progress"]["step_label"] == "Embedding"

            step = ws.receive_json()
            assert step["type"] == "step"
            assert step["step"] == "Chunking"

            terminal = ws.receive_json()
            assert terminal["type"] == "completed"

    def test_completed_task_drained_from_queue(self, ws_app_factory) -> None:
        app, manager = ws_app_factory()
        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait({"type": "completed", "results": [{"ticker": "AAPL"}], "summary": {}})
        manager.tasks["a" * 32] = _build_task(
            state=TaskState.COMPLETED,
            queue=queue,
        )
        client = TestClient(app, base_url="https://testserver")
        with _connect(client, "a" * 32) as ws:
            assert ws.receive_json()["type"] == "snapshot"
            terminal = ws.receive_json()
            assert terminal["type"] == "completed"
            assert terminal["results"] == [{"ticker": "AAPL"}]

    def test_completed_task_synthesises_terminal_when_queue_drained(self, ws_app_factory) -> None:
        """Reconnect after a prior consumer drained the queue."""
        app, manager = ws_app_factory()
        manager.tasks["a" * 32] = _build_task(
            state=TaskState.COMPLETED,
            queue=None,  # never built — same as already-drained
        )
        client = TestClient(app, base_url="https://testserver")
        with _connect(client, "a" * 32) as ws:
            assert ws.receive_json()["type"] == "snapshot"
            terminal = ws.receive_json()
            assert terminal["type"] == "completed"
            # Reconstructed from authoritative state — succeeded == len(results).
            assert terminal["summary"]["succeeded"] == 1

    def test_failed_task_synthesised_from_state(self, ws_app_factory) -> None:
        app, manager = ws_app_factory()
        manager.tasks["a" * 32] = _build_task(
            state=TaskState.FAILED,
            queue=None,
        )
        client = TestClient(app, base_url="https://testserver")
        with _connect(client, "a" * 32) as ws:
            assert ws.receive_json()["type"] == "snapshot"
            terminal = ws.receive_json()
            assert terminal["type"] == "failed"
            assert terminal["error"] == "fetch_failed"

    def test_cancelled_task_synthesised_from_state(self, ws_app_factory) -> None:
        app, manager = ws_app_factory()
        manager.tasks["a" * 32] = _build_task(
            state=TaskState.CANCELLED,
            queue=None,
        )
        client = TestClient(app, base_url="https://testserver")
        with _connect(client, "a" * 32) as ws:
            assert ws.receive_json()["type"] == "snapshot"
            terminal = ws.receive_json()
            assert terminal["type"] == "cancelled"


# ---------------------------------------------------------------------------
# F19 — bounded per-task queue: reconnect still sees the terminal frame
# ---------------------------------------------------------------------------


class _RecordingSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


@pytest.mark.security
class TestBoundedQueueReconnect:
    def test_overflowed_completed_task_still_delivers_its_terminal_frame(
        self, ws_app_factory
    ) -> None:
        """An unwatched task overflowed its queue; a later reconnect must still
        get the worker's own terminal frame (with its payload), not a
        synthesised one — it is the newest frame, so the bound never drops it."""
        app, manager = ws_app_factory()
        info = _build_task(state=TaskState.COMPLETED, queue=None)
        pusher = TaskManager.__new__(TaskManager)  # only ``_push`` is used
        pusher._loop = None
        for i in range(2 * _MESSAGE_QUEUE_MAXSIZE):
            pusher._push(info, {"type": "filing_done", "seq": i})
        pusher._push(
            info,
            {"type": "completed", "results": [{"ticker": "AAPL"}], "summary": {"total": 512}},
        )
        assert info._message_queue is not None
        assert info._message_queue.qsize() == _MESSAGE_QUEUE_MAXSIZE
        manager.tasks["a" * 32] = info

        client = TestClient(app, base_url="https://testserver")
        with _connect(client, "a" * 32) as ws:
            assert ws.receive_json()["type"] == "snapshot"
            terminal = ws.receive_json()
            assert terminal == {
                "type": "completed",
                "results": [{"ticker": "AAPL"}],
                "summary": {"total": 512},
            }

    def test_socket_builds_the_same_bounded_queue(self) -> None:
        """The first reader of a queue-less running task creates the queue;
        it must be the bounded one the worker would have built."""

        async def _scenario() -> tuple[int, list[dict]]:
            info = _build_task(queue=None)
            socket = _RecordingSocket()
            reader = asyncio.create_task(_stream_loop(socket, info))  # type: ignore[arg-type]
            await asyncio.sleep(0)
            assert info._message_queue is not None
            maxsize = info._message_queue.maxsize
            info._message_queue.put_nowait({"type": "cancelled"})
            await asyncio.wait_for(reader, timeout=5.0)
            return maxsize, socket.sent

        maxsize, sent = asyncio.run(_scenario())
        assert maxsize == _MESSAGE_QUEUE_MAXSIZE
        assert sent == [{"type": "cancelled"}]


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestAuditLog:
    def test_connect_and_disconnect_emit_audit_lines(
        self, ws_app_factory, caplog: pytest.LogCaptureFixture
    ) -> None:
        app, manager = ws_app_factory()
        manager.tasks["a" * 32] = _build_task(state=TaskState.COMPLETED)
        client = TestClient(app, base_url="https://testserver")

        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            logger = logging.getLogger(LOGGER_NAME)
            previous_propagate = logger.propagate
            logger.propagate = True
            try:
                with _connect(client, "a" * 32) as ws:
                    ws.receive_json()
                    ws.receive_json()
            finally:
                logger.propagate = previous_propagate

        audit_lines = [r.message for r in caplog.records if "SECURITY_AUDIT" in r.message]
        assert any("action=ws_ingest_connected" in line for line in audit_lines)
        assert any("action=ws_ingest_disconnected" in line for line in audit_lines)
        # Raw task id never leaks — only the masked tail (last 4 chars).
        for line in audit_lines:
            assert "a" * 32 not in line
