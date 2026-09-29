"""Tests for ``POST /api/admin/demo-reset`` (OPTIMIZATIONS F27).

The scheduled demo-corpus reset moved from a second process (a Cloud Run
Job running ``sec-rag manage clear -y`` on the shared Cloud Storage FUSE
volume, which has no file locking) into the API — the single writer.  Its
access model is a reviewed exception to rule A: a dedicated token, never the
API / admin key.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.api.tasks import IngestActiveError, IngestPausedError, TaskManager
from sec_generative_search.core.exceptions import DatabaseError

_TOKEN = "demo-reset-token-SENTINEL-0123456789abcdef"  # pragma: allowlist secret
_WRONG = "demo-reset-token-WRONG-0123456789abcdefghi"  # pragma: allowlist secret
_API_KEY = "configured-api-key-for-demo-tests"  # pragma: allowlist secret
_ADMIN_KEY = "configured-admin-key-for-demo-tests"  # pragma: allowlist secret
_PATH = "/api/admin/demo-reset"
_HEADER = "X-Demo-Reset-Token"


@dataclass
class _StubStore:
    cleared: int = 0
    raises: Exception | None = None

    def clear_all(self) -> tuple[int, int]:
        if self.raises is not None:
            raise self.raises
        self.cleared += 1
        return 1234, 7


@dataclass
class _StubManager:
    """Stands in for TaskManager's maintenance seam."""

    active: int = 0
    entered: list[bool] = field(default_factory=list)

    def exclusive_maintenance(self):
        manager = self

        class _Section:
            def __enter__(self) -> None:
                if manager.active:
                    raise IngestActiveError(manager.active)
                manager.entered.append(True)

            def __exit__(self, *_exc: object) -> None:
                return None

        return _Section()


@pytest.fixture
def reset_client(api_client_factory):
    def _build(
        *, store: _StubStore | None = None, manager: _StubManager | None = None, **env: str
    ) -> tuple[TestClient, _StubStore, _StubManager]:
        env = {"API_DEMO_MODE": "true", "API_DEMO_RESET_TOKEN": _TOKEN, **env}
        client = api_client_factory(**env)
        store = store or _StubStore()
        manager = manager or _StubManager()
        client.app.state.filing_store = store  # type: ignore[attr-defined]
        client.app.state.task_manager = manager  # type: ignore[attr-defined]
        return client, store, manager

    return _build


@pytest.mark.security
class TestTokenGate:
    def test_valid_token_resets_the_corpus(self, reset_client) -> None:
        client, store, manager = reset_client()
        response = client.post(_PATH, headers={_HEADER: _TOKEN})
        assert response.status_code == 200
        assert response.json() == {"reset": True, "filings_removed": 7, "chunks_removed": 1234}
        assert response.headers["cache-control"] == "no-store"
        assert store.cleared == 1 and manager.entered == [True]

    @pytest.mark.parametrize(
        "headers",
        [{}, {_HEADER: _WRONG}, {_HEADER: ""}, {_HEADER: _TOKEN[:-1]}, {_HEADER: b"\xe9" * 40}],
        ids=["missing", "wrong", "empty", "prefix", "non-ascii"],
    )
    def test_anything_but_the_token_is_401_and_clears_nothing(self, reset_client, headers) -> None:
        client, store, _ = reset_client()
        response = client.post(_PATH, headers=headers)
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"
        assert store.cleared == 0

    def test_no_configured_token_is_indistinguishable(self, api_client_factory) -> None:
        """Unconfigured: every request is the same 401 — the route does not
        reveal whether the deployment has a reset token."""
        client = api_client_factory(API_DEMO_MODE="true")
        client.app.state.filing_store = _StubStore()  # type: ignore[attr-defined]
        for headers in ({}, {_HEADER: _TOKEN}):
            response = client.post(_PATH, headers=headers)
            assert response.status_code == 401
            assert response.json()["error"] == "unauthorised"

    def test_api_and_admin_keys_do_not_open_it(self, reset_client) -> None:
        """Least privilege both ways: the admin credentials cannot reset the
        demo, and the token is not an admin credential."""
        client, store, _ = reset_client(API_KEY=_API_KEY, API_ADMIN_KEY=_ADMIN_KEY)
        response = client.post(_PATH, headers={"X-API-Key": _API_KEY, "X-Admin-Key": _ADMIN_KEY})
        assert response.status_code == 401
        assert store.cleared == 0
        # …and the token does not pass the admin tier elsewhere.
        other = client.delete("/api/filings/0000320193-23-000077", headers={_HEADER: _TOKEN})
        assert other.status_code == 401

    def test_token_is_compared_in_constant_time(self, reset_client, monkeypatch) -> None:
        import sec_generative_search.api.routes.demo as demo_module

        seen: list[object] = []
        real = demo_module.secure_compare

        def _spy(a: object, b: object) -> bool:
            seen.append(a)
            return real(a, b)

        monkeypatch.setattr(demo_module, "secure_compare", _spy)
        client, store, _ = reset_client()
        assert client.post(_PATH, headers={_HEADER: _TOKEN}).status_code == 200
        assert seen == [_TOKEN] and store.cleared == 1

    def test_neither_token_is_echoed(self, reset_client) -> None:
        client, _, _ = reset_client()
        bodies = [
            client.post(_PATH, headers={_HEADER: _WRONG}).text,
            client.post(_PATH, headers={_HEADER: _TOKEN}).text,
        ]
        joined = "\n".join(bodies)
        assert _TOKEN not in joined and _WRONG not in joined

    def test_rides_the_delete_rate_bucket(self) -> None:
        from sec_generative_search.api.policies import resolve_policy

        policy = resolve_policy(_PATH, "POST")
        assert policy.rate_category == "delete"
        assert policy.max_body_bytes <= 1024

    def test_header_is_redacted_from_access_logs(self) -> None:
        from sec_generative_search.api.access_log import redact_header_value

        assert _TOKEN not in redact_header_value(_HEADER, _TOKEN)
        assert _TOKEN not in redact_header_value(_HEADER.lower(), _TOKEN)


class TestResetOutcomes:
    def test_active_ingest_is_409_and_clears_nothing(self, reset_client) -> None:
        client, store, _ = reset_client(manager=_StubManager(active=2))
        response = client.post(_PATH, headers={_HEADER: _TOKEN})
        assert response.status_code == 409
        body = response.json()
        assert body["error"] == "ingest_in_progress"
        assert body["details"] == {"active": 2}
        assert store.cleared == 0

    def test_storage_failure_is_the_fixed_500(self, reset_client) -> None:
        store = _StubStore(raises=DatabaseError("boom", details="/app/data/metadata.sqlite"))
        client, _, _ = reset_client(store=store)
        response = client.post(_PATH, headers={_HEADER: _TOKEN})
        assert response.status_code == 500
        assert response.json()["error"] == "database_error"
        assert "metadata.sqlite" not in response.text


def _manager() -> TaskManager:
    return TaskManager(
        filing_store=object(),  # type: ignore[arg-type]
        registry=object(),  # type: ignore[arg-type]
        fetcher=object(),  # type: ignore[arg-type]
        orchestrator=object(),  # type: ignore[arg-type]
    )


@pytest.mark.security
class TestExclusiveMaintenance:
    """The F27 invariant on the API side: while the reset clears the store,
    no ingest task can start writing into it."""

    def test_create_task_is_refused_during_the_reset(self) -> None:
        manager = _manager()
        with manager.exclusive_maintenance(), pytest.raises(IngestPausedError):
            manager.create_task(tickers=["AAPL"], form_types=["10-K"])
        assert manager._tasks == {}

    def test_a_task_racing_the_reset_never_starts_inside_it(self) -> None:
        manager = _manager()
        inside = threading.Event()
        release = threading.Event()
        outcome: list[str] = []

        def _reset() -> None:
            with manager.exclusive_maintenance():
                inside.set()
                release.wait(timeout=5.0)

        resetter = threading.Thread(target=_reset, daemon=True)
        resetter.start()
        assert inside.wait(timeout=5.0)
        try:
            manager.create_task(tickers=["AAPL"], form_types=["10-K"])
            outcome.append("created")
        except IngestPausedError:
            outcome.append("paused")
        release.set()
        resetter.join(timeout=5.0)
        assert outcome == ["paused"]
        assert manager._maintenance is False

    def test_active_task_blocks_the_reset(self) -> None:
        manager = _manager()
        manager._gpu_semaphore.acquire()  # keep the task PENDING, never running
        try:
            manager.create_task(tickers=["AAPL"], form_types=["10-K"])
            with pytest.raises(IngestActiveError), manager.exclusive_maintenance():
                pytest.fail("entered maintenance with a pending task")
        finally:
            for info in manager._tasks.values():
                info.cancel_event.set()
            manager._gpu_semaphore.release()

    def test_concurrent_resets_are_refused(self) -> None:
        manager = _manager()
        with manager.exclusive_maintenance():
            with pytest.raises(IngestActiveError), manager.exclusive_maintenance():
                pass
            assert manager._maintenance is True
        assert manager._maintenance is False

    def test_flag_clears_when_the_reset_fails(self) -> None:
        manager = _manager()
        with pytest.raises(RuntimeError), manager.exclusive_maintenance():
            raise RuntimeError("clear failed")
        assert manager._maintenance is False
