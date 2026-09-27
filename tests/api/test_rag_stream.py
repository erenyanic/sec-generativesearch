"""Tests for the streaming RAG generation route.

Strategy
--------

Like :mod:`tests.api.test_rag_query`, the route is exercised through
hermetic in-process stubs — :func:`build_llm_provider` and the
:class:`RAGOrchestrator` factory are monkey-patched so no real LLM,
ChromaDB, or embedder is required.  The orchestrator stub yields a
controllable list of :class:`StreamEvent` instances so the test can
shape the stream's deltas + final + error transitions deterministically.

Coverage focuses on:

    - SSE framing (``event: <name>\\ndata: <json>\\n\\n``);
    - happy-path event ordering: ``delta`` * N → ``citation`` * M →
      ``final``;
    - heartbeat emission on inter-event gaps (forced via a tiny
      monkey-patched heartbeat interval);
    - response headers (``text/event-stream``, ``Cache-Control: no-cache``,
      ``X-Accel-Buffering: no``);
    - pre-stream errors map to HTTP envelopes (unknown provider,
      missing key) — they MUST NOT open the SSE response;
    - in-stream errors map to ``error`` SSE events (not HTTP errors);
    - read-tier auth gate (admin-key NOT required);
    - rate-limit classification under the ``rag`` bucket;
    - audit-log discipline: ``rag_stream`` + ``rag_stream_completed``
      lines carry metadata only — never the raw query.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from sec_generative_search.api.app import create_app
from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.edgar_identity import InMemorySessionEdgarIdentityStore
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    GenerationError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)
from sec_generative_search.core.logging import LOGGER_NAME
from sec_generative_search.core.types import (
    Citation,
    FilingIdentifier,
    GenerationResult,
    TokenUsage,
)
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.orchestrator import StreamEvent
from sec_generative_search.rag.query_understanding import QueryPlan

# ---------------------------------------------------------------------------
# In-process stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubLLM:
    provider_name: str = "openai"
    closed: bool = False

    def close(self) -> None:
        self.closed = True


@dataclass
class _StubBuildRecorder:
    """Captures ``build_llm_provider`` calls + the resolved key."""

    raise_with: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        provider_name: str,
        *,
        api_key_resolver: Any,
    ) -> _StubLLM:
        resolved = api_key_resolver(provider_name)
        self.calls.append({"provider": provider_name, "resolved_key": resolved})
        if self.raise_with is not None:
            raise self.raise_with
        return _StubLLM(provider_name=provider_name)


@dataclass
class _StubOrchestrator:
    """Stand-in for :class:`RAGOrchestrator`.

    ``events`` is the canned sequence the stub yields (deltas + an
    optional terminal final).  ``raise_after`` is raised after the listed
    events are exhausted, mimicking an LLM that fails mid-stream.
    """

    retrieval: Any = None
    llm: Any = None
    events: list[StreamEvent] = field(default_factory=list)
    raise_after: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate_stream(
        self,
        plan: QueryPlan,
        *,
        mode: AnswerMode | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        history: Any = None,
        prefer_structured_output: bool = False,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        self.calls.append(
            {
                "plan_raw_query": plan.raw_query,
                "plan_query_en": plan.query_en,
                "plan_tickers": list(plan.tickers),
                "mode": mode,
                "model": model,
                "max_output_tokens": max_output_tokens,
                "history": history,
                "prefer_structured_output": prefer_structured_output,
                "routing_hints": kwargs.get("routing_hints"),
                "max_per_section": kwargs.get("max_per_section"),
                "max_per_filing": kwargs.get("max_per_filing"),
                "rerank_over_fetch_factor": kwargs.get("rerank_over_fetch_factor"),
                "llm_provider_name": getattr(self.llm, "provider_name", None),
            }
        )
        yield from self.events
        if self.raise_after is not None:
            raise self.raise_after


_LATEST_STUB: _StubOrchestrator | None = None


def _orchestrator_factory(*, retrieval: Any, llm: Any, **_: Any) -> _StubOrchestrator:
    assert _LATEST_STUB is not None, "test forgot to install a stub orchestrator"
    _LATEST_STUB.retrieval = retrieval
    _LATEST_STUB.llm = llm
    return _LATEST_STUB


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_rag_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in list(os.environ.keys()):
        if key.startswith(("API_", "LLM_")):
            monkeypatch.delenv(key, raising=False)
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def rag_stream_app_factory(monkeypatch: pytest.MonkeyPatch):
    """Build a fresh app with build_llm_provider + RAGOrchestrator stubbed."""

    def factory(
        *,
        events: list[StreamEvent] | None = None,
        raise_after: Exception | None = None,
        build_raise: Exception | None = None,
        env: dict[str, str] | None = None,
        heartbeat_seconds: float | None = None,
    ) -> tuple[Any, _StubBuildRecorder, _StubOrchestrator]:
        if env:
            for key, value in env.items():
                monkeypatch.setenv(key, value)
        reload_settings()

        global _LATEST_STUB
        _LATEST_STUB = _StubOrchestrator(
            events=list(events or []),
            raise_after=raise_after,
        )
        build_stub = _StubBuildRecorder(raise_with=build_raise)

        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.build_llm_provider",
            build_stub,
        )
        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.RAGOrchestrator",
            _orchestrator_factory,
        )
        if heartbeat_seconds is not None:
            monkeypatch.setattr(
                "sec_generative_search.api.routes.rag._SSE_HEARTBEAT_SECONDS",
                heartbeat_seconds,
            )

        app = create_app()
        app.state.retrieval_service = object()
        app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
        app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=300)
        app.state.encrypted_credential_store = None
        return app, build_stub, _LATEST_STUB

    return factory


def _sample_plan_payload() -> dict[str, Any]:
    return {
        "raw_query": "What are AAPL's iPhone segment risks?",
        "detected_language": "en",
        "query_en": "What are AAPL's iPhone segment risks?",
        "tickers": ["AAPL"],
        "form_types": ["10-K"],
        "date_range": ["2023-01-01", "2024-01-01"],
        "intent": "Identify risks for the iPhone segment",
        "suggested_answer_mode": "analytical",
    }


def _default_citation() -> Citation:
    return Citation(
        chunk_id="chunk-1",
        filing_id=FilingIdentifier(
            ticker="AAPL",
            form_type="10-K",
            filing_date=date(2023, 9, 30),
            accession_number="0000320193-23-000077",
        ),
        section_path="Part I > Item 1A > Risk Factors",
        text_span="iPhone revenue concentration risk.",
        similarity=0.84,
        display_index=1,
    )


def _default_final_result(*, citations: list[Citation] | None = None) -> GenerationResult:
    return GenerationResult(
        answer="iPhone risks include concentration [1].",
        provider="openai",
        model="gpt-5.4-mini",
        prompt_version="rag-prompt-v1",
        citations=citations or [_default_citation()],
        retrieved_chunks=[],
        token_usage=TokenUsage(input_tokens=120, output_tokens=42),
        latency_seconds=1.23,
        streamed=True,
    )


def _parse_sse_frames(raw_text: str) -> list[tuple[str, dict]]:
    """Parse the text body of an SSE response into ``[(event_name, data), ...]``.

    Tests use ``response.text`` instead of ``iter_lines`` because the
    TestClient buffers the full streaming body.  Each frame is
    ``event: <name>\\ndata: <json>\\n\\n``.
    """
    frames: list[tuple[str, dict]] = []
    for block in raw_text.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        event_name = ""
        data_payload = ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data_payload = line[len("data: ") :]
        if event_name:
            data = json.loads(data_payload) if data_payload else {}
            frames.append((event_name, data))
    return frames


# ---------------------------------------------------------------------------
# Happy-path framing
# ---------------------------------------------------------------------------


class TestRagStreamHappyPath:
    def test_emits_deltas_then_citation_then_final(self, rag_stream_app_factory) -> None:
        events = [
            StreamEvent(delta="Hello "),
            StreamEvent(delta="world."),
            StreamEvent(final=_default_final_result()),
        ]
        app, _build, _orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"

        frames = _parse_sse_frames(response.text)
        names = [name for name, _ in frames]
        # Two deltas → one citation → one final, in that order.
        assert names == ["delta", "delta", "citation", "final"]
        assert frames[0][1] == {"text": "Hello "}
        assert frames[1][1] == {"text": "world."}

        citation = frames[2][1]
        assert citation["chunk_id"] == "chunk-1"
        assert citation["ticker"] == "AAPL"
        assert citation["filing_date"] == "2023-09-30"
        assert citation["display_index"] == 1

        final = frames[3][1]
        assert final["answer"] == "iPhone risks include concentration [1]."
        assert final["provider"] == "openai"
        assert final["model"] == "gpt-5.4-mini"
        assert final["prompt_version"] == "rag-prompt-v1"
        assert final["streamed"] is True
        assert final["refused"] is False
        assert final["token_usage"] == {
            "input_tokens": 120,
            "output_tokens": 42,
            "total_tokens": 162,
        }
        # Per-request cost estimate mirrors the non-streaming surface:
        # gpt-5.4-mini (in=0.6, out=2.4 USD/MTok) over 120+42 tokens.
        assert final["estimated_cost_usd"] == pytest.approx(0.0001728)

    def test_orchestrator_receives_lifted_plan(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")

        client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )

        call = orch.calls[0]
        assert call["plan_raw_query"] == "What are AAPL's iPhone segment risks?"
        assert call["plan_tickers"] == ["AAPL"]
        # Mode override absent in body → orchestrator decides from plan.
        assert call["mode"] is None

    def test_retrieval_tuning_forwarded(self, rag_stream_app_factory) -> None:
        # The diversity caps + over-fetch ride the stream wire through
        # the worker-thread helper into generate_stream.
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")
        client.post(
            "/api/rag/stream",
            json={
                "plan": _sample_plan_payload(),
                "max_per_section": 4,
                "max_per_filing": 2,
                "rerank_over_fetch_factor": 5,
            },
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )
        call = orch.calls[0]
        assert call["max_per_section"] == 4
        assert call["max_per_filing"] == 2
        assert call["rerank_over_fetch_factor"] == 5

    def test_body_mode_overrides_plan(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")
        client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload(), "mode": "extractive"},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )
        assert orch.calls[0]["mode"] == AnswerMode.EXTRACTIVE

    def test_header_resolver_forwards_user_key(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, build, _orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")
        secret = "sk-USER-PROVIDED-KEY-1234"  # pragma: allowlist secret
        client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": secret},
        )
        assert build.calls[0]["resolved_key"] == secret


# ---------------------------------------------------------------------------
# Refusal path
# ---------------------------------------------------------------------------


class TestRagStreamRefusal:
    def test_empty_retrieval_marks_refused(self, rag_stream_app_factory) -> None:
        # Refusal path: orchestrator yields the refusal text + a final
        # result with no chunks / no citations.
        refusal_result = GenerationResult(
            answer="I cannot answer this from the available filings.",
            provider="openai",
            model="gpt-5.4-mini",
            prompt_version="rag-prompt-v1",
            citations=[],
            retrieved_chunks=[],
            token_usage=TokenUsage(),
            latency_seconds=0.01,
            streamed=False,
        )
        events = [
            StreamEvent(delta=refusal_result.answer),
            StreamEvent(final=refusal_result),
        ]
        app, _build, _orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )

        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        names = [name for name, _ in frames]
        assert names == ["delta", "final"]
        assert frames[1][1]["refused"] is True


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


class TestRagStreamHeartbeat:
    def test_heartbeat_emitted_on_idle(
        self,
        rag_stream_app_factory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force the heartbeat to fire on every gap by setting it to a
        # tiny value, and slow the orchestrator's yields with a sleep
        # so the consumer has time to time out.
        import time as _time

        original_sleep = _time.sleep

        @dataclass
        class _SlowOrch:
            retrieval: Any = None
            llm: Any = None

            def generate_stream(self, plan: QueryPlan, **_: Any) -> Iterator[StreamEvent]:
                # First sleep > heartbeat interval, then yield, then end.
                original_sleep(0.05)
                yield StreamEvent(delta="x")
                original_sleep(0.05)
                yield StreamEvent(final=_default_final_result(citations=[]))

        slow_orch = _SlowOrch()
        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.RAGOrchestrator",
            lambda **kwargs: slow_orch,
        )

        app, _build, _orch = rag_stream_app_factory(
            events=[],  # ignored — slow_orch overrides
            heartbeat_seconds=0.01,
        )
        # Re-patch RAGOrchestrator with our slow stub since the factory
        # wires the dataclass-based stub.
        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.RAGOrchestrator",
            lambda **kwargs: slow_orch,
        )

        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        names = [name for name, _ in frames]
        # At least one heartbeat must have appeared during an idle gap.
        assert "heartbeat" in names
        # The terminal final still arrives.
        assert names[-1] == "final"


# ---------------------------------------------------------------------------
# Pre-stream errors → HTTP envelopes
# ---------------------------------------------------------------------------


class TestRagStreamPreStreamErrors:
    def test_unknown_provider_400_http(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(events=[])
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload(), "provider": "definitely-not-real"},
        )
        assert response.status_code == 400
        # Plain JSON envelope — NOT an SSE response.
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["error"] == "unknown_provider"

    def test_missing_key_400_http(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(
            build_raise=ConfigurationError("no key resolved"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
        )
        assert response.status_code == 400
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert body["error"] == "provider_key_required"


# ---------------------------------------------------------------------------
# In-stream errors → SSE error events
# ---------------------------------------------------------------------------


class TestRagStreamInStreamErrors:
    def test_provider_auth_error_emits_sse_error(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(delta="partial")]
        app, _build, _orch = rag_stream_app_factory(
            events=events,
            raise_after=ProviderAuthError("rejected mid-stream"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        # The stream opened with 200 → the error is an SSE event, not
        # an HTTP status.
        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        names = [name for name, _ in frames]
        assert names == ["delta", "error"]
        assert frames[1][1]["error"] == "provider_unauthorized"

    def test_rate_limit_emits_sse_error(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(
            events=[],
            raise_after=ProviderRateLimitError("upstream limited"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        assert any(
            name == "error" and data["error"] == "provider_unavailable" for name, data in frames
        )

    def test_connection_error_emits_provider_unavailable_sse_error(
        self, rag_stream_app_factory
    ) -> None:
        # Mirrors the /query 503 mapping: an unreachable endpoint surfaces
        # as an in-stream ``provider_unavailable`` event (the stream already
        # opened 200), with endpoint-focused guidance.
        app, _build, _orch = rag_stream_app_factory(
            events=[],
            raise_after=ProviderConnectionError("could not reach endpoint"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        err = [data for name, data in frames if name == "error"]
        assert err, "expected an SSE error frame"
        assert err[0]["error"] == "provider_unavailable"
        assert "reach" in err[0]["message"].lower()
        assert "local_llm" in err[0]["hint"].lower()

    def test_generic_provider_error_emits_sse_error(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(
            events=[],
            raise_after=ProviderError("transport blew up"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        assert any(name == "error" and data["error"] == "provider_error" for name, data in frames)

    def test_generation_error_emits_sse_error(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(
            events=[],
            raise_after=GenerationError("citation parse failed"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        assert any(name == "error" and data["error"] == "generation_error" for name, data in frames)

    def test_unknown_exception_emits_internal_error(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(
            events=[],
            raise_after=RuntimeError("unmapped"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        frames = _parse_sse_frames(response.text)
        assert any(name == "error" and data["error"] == "internal_error" for name, data in frames)


# ---------------------------------------------------------------------------
# Schema guards
# ---------------------------------------------------------------------------


class TestRagStreamSchemaGuards:
    def test_missing_plan_rejected(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(events=[])
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/rag/stream", json={})
        assert response.status_code == 422

    def test_bad_mode_rejected(self, rag_stream_app_factory) -> None:
        app, _build, _orch = rag_stream_app_factory(events=[])
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload(), "mode": "creative"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Privacy / audit-log no-leak contract
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRagStreamPrivacyContract:
    def test_audit_log_carries_metadata_not_query(
        self,
        rag_stream_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        events = [
            StreamEvent(delta="x"),
            StreamEvent(final=_default_final_result(citations=[])),
        ]
        app, _build, _orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                plan = _sample_plan_payload()
                plan["raw_query"] = "PROPRIETARY-WATCHLIST-MNEMONIC-ZZZ"
                plan["query_en"] = "PROPRIETARY-WATCHLIST-MNEMONIC-ZZZ"
                client.post(
                    "/api/rag/stream",
                    json={"plan": plan},
                    headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
                )
        finally:
            package_logger.propagate = prior_propagate

        audit = [r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()]
        # Both the open-stream and the completed-stream lines must
        # appear, and neither may carry the raw query.
        assert any("rag_stream" in line for line in audit)
        assert any("rag_stream_completed" in line for line in audit)
        assert all("PROPRIETARY-WATCHLIST-MNEMONIC-ZZZ" not in line for line in audit)

    def test_in_stream_error_event_does_not_echo_provider_key(
        self,
        rag_stream_app_factory,
    ) -> None:
        app, _build, _orch = rag_stream_app_factory(
            events=[],
            raise_after=ProviderError("kaboom"),
        )
        client = TestClient(app, base_url="https://testserver")
        secret = "sk-NEVER-ECHO-ME-1234"  # pragma: allowlist secret
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": secret},
        )
        assert response.status_code == 200
        assert secret not in response.text


# ---------------------------------------------------------------------------
# Auth tier — read-tier (no admin gate)
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRagStreamAuthGate:
    def test_api_key_required_when_configured(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, _orch = rag_stream_app_factory(
            events=events,
            env={"API_KEY": "shared-team-key"},  # pragma: allowlist secret
        )
        client = TestClient(app, base_url="https://testserver")

        unauthed = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert unauthed.status_code == 401
        assert unauthed.json()["error"] == "unauthorised"

        ok = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={
                "X-API-Key": "shared-team-key",  # pragma: allowlist secret
                "X-Provider-Key-openai": "sk-1234",  # pragma: allowlist secret
            },
        )
        assert ok.status_code == 200

    def test_admin_key_not_required(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, _orch = rag_stream_app_factory(
            events=events,
            env={
                "API_KEY": "shared-team-key",  # pragma: allowlist secret
                "API_ADMIN_KEY": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={
                "X-API-Key": "shared-team-key",  # pragma: allowlist secret
                "X-Provider-Key-openai": "sk-1234",  # pragma: allowlist secret
            },
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Rate-limit classification
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRagStreamRateLimitClassification:
    def test_stream_path_classifies_as_rag(self) -> None:
        from sec_generative_search.api.middleware import _classify_path

        assert _classify_path("/api/rag/stream", "POST") == "rag"


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------


class TestRagStreamHistory:
    """The chat surface replays prior turns through ``body.history``.

    These tests pin the shape — history forwarded into
    :meth:`RAGOrchestrator.generate_stream` as :class:`ConversationTurn`
    instances and the load-bearing privacy invariant: retrieved chunks /
    citations from prior turns are stripped at the route boundary so a
    follow-up never re-injects the prior turn's chunk text into the
    prompt.
    """

    def test_history_forwarded_as_conversation_turns(
        self,
        rag_stream_app_factory,
    ) -> None:
        events = [
            StreamEvent(delta="ok"),
            StreamEvent(final=_default_final_result(citations=[])),
        ]
        app, _build, orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")

        with client.stream(
            "POST",
            "/api/rag/stream",
            json={
                "plan": _sample_plan_payload(),
                "history": [
                    {"query": "what is revenue?", "answer": "Revenue is total sales."},
                ],
            },
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        ) as response:
            # Drain so the producer completes before we inspect calls.
            for _ in response.iter_bytes():
                pass

        passed = orch.calls[0]["history"]
        assert passed is not None
        assert len(passed) == 1
        assert passed[0].query == "what is revenue?"
        assert passed[0].generation_result.answer == "Revenue is total sales."
        # Retrieval / citations stripped at the route boundary.
        assert passed[0].retrieval_results == []
        assert passed[0].generation_result.citations == []
        assert passed[0].generation_result.retrieved_chunks == []

    def test_history_omitted_becomes_none(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")

        with client.stream(
            "POST",
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        ) as response:
            for _ in response.iter_bytes():
                pass

        assert orch.calls[0]["history"] is None

    def test_history_oversize_rejected_pre_stream(
        self,
        rag_stream_app_factory,
    ) -> None:
        app, _build, _orch = rag_stream_app_factory()
        client = TestClient(app, base_url="https://testserver")
        body = {
            "plan": _sample_plan_payload(),
            "history": [{"query": f"q{i}", "answer": f"a{i}"} for i in range(11)],
        }
        response = client.post("/api/rag/stream", json=body)
        # Pre-stream schema rejection → 422 HTTP, NOT an SSE error event.
        assert response.status_code == 422


@pytest.mark.security
class TestRagStreamHistoryPrivacyContract:
    def test_audit_lines_carry_history_count_not_text(
        self,
        rag_stream_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, _orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        secret_q = "STREAM-PRIOR-QUESTION-SENTINEL-ABC"  # pragma: allowlist secret
        secret_a = "STREAM-PRIOR-ANSWER-SENTINEL-ABC"  # pragma: allowlist secret
        try:
            with (
                caplog.at_level(logging.WARNING, logger=LOGGER_NAME),
                client.stream(
                    "POST",
                    "/api/rag/stream",
                    json={
                        "plan": _sample_plan_payload(),
                        "history": [{"query": secret_q, "answer": secret_a}],
                    },
                    headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
                ) as response,
            ):
                for _ in response.iter_bytes():
                    pass
        finally:
            package_logger.propagate = prior_propagate

        audit = [r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()]
        # Both rag_stream (open) and rag_stream_completed (close) carry
        # the history_turns count; the text never reaches any line.
        assert any("rag_stream" in line and "history_turns=1" in line for line in audit)
        assert any("rag_stream_completed" in line and "history_turns=1" in line for line in audit)
        assert all(secret_q not in line for line in audit)
        assert all(secret_a not in line for line in audit)


# ---------------------------------------------------------------------------
# OpenRouter upstream-provider routing hints (streaming)
# ---------------------------------------------------------------------------


class TestRagStreamRoutingHints:
    """Mirror of TestRagQueryRoutingHints but for the SSE surface.

    Streaming and non-streaming MUST share the same guard semantics so a
    misconfiguration cannot surface as ``invalid_flag_combination`` on one
    and a silent no-op on the other.
    """

    def test_hints_forwarded_for_openrouter(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")
        with client.stream(
            "POST",
            "/api/rag/stream",
            json={
                "plan": _sample_plan_payload(),
                "provider": "openrouter",
                "routing_hints": {
                    "order": ["anthropic"],
                    "allow_fallbacks": False,
                },
            },
            headers={
                "X-Provider-Key-openrouter": "sk-or-1234",  # pragma: allowlist secret
            },
        ) as response:
            for _ in response.iter_bytes():
                pass
            assert response.status_code == 200
        hints = orch.calls[0]["routing_hints"]
        assert hints is not None
        assert hints.order == ("anthropic",)
        assert hints.allow_fallbacks is False

    def test_hints_against_non_openrouter_provider_rejected_pre_stream(
        self,
        rag_stream_app_factory,
    ) -> None:
        # Pre-stream guard: the route returns 400 *before* opening the
        # SSE response body so the client can treat the failure as
        # do-not-retry (the SSE path is reserved for in-stream errors).
        app, _build, orch = rag_stream_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={
                "plan": _sample_plan_payload(),
                "provider": "openai",
                "routing_hints": {"order": ["anthropic"]},
            },
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 400
        body = response.json()
        assert body["error"] == "invalid_flag_combination"
        assert orch.calls == []

    def test_audit_lines_carry_or_hints_summary(
        self,
        rag_stream_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        events = [StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, _orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")
        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        sentinel = "stream-secret-upstream"
        try:
            with (
                caplog.at_level(logging.INFO, logger=LOGGER_NAME),
                client.stream(
                    "POST",
                    "/api/rag/stream",
                    json={
                        "plan": _sample_plan_payload(),
                        "provider": "openrouter",
                        "routing_hints": {
                            "order": [sentinel],
                            "allow_fallbacks": True,
                        },
                    },
                    headers={
                        "X-Provider-Key-openrouter": "sk-or-1234",  # pragma: allowlist secret
                    },
                ) as response,
            ):
                for _ in response.iter_bytes():
                    pass
        finally:
            package_logger.propagate = prior_propagate

        audit = [r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()]
        # Both ``rag_stream`` (open) and ``rag_stream_completed`` (close)
        # MUST carry the ``or_hints=...`` summary; the upstream slug
        # itself NEVER reaches any audit line.
        assert any(
            "rag_stream" in line and "or_hints=order=1 fallbacks=True" in line for line in audit
        )
        assert any(
            "rag_stream_completed" in line and "or_hints=order=1 fallbacks=True" in line
            for line in audit
        )
        assert all(sentinel not in line for line in audit)


@pytest.mark.security
class TestRagStreamClientLifecycle:
    """The streaming route closes the per-request LLM once the stream ends.

    Unlike ``/query``, the SDK client outlives the handler — it is used by
    the producer thread, which owns the close in its ``finally``.
    The producer closes the client *before* pushing the done-sentinel, so a
    fully-consumed ``response.text`` guarantees the close has run.
    """

    def test_client_closed_after_stream_completes(self, rag_stream_app_factory) -> None:
        events = [StreamEvent(delta="Hello "), StreamEvent(final=_default_final_result())]
        app, _build, orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        _ = response.text  # drain the stream fully
        assert orch.llm.closed is True

    def test_client_closed_after_midstream_failure(self, rag_stream_app_factory) -> None:
        app, _build, orch = rag_stream_app_factory(
            events=[StreamEvent(delta="partial ")],
            raise_after=ProviderRateLimitError("upstream limited"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )
        assert response.status_code == 200  # SSE opened; failure is an in-stream event
        frames = _parse_sse_frames(response.text)
        assert any(name == "error" for name, _ in frames)
        # The producer's finally closes the client even on a mid-stream raise.
        assert orch.llm.closed is True

    def test_client_closed_when_prestream_routing_hint_rejected(
        self, rag_stream_app_factory
    ) -> None:
        # Pre-stream 400 (routing hints against a non-routing provider) raises
        # before the producer thread starts; the route's ``except`` guard must
        # still close the client.
        app, _build, orch = rag_stream_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/stream",
            json={
                "plan": _sample_plan_payload(),
                "provider": "openai",  # not routing-capable
                "routing_hints": {"order": ["anthropic"]},
            },
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )
        assert response.status_code == 400
        assert orch.llm.closed is True


# ---------------------------------------------------------------------------
# F13 — a client disconnect stops generation
# ---------------------------------------------------------------------------

_TOTAL_DELTAS = 200
_DISCONNECT_AFTER = 5
_STOP_BOUND = 50


class _TeardownLog:
    """Ordered record of teardown steps and the threads they ran on."""

    def __init__(self) -> None:
        self.steps: list[tuple[str, str]] = []

    def record(self, step: str) -> None:
        self.steps.append((step, threading.current_thread().name))


@dataclass
class _RecordingLLM:
    teardown: _TeardownLog
    provider_name: str = "openai"

    def close(self) -> None:
        self.teardown.record("llm.close")


@dataclass
class _LongStreamOrch:
    """Yields 200 deltas, 20 ms apart — a long answer the client abandons."""

    teardown: _TeardownLog
    retrieval: Any = None
    llm: Any = None
    iterations: int = 0

    def generate_stream(self, plan: QueryPlan, **_: Any) -> Iterator[StreamEvent]:
        import time as _time

        try:
            for i in range(_TOTAL_DELTAS):
                self.iterations += 1
                yield StreamEvent(delta=f"token{i} ")
                _time.sleep(0.02)
            yield StreamEvent(final=_default_final_result(citations=[]))
        finally:
            self.teardown.record("generator.finally")


def _stream_scope(body: bytes, *, spec_version: str) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/api/rag/stream",
        "raw_path": b"/api/rag/stream",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"x-provider-key-openai", b"sk-key-123"),  # pragma: allowlist secret
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 443),
    }


async def _stream_then_disconnect(app: Any, *, how: str, teardown: _TeardownLog) -> int:
    """Drive the ASGI app; the client disconnects after a few deltas.

    ``how`` picks where the disconnect lands:

    - ``"listener"`` — ASGI spec 2.3 (what uvicorn advertises): Starlette's
      disconnect listener cancels the body while it awaits the queue
      (``CancelledError`` inside the SSE body);
    - ``"listener_mid_send"`` — same, but the ``send`` of the triggering
      delta is held open, so the body is paused at a ``yield`` and is left
      suspended (closed with ``GeneratorExit``);
    - ``"send_raises"`` — ASGI spec 2.4: no listener; the next ``send``
      raises ``OSError`` and Starlette raises ``ClientDisconnect``.

    After the request returns, the event loop is kept running until the
    producer has torn down (or 3 s pass), as it would be under uvicorn.
    Returning straight away would close the loop and kill the producer on
    its next ``call_soon_threadsafe`` — masking a missing stop signal.
    Returns the number of delta frames the client received.
    """
    import asyncio

    body = json.dumps({"plan": _sample_plan_payload()}).encode()
    disconnected = asyncio.Event()
    request_delivered = False
    deltas = 0

    async def receive() -> dict[str, Any]:
        nonlocal request_delivered
        if not request_delivered:
            request_delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal deltas
        if message["type"] != "http.response.body":
            return
        if how == "send_raises" and deltas >= _DISCONNECT_AFTER:
            raise OSError("client went away")
        if b"event: delta" in message.get("body", b""):
            deltas += 1
            if deltas == _DISCONNECT_AFTER:
                disconnected.set()
                if how == "listener_mid_send":
                    await asyncio.sleep(0.2)

    scope = _stream_scope(body, spec_version="2.4" if how == "send_raises" else "2.3")
    try:
        await asyncio.wait_for(app(scope, receive, send), timeout=5)
    except ClientDisconnect:
        assert how == "send_raises"
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 3.0
    while len(teardown.steps) < 2 and loop.time() < deadline:
        await asyncio.sleep(0.01)
    return deltas


@pytest.mark.security
class TestRagStreamClientDisconnect:
    """F13: an abandoned stream stops generating within one event.

    Without the stop signal the producer ran all 200 deltas into a queue no
    one read, holding the provider stream (and its billing) open to the end.
    The teardown keeps the F5(a) contract: generator first, then the client,
    exactly once, on the producer thread.
    """

    @pytest.mark.parametrize("how", ["listener", "listener_mid_send", "send_raises"])
    def test_disconnect_stops_generation_and_tears_down_on_the_producer(
        self,
        rag_stream_app_factory,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        how: str,
    ) -> None:
        import asyncio

        teardown = _TeardownLog()
        orch = _LongStreamOrch(teardown=teardown)
        app, _build, _stub = rag_stream_app_factory()
        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.build_llm_provider",
            lambda _name, **_kw: _RecordingLLM(teardown=teardown),
        )
        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.RAGOrchestrator",
            lambda **_kw: orch,
        )

        # The handler sits on the package logger itself, so each audit
        # record is captured exactly once (no root propagation needed).
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            received = asyncio.run(_stream_then_disconnect(app, how=how, teardown=teardown))

        # Bounds, not exact counts: under a loaded CI runner (xdist workers
        # sharing 2-4 vCPUs) a queued backlog can flush a few more deltas
        # before the cancellation lands.  Without the stop signal the
        # producer runs ~150 of 200 events in the 3 s window, so < 50 still
        # separates the fix from a regression.
        assert _DISCONNECT_AFTER <= received < _STOP_BOUND
        # Generation stopped shortly after the disconnect.
        assert orch.iterations < _STOP_BOUND
        # Generator torn down first, then the client — once, on the producer.
        assert teardown.steps == [
            ("generator.finally", "rag-stream-producer"),
            ("llm.close", "rag-stream-producer"),
        ]
        completed = [
            r.getMessage() for r in caplog.records if "rag_stream_completed" in r.getMessage()
        ]
        assert len(completed) == 1
        assert "status=client_disconnected" in completed[0]

    def test_a_fully_consumed_stream_is_not_reported_as_disconnected(
        self,
        rag_stream_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        events = [StreamEvent(delta="x"), StreamEvent(final=_default_final_result(citations=[]))]
        app, _build, orch = rag_stream_app_factory(events=events)
        client = TestClient(app, base_url="https://testserver")
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            response = client.post(
                "/api/rag/stream",
                json={"plan": _sample_plan_payload()},
                headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
            )
            _ = response.text
        completed = [
            r.getMessage() for r in caplog.records if "rag_stream_completed" in r.getMessage()
        ]
        assert len(completed) == 1
        assert "status=ok" in completed[0]
        assert orch.llm.closed is True

    def test_an_in_stream_error_keeps_its_own_status(
        self,
        rag_stream_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        app, _build, _orch = rag_stream_app_factory(
            events=[StreamEvent(delta="partial ")],
            raise_after=ProviderRateLimitError("upstream limited"),
        )
        client = TestClient(app, base_url="https://testserver")
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _ = client.post(
                "/api/rag/stream",
                json={"plan": _sample_plan_payload()},
                headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
            ).text
        completed = [
            r.getMessage() for r in caplog.records if "rag_stream_completed" in r.getMessage()
        ]
        assert len(completed) == 1
        assert "status=ProviderRateLimitError" in completed[0]
