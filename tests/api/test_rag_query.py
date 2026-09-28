"""Tests for the non-streaming RAG generation route.

Strategy
--------

The route is exercised through the standard ``api_client`` /
``api_client_factory`` style fixtures, mirroring :mod:`tests.api.test_rag_plan`.
Both :func:`build_llm_provider` and the :class:`RAGOrchestrator` factory
are monkey-patched so the tests stay hermetic — no real LLM, no real
ChromaDB, no real embedder.

Coverage focuses on:

    - schema guards (plan required, mode pattern, max_output_tokens
      bounds, unknown fields rejected via ``extra="forbid"``);
    - resolver-chain wiring (``X-Provider-Key-{provider}`` header is
      forwarded so the route runs on the caller's key, not admin env);
    - provider resolution (body fields override settings, settings
      defaults apply when both are omitted, unknown provider rejected);
    - orchestrator invocation contract (the route passes the lifted
      :class:`QueryPlan`, the resolved capability flag, and the right
      mode override through to :meth:`RAGOrchestrator.generate`);
    - response surface (citations are an explicit allow-list lift, no
      ``retrieved_chunks`` echoed back, ``filing_date`` rendered as ISO,
      ``token_usage.total_tokens`` populated);
    - refusal short-circuit (no retrieved chunks → ``refused=True``);
    - error mapping (``ConfigurationError`` → 400 ``provider_key_required``,
      ``ProviderAuthError`` → 401, transient ``ProviderError`` → 503,
      generic ``ProviderError`` → 502, ``GenerationError`` → 502);
    - audit-log emission (``rag_query`` action; raw query / plan body
      never appears in any audit-log line);
    - rate-limit classification (the request hits the ``rag`` bucket);
    - auth gate (read-tier — admin-key NOT required).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

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
    ProviderTimeoutError,
)
from sec_generative_search.core.logging import LOGGER_NAME
from sec_generative_search.core.types import (
    Citation,
    FilingIdentifier,
    GenerationResult,
    TokenUsage,
)
from sec_generative_search.rag.modes import AnswerMode
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

    Records the ``generate`` arguments so tests can assert delegation
    without touching retrieval / generation / citation extraction.
    """

    retrieval: Any = None
    llm: Any = None
    result: GenerationResult | None = None
    raise_with: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate(
        self,
        plan: QueryPlan,
        *,
        mode: AnswerMode | None = None,
        model: str | None = None,
        max_output_tokens: int | None = None,
        history: Any = None,
        prefer_structured_output: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        self.calls.append(
            {
                "plan_raw_query": plan.raw_query,
                "plan_query_en": plan.query_en,
                "plan_tickers": list(plan.tickers),
                "plan_date_range": plan.date_range,
                "plan_mode": plan.suggested_answer_mode,
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
        if self.raise_with is not None:
            raise self.raise_with
        if self.result is not None:
            return self.result
        return _default_result()


# Module-level handle so the patched ``RAGOrchestrator`` factory can
# return a single shared stub from every constructor call within a test.
_LATEST_STUB: _StubOrchestrator | None = None


def _orchestrator_factory(*, retrieval: Any, llm: Any, **_: Any) -> _StubOrchestrator:
    """Patched in for ``RAGOrchestrator`` — returns the per-test shared stub."""
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
def rag_query_app_factory(monkeypatch: pytest.MonkeyPatch):
    """Build a fresh app with build_llm_provider + RAGOrchestrator stubbed."""

    def factory(
        *,
        result: GenerationResult | None = None,
        generate_raise: Exception | None = None,
        build_raise: Exception | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[Any, _StubBuildRecorder, _StubOrchestrator]:
        if env:
            for key, value in env.items():
                monkeypatch.setenv(key, value)
        reload_settings()

        global _LATEST_STUB
        _LATEST_STUB = _StubOrchestrator(result=result, raise_with=generate_raise)
        build_stub = _StubBuildRecorder(raise_with=build_raise)

        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.build_llm_provider",
            build_stub,
        )
        monkeypatch.setattr(
            "sec_generative_search.api.routes.rag.RAGOrchestrator",
            _orchestrator_factory,
        )

        app = create_app()
        # Stub retrieval — the orchestrator stub never touches it.
        app.state.retrieval_service = object()
        app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
        app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=300)
        app.state.encrypted_credential_store = None
        return app, build_stub, _LATEST_STUB

    return factory


def _sample_plan_payload() -> dict[str, Any]:
    """A minimal valid plan payload for the request body."""
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


def _default_result() -> GenerationResult:
    """A small non-empty GenerationResult shared across happy-path tests."""
    citation = Citation(
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
    return GenerationResult(
        answer="iPhone risks include concentration [1].",
        provider="openai",
        model="gpt-5.4-mini",
        prompt_version="rag-prompt-v1",
        citations=[citation],
        retrieved_chunks=[],
        token_usage=TokenUsage(input_tokens=120, output_tokens=42),
        latency_seconds=1.23,
        streamed=False,
    )


# ---------------------------------------------------------------------------
# Happy-path delegation
# ---------------------------------------------------------------------------


class TestRagQueryDelegation:
    def test_returns_answer_and_citations(self, rag_query_app_factory) -> None:
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "iPhone risks include concentration [1]."
        assert body["provider"] == "openai"
        assert body["model"] == "gpt-5.4-mini"
        assert body["prompt_version"] == "rag-prompt-v1"
        assert body["streamed"] is False
        assert body["refused"] is False
        # Citation lift.
        assert len(body["citations"]) == 1
        cite = body["citations"][0]
        assert cite["chunk_id"] == "chunk-1"
        assert cite["ticker"] == "AAPL"
        assert cite["form_type"] == "10-K"
        assert cite["filing_date"] == "2023-09-30"
        assert cite["accession_number"] == "0000320193-23-000077"
        assert cite["section_path"] == "Part I > Item 1A > Risk Factors"
        assert cite["text_span"] == "iPhone revenue concentration risk."
        assert cite["display_index"] == 1
        # Token usage carries the derived total.
        usage = body["token_usage"]
        assert usage["input_tokens"] == 120
        assert usage["output_tokens"] == 42
        assert usage["total_tokens"] == 162
        # Estimated cost is the canonical token-weighted estimate for the
        # resolved model (gpt-5.4-mini: in=0.6, out=2.4 USD/MTok):
        # 120/1e6*0.6 + 42/1e6*2.4 = 0.0001728.
        assert body["estimated_cost_usd"] == pytest.approx(0.0001728)
        # Orchestrator received the lifted plan, not the raw payload.
        call = orch.calls[0]
        assert call["plan_raw_query"] == "What are AAPL's iPhone segment risks?"
        assert call["plan_query_en"] == "What are AAPL's iPhone segment risks?"
        assert call["plan_tickers"] == ["AAPL"]
        assert call["plan_date_range"] == ("2023-01-01", "2024-01-01")
        assert call["plan_mode"] == AnswerMode.ANALYTICAL

    def test_response_strips_internal_only_fields(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == {
            "answer",
            "citations",
            "provider",
            "model",
            "prompt_version",
            "token_usage",
            "estimated_cost_usd",
            "latency_seconds",
            "streamed",
            "refused",
        }
        # ``retrieved_chunks`` is intentionally NOT echoed back.
        assert "retrieved_chunks" not in body

    def test_body_provider_overrides_settings(
        self, rag_query_app_factory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_DEFAULT_PROVIDER", "openai")
        app, build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload(), "provider": "anthropic"},
            headers={"X-Provider-Key-anthropic": "sk-ant-1234"},  # pragma: allowlist secret
        )

        assert response.status_code == 200
        assert build.calls[0]["provider"] == "anthropic"

    def test_body_mode_overrides_plan(self, rag_query_app_factory) -> None:
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload(), "mode": "extractive"},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )

        assert response.status_code == 200
        # The plan said analytical; the body said extractive — body wins.
        assert orch.calls[0]["mode"] == AnswerMode.EXTRACTIVE
        assert orch.calls[0]["plan_mode"] == AnswerMode.ANALYTICAL

    def test_mode_omitted_lets_orchestrator_use_plan_mode(self, rag_query_app_factory) -> None:
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        # Route forwards ``mode=None`` so the orchestrator falls back to
        # ``plan.suggested_answer_mode``.
        assert orch.calls[0]["mode"] is None

    def test_max_output_tokens_forwarded(self, rag_query_app_factory) -> None:
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload(), "max_output_tokens": 256},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        assert orch.calls[0]["max_output_tokens"] == 256

    def test_retrieval_tuning_forwarded(self, rag_query_app_factory) -> None:
        # The diversity caps + over-fetch ride the RAG query wire into
        # the orchestrator's retrieval step.
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={
                "plan": _sample_plan_payload(),
                "max_per_section": 2,
                "max_per_filing": 1,
                "rerank_over_fetch_factor": 3,
            },
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        assert orch.calls[0]["max_per_section"] == 2
        assert orch.calls[0]["max_per_filing"] == 1
        assert orch.calls[0]["rerank_over_fetch_factor"] == 3

    def test_retrieval_tuning_defaults_to_none(self, rag_query_app_factory) -> None:
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        assert orch.calls[0]["max_per_section"] is None
        assert orch.calls[0]["max_per_filing"] is None
        assert orch.calls[0]["rerank_over_fetch_factor"] is None

    @pytest.mark.security
    def test_oversize_retrieval_tuning_rejected(self, rag_query_app_factory) -> None:
        # The same bounds as POST /api/search apply on the RAG wire so a
        # caller cannot widen the diversity cap or explode the over-fetch.
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        for tuning_field, value in (
            ("max_per_section", 51),
            ("max_per_filing", 51),
            ("rerank_over_fetch_factor", 11),
            ("rerank_over_fetch_factor", 0),
        ):
            response = client.post(
                "/api/rag/query",
                json={"plan": _sample_plan_payload(), tuning_field: value},
                headers={
                    "X-Provider-Key-openai": "sk-1234"  # pragma: allowlist secret
                },
            )
            assert response.status_code == 422, (tuning_field, value)
        assert orch.calls == []

    def test_header_resolver_forwards_user_key(self, rag_query_app_factory) -> None:
        app, build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")

        secret = "sk-USER-PROVIDED-KEY-1234"  # pragma: allowlist secret
        client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": secret},
        )

        # The route MUST run the LLM construction against the chained
        # resolver — header tier first.  Without that, an admin-env
        # fallback would silently mask a missing header.
        assert build.calls[0]["resolved_key"] == secret


# ---------------------------------------------------------------------------
# Per-request cost estimate
# ---------------------------------------------------------------------------


class TestRagQueryCostEstimate:
    """``estimated_cost_usd`` is the canonical per-request USD estimate.

    Derived from the call's token usage and the resolved model's exact
    per-MTok cost; ``None`` (the UNKNOWN case) for arbitrary-slug providers.
    """

    def test_known_model_returns_estimate(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        assert response.json()["estimated_cost_usd"] == pytest.approx(0.0001728)

    def test_arbitrary_slug_provider_cost_is_none(self, rag_query_app_factory) -> None:
        # OpenRouter carries no catalogued cost — the capability the route
        # probes has both costs None, so the estimate is None (rendered "—").
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload(), "provider": "openrouter"},
            headers={"X-Provider-Key-openrouter": "sk-or-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        assert response.json()["estimated_cost_usd"] is None

    def test_refusal_with_zero_tokens_is_zero_for_known_model(self, rag_query_app_factory) -> None:
        # A refusal short-circuits with a zero usage record; for a
        # known-cost model the estimate is a clean 0.0, not None.
        refusal = GenerationResult(
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
        app, _build, _orch = rag_query_app_factory(result=refusal)
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        assert response.json()["estimated_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# Refusal short-circuit
# ---------------------------------------------------------------------------


class TestRagQueryRefusal:
    def test_empty_retrieval_marks_refused(self, rag_query_app_factory) -> None:
        # Orchestrator returns a refusal-shaped result (no chunks, no citations).
        refusal = GenerationResult(
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
        app, _build, _orch = rag_query_app_factory(result=refusal)
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )

        assert response.status_code == 200
        body = response.json()
        assert body["refused"] is True
        assert body["citations"] == []


# ---------------------------------------------------------------------------
# Schema guards
# ---------------------------------------------------------------------------


class TestRagQuerySchemaGuards:
    def test_missing_plan_rejected(self, rag_query_app_factory) -> None:
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post("/api/rag/query", json={})
        assert response.status_code == 422
        # No orchestrator call must have happened when the schema rejects.
        assert orch.calls == []

    def test_unknown_field_rejected(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload(), "depth": 7},
        )
        assert response.status_code == 422

    def test_bad_mode_rejected(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload(), "mode": "creative"},
        )
        assert response.status_code == 422

    def test_oversize_max_output_tokens_rejected(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload(), "max_output_tokens": 999_999},
        )
        assert response.status_code == 422

    def test_missing_plan_field_rejected(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        bad_plan = _sample_plan_payload()
        del bad_plan["raw_query"]
        response = client.post("/api/rag/query", json={"plan": bad_plan})
        assert response.status_code == 422

    def test_uppercase_provider_rejected(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload(), "provider": "OPENAI"},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestRagQueryErrorMapping:
    def test_unknown_provider_400(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload(), "provider": "definitely-not-real"},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "unknown_provider"

    def test_missing_key_maps_to_400(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory(
            build_raise=ConfigurationError("no key resolved"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
        )
        assert response.status_code == 400
        body = response.json()
        assert body["error"] == "provider_key_required"
        assert "X-Provider-Key" in body["hint"]

    def test_provider_auth_error_maps_to_401(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory(
            generate_raise=ProviderAuthError("rejected by upstream"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-bad-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 401
        assert response.json()["error"] == "provider_unauthorized"

    def test_rate_limit_maps_to_503(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory(
            generate_raise=ProviderRateLimitError("upstream limited"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 503
        body = response.json()
        assert body["error"] == "provider_unavailable"
        assert "do not rotate" in body["hint"].lower()

    def test_timeout_maps_to_503(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory(
            generate_raise=ProviderTimeoutError("timed out"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 503

    def test_connection_error_maps_to_503(self, rag_query_app_factory) -> None:
        # An unreachable endpoint (chiefly an unstarted local_llm server) is
        # transient → 503, with endpoint-focused guidance, not a key rotation.
        app, _build, _orch = rag_query_app_factory(
            generate_raise=ProviderConnectionError("could not reach endpoint"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 503
        body = response.json()
        assert body["error"] == "provider_unavailable"
        assert "reach" in body["message"].lower()
        assert "local_llm" in body["hint"].lower()

    def test_generic_provider_error_maps_to_502(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory(
            generate_raise=ProviderError("transport blew up"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 502
        assert response.json()["error"] == "provider_error"

    def test_generation_error_maps_to_502(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory(
            generate_raise=GenerationError("citation parse failed"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 502
        assert response.json()["error"] == "generation_error"


# ---------------------------------------------------------------------------
# Privacy / audit-log no-leak contract
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRagQueryPrivacyContract:
    def test_audit_log_carries_metadata_not_query(
        self,
        rag_query_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        app, _build, _orch = rag_query_app_factory()
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
                    "/api/rag/query",
                    json={"plan": plan},
                    headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
                )
        finally:
            package_logger.propagate = prior_propagate

        audit = [r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()]
        assert any("rag_query" in line for line in audit)
        assert all("PROPRIETARY-WATCHLIST-MNEMONIC-ZZZ" not in line for line in audit)

    def test_error_envelope_does_not_echo_provider_key(
        self,
        rag_query_app_factory,
    ) -> None:
        app, _build, _orch = rag_query_app_factory(
            generate_raise=ProviderError("transport blew up"),
        )
        client = TestClient(app, base_url="https://testserver")
        secret = "sk-NEVER-ECHO-ME-1234"  # pragma: allowlist secret
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": secret},
        )
        assert response.status_code == 502
        assert secret not in response.text

    def test_error_envelope_does_not_echo_plan_body(
        self,
        rag_query_app_factory,
    ) -> None:
        app, _build, _orch = rag_query_app_factory(
            generate_raise=ProviderError("kaboom"),
        )
        client = TestClient(app, base_url="https://testserver")
        plan = _sample_plan_payload()
        secret_query = "PROPRIETARY-WATCHLIST-MNEMONIC-ZZZ"  # pragma: allowlist secret
        plan["raw_query"] = secret_query
        plan["query_en"] = secret_query
        response = client.post(
            "/api/rag/query",
            json={"plan": plan},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 502
        # Even though the plan body unavoidably echoes the query in
        # 200-success responses (chip UI), the error envelope MUST stay
        # clean.
        assert secret_query not in response.text


# ---------------------------------------------------------------------------
# Auth tier — read-tier (no admin gate)
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRagQueryAuthGate:
    def test_api_key_required_when_configured(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory(
            env={"API_KEY": "shared-team-key"},  # pragma: allowlist secret
        )
        client = TestClient(app, base_url="https://testserver")

        unauthed = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert unauthed.status_code == 401
        assert unauthed.json()["error"] == "unauthorised"

        ok = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={
                "X-API-Key": "shared-team-key",  # pragma: allowlist secret
                "X-Provider-Key-openai": "sk-1234",  # pragma: allowlist secret
            },
        )
        assert ok.status_code == 200

    def test_admin_key_not_required(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory(
            env={
                "API_KEY": "shared-team-key",  # pragma: allowlist secret
                "API_ADMIN_KEY": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
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
class TestRagQueryRateLimitClassification:
    def test_query_path_classifies_as_rag(self) -> None:
        from sec_generative_search.api.middleware import _rate_category_for_scope

        assert (
            _rate_category_for_scope({"type": "http", "path": "/api/rag/query", "method": "POST"})
            == "rag"
        )

    def test_query_respects_per_ip_window(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory(
            env={"API_RATE_LIMIT_RAG": "3"},
        )
        client = TestClient(app, base_url="https://testserver")
        statuses: list[int] = []
        for _ in range(6):
            r = client.post(
                "/api/rag/query",
                json={"plan": _sample_plan_payload()},
                headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
            )
            statuses.append(r.status_code)
        assert statuses.count(200) == 3
        assert 429 in statuses


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------


class TestRagQueryHistory:
    """The chat surface replays prior turns through ``body.history``.

    These tests pin the shape — history forwarded into
    :meth:`RAGOrchestrator.generate` as :class:`ConversationTurn` instances,
    bounded by :class:`ConversationTurnSchema` length caps — and the privacy
    guarantee that retrieved chunks and citations from prior turns never
    re-enter the prompt (the route's lift drops them).
    """

    def test_history_forwarded_as_conversation_turns(
        self,
        rag_query_app_factory,
    ) -> None:
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")

        body = {
            "plan": _sample_plan_payload(),
            "history": [
                {"query": "what is revenue?", "answer": "Revenue is total sales."},
                {"query": "and net income?", "answer": "Net income subtracts costs."},
            ],
        }
        response = client.post(
            "/api/rag/query",
            json=body,
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        passed = orch.calls[0]["history"]
        assert passed is not None
        assert len(passed) == 2
        # Only the query + answer are surfaced into the synthesised
        # ConversationTurn; retrieval_results is empty by route contract.
        assert passed[0].query == "what is revenue?"
        assert passed[0].generation_result.answer == "Revenue is total sales."
        assert passed[0].retrieval_results == []
        assert passed[0].generation_result.citations == []
        assert passed[0].generation_result.retrieved_chunks == []

    def test_history_omitted_becomes_none(self, rag_query_app_factory) -> None:
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")

        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        # Empty list lifts to ``None`` so the orchestrator's existing
        # short-circuit applies (no history block rendered, no token spent).
        assert orch.calls[0]["history"] is None

    def test_history_oversize_rejected(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        body = {
            "plan": _sample_plan_payload(),
            "history": [{"query": f"q{i}", "answer": f"a{i}"} for i in range(11)],
        }
        response = client.post("/api/rag/query", json=body)
        assert response.status_code == 422

    def test_history_turn_answer_oversize_rejected(
        self,
        rag_query_app_factory,
    ) -> None:
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        body = {
            "plan": _sample_plan_payload(),
            "history": [{"query": "q", "answer": "a" * 5000}],
        }
        response = client.post("/api/rag/query", json=body)
        assert response.status_code == 422

    def test_history_empty_strings_rejected(self, rag_query_app_factory) -> None:
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        body = {
            "plan": _sample_plan_payload(),
            "history": [{"query": "", "answer": "ok"}],
        }
        response = client.post("/api/rag/query", json=body)
        assert response.status_code == 422


@pytest.mark.security
class TestRagQueryHistoryPrivacyContract:
    def test_audit_log_carries_history_count_not_text(
        self,
        rag_query_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")

        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        secret_q = "PRIVATE-PRIOR-QUESTION-SENTINEL-XYZ"  # pragma: allowlist secret
        secret_a = "PRIVATE-PRIOR-ANSWER-SENTINEL-XYZ"  # pragma: allowlist secret
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                client.post(
                    "/api/rag/query",
                    json={
                        "plan": _sample_plan_payload(),
                        "history": [{"query": secret_q, "answer": secret_a}],
                    },
                    headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
                )
        finally:
            package_logger.propagate = prior_propagate

        audit = [r.getMessage() for r in caplog.records if "SECURITY_AUDIT:" in r.getMessage()]
        assert any("rag_query" in line for line in audit)
        assert any("history_turns=1" in line for line in audit)
        assert all(secret_q not in line for line in audit)
        assert all(secret_a not in line for line in audit)

    def test_error_envelope_does_not_echo_history(
        self,
        rag_query_app_factory,
    ) -> None:
        app, _build, _orch = rag_query_app_factory(
            generate_raise=ProviderError("transport blew up"),
        )
        client = TestClient(app, base_url="https://testserver")
        secret_q = "HISTORY-NEVER-ECHO-Q-SENTINEL"  # pragma: allowlist secret
        secret_a = "HISTORY-NEVER-ECHO-A-SENTINEL"  # pragma: allowlist secret
        response = client.post(
            "/api/rag/query",
            json={
                "plan": _sample_plan_payload(),
                "history": [{"query": secret_q, "answer": secret_a}],
            },
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 502
        assert secret_q not in response.text
        assert secret_a not in response.text


# ---------------------------------------------------------------------------
# OpenRouter upstream-provider routing hints
# ---------------------------------------------------------------------------


class TestRagQueryRoutingHints:
    """Mirror of the CLI's ``--openrouter-*`` flag plumbing.

    The route forwards ``routing_hints`` verbatim into
    :meth:`RAGOrchestrator.generate`; only :class:`OpenRouterProvider`
    consumes it.  Supplying hints against any other provider must fail
    closed with 400 ``invalid_flag_combination`` so a caller never gets
    a silently-no-op generation.
    """

    def test_hints_forwarded_for_openrouter(self, rag_query_app_factory) -> None:
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={
                "plan": _sample_plan_payload(),
                "provider": "openrouter",
                "routing_hints": {
                    "order": ["anthropic", "openai"],
                    "allow_fallbacks": False,
                },
            },
            headers={
                "X-Provider-Key-openrouter": "sk-or-1234",  # pragma: allowlist secret
            },
        )
        assert response.status_code == 200
        hints = orch.calls[0]["routing_hints"]
        assert hints is not None
        assert hints.order == ("anthropic", "openai")
        assert hints.allow_fallbacks is False

    def test_hints_omitted_yields_none(self, rag_query_app_factory) -> None:
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        assert orch.calls[0]["routing_hints"] is None

    def test_hints_against_non_openrouter_provider_rejected(self, rag_query_app_factory) -> None:
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
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
        # Orchestrator MUST NOT have run — the guard rejects before
        # generation is attempted.
        assert orch.calls == []

    def test_data_collection_invalid_value_rejected(self, rag_query_app_factory) -> None:
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={
                "plan": _sample_plan_payload(),
                "provider": "openrouter",
                "routing_hints": {"data_collection": "perhaps"},
            },
            headers={
                "X-Provider-Key-openrouter": "sk-or-1234",  # pragma: allowlist secret
            },
        )
        # Schema-layer rejection — 422, not 400.
        assert response.status_code == 422
        assert orch.calls == []

    def test_upstream_slug_shape_violation_rejected(self, rag_query_app_factory) -> None:
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={
                "plan": _sample_plan_payload(),
                "provider": "openrouter",
                "routing_hints": {"order": ["UPPERCASE-NOT-OK"]},
            },
            headers={
                "X-Provider-Key-openrouter": "sk-or-1234",  # pragma: allowlist secret
            },
        )
        assert response.status_code == 422
        assert orch.calls == []

    def test_audit_log_carries_or_hints_summary_not_slugs(
        self,
        rag_query_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # The audit log MUST carry the ``or_hints=...`` count summary
        # without echoing the upstream-provider slug list.  Mirrors the
        # CLI's ``_format_routing_hints_audit`` privacy stance.
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        sentinel = "very-secret-upstream-slug"
        logging.getLogger(LOGGER_NAME).propagate = True
        try:
            with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
                response = client.post(
                    "/api/rag/query",
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
                )
        finally:
            logging.getLogger(LOGGER_NAME).propagate = False
        assert response.status_code == 200
        audit = [r.getMessage() for r in caplog.records if "SECURITY_AUDIT" in r.getMessage()]
        assert any("or_hints=order=1 fallbacks=True" in line for line in audit)
        # The slug itself MUST NOT appear in any audit line.
        assert all(sentinel not in line for line in audit)

    def test_audit_log_shows_none_when_no_hints(
        self,
        rag_query_app_factory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        logging.getLogger(LOGGER_NAME).propagate = True
        try:
            with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
                response = client.post(
                    "/api/rag/query",
                    json={"plan": _sample_plan_payload()},
                    headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
                )
        finally:
            logging.getLogger(LOGGER_NAME).propagate = False
        assert response.status_code == 200
        audit = [r.getMessage() for r in caplog.records if "SECURITY_AUDIT" in r.getMessage()]
        assert any("or_hints=none" in line for line in audit)


@pytest.mark.security
class TestRagQueryRoutingHintsSecurityContract:
    """Pin the load-bearing privacy invariants for routing hints.

    Re-run these when touching :func:`_resolve_routing_hints_api` or
    :func:`_format_routing_hints_audit`.
    """

    def test_routing_hints_schema_has_no_credential_fields(self) -> None:
        # The OpenRouter API key already flows on the request's
        # ``Authorization`` header.  A redundant key field on the hint
        # would silently widen the audit surface.
        from sec_generative_search.api.schemas import OpenRouterRoutingHintsSchema

        forbidden = {"api_key", "authorization", "bearer", "token", "secret"}
        field_names = set(OpenRouterRoutingHintsSchema.model_fields.keys())
        assert field_names.isdisjoint(forbidden), (
            f"OpenRouterRoutingHintsSchema must not carry credential-shaped "
            f"fields; found: {field_names & forbidden}"
        )

    def test_invalid_combination_envelope_redacts_slug(self, rag_query_app_factory) -> None:
        # The 400 envelope MUST NOT echo the supplied upstream slugs —
        # the message names the provider and the remediation only.
        app, _build, _orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        sentinel = "anthropic-redacted-sentinel"
        response = client.post(
            "/api/rag/query",
            json={
                "plan": _sample_plan_payload(),
                "provider": "openai",
                "routing_hints": {"order": [sentinel]},
            },
            headers={"X-Provider-Key-openai": "sk-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 400
        assert sentinel not in response.text


@pytest.mark.security
class TestRagQueryClientLifecycle:
    """The route closes the per-request LLM on every exit path.

    Deterministic teardown releases the SDK client's connection pool + the
    credential-bearing transport at end-of-request rather than at GC.
    """

    def test_client_closed_after_success(self, rag_query_app_factory) -> None:
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        assert orch.llm.closed is True

    def test_client_closed_after_provider_error(self, rag_query_app_factory) -> None:
        app, _build, orch = rag_query_app_factory(
            generate_raise=ProviderRateLimitError("upstream limited"),
        )
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={"plan": _sample_plan_payload()},
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )
        assert response.status_code == 503
        # The finally still runs even though generation raised.
        assert orch.llm.closed is True

    def test_client_closed_after_routing_hint_rejection(self, rag_query_app_factory) -> None:
        # ``_resolve_routing_hints_api`` raises 400 *after* the LLM is built
        # (and the orchestrator stub has captured it) but *before* generation
        # — the finally must still close it.
        app, _build, orch = rag_query_app_factory()
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/api/rag/query",
            json={
                "plan": _sample_plan_payload(),
                "provider": "openai",  # not routing-capable
                "routing_hints": {"order": ["anthropic-claude"]},
            },
            headers={"X-Provider-Key-openai": "sk-key-123"},  # pragma: allowlist secret
        )
        assert response.status_code == 400
        assert orch.llm.closed is True
