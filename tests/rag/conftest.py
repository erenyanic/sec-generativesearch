"""Shared fixtures for the RAG tests.

Test doubles live here rather than in each test file so the
orchestrator and citation tests share a single source of truth for the
shape of a retrieval result. The doubles are intentionally minimal —
just enough surface to exercise the path under test.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from sec_generative_search.core.types import (
    ContentType,
    ProviderCapability,
    RetrievalResult,
    TokenUsage,
)
from sec_generative_search.providers.base import (
    BaseLLMProvider,
    GenerationRequest,
    GenerationResponse,
)

if TYPE_CHECKING:
    pass


def _make_chunk(
    *,
    index: int,
    ticker: str = "AAPL",
    form_type: str = "10-K",
    filing_date: str = "2023-11-03",
    accession_number: str = "0000320193-23-000077",
    path: str = "Part I > Item 1A > Risk Factors",
    content: str = "Risk factor text.",
    similarity: float = 0.9,
) -> RetrievalResult:
    """Construct a synthetic :class:`RetrievalResult` with a stable chunk_id."""
    chunk_id = f"{ticker}_{form_type}_{filing_date}_{index:03d}"
    return RetrievalResult(
        content=content,
        path=path,
        content_type=ContentType.TEXT,
        ticker=ticker,
        form_type=form_type,
        similarity=similarity,
        filing_date=filing_date,
        accession_number=accession_number,
        chunk_id=chunk_id,
        token_count=10,
        section_boundaries=tuple(p.strip() for p in path.split(">") if p.strip()),
    )


@pytest.fixture
def make_chunk():
    """Factory fixture that yields :func:`_make_chunk`."""
    return _make_chunk


@pytest.fixture
def sample_chunks() -> list[RetrievalResult]:
    """Three retrieval results from one filing — used by most tests."""
    return [
        _make_chunk(index=1, content="Revenue grew 8% year over year."),
        _make_chunk(index=2, content="Operating margin compressed in Q3."),
        _make_chunk(
            index=3,
            content="Foreign exchange impacts noted in MD&A.",
            path="Part II > Item 7 > MD&A",
        ),
    ]


# ---------------------------------------------------------------------------
# LLM provider doubles
# ---------------------------------------------------------------------------


class FakeLLMProvider(BaseLLMProvider):
    """Records the last :class:`GenerationRequest` and returns a canned reply.

    Streaming yields the canned reply split into a few deltas plus a
    final usage-only event, mirroring the real OpenAI-compatible
    streaming shape.
    """

    provider_name = "fake-llm"

    def __init__(
        self,
        *,
        reply: str = "Answer text.",
        usage: TokenUsage | None = None,
        capability: ProviderCapability | None = None,
    ) -> None:
        super().__init__(api_key="fake-key-padding-1234")  # pragma: allowlist secret
        self.reply = reply
        self.usage = usage or TokenUsage(input_tokens=12, output_tokens=8)
        self.capability = capability or ProviderCapability(
            chat=True,
            streaming=True,
            structured_output=False,
            context_window_tokens=8000,
            max_output_tokens=2048,
        )
        self.last_request: GenerationRequest | None = None
        self.call_count = 0

    def validate_key(self) -> bool:  # pragma: no cover - unused in tests
        return True

    def get_capabilities(self) -> ProviderCapability:
        return self.capability

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.last_request = request
        self.call_count += 1
        return GenerationResponse(
            text=self.reply,
            model=request.model or "fake-model",
            token_usage=self.usage,
            finish_reason="stop",
        )

    def generate_stream(self, request: GenerationRequest) -> Iterator[GenerationResponse]:
        self.last_request = request
        self.call_count += 1
        # Split into 2-3 chunks so tests can assert deltas accumulate.
        midpoint = len(self.reply) // 2 or 1
        first = self.reply[:midpoint]
        second = self.reply[midpoint:]
        if first:
            yield GenerationResponse(
                text=first,
                model=request.model or "fake-model",
                token_usage=TokenUsage(),
            )
        if second:
            yield GenerationResponse(
                text=second,
                model=request.model or "fake-model",
                token_usage=TokenUsage(),
            )
        # Final usage-only event.
        yield GenerationResponse(
            text="",
            model=request.model or "fake-model",
            token_usage=self.usage,
        )


@pytest.fixture
def fake_llm() -> FakeLLMProvider:
    """A reply-with-citation LLM by default — exercises the inline-marker path."""
    return FakeLLMProvider(
        reply="Revenue grew 8% [1] and margin compressed [2].",
    )


# ---------------------------------------------------------------------------
# Retrieval service double
# ---------------------------------------------------------------------------


class FakeRetrievalService:
    """Stand-in for :class:`RetrievalService` — records call args and returns canned chunks.

    Matches the public ``retrieve(query, *, top_k, ticker, form_type, ...)``
    signature of the real service. Returns a copy of the configured
    chunks each call so tests can assert call ordering without
    cross-call mutation.
    """

    def __init__(
        self,
        results: list[RetrievalResult] | None = None,
        *,
        per_call_results: list[list[RetrievalResult]] | None = None,
    ) -> None:
        self._results = results or []
        self._per_call_results = per_call_results
        self.calls: list[dict] = []
        self.embed_calls: list[str] = []

    def embed_query(self, query: str) -> tuple[str, str]:
        """Record the call; return an opaque stand-in for the query vector."""
        self.embed_calls.append(query)
        return ("fake-vector", query)

    def retrieve(self, query: str, **kwargs) -> list[RetrievalResult]:
        self.calls.append({"query": query, **kwargs})
        if self._per_call_results is not None:
            idx = len(self.calls) - 1
            if idx < len(self._per_call_results):
                return list(self._per_call_results[idx])
            return []
        return list(self._results)


@pytest.fixture
def fake_retrieval(sample_chunks: list[RetrievalResult]) -> FakeRetrievalService:
    return FakeRetrievalService(results=sample_chunks)


# ---------------------------------------------------------------------------
# Settings reset — per-test isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_settings_singleton():
    """Reset the settings singleton between tests.

    Pydantic-settings v2 evaluates nested-settings defaults at class
    definition time, so ``reload_settings()`` does not flow ``RAG_*``
    env-var changes into ``settings.rag``.  Tests that need RAG-knob
    overrides mutate ``orchestrator._rag_settings`` directly (the
    pattern used in :mod:`tests.rag.test_orchestrator`).  This fixture
    just rebuilds the singleton so its top-level state stays clean
    across tests.
    """
    from sec_generative_search.config.settings import reload_settings

    reload_settings()
    yield
    reload_settings()


__all__ = ["FakeLLMProvider", "FakeRetrievalService"]
