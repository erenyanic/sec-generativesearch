"""Tests for :mod:`sec_generative_search.providers.base`.

Covers:

- ABC instantiation guards (missing abstract methods must refuse to
  instantiate).
- ``provider_name`` class-attribute check.
- ``api_key`` validation (non-empty, correct type).
- ``__repr__`` / ``__str__`` redact the stored API key via
  :func:`mask_secret`.
- ``__repr__`` is marked :data:`typing.final` so type checkers refuse
  subclass overrides.
- Default :meth:`embed_chunks` delegates to :meth:`embed_texts` and
  satisfies the orchestrator's ``ChunkEmbedder`` protocol.
- Security: raw API keys never appear in ``repr()``, ``str()``, or
  :mod:`logging` records emitted by the base layer.

Concrete-provider suites cover vendor-specific behaviour elsewhere;
these tests focus on the interface contract only.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import fields
from datetime import date
from typing import TYPE_CHECKING

import pytest

from sec_generative_search.core.logging import LOGGER_NAME
from sec_generative_search.core.types import (
    Chunk,
    ContentType,
    FilingIdentifier,
    ProviderCapability,
    TokenUsage,
)
from sec_generative_search.pipeline.orchestrator import ChunkEmbedder
from sec_generative_search.providers import base
from sec_generative_search.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    BaseRerankerProvider,
    GenerationRequest,
    GenerationResponse,
    RerankResult,
)

if TYPE_CHECKING:
    import numpy as np
else:
    np = pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# Fixtures — concrete test doubles
# ---------------------------------------------------------------------------


_LONG_KEY = "sk-proj-ABCDEFGHIJKLMNOPQRSTUV"
_LONG_KEY_TAIL = _LONG_KEY[-4:]


class _FakeLLM(BaseLLMProvider):
    """Minimal concrete provider used to exercise the ABC surface."""

    provider_name = "fake-llm"

    def validate_key(self) -> bool:
        return True

    def get_capabilities(self) -> ProviderCapability:
        return ProviderCapability(chat=True)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text=f"echo: {request.prompt}",
            model=request.model,
            token_usage=TokenUsage(input_tokens=1, output_tokens=1),
        )

    def generate_stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationResponse]:
        yield GenerationResponse(text="part", model=request.model)
        yield GenerationResponse(
            text="",
            model=request.model,
            token_usage=TokenUsage(input_tokens=2, output_tokens=2),
        )


class _FakeEmbedder(BaseEmbeddingProvider):
    provider_name = "fake-embed"

    def validate_key(self) -> bool:
        return True

    def get_capabilities(self) -> ProviderCapability:
        return ProviderCapability(embeddings=True)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 4), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        del text
        return np.zeros(4, dtype=np.float32)

    def get_dimension(self) -> int:
        return 4


class _FakeReranker(BaseRerankerProvider):
    provider_name = "fake-rerank"

    def validate_key(self) -> bool:
        return True

    def get_capabilities(self) -> ProviderCapability:
        return ProviderCapability()

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        del query
        results = [RerankResult(index=i, score=1.0 / (i + 1)) for i in range(len(documents))]
        return results[:top_k] if top_k is not None else results


class _LLMMissingProviderName(BaseLLMProvider):
    # No provider_name override — instantiation must fail.

    def validate_key(self) -> bool:  # pragma: no cover - never reached
        return True

    def get_capabilities(self) -> ProviderCapability:  # pragma: no cover
        return ProviderCapability()

    def generate(self, request: GenerationRequest) -> GenerationResponse:  # pragma: no cover
        return GenerationResponse(text="", model=request.model)

    def generate_stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationResponse]:  # pragma: no cover
        yield GenerationResponse(text="", model=request.model)


# ---------------------------------------------------------------------------
# ABC instantiation guards
# ---------------------------------------------------------------------------


class TestAbstractnessGuards:
    def test_llm_abc_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseLLMProvider("key-value-12345")  # type: ignore[abstract]

    def test_embedding_abc_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseEmbeddingProvider("key-value-12345")  # type: ignore[abstract]

    def test_reranker_abc_cannot_be_instantiated_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseRerankerProvider("key-value-12345")  # type: ignore[abstract]

    def test_missing_provider_name_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="provider_name"):
            _LLMMissingProviderName("key-value-12345")


# ---------------------------------------------------------------------------
# API key validation
# ---------------------------------------------------------------------------


class TestApiKeyValidation:
    def test_empty_string_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            _FakeLLM("")

    def test_non_string_key_rejected(self) -> None:
        with pytest.raises(TypeError, match="api_key"):
            _FakeLLM(b"bytes-are-not-str")  # type: ignore[arg-type]

    def test_valid_key_accepted(self) -> None:
        provider = _FakeLLM(_LONG_KEY)
        # Stored under a leading-underscore attribute — the name signals
        # 'private' to tooling and reviewers.
        assert provider._api_key == _LONG_KEY


# ---------------------------------------------------------------------------
# Repr / str redaction
# ---------------------------------------------------------------------------


class TestReprRedaction:
    def test_repr_masks_long_key(self) -> None:
        provider = _FakeLLM(_LONG_KEY)
        text = repr(provider)
        assert _LONG_KEY not in text
        # Only the trailing four characters are allowed to leak.
        assert _LONG_KEY_TAIL in text
        assert "***" in text

    def test_str_mirrors_repr(self) -> None:
        provider = _FakeLLM(_LONG_KEY)
        assert str(provider) == repr(provider)

    def test_repr_masks_short_key_fully(self) -> None:
        provider = _FakeLLM("tiny-key")  # exactly 8 chars — tail allowed
        text = repr(provider)
        # Tail of length 4 is exposed by the mask_secret contract.
        assert text.count("tiny") == 0  # prefix is always obscured
        assert "y-ke" not in text  # only the final 4 chars may appear
        assert "-key" in text

    def test_repr_fully_masks_very_short_key(self) -> None:
        provider = _FakeLLM("abc")  # below min-length threshold
        text = repr(provider)
        assert "abc" not in text

    def test_repr_includes_provider_name(self) -> None:
        assert "fake-llm" in repr(_FakeLLM(_LONG_KEY))


# ---------------------------------------------------------------------------
# Final decorator sanity
# ---------------------------------------------------------------------------


class TestFinalRepr:
    def test_repr_is_marked_final(self) -> None:
        """``typing.final`` sets a dunder on the wrapped function."""
        # ``@final`` sets ``__final__ = True`` on the decorated callable.
        assert getattr(base._ProviderBase.__repr__, "__final__", False) is True
        assert getattr(base._ProviderBase.__str__, "__final__", False) is True


# ---------------------------------------------------------------------------
# Embedding ABC — default embed_chunks
# ---------------------------------------------------------------------------


class TestEmbeddingBaseDefaults:
    def test_embed_chunks_default_delegates_to_embed_texts(self) -> None:
        embedder = _FakeEmbedder(_LONG_KEY)
        filing = FilingIdentifier(
            ticker="AAPL",
            form_type="10-K",
            filing_date=date(2023, 11, 3),
            accession_number="acc-1",
        )
        chunks = [
            Chunk(
                content="chunk one",
                path="Part I > Item 1",
                content_type=ContentType.TEXT,
                filing_id=filing,
                chunk_index=0,
            ),
            Chunk(
                content="chunk two",
                path="Part I > Item 2",
                content_type=ContentType.TEXT,
                filing_id=filing,
                chunk_index=1,
            ),
        ]
        out = embedder.embed_chunks(chunks, show_progress=True)  # flag ignored
        assert out.shape == (2, 4)

    def test_embedding_provider_is_a_chunk_embedder(self) -> None:
        """Every concrete embedding provider must satisfy the orchestrator's
        structural :class:`ChunkEmbedder` protocol so that
        :class:`PipelineOrchestrator` accepts it without an adapter.

        ``ChunkEmbedder`` is not ``@runtime_checkable``, so we verify the
        structural contract by attribute lookup and a smoke invocation
        rather than ``isinstance``.
        """
        # Keep the import reachable — it documents the dependency the
        # assertions below are enforcing.
        assert ChunkEmbedder is not None
        embedder = _FakeEmbedder(_LONG_KEY)
        assert callable(embedder.embed_chunks)
        filing = FilingIdentifier(
            ticker="AAPL",
            form_type="10-K",
            filing_date=date(2023, 11, 3),
            accession_number="acc-1",
        )
        smoke_chunks = [
            Chunk(
                content="smoke",
                path="Part I",
                content_type=ContentType.TEXT,
                filing_id=filing,
                chunk_index=0,
            )
        ]
        assert embedder.embed_chunks(smoke_chunks, show_progress=False).shape == (1, 4)


# ---------------------------------------------------------------------------
# Reranker behaviour
# ---------------------------------------------------------------------------


class TestRerankerBase:
    def test_rerank_returns_top_k(self) -> None:
        reranker = _FakeReranker(_LONG_KEY)
        results = reranker.rerank("q", ["a", "b", "c"], top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, RerankResult) for r in results)

    def test_rerank_without_top_k_returns_all(self) -> None:
        reranker = _FakeReranker(_LONG_KEY)
        results = reranker.rerank("q", ["a", "b", "c"])
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Request / response dataclass sanity
# ---------------------------------------------------------------------------


class TestRequestResponseDataclasses:
    def test_generation_request_defaults(self) -> None:
        req = GenerationRequest(prompt="hello", model="gpt-4o")
        assert req.temperature == pytest.approx(0.1)
        assert req.max_output_tokens == 2048
        assert req.system is None

    def test_generation_response_token_usage_defaults(self) -> None:
        resp = GenerationResponse(text="hi", model="gpt-4o")
        assert resp.token_usage.total_tokens == 0
        assert resp.finish_reason == "stop"


# ---------------------------------------------------------------------------
# close() / context-manager — deterministic client teardown
# ---------------------------------------------------------------------------


class _RecordingClient:
    """Stand-in for an SDK client's httpx pool that records teardown."""

    def __init__(self) -> None:
        self.is_closed = False
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.is_closed = True


class _RaisingClient:
    """Client whose close() raises — proves ``provider.close()`` is quiet."""

    def close(self) -> None:
        raise RuntimeError("transport teardown boom")


class _ClientOwningLLM(_FakeLLM):
    """``_FakeLLM`` that owns a ``_client`` like every hosted adapter."""

    provider_name = "fake-llm-with-client"

    def __init__(self, api_key: str, client: object) -> None:
        super().__init__(api_key)
        self._client = client


class TestProviderClose:
    def test_close_releases_owned_client(self) -> None:
        client = _RecordingClient()
        provider = _ClientOwningLLM(_LONG_KEY, client)
        provider.close()
        assert client.is_closed is True
        assert client.close_calls == 1

    def test_context_manager_closes_on_exit(self) -> None:
        client = _RecordingClient()
        with _ClientOwningLLM(_LONG_KEY, client) as provider:
            assert provider is not None
            assert client.is_closed is False
        assert client.is_closed is True

    def test_context_manager_closes_even_on_exception(self) -> None:
        client = _RecordingClient()
        with pytest.raises(ValueError, match="boom"), _ClientOwningLLM(_LONG_KEY, client):
            raise ValueError("boom")
        assert client.is_closed is True

    def test_close_is_noop_without_client(self) -> None:
        # The on-device embedder owns a torch model, not an httpx client;
        # the inherited close() must be a safe no-op there.
        provider = _FakeLLM(_LONG_KEY)
        assert not hasattr(provider, "_client")
        assert provider.close() is None  # must not raise

    def test_close_is_idempotent(self) -> None:
        client = _RecordingClient()
        provider = _ClientOwningLLM(_LONG_KEY, client)
        provider.close()
        provider.close()
        assert client.close_calls == 2  # each call forwards; never raises

    def test_close_is_quiet_when_client_close_raises(self) -> None:
        # close() is called from route/thread ``finally`` blocks — a failed
        # transport teardown must never mask the request's own exception.
        provider = _ClientOwningLLM(_LONG_KEY, _RaisingClient())
        assert provider.close() is None  # swallowed, not propagated


# ---------------------------------------------------------------------------
# Security — raw key must never appear in logs emitted by the base layer
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestKeyNeverLeaksToLogs:
    def test_repr_never_contains_full_key(self) -> None:
        provider = _FakeLLM(_LONG_KEY)
        assert _LONG_KEY not in repr(provider)

    def test_format_never_contains_full_key(self) -> None:
        provider = _FakeLLM(_LONG_KEY)
        assert _LONG_KEY not in f"{provider}"
        assert _LONG_KEY not in f"{provider!r}"

    def test_package_loggers_do_not_see_key_from_repr(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The base ABC must not route the raw key through any logger.

        ``configure_logging`` disables propagation on the package
        logger, so we force-propagate for the duration of this test to
        let ``caplog`` observe emissions (same pattern used elsewhere
        in ``tests/core/test_logging.py``).
        """
        package_logger = logging.getLogger(LOGGER_NAME)
        previous = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
                provider = _FakeLLM(_LONG_KEY)
                # Any code path that touches ``repr`` should still not
                # emit the raw key.
                package_logger.info("provider=%s", provider)
        finally:
            package_logger.propagate = previous

        for record in caplog.records:
            assert _LONG_KEY not in record.getMessage()
            # And format a second time via the handler's formatter to
            # catch the case where the attribute itself carries the key.
            assert _LONG_KEY not in str(record.args)


# ---------------------------------------------------------------------------
# Security — no credential-bearing field names on provider dataclasses
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
class TestNoCredentialFieldsOnProviderDataclasses:
    """Mirror the domain-type credential-field check for every
    dataclass introduced in :mod:`providers.base`.

    Domain types already fail fast on this via
    :mod:`tests.core.test_types`; provider-layer dataclasses get the
    same guard so a future refactor cannot sneak a key onto a request
    or response body.
    """

    @pytest.mark.parametrize(
        "cls",
        [GenerationRequest, GenerationResponse, RerankResult],
    )
    def test_no_secret_looking_fields(self, cls: type) -> None:
        for f in fields(cls):
            lowered = f.name.lower()
            for hint in _SECRET_FIELD_HINTS:
                assert hint not in lowered, (
                    f"{cls.__name__}.{f.name} looks credential-bearing; "
                    "provider dataclasses must not carry secrets."
                )
