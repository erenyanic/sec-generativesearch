"""Tests for :mod:`sec_generative_search.providers.gemini`.

Covers both the chat and embedding surfaces:

- Non-streaming generation: text extraction, usage accounting, finish-
  reason normalisation.
- Streaming generation: per-chunk deltas, final usage-only frame,
  mid-stream safety blocks.
- Error translation: the genai SDK's single :class:`APIError` hierarchy
  is classified by HTTP status into auth / rate-limit / timeout /
  generic buckets, and terminal errors are not retried.
- Prompt-level safety blocks (``prompt_feedback.block_reason``) raise
  :class:`ProviderContentFilterError` directly.
- Capability probe is O(1) — no SDK call required.
- Embeddings: empty input avoids the network; non-empty input builds a
  float32 array of the declared dimension; unknown models are rejected
  at construction time.
- Security: keys never appear in ``repr``, errors, or logs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from sec_generative_search.core.exceptions import (
    ProviderAuthError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.logging import LOGGER_NAME
from sec_generative_search.core.resilience import RetryPolicy
from sec_generative_search.core.types import PricingTier, ProviderCapability
from sec_generative_search.providers import gemini as gemini_mod
from sec_generative_search.providers.base import GenerationRequest
from sec_generative_search.providers.gemini import (
    GEMINI_EXCEPTION_MAPPING,
    GeminiEmbeddingProvider,
    GeminiProvider,
)
from sec_generative_search.providers.registry import ProviderRegistry, ProviderSurface

# ---------------------------------------------------------------------------
# Fake ``APIError`` — bypasses the real ``__init__`` so tests can
# construct one with just the status code.  ``isinstance`` checks in
# the translation layer still succeed because we subclass the genuine
# SDK exception types.
# ---------------------------------------------------------------------------


class _FakeAPIError(gemini_mod.errors.APIError):
    def __init__(self, code: int, message: str = "boom") -> None:
        Exception.__init__(self, f"{code} {message}")
        self.code = code
        self.message = message
        self.status = None
        self.details = None
        self.response = None


class _FakeClientError(gemini_mod.errors.ClientError, _FakeAPIError):
    pass


class _FakeServerError(gemini_mod.errors.ServerError, _FakeAPIError):
    pass


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


_LONG_KEY = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWX"  # pragma: allowlist secret
_KEY_TAIL = _LONG_KEY[-4:]


def _fast_retry() -> RetryPolicy:
    return RetryPolicy(max_retries=2, backoff_base=2.0, initial_delay=0.0, max_delay=0.0)


def _candidate(text: str, finish_reason: str = "STOP") -> MagicMock:
    """Build a ``Candidate``-shaped mock with a text part."""
    part = MagicMock()
    part.text = text
    content = MagicMock()
    content.parts = [part]
    candidate = MagicMock()
    candidate.content = content
    # Mirror the enum shape used by the real SDK.
    reason = MagicMock()
    reason.name = finish_reason
    candidate.finish_reason = reason
    return candidate


def _make_response(
    *,
    text: str = "hello",
    finish_reason: str = "STOP",
    prompt_tokens: int = 10,
    candidate_tokens: int = 5,
    model_version: str = "gemini-2.5-flash",
    prompt_block_reason: str | None = None,
) -> MagicMock:
    response = MagicMock()
    response.candidates = [_candidate(text, finish_reason)]
    response.text = text
    response.model_version = model_version
    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = candidate_tokens
    response.usage_metadata = usage
    if prompt_block_reason is not None:
        block = MagicMock()
        block.name = prompt_block_reason
        feedback = MagicMock()
        feedback.block_reason = block
        response.prompt_feedback = feedback
    else:
        response.prompt_feedback = None
    return response


def _stream_chunk(
    *,
    text: str,
    finish_reason: str | None = None,
    prompt_tokens: int | None = None,
    candidate_tokens: int | None = None,
) -> MagicMock:
    chunk = MagicMock()
    chunk.candidates = [_candidate(text, finish_reason or "STOP")]
    if finish_reason is None:
        # Suppress the finish-reason lookup on delta chunks.
        chunk.candidates[0].finish_reason = None
    chunk.text = text
    chunk.prompt_feedback = None
    if prompt_tokens is not None or candidate_tokens is not None:
        usage = MagicMock()
        usage.prompt_token_count = prompt_tokens or 0
        usage.candidates_token_count = candidate_tokens or 0
        chunk.usage_metadata = usage
    else:
        chunk.usage_metadata = None
    return chunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> GeminiProvider:
    fake_client = MagicMock()
    monkeypatch.setattr(gemini_mod.genai, "Client", lambda **_kwargs: fake_client)
    prov = GeminiProvider(_LONG_KEY, retry_policy=_fast_retry())
    prov._fake_client = fake_client  # type: ignore[attr-defined]
    return prov


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> GeminiEmbeddingProvider:
    fake_client = MagicMock()
    monkeypatch.setattr(gemini_mod.genai, "Client", lambda **_kwargs: fake_client)
    prov = GeminiEmbeddingProvider(_LONG_KEY, retry_policy=_fast_retry())
    prov._fake_client = fake_client  # type: ignore[attr-defined]
    return prov


# ---------------------------------------------------------------------------
# Metadata and construction
# ---------------------------------------------------------------------------


class TestProviderMetadata:
    def test_provider_name_consistent(self) -> None:
        assert GeminiProvider.provider_name == "gemini"
        assert GeminiEmbeddingProvider.provider_name == "gemini"

    def test_default_models(self) -> None:
        assert GeminiProvider.default_model == "gemini-3-flash-preview"
        assert GeminiEmbeddingProvider.default_model == "gemini-embedding-2"

    def test_catalogue_has_tiered_models(self) -> None:
        def cap(slug: str) -> ProviderCapability:
            return ProviderRegistry.get_capability("gemini", ProviderSurface.LLM, slug)

        assert cap("gemini-3.1-pro-preview").pricing_tier == PricingTier.PREMIUM
        assert cap("gemini-3.1-pro-preview-customtools").tool_use is True
        assert cap("gemini-3-flash-preview").pricing_tier == PricingTier.STANDARD
        assert cap("gemini-2.5-pro").pricing_tier == PricingTier.PREMIUM
        assert cap("gemini-2.5-flash-lite").pricing_tier == PricingTier.LOW
        assert cap("gemini-2.5-flash").pricing_tier == PricingTier.STANDARD

    def test_embedding_dimensions(self) -> None:
        assert GeminiEmbeddingProvider.MODEL_DIMENSIONS["gemini-embedding-2"] == 3072
        assert GeminiEmbeddingProvider.MODEL_DIMENSIONS["gemini-embedding-2-preview"] == 3072
        assert GeminiEmbeddingProvider.MODEL_DIMENSIONS["gemini-embedding-001"] == 3072

    def test_sdk_client_constructed_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def factory(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(gemini_mod.genai, "Client", factory)
        GeminiProvider(_LONG_KEY)
        assert captured["api_key"] == _LONG_KEY
        # ``http_options`` exists and carries the timeout.
        assert captured["http_options"] is not None

    def test_embed_rejects_unknown_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gemini_mod.genai, "Client", lambda **_kwargs: MagicMock())
        with pytest.raises(ValueError, match="Unknown embedding model"):
            GeminiEmbeddingProvider(_LONG_KEY, model="text-embedding-99")


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


class TestCapabilityProbe:
    def test_known_model_cheap_probe(self, provider: GeminiProvider) -> None:
        caps = provider.get_capabilities("gemini-2.5-flash-lite")
        assert caps.streaming is True
        assert caps.context_window_tokens == 1_048_576
        provider._fake_client.models.generate_content.assert_not_called()
        provider._fake_client.models.list.assert_not_called()

    def test_unknown_model_is_permissive(self, provider: GeminiProvider) -> None:
        caps = provider.get_capabilities("gemini-99")
        assert caps.chat is True
        assert caps.streaming is True
        assert caps.context_window_tokens == 0


# ---------------------------------------------------------------------------
# Generation — non-streaming
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_text_and_usage(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.return_value = _make_response(
            text="42",
            prompt_tokens=12,
            candidate_tokens=3,
        )
        response = provider.generate(
            GenerationRequest(prompt="ultimate?", model="gemini-2.5-flash")
        )
        assert response.text == "42"
        assert response.token_usage.input_tokens == 12
        assert response.token_usage.output_tokens == 3
        assert response.finish_reason == "stop"

    def test_uses_default_model_when_empty(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.return_value = _make_response()
        provider.generate(GenerationRequest(prompt="hi", model=""))
        kwargs = provider._fake_client.models.generate_content.call_args.kwargs
        assert kwargs["model"] == GeminiProvider.default_model

    def test_system_prompt_is_passed_on_config(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.return_value = _make_response()
        provider.generate(
            GenerationRequest(
                prompt="user message",
                model="gemini-2.5-flash",
                system="be terse",
            )
        )
        config = provider._fake_client.models.generate_content.call_args.kwargs["config"]
        assert config.system_instruction == "be terse"

    def test_safety_finish_reason_raises(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.return_value = _make_response(
            finish_reason="SAFETY",
            text="",
        )
        with pytest.raises(ProviderContentFilterError):
            provider.generate(GenerationRequest(prompt="bad", model="gemini-2.5-flash"))

    def test_prohibited_content_finish_reason_raises(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.return_value = _make_response(
            finish_reason="PROHIBITED_CONTENT",
            text="",
        )
        with pytest.raises(ProviderContentFilterError):
            provider.generate(GenerationRequest(prompt="bad", model="gemini-2.5-flash"))

    def test_max_tokens_maps_to_length(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.return_value = _make_response(
            finish_reason="MAX_TOKENS",
        )
        response = provider.generate(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        assert response.finish_reason == "length"

    def test_prompt_feedback_block_raises(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.return_value = _make_response(
            text="",
            finish_reason="STOP",
            prompt_block_reason="SAFETY",
        )
        with pytest.raises(ProviderContentFilterError):
            provider.generate(GenerationRequest(prompt="bad", model="gemini-2.5-flash"))


# ---------------------------------------------------------------------------
# Generation — streaming
# ---------------------------------------------------------------------------


class TestGenerateStream:
    def test_yields_deltas_and_final_usage(self, provider: GeminiProvider) -> None:
        chunks = [
            _stream_chunk(text="hel"),
            _stream_chunk(text="lo", prompt_tokens=20, candidate_tokens=4),
            _stream_chunk(text="", finish_reason="STOP", prompt_tokens=20, candidate_tokens=7),
        ]
        provider._fake_client.models.generate_content_stream.return_value = iter(chunks)

        results = list(
            provider.generate_stream(GenerationRequest(prompt="hi", model="gemini-2.5-flash"))
        )
        text_frames = [r for r in results if r.text]
        assert [r.text for r in text_frames] == ["hel", "lo"]
        final = results[-1]
        assert final.text == ""
        assert final.token_usage.input_tokens == 20
        assert final.token_usage.output_tokens == 7
        assert final.finish_reason == "stop"

    def test_mid_stream_safety_raises(self, provider: GeminiProvider) -> None:
        chunks = [
            _stream_chunk(text="ok"),
            _stream_chunk(text="", finish_reason="SAFETY"),
        ]
        provider._fake_client.models.generate_content_stream.return_value = iter(chunks)
        gen = provider.generate_stream(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        first = next(gen)
        assert first.text == "ok"
        with pytest.raises(ProviderContentFilterError):
            next(gen)

    def test_stream_passes_correct_model(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content_stream.return_value = iter(
            [_stream_chunk(text="", finish_reason="STOP")]
        )
        list(provider.generate_stream(GenerationRequest(prompt="hi", model="gemini-2.5-pro")))
        kwargs = provider._fake_client.models.generate_content_stream.call_args.kwargs
        assert kwargs["model"] == "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Mid-stream SDK failure normalisation (F2)
#
# The genai stream is iterated outside ``_call``, so a mid-iteration
# ``errors.APIError`` must be translated through the same
# ``_translate_api_error`` the opening call uses, and any other SDK
# failure through ``normalise_exception``.  Otherwise the raw SDK type
# escapes the orchestrator's ``except ProviderError`` and is
# misclassified as ``internal_error``.
# ---------------------------------------------------------------------------


def _stream_then_raise(chunks: list[Any], exc: BaseException) -> Iterator[Any]:
    """Yield *chunks*, then raise *exc* — an SDK stream failing mid-flight."""
    yield from chunks
    raise exc


class TestMidStreamFailureNormalised:
    def test_rate_limit_api_error_mid_stream_translated(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content_stream.return_value = _stream_then_raise(
            [_stream_chunk(text="par")], _FakeClientError(429)
        )
        gen = provider.generate_stream(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        assert next(gen).text == "par"
        with pytest.raises(ProviderRateLimitError):
            next(gen)

    def test_auth_api_error_mid_stream_translated(self, provider: GeminiProvider) -> None:
        # 401 is terminal — the translation still applies mid-stream.
        provider._fake_client.models.generate_content_stream.return_value = _stream_then_raise(
            [_stream_chunk(text="par")], _FakeClientError(401)
        )
        gen = provider.generate_stream(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        assert next(gen).text == "par"
        with pytest.raises(ProviderAuthError):
            next(gen)

    def test_server_api_error_mid_stream_translated(self, provider: GeminiProvider) -> None:
        # 503 falls outside the known buckets → generic ProviderError,
        # still a ``ProviderError`` the orchestrator can classify.
        provider._fake_client.models.generate_content_stream.return_value = _stream_then_raise(
            [_stream_chunk(text="par")], _FakeServerError(503)
        )
        gen = provider.generate_stream(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        assert next(gen).text == "par"
        with pytest.raises(ProviderError):
            next(gen)

    def test_timeout_mid_stream_normalised(self, provider: GeminiProvider) -> None:
        # A stdlib ``TimeoutError`` (not an ``errors.APIError``) routes
        # through the mapping to ProviderTimeoutError.
        provider._fake_client.models.generate_content_stream.return_value = _stream_then_raise(
            [], TimeoutError("slow")
        )
        gen = provider.generate_stream(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        with pytest.raises(ProviderTimeoutError):
            next(gen)

    def test_unknown_error_mid_stream_falls_back(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content_stream.return_value = _stream_then_raise(
            [], RuntimeError("boom")
        )
        gen = provider.generate_stream(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        with pytest.raises(ProviderError):
            next(gen)

    def test_mid_stream_failure_not_retried(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content_stream.return_value = _stream_then_raise(
            [_stream_chunk(text="a")], _FakeClientError(429)
        )
        gen = provider.generate_stream(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        next(gen)
        with pytest.raises(ProviderRateLimitError):
            next(gen)
        assert provider._fake_client.models.generate_content_stream.call_count == 1


def test_generate_stream_returns_iterator(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.models.generate_content_stream.return_value = iter(
        [_stream_chunk(text="", finish_reason="STOP")]
    )
    monkeypatch.setattr(gemini_mod.genai, "Client", lambda **_kwargs: fake)
    prov = GeminiProvider(_LONG_KEY, retry_policy=_fast_retry())
    stream = prov.generate_stream(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
    assert isinstance(stream, Iterator)


# ---------------------------------------------------------------------------
# Error translation — APIError classification
# ---------------------------------------------------------------------------


def _retry_info_body(delay: object) -> dict[str, Any]:
    """A Gemini 429 body carrying a ``google.rpc.RetryInfo`` detail."""
    return {
        "error": {
            "code": 429,
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.QuotaFailure"},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
            ],
        }
    }


@pytest.mark.security
class TestRetryAfter:
    """F16(b): a Gemini 429's retry delay reaches the retry loop.

    Google APIs carry it as ``google.rpc.RetryInfo`` in the (untrusted)
    JSON body; a ``Retry-After`` header, when present, wins.
    """

    @staticmethod
    def _rate_limit(*, body: object = None, headers: dict[str, str] | None = None) -> Any:
        exc = _FakeClientError(429)
        exc.details = body
        if headers is not None:
            exc.response = SimpleNamespace(headers=httpx.Headers(headers))
        return exc

    @pytest.mark.parametrize(("delay", "expected"), [("37s", 37.0), ("0.5s", 0.5)])
    def test_retry_info_body_is_parsed(self, delay: str, expected: float) -> None:
        err = gemini_mod._translate_api_error(
            self._rate_limit(body=_retry_info_body(delay)), provider="gemini"
        )
        assert isinstance(err, ProviderRateLimitError)
        assert err.retry_after == expected

    def test_header_wins_over_body(self) -> None:
        exc = self._rate_limit(body=_retry_info_body("37s"), headers={"retry-after": "4"})
        err = gemini_mod._translate_api_error(exc, provider="gemini")
        assert err.retry_after == 4.0  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "delay",
        ["abc", "-1s", "1e9s", "37", "9999999999s", "nans", "infs", 37, None, "37s " * 20],
    )
    def test_malformed_retry_delay_is_ignored(self, delay: object) -> None:
        err = gemini_mod._translate_api_error(
            self._rate_limit(body=_retry_info_body(delay)), provider="gemini"
        )
        assert err.retry_after is None  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "body",
        [None, "oops", {"error": "oops"}, {"error": {"details": "oops"}}, [1, 2]],
    )
    def test_malformed_body_is_ignored(self, body: object) -> None:
        err = gemini_mod._translate_api_error(self._rate_limit(body=body), provider="gemini")
        assert err.retry_after is None  # type: ignore[attr-defined]

    def test_detail_scan_is_bounded(self) -> None:
        body = _retry_info_body("5s")
        filler = [{"@type": "type.googleapis.com/google.rpc.Help"}] * 16
        body["error"]["details"] = filler + body["error"]["details"]
        err = gemini_mod._translate_api_error(self._rate_limit(body=body), provider="gemini")
        assert err.retry_after is None  # type: ignore[attr-defined]

    def test_retry_delay_beyond_the_cap_is_not_retried(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.side_effect = self._rate_limit(
            body=_retry_info_body("56s")
        )
        with pytest.raises(ProviderRateLimitError) as info:
            provider.generate(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        assert info.value.retry_after == 56.0
        assert provider._fake_client.models.generate_content.call_count == 1


class TestErrorMapping:
    def test_401_is_auth_and_terminal(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.side_effect = _FakeClientError(401)
        with pytest.raises(ProviderAuthError):
            provider.generate(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        assert provider._fake_client.models.generate_content.call_count == 1

    def test_403_is_auth_and_terminal(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.side_effect = _FakeClientError(403)
        with pytest.raises(ProviderAuthError):
            provider.generate(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        assert provider._fake_client.models.generate_content.call_count == 1

    def test_429_is_rate_limit_and_retried(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.side_effect = _FakeClientError(429)
        with pytest.raises(ProviderRateLimitError):
            provider.generate(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        # Initial + 2 retries.
        assert provider._fake_client.models.generate_content.call_count == 3

    def test_504_is_timeout_and_retried(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.side_effect = _FakeServerError(504)
        with pytest.raises(ProviderTimeoutError):
            provider.generate(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        assert provider._fake_client.models.generate_content.call_count == 3

    def test_500_falls_back_to_generic_and_retried(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.side_effect = _FakeServerError(500)
        with pytest.raises(ProviderError):
            provider.generate(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        assert provider._fake_client.models.generate_content.call_count == 3

    def test_builtin_timeout_also_mapped(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.side_effect = TimeoutError("slow")
        with pytest.raises(ProviderTimeoutError):
            provider.generate(GenerationRequest(prompt="x", model="gemini-2.5-flash"))

    def test_eventual_success_after_retry(self, provider: GeminiProvider) -> None:
        attempts: list[int] = []

        def side_effect(**_kwargs: Any) -> Any:
            attempts.append(1)
            if len(attempts) < 2:
                raise _FakeClientError(429)
            return _make_response(text="ok")

        provider._fake_client.models.generate_content.side_effect = side_effect
        response = provider.generate(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        assert response.text == "ok"
        assert len(attempts) == 2


class TestValidation:
    def test_validate_key_pulls_models_list(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.list.return_value = iter([MagicMock()])
        assert provider.validate_key() is True
        provider._fake_client.models.list.assert_called()

    def test_validate_key_propagates_auth_error(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.list.side_effect = _FakeClientError(401)
        with pytest.raises(ProviderAuthError):
            provider.validate_key()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_empty_input_avoids_network(self, embedder: GeminiEmbeddingProvider) -> None:
        out = embedder.embed_texts([])
        assert out.shape == (0, 3072)
        embedder._fake_client.models.embed_content.assert_not_called()

    def test_embed_texts_returns_array(self, embedder: GeminiEmbeddingProvider) -> None:
        item_a = MagicMock()
        item_a.values = [0.1] * 3072
        item_b = MagicMock()
        item_b.values = [0.2] * 3072
        response = MagicMock()
        response.embeddings = [item_a, item_b]
        embedder._fake_client.models.embed_content.return_value = response

        out = embedder.embed_texts(["one", "two"])
        assert out.shape == (2, 3072)
        assert out.dtype == np.float32

    def test_embed_query_returns_1d_vector(self, embedder: GeminiEmbeddingProvider) -> None:
        item = MagicMock()
        item.values = [0.3] * 3072
        response = MagicMock()
        response.embeddings = [item]
        embedder._fake_client.models.embed_content.return_value = response
        out = embedder.embed_query("hello")
        assert out.shape == (3072,)

    def test_embed_passes_correct_model(self, embedder: GeminiEmbeddingProvider) -> None:
        item = MagicMock()
        item.values = [0.0] * 3072
        response = MagicMock()
        response.embeddings = [item]
        embedder._fake_client.models.embed_content.return_value = response
        embedder.embed_texts(["x"])
        assert embedder._fake_client.models.embed_content.call_args.kwargs["model"] == (
            "gemini-embedding-2"
        )

    def test_embed_auth_error_terminal(self, embedder: GeminiEmbeddingProvider) -> None:
        embedder._fake_client.models.embed_content.side_effect = _FakeClientError(401)
        with pytest.raises(ProviderAuthError):
            embedder.embed_texts(["x"])
        assert embedder._fake_client.models.embed_content.call_count == 1


# ---------------------------------------------------------------------------
# Mapping and token counting
# ---------------------------------------------------------------------------


class TestExceptionMapping:
    def test_mapping_only_contains_timeout_error(self) -> None:
        # Auth and rate limit come in as ``errors.APIError`` with a
        # status code, so type-based mapping would miss them.  They are
        # translated inside the call wrapper instead.
        assert GEMINI_EXCEPTION_MAPPING.auth == ()
        assert GEMINI_EXCEPTION_MAPPING.rate_limit == ()
        assert GEMINI_EXCEPTION_MAPPING.timeout == (TimeoutError,)


# ---------------------------------------------------------------------------
# Security — keys must never leak
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSecretSafety:
    def test_repr_masks_key(self, provider: GeminiProvider) -> None:
        text = repr(provider)
        assert _LONG_KEY not in text
        assert _KEY_TAIL in text

    def test_embed_repr_masks_key(self, embedder: GeminiEmbeddingProvider) -> None:
        text = repr(embedder)
        assert _LONG_KEY not in text
        assert _KEY_TAIL in text

    def test_provider_error_does_not_leak_key(self, provider: GeminiProvider) -> None:
        provider._fake_client.models.generate_content.side_effect = _FakeClientError(
            401, f"mentions {_LONG_KEY} maybe"
        )
        with pytest.raises(ProviderAuthError) as excinfo:
            provider.generate(GenerationRequest(prompt="x", model="gemini-2.5-flash"))
        # The translator copies ``str(exc)`` into ``details`` — we
        # verify the outer ``message`` and ``hint`` never carry the key.
        # ``details`` intentionally carries the SDK message verbatim
        # for debugging; sanitising it here is the caller's job
        # (downstream logging uses mask_secret).
        assert _LONG_KEY not in excinfo.value.message
        assert _LONG_KEY not in str(excinfo.value.hint or "")

    def test_logger_emissions_redact_key(
        self,
        caplog: pytest.LogCaptureFixture,
        provider: GeminiProvider,
    ) -> None:
        package_logger = logging.getLogger(LOGGER_NAME)
        previous = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
                provider._fake_client.models.generate_content.return_value = _make_response()
                provider.generate(GenerationRequest(prompt="hi", model="gemini-2.5-flash"))
                package_logger.info("constructed %s", provider)
        finally:
            package_logger.propagate = previous
        for record in caplog.records:
            assert _LONG_KEY not in record.getMessage()
            assert _LONG_KEY not in str(record.args)


# ---------------------------------------------------------------------------
# Security — per-request client teardown
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestClientTeardown:
    """``close()`` reaches the genai client so its transport is torn down.

    The genai ``Client`` exposes no observable ``is_closed`` flag (unlike the
    OpenAI / Anthropic wrappers), so closure is asserted by forwarding to the
    client's own ``close()``. The per-request build sites close the provider
    in a ``finally`` to release the credential-bearing transport at
    end-of-request rather than at GC.
    """

    def test_close_forwards_to_genai_client(self, provider: GeminiProvider) -> None:
        provider.close()
        provider._fake_client.close.assert_called_once_with()

    def test_close_on_real_client_is_quiet(self) -> None:
        prov = GeminiProvider(_LONG_KEY)
        # Real genai client; close must not raise and stays idempotent when
        # called again from a route / thread ``finally``.
        assert prov.close() is None
        assert prov.close() is None
