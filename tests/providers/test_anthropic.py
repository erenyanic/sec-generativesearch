"""Tests for :mod:`sec_generative_search.providers.anthropic`.

Covers the Anthropic-specific provider surface:

- Non-streaming generation: text extraction from typed content blocks,
  usage accounting, stop-reason normalisation.
- Streaming generation: per-chunk text deltas plus a final usage-only
  frame, stop-reason propagation, mid-stream refusal handling.
- Content-filter refusals (``stop_reason="refusal"``) raise
  :class:`ProviderContentFilterError` and are terminal.
- Error normalisation: ``AuthenticationError`` /
  ``PermissionDeniedError`` / ``RateLimitError`` /
  ``APITimeoutError`` / stdlib ``TimeoutError`` all map onto our
  :class:`ProviderError` subclasses via the shared resilience layer.
- Capability probe is O(1) — no SDK call required.
- Security: keys never appear in ``repr``, errors, or logs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from sec_generative_search.core.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.logging import LOGGER_NAME
from sec_generative_search.core.resilience import RetryPolicy
from sec_generative_search.core.types import PricingTier
from sec_generative_search.providers import anthropic as anthropic_mod
from sec_generative_search.providers.anthropic import (
    ANTHROPIC_EXCEPTION_MAPPING,
    AnthropicProvider,
)
from sec_generative_search.providers.base import GenerationRequest
from sec_generative_search.providers.registry import ProviderRegistry, ProviderSurface

# ---------------------------------------------------------------------------
# SDK exception subclasses that bypass the real SDK constructors.  We
# subclass the genuine Anthropic exception types so ``isinstance``
# checks in ``normalise_exception`` still match but the heavy
# ``httpx.Response`` construction is skipped.
# ---------------------------------------------------------------------------


class _FakeAuthError(anthropic_mod.AuthenticationError):
    def __init__(self, message: str = "bad key") -> None:
        Exception.__init__(self, message)


class _FakePermissionDenied(anthropic_mod.PermissionDeniedError):
    def __init__(self, message: str = "no access") -> None:
        Exception.__init__(self, message)


class _FakeRateLimit(anthropic_mod.RateLimitError):
    def __init__(self, message: str = "429") -> None:
        Exception.__init__(self, message)


class _FakeAPITimeout(anthropic_mod.APITimeoutError):
    def __init__(self, message: str = "timed out") -> None:
        Exception.__init__(self, message)


class _FakeAPIConnection(anthropic_mod.APIConnectionError):
    def __init__(self, message: str = "connection refused") -> None:
        Exception.__init__(self, message)


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


_LONG_KEY = "sk-ant-ABCDEFGHIJKLMNOPQRSTUVWX"  # pragma: allowlist secret
_KEY_TAIL = _LONG_KEY[-4:]


def _fast_retry() -> RetryPolicy:
    """Millisecond-scale retry policy so tests stay fast."""
    return RetryPolicy(max_retries=2, backoff_base=2.0, initial_delay=0.0, max_delay=0.0)


def _text_block(text: str, block_type: str = "text") -> MagicMock:
    block = MagicMock()
    block.type = block_type
    block.text = text
    return block


def _make_message(
    *,
    text: str = "hello",
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
    model: str = "claude-haiku-4-5",
) -> MagicMock:
    """Build a non-streaming ``Message`` shape."""
    message = MagicMock()
    message.content = [_text_block(text)] if text else []
    message.stop_reason = stop_reason
    message.model = model
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    message.usage = usage
    return message


def _make_event(event_type: str, **fields: Any) -> MagicMock:
    """Build a streaming event of the given type."""
    event = MagicMock()
    event.type = event_type
    for key, value in fields.items():
        setattr(event, key, value)
    return event


def _start_event(*, input_tokens: int = 10, output_tokens: int = 0) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    message = MagicMock()
    message.usage = usage
    return _make_event("message_start", message=message)


def _delta_event(text: str) -> MagicMock:
    delta = MagicMock()
    delta.text = text
    return _make_event("content_block_delta", delta=delta)


def _message_delta(*, stop_reason: str | None = "end_turn", output_tokens: int = 5) -> MagicMock:
    delta = MagicMock()
    delta.stop_reason = stop_reason
    usage = MagicMock()
    usage.output_tokens = output_tokens
    return _make_event("message_delta", delta=delta, usage=usage)


def _stop_event() -> MagicMock:
    return _make_event("message_stop")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> AnthropicProvider:
    fake_client = MagicMock()
    monkeypatch.setattr(anthropic_mod, "Anthropic", lambda **_kwargs: fake_client)
    prov = AnthropicProvider(_LONG_KEY, retry_policy=_fast_retry())
    prov._fake_client = fake_client  # type: ignore[attr-defined]
    return prov


# ---------------------------------------------------------------------------
# Metadata and catalogue
# ---------------------------------------------------------------------------


class TestProviderMetadata:
    def test_provider_name(self) -> None:
        assert AnthropicProvider.provider_name == "anthropic"

    def test_default_model(self) -> None:
        assert AnthropicProvider.default_model == "claude-haiku-4-5"

    def test_catalogue_has_tiered_models(self) -> None:
        def tier(slug: str) -> PricingTier:
            return ProviderRegistry.get_capability(
                "anthropic", ProviderSurface.LLM, slug
            ).pricing_tier

        assert tier("claude-opus-4-7") == PricingTier.PREMIUM
        assert tier("claude-sonnet-4-6") == PricingTier.PREMIUM
        assert tier("claude-haiku-4-5") == PricingTier.STANDARD

    def test_sdk_retries_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def factory(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return MagicMock()

        monkeypatch.setattr(anthropic_mod, "Anthropic", factory)
        AnthropicProvider(_LONG_KEY)
        # ``resilient_call`` owns the retry schedule.
        assert captured["max_retries"] == 0
        assert captured["api_key"] == _LONG_KEY


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


class TestCapabilityProbe:
    def test_known_model_cheap_probe(self, provider: AnthropicProvider) -> None:
        caps = provider.get_capabilities("claude-sonnet-4-6")
        assert caps.streaming is True
        assert caps.context_window_tokens == 1_000_000
        # No SDK call required — catalogue lookup is O(1).
        provider._fake_client.messages.create.assert_not_called()
        provider._fake_client.models.list.assert_not_called()

    def test_unknown_model_is_permissive(self, provider: AnthropicProvider) -> None:
        caps = provider.get_capabilities("claude-2023-hypothetical")
        assert caps.chat is True
        assert caps.streaming is True
        # The SDK gets a chance to reject the slug at call time.
        assert caps.context_window_tokens == 0


# ---------------------------------------------------------------------------
# Generation — non-streaming
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_text_and_usage(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.return_value = _make_message(
            text="42",
            input_tokens=12,
            output_tokens=3,
        )
        response = provider.generate(
            GenerationRequest(prompt="ultimate?", model="claude-haiku-4-5")
        )
        assert response.text == "42"
        assert response.token_usage.input_tokens == 12
        assert response.token_usage.output_tokens == 3
        assert response.finish_reason == "stop"

    def test_uses_default_model_when_empty(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.return_value = _make_message()
        provider.generate(GenerationRequest(prompt="hi", model=""))
        kwargs = provider._fake_client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-haiku-4-5"
        assert kwargs["stream"] is False

    def test_system_prompt_passed_as_field(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.return_value = _make_message()
        provider.generate(
            GenerationRequest(
                prompt="user message",
                model="claude-haiku-4-5",
                system="be terse",
            )
        )
        kwargs = provider._fake_client.messages.create.call_args.kwargs
        # Anthropic uses a top-level ``system`` field, distinct from messages.
        assert kwargs["system"] == "be terse"
        assert kwargs["messages"] == [{"role": "user", "content": "user message"}]

    def test_refusal_stop_reason_raises(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.return_value = _make_message(
            stop_reason="refusal",
        )
        with pytest.raises(ProviderContentFilterError):
            provider.generate(GenerationRequest(prompt="bad", model="claude-haiku-4-5"))

    def test_max_tokens_maps_to_length(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.return_value = _make_message(
            stop_reason="max_tokens",
        )
        response = provider.generate(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        assert response.finish_reason == "length"

    def test_non_text_blocks_are_ignored(self, provider: AnthropicProvider) -> None:
        message = _make_message()
        # A tool_use block should not bleed into the text surface.
        message.content = [_text_block("visible"), _text_block("", block_type="tool_use")]
        provider._fake_client.messages.create.return_value = message
        response = provider.generate(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        assert response.text == "visible"


# ---------------------------------------------------------------------------
# Generation — streaming
# ---------------------------------------------------------------------------


class TestGenerateStream:
    def test_yields_deltas_and_final_usage(self, provider: AnthropicProvider) -> None:
        events = [
            _start_event(input_tokens=20),
            _delta_event("hel"),
            _delta_event("lo"),
            _message_delta(stop_reason="end_turn", output_tokens=7),
            _stop_event(),
        ]
        provider._fake_client.messages.create.return_value = iter(events)

        results = list(
            provider.generate_stream(GenerationRequest(prompt="hi", model="claude-haiku-4-5"))
        )
        assert [r.text for r in results[:2]] == ["hel", "lo"]
        final = results[-1]
        assert final.text == ""
        assert final.token_usage.input_tokens == 20
        assert final.token_usage.output_tokens == 7
        assert final.finish_reason == "stop"

    def test_stream_passes_stream_flag(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.return_value = iter([_start_event(), _stop_event()])
        list(provider.generate_stream(GenerationRequest(prompt="hi", model="claude-haiku-4-5")))
        kwargs = provider._fake_client.messages.create.call_args.kwargs
        assert kwargs["stream"] is True

    def test_mid_stream_refusal_raises(self, provider: AnthropicProvider) -> None:
        events = [
            _start_event(),
            _delta_event("ok"),
            _message_delta(stop_reason="refusal", output_tokens=3),
        ]
        provider._fake_client.messages.create.return_value = iter(events)
        gen = provider.generate_stream(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        first = next(gen)
        assert first.text == "ok"
        with pytest.raises(ProviderContentFilterError):
            next(gen)

    def test_empty_delta_skipped(self, provider: AnthropicProvider) -> None:
        events = [
            _start_event(),
            _delta_event(""),
            _delta_event("real text"),
            _message_delta(output_tokens=2),
            _stop_event(),
        ]
        provider._fake_client.messages.create.return_value = iter(events)
        out = list(
            provider.generate_stream(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        )
        # Empty delta dropped; one text frame plus the final usage frame.
        assert [r.text for r in out] == ["real text", ""]


# ---------------------------------------------------------------------------
# Mid-stream SDK failure normalisation (F2)
#
# Iterating the returned event stream spans network round-trips outside
# the ``_call`` wrapper.  A connection drop / in-flight rate-limit mid
# iteration MUST normalise to the same ``ProviderError`` subclass the
# opening call would have produced, else the raw SDK type escapes the
# orchestrator's ``except ProviderError`` and is misclassified.
# ---------------------------------------------------------------------------


def _stream_then_raise(events: list[Any], exc: BaseException) -> Iterator[Any]:
    """Yield *events*, then raise *exc* — an SDK stream failing mid-flight."""
    yield from events
    raise exc


class TestMidStreamFailureNormalised:
    def test_connection_error_mid_stream_normalised(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.return_value = _stream_then_raise(
            [_start_event(), _delta_event("par")], _FakeAPIConnection()
        )
        gen = provider.generate_stream(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        assert next(gen).text == "par"
        with pytest.raises(ProviderConnectionError):
            next(gen)

    def test_rate_limit_mid_stream_normalised(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.return_value = _stream_then_raise(
            [_start_event(), _delta_event("hi")], _FakeRateLimit()
        )
        gen = provider.generate_stream(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        assert next(gen).text == "hi"
        with pytest.raises(ProviderRateLimitError):
            next(gen)

    def test_timeout_mid_stream_wins_over_connection(self, provider: AnthropicProvider) -> None:
        # ``APITimeoutError`` subclasses ``APIConnectionError``; a slow
        # read mid-stream must resolve to ProviderTimeoutError.
        provider._fake_client.messages.create.return_value = _stream_then_raise(
            [_start_event()], _FakeAPITimeout()
        )
        gen = provider.generate_stream(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        with pytest.raises(ProviderTimeoutError) as excinfo:
            next(gen)
        assert not isinstance(excinfo.value, ProviderConnectionError)

    def test_unknown_error_mid_stream_falls_back(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.return_value = _stream_then_raise(
            [_start_event()], RuntimeError("boom")
        )
        gen = provider.generate_stream(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        with pytest.raises(ProviderError):
            next(gen)

    def test_mid_stream_failure_not_retried(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.return_value = _stream_then_raise(
            [_start_event(), _delta_event("a")], _FakeRateLimit()
        )
        gen = provider.generate_stream(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        next(gen)
        with pytest.raises(ProviderRateLimitError):
            next(gen)
        assert provider._fake_client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# Iterator typing
# ---------------------------------------------------------------------------


def test_generate_stream_returns_iterator(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.messages.create.return_value = iter([_start_event(), _stop_event()])
    monkeypatch.setattr(anthropic_mod, "Anthropic", lambda **_kwargs: fake)
    prov = AnthropicProvider(_LONG_KEY, retry_policy=_fast_retry())
    stream = prov.generate_stream(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
    assert isinstance(stream, Iterator)


# ---------------------------------------------------------------------------
# Error normalisation
# ---------------------------------------------------------------------------


class TestErrorMapping:
    def test_auth_error_terminal(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.side_effect = _FakeAuthError()
        with pytest.raises(ProviderAuthError):
            provider.generate(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        assert provider._fake_client.messages.create.call_count == 1

    def test_permission_denied_is_auth_terminal(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.side_effect = _FakePermissionDenied()
        with pytest.raises(ProviderAuthError):
            provider.generate(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        assert provider._fake_client.messages.create.call_count == 1

    def test_rate_limit_retried(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.side_effect = _FakeRateLimit()
        with pytest.raises(ProviderRateLimitError):
            provider.generate(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        # Initial + 2 retries = 3 attempts under the fast retry policy.
        assert provider._fake_client.messages.create.call_count == 3

    def test_api_timeout_retried(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.side_effect = _FakeAPITimeout()
        with pytest.raises(ProviderTimeoutError):
            provider.generate(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        assert provider._fake_client.messages.create.call_count == 3

    def test_builtin_timeout_also_mapped(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.side_effect = TimeoutError("slow")
        with pytest.raises(ProviderTimeoutError):
            provider.generate(GenerationRequest(prompt="x", model="claude-haiku-4-5"))

    def test_eventual_success_after_retry(self, provider: AnthropicProvider) -> None:
        attempts: list[int] = []

        def side_effect(**_kwargs: Any) -> Any:
            attempts.append(1)
            if len(attempts) < 2:
                raise _FakeRateLimit()
            return _make_message(text="ok")

        provider._fake_client.messages.create.side_effect = side_effect
        response = provider.generate(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        assert response.text == "ok"
        assert len(attempts) == 2

    def test_unknown_exception_normalised(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.side_effect = RuntimeError("boom")
        with pytest.raises(ProviderError):
            provider.generate(GenerationRequest(prompt="x", model="claude-haiku-4-5"))


@pytest.mark.security
class TestRetryAfterFromTheSdk:
    """F16(b): Anthropic's ``Retry-After`` header reaches the retry loop."""

    @staticmethod
    def _sdk_rate_limit(retry_after: str) -> anthropic_mod.RateLimitError:
        response = httpx.Response(
            429,
            headers={"retry-after": retry_after},
            request=httpx.Request("POST", "https://example.invalid/v1/messages"),
        )
        return anthropic_mod.RateLimitError("rate limited", response=response, body=None)

    def test_retry_after_beyond_the_cap_is_not_retried(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.side_effect = self._sdk_rate_limit("45")
        with pytest.raises(ProviderRateLimitError) as info:
            provider.generate(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        assert info.value.retry_after == 45.0
        assert provider._fake_client.messages.create.call_count == 1

    def test_zero_retry_after_keeps_the_budget(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.side_effect = self._sdk_rate_limit("0")
        with pytest.raises(ProviderRateLimitError):
            provider.generate(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        assert provider._fake_client.messages.create.call_count == 3


class TestValidation:
    def test_validate_key_calls_models_list(self, provider: AnthropicProvider) -> None:
        provider._fake_client.models.list.return_value = MagicMock()
        assert provider.validate_key() is True
        assert provider._fake_client.models.list.call_count == 1

    def test_validate_key_auth_failure_propagates(self, provider: AnthropicProvider) -> None:
        provider._fake_client.models.list.side_effect = _FakeAuthError()
        with pytest.raises(ProviderAuthError):
            provider.validate_key()


# ---------------------------------------------------------------------------
# Mapping invariants
# ---------------------------------------------------------------------------


class TestExceptionMapping:
    def test_mapping_contains_expected_types(self) -> None:
        assert anthropic_mod.AuthenticationError in ANTHROPIC_EXCEPTION_MAPPING.auth
        assert anthropic_mod.PermissionDeniedError in ANTHROPIC_EXCEPTION_MAPPING.auth
        assert anthropic_mod.RateLimitError in ANTHROPIC_EXCEPTION_MAPPING.rate_limit
        assert anthropic_mod.APITimeoutError in ANTHROPIC_EXCEPTION_MAPPING.timeout

    def test_content_filter_is_response_signal_only(self) -> None:
        # Safety refusals come from ``stop_reason="refusal"``, never as
        # an SDK exception — keep the mapping clean of content-filter
        # types so we never normalise an unrelated error into one.
        assert ANTHROPIC_EXCEPTION_MAPPING.content_filter == ()


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_count_tokens_is_positive(self, provider: AnthropicProvider) -> None:
        assert provider.count_tokens("Apple Inc. reported strong revenues.") > 0

    def test_count_tokens_caches_encoder(self, provider: AnthropicProvider) -> None:
        provider.count_tokens("warm up")
        encoder = provider._encoder
        assert encoder is not None
        provider.count_tokens("again")
        assert provider._encoder is encoder


# ---------------------------------------------------------------------------
# Security — keys must never leak
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSecretSafety:
    def test_repr_masks_key(self, provider: AnthropicProvider) -> None:
        text = repr(provider)
        assert _LONG_KEY not in text
        assert _KEY_TAIL in text

    def test_provider_error_does_not_leak_key(self, provider: AnthropicProvider) -> None:
        provider._fake_client.messages.create.side_effect = _FakeAuthError("mentions key maybe")
        with pytest.raises(ProviderAuthError) as excinfo:
            provider.generate(GenerationRequest(prompt="x", model="claude-haiku-4-5"))
        assert _LONG_KEY not in str(excinfo.value)
        assert _LONG_KEY not in str(excinfo.value.details or "")
        assert _LONG_KEY not in str(excinfo.value.hint or "")

    def test_logger_emissions_redact_key(
        self,
        caplog: pytest.LogCaptureFixture,
        provider: AnthropicProvider,
    ) -> None:
        package_logger = logging.getLogger(LOGGER_NAME)
        previous = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
                provider._fake_client.messages.create.return_value = _make_message()
                provider.generate(GenerationRequest(prompt="hi", model="claude-haiku-4-5"))
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
    """``close()`` deterministically tears down the real Anthropic transport.

    The per-request build sites close the provider in a ``finally`` so the
    credential-bearing httpx pool is released at end-of-request rather than at
    GC. Uses a *real* ``Anthropic`` client (not the mocked fixture) so the
    teardown is genuinely observable.
    """

    def test_close_closes_real_transport(self) -> None:
        prov = AnthropicProvider(_LONG_KEY)
        assert prov._client.is_closed() is False
        prov.close()
        assert prov._client.is_closed() is True

    def test_close_is_idempotent_and_quiet(self) -> None:
        prov = AnthropicProvider(_LONG_KEY)
        prov.close()
        # Called from route / producer-thread ``finally`` — must not raise.
        assert prov.close() is None
        assert prov._client.is_closed() is True
