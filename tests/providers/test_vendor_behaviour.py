"""End-to-end behavioural + invariant tests for the OpenAI-compatible vendors.

The eight OpenAI-compatible vendor adapters — Mistral, Kimi, DeepSeek,
Qwen, Z.ai, Grok, MiniMax, MiMo — are purely declarative: they add
nothing but a ``provider_name``, a ``default_base_url``, a
``default_model``, and a static catalogue on top of the shared
:mod:`providers.openai_compat` surface.

``test_openai_compat`` already exercises that shared surface exhaustively
through a synthetic ``_DemoLLM`` / ``_DemoEmbed`` double, and
``test_openai_compat_vendors`` pins each vendor's *static* declarations
(name / base_url / catalogue / repr).  Neither runs a real vendor through
``generate`` / ``generate_stream`` / ``embed_texts``.  This suite closes
that gap: every vendor is driven end-to-end against a mocked SDK so the
**vendor's own** ``default_model`` and catalogue are proven to flow into
the wire call, and the load-bearing vendor invariants get a regression
lock of their own:

- **Routing-hint isolation (security).**  Only
  :class:`OpenRouterProvider` forwards
  :class:`OpenRouterRoutingHints`; every other vendor MUST drop them
  before the SDK call so a routing channel can never silently ride a
  request to an unrelated upstream.  Asserted on both the streaming and
  non-streaming paths.
- **Cost safety (security).**  Omitting ``model`` must never land a
  caller on a ``PREMIUM``-tier model.
- **No adapter carries a tokeniser.**  Prompt budgeting uses the one
  shared offline ``cl100k_base`` counter (``search/retrieval.py``); the
  dead per-adapter ``count_tokens`` (F34) must not come back as a
  second, network-capable budgeting path.
- Vendor-specific endpoint / slug invariants (Grok's dated slugs,
    Z.ai's general-PaaS endpoint).

Every test mocks the ``OpenAI`` SDK factory — no real network call is
issued and no live API key is required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from sec_generative_search.core.exceptions import (
    ProviderAuthError,
    ProviderContentFilterError,
)
from sec_generative_search.core.resilience import RetryPolicy
from sec_generative_search.core.types import PricingTier
from sec_generative_search.providers import openai_compat
from sec_generative_search.providers.base import GenerationRequest
from sec_generative_search.providers.deepseek import DeepSeekProvider
from sec_generative_search.providers.grok import GrokProvider
from sec_generative_search.providers.kimi import KimiProvider
from sec_generative_search.providers.mimo import MimoProvider
from sec_generative_search.providers.minimax import MiniMaxProvider
from sec_generative_search.providers.mistral import (
    MistralEmbeddingProvider,
    MistralProvider,
)
from sec_generative_search.providers.openrouter import OpenRouterRoutingHints
from sec_generative_search.providers.qwen import QwenEmbeddingProvider, QwenProvider
from sec_generative_search.providers.registry import ProviderRegistry, ProviderSurface
from sec_generative_search.providers.zai import ZaiProvider

# ---------------------------------------------------------------------------
# Shared constants + deterministic SDK doubles
# ---------------------------------------------------------------------------


_LONG_KEY = "sk-test-BEHAVIOUR-ABCDEFGHIJKLMNOPQRSTUVWX"  # pragma: allowlist secret


# Every closed-catalogue vendor paired with the slug it serves by default.
# Kept as a literal table (not derived from ``cls.default_model``) so a
# silent default-model change trips a review here as well as in the
# declaration suite.
_LLM_VENDORS: list[tuple[type, str]] = [
    (MistralProvider, "mistral-small-2603"),
    (KimiProvider, "kimi-k2.5"),
    (DeepSeekProvider, "deepseek-v4-flash"),
    (QwenProvider, "qwen3.6-plus"),
    (ZaiProvider, "glm-5"),
    (GrokProvider, "grok-4-1-fast-non-reasoning"),
    (MiniMaxProvider, "minimax-m2.7"),
    (MimoProvider, "mimo-v2.5"),
]

_LLM_PARAMS = [pytest.param(cls, model, id=cls.provider_name) for cls, model in _LLM_VENDORS]


# Embedding vendors paired with default model + native dimension.
_EMBED_VENDORS: list[tuple[type, str, int]] = [
    (MistralEmbeddingProvider, "mistral-embed", 1024),
    (QwenEmbeddingProvider, "text-embedding-v4", 1024),
]

_EMBED_PARAMS = [
    pytest.param(cls, model, dim, id=cls.provider_name) for cls, model, dim in _EMBED_VENDORS
]


class _FakeAuthError(openai_compat.AuthenticationError):
    """SDK ``AuthenticationError`` that skips the heavy httpx constructor."""

    def __init__(self, message: str = "bad key") -> None:
        Exception.__init__(self, message)


def _fast_retry() -> RetryPolicy:
    """Zero-delay retry schedule so retried paths stay sub-millisecond."""
    return RetryPolicy(max_retries=2, backoff_base=2.0, initial_delay=0.0, max_delay=0.0)


def _make_completion(
    *,
    text: str = "hello",
    finish_reason: str = "stop",
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    model: str = "echo-model",
) -> MagicMock:
    """Build a non-streaming completion shape mirroring the SDK response."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    completion.model = model
    return completion


def _make_chunk(
    *,
    content: str | None = None,
    finish_reason: str | None = None,
    model: str = "echo-model",
    usage: tuple[int, int] | None = None,
) -> MagicMock:
    """Build a streaming-chunk shape (``choices[0].delta`` + optional usage)."""
    chunk = MagicMock()
    chunk.model = model
    if usage is not None:
        usage_obj = MagicMock()
        usage_obj.prompt_tokens, usage_obj.completion_tokens = usage
        chunk.usage = usage_obj
    else:
        chunk.usage = None
    if content is None and finish_reason is None and usage is not None:
        chunk.choices = []
        return chunk
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    return chunk


def _build(provider_cls: type, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, MagicMock]:
    """Instantiate *provider_cls* with its SDK client swapped for a MagicMock.

    Returns ``(provider, fake_client)`` so a test can stub responses on
    the client and assert on the captured call kwargs.  The fast retry
    policy keeps the retried error paths instantaneous.
    """
    fake_client = MagicMock()
    monkeypatch.setattr(openai_compat, "OpenAI", lambda **_kwargs: fake_client)
    provider = provider_cls(_LONG_KEY, retry_policy=_fast_retry())
    return provider, fake_client


# ---------------------------------------------------------------------------
# Generation — non-streaming + streaming, driven through the real vendor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("provider_cls", "default_model"), _LLM_PARAMS)
class TestVendorGeneration:
    def test_generate_uses_vendor_default_model_and_reports_usage(
        self,
        provider_cls: type,
        default_model: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty ``model`` resolves to the vendor's own ``default_model``.

        Proves the declared default actually reaches the wire call (the
        synthetic ``_DemoLLM`` in ``test_openai_compat`` cannot catch a
        per-vendor default-model regression) and that token accounting
        is wired through unchanged.
        """
        provider, client = _build(provider_cls, monkeypatch)
        client.chat.completions.create.return_value = _make_completion(
            text="grounded answer",
            prompt_tokens=13,
            completion_tokens=4,
        )

        response = provider.generate(GenerationRequest(prompt="summarise the 10-K", model=""))

        assert response.text == "grounded answer"
        assert response.token_usage.input_tokens == 13
        assert response.token_usage.output_tokens == 4
        assert response.finish_reason == "stop"
        sent = client.chat.completions.create.call_args.kwargs
        assert sent["model"] == default_model
        assert sent["stream"] is False

    def test_content_filter_finish_reason_is_terminal(
        self,
        provider_cls: type,
        default_model: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A ``content_filter`` stop maps to a terminal error, no retry."""
        del default_model
        provider, client = _build(provider_cls, monkeypatch)
        client.chat.completions.create.return_value = _make_completion(
            finish_reason="content_filter",
        )

        with pytest.raises(ProviderContentFilterError):
            provider.generate(GenerationRequest(prompt="x", model=""))
        # Terminal — surfaced from the body, never retried.
        assert client.chat.completions.create.call_count == 1

    def test_stream_yields_text_then_final_usage_frame(
        self,
        provider_cls: type,
        default_model: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Streaming emits per-chunk text then a usage-only closing frame."""
        provider, client = _build(provider_cls, monkeypatch)
        client.chat.completions.create.return_value = iter(
            [
                _make_chunk(content="part-", model=default_model),
                _make_chunk(content="two", model=default_model),
                _make_chunk(content=None, usage=(20, 6), model=default_model),
            ]
        )

        out = list(provider.generate_stream(GenerationRequest(prompt="hi", model="")))

        assert [r.text for r in out[:2]] == ["part-", "two"]
        assert out[-1].text == ""
        assert out[-1].token_usage.input_tokens == 20
        assert out[-1].token_usage.output_tokens == 6
        sent = client.chat.completions.create.call_args.kwargs
        assert sent["stream"] is True
        assert sent["stream_options"] == {"include_usage": True}

    def test_auth_error_is_terminal_and_not_retried(
        self,
        provider_cls: type,
        default_model: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An SDK ``AuthenticationError`` normalises to a terminal
        :class:`ProviderAuthError` with a single attempt for every vendor."""
        del default_model
        provider, client = _build(provider_cls, monkeypatch)
        client.chat.completions.create.side_effect = _FakeAuthError()

        with pytest.raises(ProviderAuthError):
            provider.generate(GenerationRequest(prompt="x", model=""))
        assert client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# Token budgeting stays offline (no SDK round-trip on the hot path)
# ---------------------------------------------------------------------------


def test_no_llm_adapter_carries_a_tokeniser() -> None:
    """F34: ``count_tokens`` had no caller; budgeting is the shared counter.

    Walks every registered LLM adapter (not just the OpenAI-wire vendors
    above) and the ABC itself.
    """
    from sec_generative_search.providers.base import BaseLLMProvider
    from sec_generative_search.providers.registry import ProviderRegistry, ProviderSurface

    classes = [
        entry.provider_cls
        for entry in ProviderRegistry.all_entries(ProviderSurface.LLM, include_unavailable=True)
    ]
    assert len(classes) >= 13
    for cls in [BaseLLMProvider, *classes]:
        assert not hasattr(cls, "count_tokens"), cls.__name__


# ---------------------------------------------------------------------------
# Security — routing hints never ride a non-OpenRouter request
# ---------------------------------------------------------------------------


_ROUTING_HINTS = OpenRouterRoutingHints(
    order=("openai", "anthropic"),
    allow_fallbacks=False,
    data_collection="deny",
)


@pytest.mark.security
@pytest.mark.parametrize(("provider_cls", "default_model"), _LLM_PARAMS)
class TestRoutingHintIsolation:
    """Only OpenRouter honours :class:`OpenRouterRoutingHints`.

    The hint is a routing channel, not an auth one, but letting it reach
    an unrelated vendor's wire would silently alter where a request is
    served and defeat the fail-closed guard the CLI / API enforce around
    ``supports_upstream_routing``.  Every non-OpenRouter vendor MUST drop
    the hint before the SDK call.
    """

    def test_hints_absent_from_non_streaming_call(
        self,
        provider_cls: type,
        default_model: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        del default_model
        provider, client = _build(provider_cls, monkeypatch)
        client.chat.completions.create.return_value = _make_completion()

        provider.generate(GenerationRequest(prompt="x", model="", routing_hints=_ROUTING_HINTS))

        kwargs = client.chat.completions.create.call_args.kwargs
        assert "extra_body" not in kwargs
        # Belt-and-braces: the hint must not have leaked under any key.
        assert "provider" not in kwargs
        assert "openai" not in repr(kwargs)

    def test_hints_absent_from_streaming_call(
        self,
        provider_cls: type,
        default_model: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        del default_model
        provider, client = _build(provider_cls, monkeypatch)
        client.chat.completions.create.return_value = iter(
            [_make_chunk(content=None, usage=(1, 1))]
        )

        list(
            provider.generate_stream(
                GenerationRequest(prompt="x", model="", routing_hints=_ROUTING_HINTS)
            )
        )

        kwargs = client.chat.completions.create.call_args.kwargs
        assert "extra_body" not in kwargs
        assert "provider" not in kwargs
        assert "openai" not in repr(kwargs)


# ---------------------------------------------------------------------------
# Security / cost safety — the default model is never PREMIUM-priced
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.parametrize(("provider_cls", "default_model"), _LLM_PARAMS)
def test_default_model_is_not_premium_tier(provider_cls: type, default_model: str) -> None:
    """Omitting ``model`` must not silently bill a caller at PREMIUM rates.

    A vendor whose flagship is its costliest slug must still default to a
    cheaper tier.
    """
    cap = ProviderRegistry.get_capability(
        provider_cls.provider_name, ProviderSurface.LLM, default_model
    )
    assert cap.pricing_tier is not PricingTier.PREMIUM
    # And the declared default must be a real, catalogued slug (not the
    # permissive-default fall-through) so the tier above is meaningful.
    assert provider_cls.default_model == default_model
    assert default_model in ProviderRegistry.list_models(
        provider_cls.provider_name, ProviderSurface.LLM
    )


# ---------------------------------------------------------------------------
# Vendor-specific invariants
# ---------------------------------------------------------------------------


def test_grok_preserves_dated_slugs_not_moving_aliases() -> None:
    """Grok pins dated slugs — a shorter moving alias 404s at xAI.

    The catalogue intentionally preserves dated slugs.
    """
    models = ProviderRegistry.list_models("grok", ProviderSurface.LLM)
    assert "grok-4.20-0309-reasoning" in models
    assert "grok-4-1-fast-non-reasoning" in models
    # Moving aliases that xAI does not serve authoritatively are absent.
    assert "grok-4" not in models
    assert "grok" not in models


def test_zai_targets_general_paas_not_coding_endpoint() -> None:
    """Z.ai uses the general PaaS endpoint, never the Coding-plan mirror.

    The catalogue intentionally targets the general PaaS endpoint.
    """
    assert ZaiProvider.default_base_url == "https://api.z.ai/api/paas/v4"
    assert "coding" not in ZaiProvider.default_base_url


# ---------------------------------------------------------------------------
# Embedding vendors — real adapter driven through the wire
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("provider_cls", "default_model", "dimension"), _EMBED_PARAMS)
class TestEmbeddingVendorBehaviour:
    def test_empty_input_bypasses_network(
        self,
        provider_cls: type,
        default_model: str,
        dimension: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        del default_model
        provider, client = _build(provider_cls, monkeypatch)

        out = provider.embed_texts([])

        assert out.shape == (0, dimension)
        client.embeddings.create.assert_not_called()

    def test_embed_texts_returns_dimension_shaped_float32(
        self,
        provider_cls: type,
        default_model: str,
        dimension: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider, client = _build(provider_cls, monkeypatch)
        item_a = MagicMock()
        item_a.embedding = [0.1] * dimension
        item_b = MagicMock()
        item_b.embedding = [0.2] * dimension
        response = MagicMock()
        response.data = [item_a, item_b]
        client.embeddings.create.return_value = response

        out = provider.embed_texts(["one", "two"])

        assert out.shape == (2, dimension)
        assert out.dtype == np.float32
        # The vendor's own default embedding model reaches the wire call.
        assert client.embeddings.create.call_args.kwargs["model"] == default_model

    def test_embed_query_returns_1d_vector(
        self,
        provider_cls: type,
        default_model: str,
        dimension: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        del default_model
        provider, client = _build(provider_cls, monkeypatch)
        item = MagicMock()
        item.embedding = [0.3] * dimension
        response = MagicMock()
        response.data = [item]
        client.embeddings.create.return_value = response

        out = provider.embed_query("what was net income?")

        assert out.shape == (dimension,)
