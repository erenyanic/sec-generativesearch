"""Parametrised vendor tests for the OpenAI-compatible subclasses.

Every vendor ships by configuring the shared
:mod:`providers.openai_compat` surface with nothing more than a
``provider_name``, a ``default_base_url``, a ``default_model``, and a
static catalogue.  The heavy behaviour — retry/circuit-breaker, content-
filter handling, streaming, token accounting — is already exhaustively
covered in ``test_openai_compat``; this suite therefore focuses on the
vendor-specific contract:

- Each provider declares a non-empty, reachable-looking ``base_url``.
- Each provider's ``default_model`` is present in its catalogue (or the
  catalogue is deliberately empty for the :class:`OpenRouterProvider`
  meta-provider).
- The O(1) capability probe returns the declared catalogue entry
  without any SDK round-trip.
- Unknown slugs on the meta-provider fall back to the permissive
  ``ProviderCapability(chat=True, streaming=True)`` default, preserving
  the "accepts any model slug" semantics of OpenRouter.
- Embedding variants expose their declared native dimension without a
  network call.
- Construction plumbing wires the vendor-specific ``base_url`` into the
  underlying ``OpenAI`` client exactly once.
- Security: the API key is never rendered in the provider's ``repr``.

Every test mocks the ``OpenAI`` SDK factory — no real network calls are
issued and no live API key is required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from sec_generative_search.providers import openai_compat
from sec_generative_search.providers.deepseek import DeepSeekProvider
from sec_generative_search.providers.grok import GrokProvider
from sec_generative_search.providers.kimi import KimiProvider
from sec_generative_search.providers.mimo import MimoProvider
from sec_generative_search.providers.minimax import MiniMaxProvider
from sec_generative_search.providers.mistral import (
    MistralEmbeddingProvider,
    MistralProvider,
)
from sec_generative_search.providers.openrouter import OpenRouterProvider
from sec_generative_search.providers.qwen import QwenEmbeddingProvider, QwenProvider
from sec_generative_search.providers.registry import ProviderRegistry, ProviderSurface
from sec_generative_search.providers.zai import ZaiProvider

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


_LONG_KEY = "sk-test-VENDOR-ABCDEFGHIJKLMNOPQRSTUVWX"  # pragma: allowlist secret
_KEY_TAIL = _LONG_KEY[-4:]


@pytest.fixture
def patched_openai(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the OpenAI client class with a MagicMock factory.

    Captures the kwargs so each test can assert on the ``base_url``
    passed by the vendor subclass.
    """
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> MagicMock:
        client = MagicMock()
        captured["client"] = client
        captured["kwargs"] = kwargs
        return client

    monkeypatch.setattr(openai_compat, "OpenAI", factory)
    return captured


# ---------------------------------------------------------------------------
# LLM vendor tables
#
# Two matrices so the OpenRouter meta-provider (empty catalogue by
# design) does not have to opt out at runtime via ``pytest.skip``:
#
# - ``_LLM_VENDORS`` covers every vendor and is used for the tests that
#   apply universally (name, base_url, default_model, construction,
#   capability probe, repr redaction).
# - ``_LLM_VENDORS_WITH_CATALOGUE`` adds a per-vendor spot-check slug
#   and therefore excludes :class:`OpenRouterProvider` — a meta-provider
#   that ships an empty catalogue cannot have a "non-default slug
#   declared in the catalogue".  The permissive-default semantics for
#   OpenRouter are asserted directly in ``test_openrouter_accepts_any_slug``.
# ---------------------------------------------------------------------------


_LLM_VENDORS: list[tuple[type, str, str, str]] = [
    (
        MistralProvider,
        "mistral",
        "https://api.mistral.ai/v1",
        "mistral-small-2603",
    ),
    (
        KimiProvider,
        "kimi",
        "https://api.moonshot.ai/v1",
        "kimi-k2.5",
    ),
    (
        DeepSeekProvider,
        "deepseek",
        "https://api.deepseek.com",
        "deepseek-flash",
    ),
    (
        QwenProvider,
        "qwen",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "qwen3.6-plus",
    ),
    (
        ZaiProvider,
        "zai",
        "https://api.z.ai/api/paas/v4",
        "glm-5",
    ),
    (
        GrokProvider,
        "grok",
        "https://api.x.ai/v1",
        "grok-4.3",
    ),
    (
        MiniMaxProvider,
        "minimax",
        "https://api.minimax.io/v1",
        "minimax-m2.7",
    ),
    (
        MimoProvider,
        "mimo",
        "https://api.xiaomimimo.com/v1",
        "mimo-v2.5",
    ),
    (
        OpenRouterProvider,
        "openrouter",
        "https://openrouter.ai/api/v1",
        "qwen/qwen3.6-plus",
    ),
]


# (ProviderClass, extra non-default catalogue slug to spot-check).  Keeps
# the parametrisation trivially introspectable and avoids an in-test
# ``pytest.skip`` branch for the meta-provider.
_LLM_VENDORS_WITH_CATALOGUE: list[tuple[type, str]] = [
    (MistralProvider, "mistral-large-2512"),
    (KimiProvider, "kimi-k2.6"),
    (DeepSeekProvider, "deepseek-v4-pro"),
    (QwenProvider, "qwen3-max"),
    (ZaiProvider, "glm-5"),
    (GrokProvider, "grok-4.20-0309-reasoning"),
    (MiniMaxProvider, "minimax-m2"),
    (MimoProvider, "mimo-v2-flash"),
]


_EMBED_VENDORS: list[tuple[type, str, str, str, int]] = [
    (
        MistralEmbeddingProvider,
        "mistral",
        "https://api.mistral.ai/v1",
        "mistral-embed",
        1024,
    ),
    (
        QwenEmbeddingProvider,
        "qwen",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "text-embedding-v4",
        1024,
    ),
]


# ---------------------------------------------------------------------------
# Class-level contract: provider_name, base_url, default_model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider_cls", "provider_name", "base_url", "default_model"),
    _LLM_VENDORS,
)
class TestLLMVendorDeclarations:
    def test_provider_name_matches(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
    ) -> None:
        assert provider_cls.provider_name == provider_name

    def test_default_base_url_matches(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
    ) -> None:
        assert provider_cls.default_base_url == base_url

    def test_default_model_is_declared(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
    ) -> None:
        assert provider_cls.default_model == default_model


@pytest.mark.parametrize(
    ("provider_cls", "default_model"),
    [(cls, model) for cls, _name, _url, model in _LLM_VENDORS if cls is not OpenRouterProvider],
)
def test_default_model_is_in_catalogue(provider_cls: type, default_model: str) -> None:
    """Non-meta providers pin ``default_model`` to a catalogue entry.

    This is what makes the capability probe truly O(1) — an unknown
    ``default_model`` would silently fall through to the permissive
    default branch and mask a copy-paste mistake in the catalogue.
    """
    models = ProviderRegistry.list_models(provider_cls.provider_name, ProviderSurface.LLM)
    assert default_model in models, (
        f"{provider_cls.__name__} default_model '{default_model}' missing from the catalogue"
    )


def test_openrouter_catalogue_is_empty_by_design() -> None:
    """The meta-provider's empty catalogue is load-bearing.

    Guarded here so a future "helpful" patch that adds OpenRouter rows to
    the vendored catalogue trips a test instead of silently narrowing the
    set of accepted slugs.  The capability probe relies on the permissive-
    default branch for any slug not in the catalogue; cataloguing
    OpenRouter would start rejecting slugs OpenRouter actually serves.
    """
    assert ProviderRegistry.list_models("openrouter", ProviderSurface.LLM) == []


@pytest.mark.parametrize(("provider_cls", "extra_slug"), _LLM_VENDORS_WITH_CATALOGUE)
def test_extra_catalogue_slug_present(provider_cls: type, extra_slug: str) -> None:
    """Spot-check a non-default slug per vendor to catch catalogue typos."""
    assert extra_slug in ProviderRegistry.list_models(
        provider_cls.provider_name, ProviderSurface.LLM
    )


# ---------------------------------------------------------------------------
# Construction contract: base_url reaches the SDK client unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider_cls", "provider_name", "base_url", "default_model"),
    _LLM_VENDORS,
)
def test_base_url_wired_into_client(
    provider_cls: type,
    provider_name: str,
    base_url: str,
    default_model: str,
    patched_openai: dict[str, Any],
) -> None:
    """Every LLM vendor subclass hands its ``default_base_url`` to the SDK."""
    del provider_name, default_model
    provider_cls(_LONG_KEY)
    kwargs = patched_openai["kwargs"]
    assert kwargs["base_url"] == base_url
    # Resilience is owned by ``resilient_call`` — the SDK's own retry
    # loop must stay disabled on every OpenAI-compatible subclass.
    assert kwargs["max_retries"] == 0


# ---------------------------------------------------------------------------
# Capability probe contract: O(1), no SDK round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider_cls", "provider_name", "base_url", "default_model"),
    _LLM_VENDORS,
)
def test_capability_probe_is_offline(
    provider_cls: type,
    provider_name: str,
    base_url: str,
    default_model: str,
    patched_openai: dict[str, Any],
) -> None:
    del provider_name, base_url
    provider = provider_cls(_LONG_KEY)
    client = patched_openai["client"]
    client.reset_mock()
    caps = provider.get_capabilities(default_model)
    assert caps.chat is True
    assert caps.streaming is True
    # O(1) probe: no SDK call required to inspect declared capabilities.
    client.models.list.assert_not_called()
    client.chat.completions.create.assert_not_called()


def test_openrouter_accepts_any_slug(patched_openai: dict[str, Any]) -> None:
    """OpenRouter's meta-provider contract: unknown slug → permissive default.

    This is the "accepts any model slug; lazy validation
    against OpenRouter's model list" contract.  Validation against the
    live model list is delegated to ``validate_key``/the registry
    registry; the capability probe deliberately stays permissive so the
    SDK call can proceed and the upstream API produces the authoritative
    error if the slug is unserviceable.
    """
    provider = OpenRouterProvider(_LONG_KEY)
    caps = provider.get_capabilities("some-vendor/model-that-may-not-exist")
    assert caps.chat is True
    assert caps.streaming is True
    # Permissive default carries no context window or output budget — the
    # true values depend on whichever upstream OpenRouter routes the slug
    # to, and the SDK call will surface any mismatch.
    assert not caps.context_window_tokens
    assert not caps.max_output_tokens


# ---------------------------------------------------------------------------
# Embedding providers: declared native dimensions, no network on empty input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider_cls", "provider_name", "base_url", "default_model", "dimension"),
    _EMBED_VENDORS,
)
class TestEmbeddingVendorDeclarations:
    def test_provider_name_matches(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
        dimension: int,
    ) -> None:
        assert provider_cls.provider_name == provider_name

    def test_base_url_matches(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
        dimension: int,
    ) -> None:
        assert provider_cls.default_base_url == base_url

    def test_default_model_has_dimension(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
        dimension: int,
    ) -> None:
        # Dimension is declared statically; the constructor rejects any
        # slug missing from MODEL_DIMENSIONS, so we can rely on the
        # lookup without a network call.
        assert provider_cls.MODEL_DIMENSIONS[default_model] == dimension

    def test_get_dimension_matches(
        self,
        provider_cls: type,
        provider_name: str,
        base_url: str,
        default_model: str,
        dimension: int,
        patched_openai: dict[str, Any],
    ) -> None:
        provider = provider_cls(_LONG_KEY)
        assert provider.get_dimension() == dimension


# ---------------------------------------------------------------------------
# Security — vendor subclasses inherit the base redaction, and we re-assert
# per-vendor so the regression fails loudly on any accidental override.
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.parametrize("provider_cls", [row[0] for row in _LLM_VENDORS])
def test_llm_repr_never_exposes_key(
    provider_cls: type,
    patched_openai: dict[str, Any],
) -> None:
    del patched_openai  # fixture is only needed to stub the OpenAI client
    text = repr(provider_cls(_LONG_KEY))
    assert _LONG_KEY not in text
    assert _KEY_TAIL in text


@pytest.mark.security
@pytest.mark.parametrize("provider_cls", [row[0] for row in _EMBED_VENDORS])
def test_embed_repr_never_exposes_key(
    provider_cls: type,
    patched_openai: dict[str, Any],
) -> None:
    del patched_openai
    text = repr(provider_cls(_LONG_KEY))
    assert _LONG_KEY not in text
    assert _KEY_TAIL in text
