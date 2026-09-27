"""Tests for :mod:`sec_generative_search.providers.registry`.

Covers:

- Listings: which providers a build exposes per surface.
- Optional-extras gating: ``LocalEmbeddingProvider`` hides when
  :func:`importlib.util.find_spec` cannot resolve ``sentence_transformers``.
- Class lookup by ``(name, surface)``, including the OpenAI / OpenAI-
  embedding name collision.
- Model listings derived from the vendored catalogue / ``MODEL_DIMENSIONS``.
- ``supports_arbitrary_models`` flag for OpenRouter.
- Capability lookup: O(1), credential-free, no network call; permissive
  default for unknown LLM slugs; ``ValueError`` for unknown embedding
  slugs.
- Dimension lookup for the storage layer.
- ``validate_key`` semantics: ``ProviderAuthError → False``; every other
  ``ProviderError`` propagates; never logs the key.
- Security: registry holds no per-instance state, has no credential-
  bearing field names, and the validated key is not retained on either
  the registry class or the entry tuple after ``validate_key`` returns.

Real provider classes are exercised throughout — only the network-
touching ``validate_key()`` method is stubbed, so the tests cover the
actual catalogues, dimensions, and capability shapes that ship with the
package.
"""

from __future__ import annotations

import importlib.util
from dataclasses import fields

import pytest

from sec_generative_search.core.exceptions import (
    ProviderAuthError,
    ProviderContentFilterError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.types import PricingTier, ProviderCapability
from sec_generative_search.providers.anthropic import AnthropicProvider
from sec_generative_search.providers.catalogue import model_catalogue
from sec_generative_search.providers.deepseek import DeepSeekProvider
from sec_generative_search.providers.gemini import (
    GeminiEmbeddingProvider,
    GeminiProvider,
)
from sec_generative_search.providers.grok import GrokProvider
from sec_generative_search.providers.kimi import KimiProvider
from sec_generative_search.providers.local import LocalEmbeddingProvider
from sec_generative_search.providers.mimo import MimoProvider
from sec_generative_search.providers.minimax import MiniMaxProvider
from sec_generative_search.providers.mistral import (
    MistralEmbeddingProvider,
    MistralProvider,
)
from sec_generative_search.providers.openai import (
    OpenAIEmbeddingProvider,
    OpenAIProvider,
)
from sec_generative_search.providers.openrouter import OpenRouterProvider
from sec_generative_search.providers.qwen import (
    QwenEmbeddingProvider,
    QwenProvider,
)
from sec_generative_search.providers.registry import (
    ProviderEntry,
    ProviderRegistry,
    ProviderSurface,
)
from sec_generative_search.providers.zai import ZaiProvider

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------


_FAKE_KEY = "sk-test-ABCDEFGHIJKLMNOP"
_FAKE_HF_KEY = "hf_ABCDEFGHIJKLMNOP"


def _sentence_transformers_available() -> bool:
    """Whether the optional ``[local-embeddings]`` extra is installed.

    The dev-install profile (``uv pip install -e ".[dev,test]"``) does
    *not* pull ``sentence-transformers`` — only profiles that
    explicitly add ``[local-embeddings]`` do.  Tests that depend on the
    local provider being visible in the registry skip cleanly when the
    extra is absent.
    """
    return importlib.util.find_spec("sentence_transformers") is not None


_LOCAL_AVAILABLE = _sentence_transformers_available()
_requires_local = pytest.mark.skipif(
    not _LOCAL_AVAILABLE,
    reason="LocalEmbeddingProvider requires the [local-embeddings] extra",
)


@pytest.fixture(autouse=True)
def _clear_availability_cache() -> None:
    """Force every test to start from a clean availability cache.

    The registry caches :func:`find_spec` results across calls; tests
    that monkeypatch that probe must start from a known state, and the
    other tests benefit from a deterministic baseline too.
    """
    ProviderRegistry._reset_availability_cache()


# ---------------------------------------------------------------------------
# Surface enum
# ---------------------------------------------------------------------------


class TestProviderSurface:
    def test_values_are_string_serialisable(self) -> None:
        # ``ProviderSurface`` inherits from ``str`` so the values can be
        # used directly in JSON / form payloads without coercion.
        assert ProviderSurface.LLM == "llm"
        assert ProviderSurface.EMBEDDING == "embedding"
        assert ProviderSurface.RERANKER == "reranker"


# ---------------------------------------------------------------------------
# Listings
# ---------------------------------------------------------------------------


_EXPECTED_LLM_NAMES = (
    "openai",
    "anthropic",
    "gemini",
    "deepseek",
    "kimi",
    "mistral",
    "qwen",
    "zai",
    "grok",
    "minimax",
    "mimo",
    "openrouter",
    "local_llm",
)

_EXPECTED_EMBEDDING_NAMES_BASE = (
    "openai",
    "gemini",
    "mistral",
    "qwen",
)
# ``local`` only ships when the optional ``[local-embeddings]`` extra is
# installed.  The dev/test profile skips it, so the canonical list is
# computed at import time against the running interpreter.
_EXPECTED_EMBEDDING_NAMES = (
    (*_EXPECTED_EMBEDDING_NAMES_BASE, "local")
    if _LOCAL_AVAILABLE
    else _EXPECTED_EMBEDDING_NAMES_BASE
)


class TestListings:
    def test_llm_surface_lists_every_shipped_llm_provider(self) -> None:
        names = ProviderRegistry.list_providers(ProviderSurface.LLM)
        assert tuple(names) == _EXPECTED_LLM_NAMES

    def test_embedding_surface_lists_every_shipped_embedding_provider(self) -> None:
        names = ProviderRegistry.list_providers(ProviderSurface.EMBEDDING)
        assert tuple(names) == _EXPECTED_EMBEDDING_NAMES

    def test_reranker_surface_is_empty(self) -> None:
        # No first-party reranker ships yet. An empty list today is
        # both correct and the contract callers should expect.
        assert ProviderRegistry.list_providers(ProviderSurface.RERANKER) == []

    def test_all_entries_returns_curated_tuple(self) -> None:
        entries = ProviderRegistry.all_entries()
        assert all(isinstance(e, ProviderEntry) for e in entries)
        # Every entry's ``provider_name`` matches the registered name —
        # the registry must not lie about what class it returns.
        for entry in entries:
            assert entry.provider_cls.provider_name == entry.name

    def test_all_entries_filters_by_surface(self) -> None:
        llm_only = ProviderRegistry.all_entries(ProviderSurface.LLM)
        assert len(llm_only) == len(_EXPECTED_LLM_NAMES)
        assert {e.name for e in llm_only} == set(_EXPECTED_LLM_NAMES)


# ---------------------------------------------------------------------------
# Optional-extras gating
# ---------------------------------------------------------------------------


class TestOptionalExtrasGating:
    def test_local_provider_hidden_when_sentence_transformers_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``find_spec`` returning ``None`` must collapse the entry."""
        original = importlib.util.find_spec

        def fake_find_spec(name: str, *args: object, **kwargs: object) -> object | None:
            if name == "sentence_transformers":
                return None
            return original(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        ProviderRegistry._reset_availability_cache()

        names = ProviderRegistry.list_providers(ProviderSurface.EMBEDDING)
        assert "local" not in names

    @_requires_local
    def test_local_provider_visible_when_sentence_transformers_present(self) -> None:
        # When the optional extra is installed, ``local`` must appear in
        # the list — this guards against a regression that gates the
        # entry for the wrong reason.
        names = ProviderRegistry.list_providers(ProviderSurface.EMBEDDING)
        assert "local" in names

    def test_include_unavailable_returns_gated_entries_too(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Web UIs need to *show* unavailable entries with a hint."""
        # Capture the real ``find_spec`` *before* patching so the
        # delegate does not recurse into the patched copy.
        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **kw: (
                None if name == "sentence_transformers" else real_find_spec(name, *a, **kw)
            ),
        )
        ProviderRegistry._reset_availability_cache()

        gated = ProviderRegistry.all_entries(ProviderSurface.EMBEDDING, include_unavailable=True)
        names = [e.name for e in gated]
        assert "local" in names

    def test_get_entry_raises_when_extras_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **kw: (
                None if name == "sentence_transformers" else real_find_spec(name, *a, **kw)
            ),
        )
        ProviderRegistry._reset_availability_cache()

        with pytest.raises(KeyError, match="optional extras"):
            ProviderRegistry.get_entry("local", ProviderSurface.EMBEDDING)

    def test_availability_probe_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One ``find_spec`` call per module per process is the contract."""
        call_count = {"n": 0}
        real_find_spec = importlib.util.find_spec

        def counting_find_spec(name: str, *a: object, **kw: object) -> object | None:
            if name == "sentence_transformers":
                call_count["n"] += 1
            return real_find_spec(name, *a, **kw)

        monkeypatch.setattr(importlib.util, "find_spec", counting_find_spec)
        ProviderRegistry._reset_availability_cache()

        # Run two listings — the second must hit the cache.
        ProviderRegistry.list_providers(ProviderSurface.EMBEDDING)
        ProviderRegistry.list_providers(ProviderSurface.EMBEDDING)
        assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# Class lookup
# ---------------------------------------------------------------------------


_GET_CLASS_CASES: list[tuple[str, ProviderSurface, type]] = [
    ("openai", ProviderSurface.LLM, OpenAIProvider),
    ("anthropic", ProviderSurface.LLM, AnthropicProvider),
    ("gemini", ProviderSurface.LLM, GeminiProvider),
    ("deepseek", ProviderSurface.LLM, DeepSeekProvider),
    ("kimi", ProviderSurface.LLM, KimiProvider),
    ("mistral", ProviderSurface.LLM, MistralProvider),
    ("qwen", ProviderSurface.LLM, QwenProvider),
    ("zai", ProviderSurface.LLM, ZaiProvider),
    ("grok", ProviderSurface.LLM, GrokProvider),
    ("minimax", ProviderSurface.LLM, MiniMaxProvider),
    ("mimo", ProviderSurface.LLM, MimoProvider),
    ("openrouter", ProviderSurface.LLM, OpenRouterProvider),
    ("openai", ProviderSurface.EMBEDDING, OpenAIEmbeddingProvider),
    ("gemini", ProviderSurface.EMBEDDING, GeminiEmbeddingProvider),
    ("mistral", ProviderSurface.EMBEDDING, MistralEmbeddingProvider),
    ("qwen", ProviderSurface.EMBEDDING, QwenEmbeddingProvider),
]
if _LOCAL_AVAILABLE:
    # The local entry only resolves when ``sentence-transformers`` is
    # installed; otherwise ``get_class`` raises ``KeyError`` by design.
    _GET_CLASS_CASES.append(("local", ProviderSurface.EMBEDDING, LocalEmbeddingProvider))


@pytest.mark.parametrize(("name", "surface", "expected_cls"), _GET_CLASS_CASES)
def test_get_class_returns_concrete_provider(
    name: str,
    surface: ProviderSurface,
    expected_cls: type,
) -> None:
    assert ProviderRegistry.get_class(name, surface) is expected_cls


class TestGetClassCollisions:
    def test_openai_name_resolves_to_distinct_classes_per_surface(self) -> None:
        """``"openai"`` is registered on both surfaces — they must
        return distinct classes, not collapse to one."""
        llm = ProviderRegistry.get_class("openai", ProviderSurface.LLM)
        emb = ProviderRegistry.get_class("openai", ProviderSurface.EMBEDDING)
        assert llm is OpenAIProvider
        assert emb is OpenAIEmbeddingProvider
        assert llm is not emb

    def test_unknown_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="No provider registered"):
            ProviderRegistry.get_class("not-a-real-provider", ProviderSurface.LLM)

    def test_unknown_surface_for_known_name_raises(self) -> None:
        # Anthropic only registers an LLM surface — asking for its
        # embedding adapter must fail rather than silently return
        # ``None`` or fall back to a different vendor.
        with pytest.raises(KeyError):
            ProviderRegistry.get_class("anthropic", ProviderSurface.EMBEDDING)


# ---------------------------------------------------------------------------
# Model listings
# ---------------------------------------------------------------------------


class TestListModels:
    def test_llm_models_match_provider_catalogue(self) -> None:
        models = set(ProviderRegistry.list_models("openai", ProviderSurface.LLM))
        # The registry delegates LLM model listing to the vendored
        # catalogue keyed by provider name — assert that delegation is
        # faithful, plus a spot-check that a known slug is present.
        assert models == set(model_catalogue().list_llm_models("openai"))
        assert "gpt-4o" in models

    def test_embedding_models_match_provider_dimensions(self) -> None:
        models = set(ProviderRegistry.list_models("openai", ProviderSurface.EMBEDDING))
        assert models == set(OpenAIEmbeddingProvider.MODEL_DIMENSIONS.keys())

    def test_openrouter_returns_empty_list_by_design(self) -> None:
        # OpenRouter is the meta-provider — the catalogue is *meant* to
        # be empty and the UI should fall back to a free-text input.
        assert ProviderRegistry.list_models("openrouter", ProviderSurface.LLM) == []

    @_requires_local
    def test_local_models_match_dimensions(self) -> None:
        models = set(ProviderRegistry.list_models("local", ProviderSurface.EMBEDDING))
        assert models == set(LocalEmbeddingProvider.MODEL_DIMENSIONS.keys())


# ---------------------------------------------------------------------------
# Arbitrary-models flag
# ---------------------------------------------------------------------------


class TestArbitraryModels:
    def test_openrouter_supports_arbitrary_models(self) -> None:
        assert ProviderRegistry.supports_arbitrary_models("openrouter", ProviderSurface.LLM) is True

    @pytest.mark.parametrize(
        "name",
        [
            "openai",
            "anthropic",
            "gemini",
            "deepseek",
            "kimi",
            "mistral",
            "qwen",
            "zai",
            "grok",
            "minimax",
            "mimo",
        ],
    )
    def test_closed_catalogue_providers_do_not(self, name: str) -> None:
        assert ProviderRegistry.supports_arbitrary_models(name, ProviderSurface.LLM) is False


# ---------------------------------------------------------------------------
# Upstream-routing flag
# ---------------------------------------------------------------------------


class TestUpstreamRoutingFlag:
    """Only OpenRouter honours
    :class:`~sec_generative_search.providers.openrouter.OpenRouterRoutingHints`.

    The flag is what CLI and web UI keys off to decide whether to
    surface an upstream-provider picker.  A direct upstream (OpenAI,
    Anthropic, …) would render the picker as a no-op at best and a
    misleading UX at worst — the non-OpenRouter subclasses silently
    ignore ``routing_hints`` via the OpenAI-compatible base's empty
    default ``_extra_request_kwargs`` hook.
    """

    def test_openrouter_supports_upstream_routing(self) -> None:
        assert ProviderRegistry.supports_upstream_routing("openrouter", ProviderSurface.LLM) is True

    @pytest.mark.parametrize(
        "name",
        [
            "openai",
            "anthropic",
            "gemini",
            "deepseek",
            "kimi",
            "mistral",
            "qwen",
            "zai",
            "grok",
            "minimax",
            "mimo",
        ],
    )
    def test_direct_upstreams_do_not(self, name: str) -> None:
        assert ProviderRegistry.supports_upstream_routing(name, ProviderSurface.LLM) is False

    def test_entry_flag_matches_classmethod(self) -> None:
        """Single source of truth: the entry's flag and the registry
        classmethod never disagree."""
        entry = ProviderRegistry.get_entry("openrouter", ProviderSurface.LLM)
        assert entry.supports_upstream_routing is True
        assert ProviderRegistry.supports_upstream_routing(entry.name, entry.surface) is True


# ---------------------------------------------------------------------------
# Pricing-tier single source of truth
# ---------------------------------------------------------------------------


class TestPricingTierSingleSourceOfTruth:
    """The **vendored baseline** stays fully priced; overlay rows may not.

    The cost-derived ``pricing_tier`` is the single source of truth
    consumed by both the metrics facade
    (:meth:`ProviderRegistry.get_capability`) and the web UI
    (``GET /api/providers/{provider}/models``). A *baseline* slug whose
    tier silently defaulted to ``UNKNOWN`` would collapse the metric's
    ``pricing_tier`` label to noise and leave the UI unable to render a
    cheap-vs-premium hint — so the invariant is locked here rather than
    left to per-provider review.

    Overlay / auto-discovered models — and OpenRouter's arbitrary slugs —
    *may* carry ``UNKNOWN`` (a freshly-fetched row can legitimately arrive
    without pricing, exactly like a slug a user types into the free-text
    input). The baseline is the part that must never be ``UNKNOWN``; the
    consumer surfaces must *tolerate* an ``UNKNOWN`` row from anywhere else
    (see :meth:`test_overlay_only_model_may_carry_unknown_tier`).
    """

    @pytest.mark.security
    def test_no_baseline_model_is_unknown_tier(self) -> None:
        from sec_generative_search.core.types import PricingTier
        from sec_generative_search.providers.catalogue import (
            ModelCatalogue,
            reset_catalogue,
            set_catalogue,
        )

        # Pin the active catalogue to the pure baseline so the lock is
        # explicitly baseline-scoped and immune to any overlay another
        # test might have left installed — this is the row set that must
        # never surface ``UNKNOWN``.
        set_catalogue(ModelCatalogue.load_baseline())
        try:
            offenders: list[str] = []
            for name in ProviderRegistry.list_providers(ProviderSurface.LLM):
                if ProviderRegistry.supports_arbitrary_models(name, ProviderSurface.LLM):
                    continue  # OpenRouter — empty catalogue by design.
                for slug in ProviderRegistry.list_models(name, ProviderSurface.LLM):
                    tier = ProviderRegistry.get_capability(
                        name, ProviderSurface.LLM, slug
                    ).pricing_tier
                    if tier is PricingTier.UNKNOWN:
                        offenders.append(f"{name}/{slug}")
        finally:
            reset_catalogue()
        assert not offenders, (
            "every vendored baseline LLM model must carry a non-UNKNOWN pricing tier; "
            f"offenders: {offenders}"
        )

    @pytest.mark.security
    def test_overlay_only_model_may_carry_unknown_tier(self) -> None:
        # An overlay / auto-discovered model that arrives without pricing
        # surfaces through the registry as ``UNKNOWN`` — tolerated, never
        # an error. This mirrors the OpenRouter free-text contract for a
        # closed-catalogue provider once a price-less row is merged on top
        # of the baseline.
        from sec_generative_search.core.types import PricingTier, ProviderCapability
        from sec_generative_search.providers.catalogue import (
            model_catalogue,
            reset_catalogue,
            set_catalogue,
        )

        unpriced = ProviderCapability(chat=True, streaming=True)  # both costs None
        assert unpriced.pricing_tier is PricingTier.UNKNOWN
        augmented = model_catalogue().with_provider("openai", {"gpt-overlay-x": unpriced})
        set_catalogue(augmented)
        try:
            cap = ProviderRegistry.get_capability("openai", ProviderSurface.LLM, "gpt-overlay-x")
            assert cap.pricing_tier is PricingTier.UNKNOWN
            assert "gpt-overlay-x" in ProviderRegistry.list_models("openai", ProviderSurface.LLM)
        finally:
            reset_catalogue()

    def test_closed_catalogue_providers_have_at_least_one_model(self) -> None:
        # Defensive: the UNKNOWN sweep above is vacuously true for an
        # empty catalogue. A closed-catalogue provider that lost its
        # catalogue (e.g. a botched edit) would silently pass the sweep —
        # this guards against that.
        for name in ProviderRegistry.list_providers(ProviderSurface.LLM):
            if ProviderRegistry.supports_arbitrary_models(name, ProviderSurface.LLM):
                continue
            models = ProviderRegistry.list_models(name, ProviderSurface.LLM)
            assert models, f"closed-catalogue provider {name!r} has no catalogued models"


# ---------------------------------------------------------------------------
# Capability lookup
# ---------------------------------------------------------------------------


class TestGetCapability:
    def test_known_llm_slug_returns_catalogued_capability(self) -> None:
        cap = ProviderRegistry.get_capability("openai", ProviderSurface.LLM, "gpt-4o")
        # Identity check — must be the same ``ProviderCapability`` the
        # vendored catalogue holds, not a copy with subtly different flags.
        assert cap == model_catalogue().get_llm_capability("openai", "gpt-4o")
        assert cap.chat is True
        assert cap.streaming is True
        assert cap.pricing_tier == PricingTier.PREMIUM

    def test_unknown_llm_slug_returns_permissive_default(self) -> None:
        cap = ProviderRegistry.get_capability("openai", ProviderSurface.LLM, "gpt-future-x")
        assert cap == ProviderCapability(chat=True, streaming=True)

    def test_openrouter_unknown_slug_returns_permissive_default(self) -> None:
        cap = ProviderRegistry.get_capability("openrouter", ProviderSurface.LLM, "vendor/model-x")
        assert cap.chat is True
        assert cap.streaming is True
        # OpenRouter's arbitrary slugs are honestly UNKNOWN-cost — never
        # silently priced as free.
        assert cap.pricing_tier is PricingTier.UNKNOWN

    def test_local_llm_uncatalogued_slug_returns_free_capability(self) -> None:
        # local_llm is the only ``free_tier`` entry: an uncatalogued slug
        # resolves to a 0.0-cost capability whose tier *derives* to FREE.
        cap = ProviderRegistry.get_capability("local_llm", ProviderSurface.LLM, "mixtral")
        assert cap.chat is True
        assert cap.streaming is True
        assert cap.input_cost_per_mtok == 0.0
        assert cap.output_cost_per_mtok == 0.0
        assert cap.pricing_tier is PricingTier.FREE

    def test_default_model_used_when_none_passed(self) -> None:
        # ``OpenAIProvider.default_model`` is ``"gpt-5.4-mini"``. The
        # registry must fall through to it.
        cap = ProviderRegistry.get_capability("openai", ProviderSurface.LLM)
        expected = model_catalogue().get_llm_capability("openai", "gpt-5.4-mini")
        assert cap == expected

    def test_known_embedding_slug_returns_embeddings_capability(self) -> None:
        cap = ProviderRegistry.get_capability(
            "openai", ProviderSurface.EMBEDDING, "text-embedding-3-small"
        )
        assert cap.embeddings is True
        # No chat / streaming flags on an embedding model.
        assert cap.chat is False
        assert cap.streaming is False

    def test_unknown_embedding_slug_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown embedding model"):
            ProviderRegistry.get_capability(
                "openai", ProviderSurface.EMBEDDING, "text-embedding-future"
            )


# ---------------------------------------------------------------------------
# Dimension lookup (storage layer)
# ---------------------------------------------------------------------------


class TestGetDimension:
    def test_returns_native_dimension_for_known_model(self) -> None:
        assert ProviderRegistry.get_dimension("openai", "text-embedding-3-small") == 1536
        assert ProviderRegistry.get_dimension("openai", "text-embedding-3-large") == 3072

    def test_default_model_used_when_none_passed(self) -> None:
        # ``OpenAIEmbeddingProvider.default_model`` is
        # ``"text-embedding-3-small"`` — fall through must hit it.
        assert ProviderRegistry.get_dimension("openai") == 1536

    def test_unknown_model_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown embedding model"):
            ProviderRegistry.get_dimension("openai", "text-embedding-not-real")

    @_requires_local
    def test_local_dimension_lookup_works(self) -> None:
        assert ProviderRegistry.get_dimension("local", "google/embeddinggemma-300m") == 768


# ---------------------------------------------------------------------------
# validate_key
# ---------------------------------------------------------------------------


class TestValidateKey:
    def test_returns_true_when_provider_accepts_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(OpenAIProvider, "validate_key", lambda self: True)
        assert ProviderRegistry.validate_key("openai", ProviderSurface.LLM, _FAKE_KEY) is True

    def test_returns_false_on_provider_auth_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake(self: OpenAIProvider) -> bool:
            raise ProviderAuthError("invalid key", provider="openai", hint="check the key")

        monkeypatch.setattr(OpenAIProvider, "validate_key", fake)
        assert ProviderRegistry.validate_key("openai", ProviderSurface.LLM, _FAKE_KEY) is False

    def test_propagates_rate_limit_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Rate-limit is *not* a verdict on the key — surfacing it as
        # ``False`` would mislead the UI into a key rotation.
        def fake(self: OpenAIProvider) -> bool:
            raise ProviderRateLimitError("429", provider="openai")

        monkeypatch.setattr(OpenAIProvider, "validate_key", fake)
        with pytest.raises(ProviderRateLimitError):
            ProviderRegistry.validate_key("openai", ProviderSurface.LLM, _FAKE_KEY)

    def test_propagates_timeout_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(self: OpenAIProvider) -> bool:
            raise ProviderTimeoutError("timeout", provider="openai")

        monkeypatch.setattr(OpenAIProvider, "validate_key", fake)
        with pytest.raises(ProviderTimeoutError):
            ProviderRegistry.validate_key("openai", ProviderSurface.LLM, _FAKE_KEY)

    def test_propagates_content_filter_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake(self: OpenAIProvider) -> bool:
            raise ProviderContentFilterError("blocked", provider="openai", hint="reformulate")

        monkeypatch.setattr(OpenAIProvider, "validate_key", fake)
        with pytest.raises(ProviderContentFilterError):
            ProviderRegistry.validate_key("openai", ProviderSurface.LLM, _FAKE_KEY)

    def test_unknown_provider_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            ProviderRegistry.validate_key("no-such-vendor", ProviderSurface.LLM, _FAKE_KEY)

    def test_embedding_path_passes_model_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, str] = {}

        original_init = OpenAIEmbeddingProvider.__init__

        def spy_init(
            self: OpenAIEmbeddingProvider,
            api_key: str,
            *,
            model: str | None = None,
            **kwargs: object,
        ) -> None:
            captured["api_key"] = api_key
            captured["model"] = model or ""
            original_init(self, api_key, model=model, **kwargs)

        monkeypatch.setattr(OpenAIEmbeddingProvider, "__init__", spy_init)
        monkeypatch.setattr(OpenAIEmbeddingProvider, "validate_key", lambda self: True)

        ok = ProviderRegistry.validate_key(
            "openai",
            ProviderSurface.EMBEDDING,
            _FAKE_KEY,
            model="text-embedding-3-large",
        )
        assert ok is True
        assert captured["model"] == "text-embedding-3-large"
        # Key reaches the constructor exactly as supplied — no mutation.
        assert captured["api_key"] == _FAKE_KEY

    @_requires_local
    def test_local_provider_receives_key_as_hf_token(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sanity check on the positional-arg construction strategy.

        :class:`LocalEmbeddingProvider`'s first positional parameter is
        ``hf_token``, not ``api_key``.  The registry must still pass the
        key through unchanged.
        """
        captured: dict[str, object] = {}

        original_init = LocalEmbeddingProvider.__init__

        def spy_init(
            self: LocalEmbeddingProvider,
            hf_token: str | None = None,
            **kwargs: object,
        ) -> None:
            captured["hf_token"] = hf_token
            captured["model"] = kwargs.get("model")
            original_init(self, hf_token, **kwargs)

        monkeypatch.setattr(LocalEmbeddingProvider, "__init__", spy_init)
        monkeypatch.setattr(LocalEmbeddingProvider, "validate_key", lambda self: True)

        ok = ProviderRegistry.validate_key(
            "local",
            ProviderSurface.EMBEDDING,
            _FAKE_HF_KEY,
            model="google/embeddinggemma-300m",
        )
        assert ok is True
        assert captured["hf_token"] == _FAKE_HF_KEY
        assert captured["model"] == "google/embeddinggemma-300m"

    def test_construct_time_auth_error_collapses_to_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A provider that fail-fasts at construction must also report
        ``False`` — anything else would mean the registry's contract
        depends on whether the key check happens early or late."""

        def fake_init(self: OpenAIProvider, api_key: str, **_: object) -> None:
            raise ProviderAuthError("rejected at construction", provider="openai")

        monkeypatch.setattr(OpenAIProvider, "__init__", fake_init)
        assert ProviderRegistry.validate_key("openai", ProviderSurface.LLM, _FAKE_KEY) is False


# ---------------------------------------------------------------------------
# Security
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
class TestRegistryHoldsNoSecrets:
    """The registry must not accidentally
    become a credential store the way a naive cache layer would."""

    def test_provider_entry_has_no_credential_fields(self) -> None:
        for f in fields(ProviderEntry):
            # Boolean *flags* (e.g. ``requires_api_key``) describe whether a
            # credential is needed — they cannot hold one, so the credential-
            # name heuristic must not fire on them.  A field that could
            # actually carry a secret is str/bytes-shaped; those are still
            # caught below, and the value-based lock
            # (``test_validate_key_does_not_appear_in_returned_entries``)
            # is the belt to this suspenders.
            if isinstance(f.default, bool):
                continue
            lowered = f.name.lower()
            for hint in _SECRET_FIELD_HINTS:
                assert hint not in lowered, (
                    f"ProviderEntry.{f.name} looks credential-bearing; "
                    "registry rows must never carry secrets."
                )

    def test_registry_class_has_no_credential_attributes(self) -> None:
        # The registry has class-level attributes only.  Walk every
        # public attribute name and refuse credential-shaped names.
        for attr in vars(ProviderRegistry):
            lowered = attr.lower()
            for hint in _SECRET_FIELD_HINTS:
                assert hint not in lowered, (
                    f"ProviderRegistry.{attr} looks credential-bearing; "
                    "registry must not store secrets."
                )

    def test_validate_key_does_not_retain_key_on_class(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After ``validate_key`` returns, no class-level state on the
        registry should contain the key.

        We compare the textual snapshot of every class-level attribute
        before and after; any mention of the (deliberately distinctive)
        test key would indicate the registry held onto it.
        """

        monkeypatch.setattr(OpenAIProvider, "validate_key", lambda self: True)

        before = {k: repr(v) for k, v in vars(ProviderRegistry).items()}
        ProviderRegistry.validate_key("openai", ProviderSurface.LLM, _FAKE_KEY)
        after = {k: repr(v) for k, v in vars(ProviderRegistry).items()}

        for attr, value in after.items():
            assert _FAKE_KEY not in value, (
                f"ProviderRegistry.{attr} contains the validated key — "
                "registry is leaking credentials into class state."
            )
        # And the structure of the class state must be unchanged
        # (no new attributes appearing as a side effect).
        assert set(before.keys()) == set(after.keys())

    def test_validate_key_does_not_appear_in_returned_entries(self) -> None:
        """The curated entries themselves must not gain a key field
        after a validation cycle (defence in depth — covers the case
        where a future refactor adds a per-entry instance store)."""
        for entry in ProviderRegistry.all_entries(include_unavailable=True):
            for f in fields(entry):
                value = getattr(entry, f.name)
                if isinstance(value, str):
                    assert _FAKE_KEY not in value
                    assert _FAKE_HF_KEY not in value


@pytest.mark.security
class TestRegistryReturnsRedactedReprs:
    """Sanity: any provider built via the registry must still redact
    the API key from ``repr()``.  This is enforced at the ABC level
    by ``_ProviderBase``, but the registry path is a new entry point
    — re-check it end-to-end."""

    def test_validated_provider_repr_redacts_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, str] = {}

        def fake(self: OpenAIProvider) -> bool:
            captured["repr"] = repr(self)
            return True

        monkeypatch.setattr(OpenAIProvider, "validate_key", fake)

        ProviderRegistry.validate_key("openai", ProviderSurface.LLM, _FAKE_KEY)

        text = captured["repr"]
        assert _FAKE_KEY not in text
        # ``mask_secret`` exposes the trailing four characters only.
        assert _FAKE_KEY[-4:] in text


@pytest.mark.security
@pytest.mark.security
class TestValidateKeyUsesTheInteractiveBudget:
    """F16(d): the key-validation probe is interactive — one retry, not three."""

    def test_llm_probe_is_built_with_the_interactive_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sec_generative_search.core.resilience import INTERACTIVE_RETRY_POLICY

        built: list[OpenAIProvider] = []

        def fake(self: OpenAIProvider) -> bool:
            built.append(self)
            return True

        monkeypatch.setattr(OpenAIProvider, "validate_key", fake)
        assert ProviderRegistry.validate_key("openai", ProviderSurface.LLM, _FAKE_KEY) is True
        assert built[0]._policy.retry_policy is INTERACTIVE_RETRY_POLICY


class TestValidateKeyClosesClient:
    """``validate_key`` closes the provider it constructs on every path.

    The instance is local to the call; closing it in a ``finally`` releases
    the SDK client's connection pool and credential-bearing transport
    deterministically rather than at GC.  Covers both the API ``/validate``
    route and the CLI ``provider validate`` — both flow through here.
    """

    def test_closes_on_accept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        closed: list[bool] = []
        monkeypatch.setattr(OpenAIProvider, "validate_key", lambda self: True)
        monkeypatch.setattr(OpenAIProvider, "close", lambda self: closed.append(True))
        assert ProviderRegistry.validate_key("openai", ProviderSurface.LLM, _FAKE_KEY) is True
        assert closed == [True]

    def test_closes_on_auth_rejection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        closed: list[bool] = []

        def fake(self: OpenAIProvider) -> bool:
            raise ProviderAuthError("invalid", provider="openai", hint="check the key")

        monkeypatch.setattr(OpenAIProvider, "validate_key", fake)
        monkeypatch.setattr(OpenAIProvider, "close", lambda self: closed.append(True))
        assert ProviderRegistry.validate_key("openai", ProviderSurface.LLM, _FAKE_KEY) is False
        assert closed == [True]

    def test_closes_when_error_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        closed: list[bool] = []

        def fake(self: OpenAIProvider) -> bool:
            raise ProviderRateLimitError("429", provider="openai")

        monkeypatch.setattr(OpenAIProvider, "validate_key", fake)
        monkeypatch.setattr(OpenAIProvider, "close", lambda self: closed.append(True))
        with pytest.raises(ProviderRateLimitError):
            ProviderRegistry.validate_key("openai", ProviderSurface.LLM, _FAKE_KEY)
        # The finally closes the client even though the error propagated.
        assert closed == [True]
