"""Tests for the opt-in model-catalogue refresh seam.

Covers the three bounded stages and their security envelope:

- **Bounded fetch** (``@pytest.mark.security``): https-only, no-redirect,
  response-size cap, transport / decode failures collapse to a content-free
  :class:`CatalogueRefreshError`.
- **Untrusted-input validation** (``@pytest.mark.security``): provider / model
  slug shape, count bounds, strict typing, finite / non-negative / bounded
  cost, control-character rejection, and the allow-list projection.
- **Normalisers**: models.dev + LiteLLM shape mapping, provider aliasing, and
  served-provider filtering.
- **Overlay writer**: atomic ``{_meta, providers}`` write that never stores a
  ``pricing_tier``.
- **Orchestration**: fail-closed — a fetch / validation failure leaves any
  prior overlay untouched.

All fetches use an injected ``httpx.Client`` bound to a ``MockTransport`` — no
test in this module touches the real network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from sec_generative_search.core.exceptions import CatalogueRefreshError
from sec_generative_search.core.types import PricingTier
from sec_generative_search.providers import refresh
from sec_generative_search.providers.refresh import (
    MAX_MODELS_PER_PROVIDER,
    MAX_PROVIDERS,
    fetch_json,
    normalise_litellm,
    normalise_models_dev,
    refresh_overlay,
    validate_catalogue_payload,
    write_overlay,
)

_KNOWN = {"openai", "anthropic", "gemini", "deepseek", "kimi", "mistral"}


def _client_factory(handler):
    """Return a ``client_factory`` yielding a MockTransport-backed client."""

    def factory() -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    return factory


def _json_handler(payload: Any):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return handler


# ---------------------------------------------------------------------------
# Bounded fetch
# ---------------------------------------------------------------------------


class TestFetchJson:
    @pytest.mark.security
    def test_rejects_non_https_scheme(self) -> None:
        # No socket should open for an http:// (or any non-TLS) URL.
        for url in ("http://models.dev/api.json", "ftp://x/y", "file:///etc/passwd"):
            with pytest.raises(CatalogueRefreshError, match="https"):
                fetch_json(url)

    def test_parses_json_body(self) -> None:
        result = fetch_json(
            "https://models.dev/api.json",
            client_factory=_client_factory(_json_handler({"ok": True})),
        )
        assert result == {"ok": True}

    @pytest.mark.security
    def test_response_size_cap_enforced(self) -> None:
        big = b'{"x": "' + b"a" * 4096 + b'"}'

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=big)

        with pytest.raises(CatalogueRefreshError, match="size cap"):
            fetch_json(
                "https://models.dev/api.json",
                max_bytes=1024,
                client_factory=_client_factory(handler),
            )

    @pytest.mark.security
    def test_http_error_collapses_to_refresh_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        with pytest.raises(CatalogueRefreshError, match="fetch failed"):
            fetch_json(
                "https://models.dev/api.json",
                client_factory=_client_factory(handler),
            )

    @pytest.mark.security
    def test_transport_error_collapses_to_refresh_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("boom")

        with pytest.raises(CatalogueRefreshError, match="fetch failed"):
            fetch_json(
                "https://models.dev/api.json",
                client_factory=_client_factory(handler),
            )

    def test_invalid_json_collapses_to_refresh_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not json{{")

        with pytest.raises(CatalogueRefreshError, match="valid JSON"):
            fetch_json(
                "https://models.dev/api.json",
                client_factory=_client_factory(handler),
            )

    @pytest.mark.security
    def test_client_built_here_disables_redirects(self) -> None:
        # When no factory is injected, the default client must not follow
        # redirects (SSRF defence) and must verify TLS.  We assert on the
        # constructed client's configuration rather than hitting the network.
        captured: dict[str, Any] = {}
        real_client = httpx.Client

        def spy(*args: Any, **kwargs: Any) -> httpx.Client:
            captured.update(kwargs)
            # Force an immediate, offline failure so no real request is made.
            kwargs["transport"] = httpx.MockTransport(
                lambda _req: (_ for _ in ()).throw(httpx.ConnectError("offline"))
            )
            return real_client(*args, **kwargs)

        import sec_generative_search.providers.refresh as mod

        original = httpx.Client
        httpx.Client = spy  # type: ignore[assignment, misc]
        try:
            with pytest.raises(CatalogueRefreshError):
                mod.fetch_json("https://models.dev/api.json")
        finally:
            httpx.Client = original  # type: ignore[assignment, misc]

        assert captured.get("follow_redirects") is False
        assert captured.get("verify") is True


# ---------------------------------------------------------------------------
# Validation — the untrusted-input gate
# ---------------------------------------------------------------------------


class TestValidation:
    def _row(self, **overrides: Any) -> dict[str, Any]:
        base = {
            "streaming": True,
            "tool_use": False,
            "structured_output": False,
            "prompt_caching": False,
            "vision": False,
            "context_window_tokens": 128000,
            "max_output_tokens": 4096,
            "input_cost_per_mtok": 1.0,
            "output_cost_per_mtok": 2.0,
        }
        base.update(overrides)
        return base

    def test_valid_payload_round_trips_to_clean_rows(self) -> None:
        out = validate_catalogue_payload({"openai": {"gpt-5.4-mini": self._row()}})
        row = out["openai"]["gpt-5.4-mini"]
        assert row["input_cost_per_mtok"] == 1.0
        # Only allow-listed keys survive — no smuggled field.
        assert set(row) == {
            "streaming",
            "tool_use",
            "structured_output",
            "prompt_caching",
            "vision",
            "context_window_tokens",
            "max_output_tokens",
            "input_cost_per_mtok",
            "output_cost_per_mtok",
        }

    @pytest.mark.security
    def test_pricing_tier_never_in_validated_output(self) -> None:
        # An overlay must never carry a tier (derived from cost at load).
        out = validate_catalogue_payload({"openai": {"m": self._row(pricing_tier="premium")}})
        assert "pricing_tier" not in out["openai"]["m"]

    @pytest.mark.security
    @pytest.mark.parametrize(
        "provider",
        ["Openai", "open ai", "openai!", "a" * 33, "", "open\nai", "-bad"],
    )
    def test_malformed_provider_slug_rejected(self, provider: str) -> None:
        with pytest.raises(CatalogueRefreshError, match="provider key"):
            validate_catalogue_payload({provider: {"m": self._row()}})

    @pytest.mark.security
    @pytest.mark.parametrize(
        "slug",
        ["bad slug", "slug\twith\ttab", "slug\nnewline", "a" * 129, "", ".leading"],
    )
    def test_malformed_model_slug_rejected(self, slug: str) -> None:
        with pytest.raises(CatalogueRefreshError, match="model slug"):
            validate_catalogue_payload({"openai": {slug: self._row()}})

    def test_realistic_model_slugs_accepted(self) -> None:
        for slug in ("openai/gpt-5.4-mini", "glm-4.5-air", "qwen3.5-omni-plus", "o4-mini"):
            out = validate_catalogue_payload({"openai": {slug: self._row()}})
            assert slug in out["openai"]

    @pytest.mark.security
    def test_provider_count_bound(self) -> None:
        payload = {f"p{i}": {"m": self._row()} for i in range(MAX_PROVIDERS + 1)}
        with pytest.raises(CatalogueRefreshError, match="provider bound"):
            validate_catalogue_payload(payload)

    @pytest.mark.security
    def test_model_count_bound(self) -> None:
        models = {f"m{i}": self._row() for i in range(MAX_MODELS_PER_PROVIDER + 1)}
        with pytest.raises(CatalogueRefreshError, match="model bound"):
            validate_catalogue_payload({"openai": models})

    @pytest.mark.security
    @pytest.mark.parametrize("bad", [-1.0, float("inf"), float("nan"), 2_000_000.0])
    def test_invalid_cost_rejected(self, bad: float) -> None:
        with pytest.raises(CatalogueRefreshError):
            validate_catalogue_payload({"openai": {"m": self._row(input_cost_per_mtok=bad)}})

    @pytest.mark.security
    def test_string_cost_rejected(self) -> None:
        # A JSON string where a number is expected must not be coerced.
        with pytest.raises(CatalogueRefreshError, match="number or null"):
            validate_catalogue_payload({"openai": {"m": self._row(input_cost_per_mtok="9.99")}})

    @pytest.mark.security
    def test_non_bool_flag_rejected(self) -> None:
        with pytest.raises(CatalogueRefreshError, match="boolean"):
            validate_catalogue_payload({"openai": {"m": self._row(vision="yes")}})

    @pytest.mark.security
    def test_non_int_token_field_rejected(self) -> None:
        with pytest.raises(CatalogueRefreshError, match="integer"):
            validate_catalogue_payload({"openai": {"m": self._row(context_window_tokens=1.5)}})

    def test_null_cost_yields_unknown_tier_capability(self) -> None:
        # A row with both costs null is valid (overlay models may be UNKNOWN);
        # the derived capability buckets to UNKNOWN.
        out = validate_catalogue_payload(
            {"openai": {"m": self._row(input_cost_per_mtok=None, output_cost_per_mtok=None)}}
        )
        assert out["openai"]["m"]["input_cost_per_mtok"] is None
        from sec_generative_search.providers.catalogue import capability_from_row

        assert capability_from_row(out["openai"]["m"]).pricing_tier is PricingTier.UNKNOWN

    @pytest.mark.security
    def test_non_object_payload_rejected(self) -> None:
        with pytest.raises(CatalogueRefreshError, match="JSON object"):
            validate_catalogue_payload([1, 2, 3])

    def test_derived_tier_matches_baseline_function(self) -> None:
        # (0.9 + 3.6) / 2 = 2.25 -> STANDARD, same boundaries as the baseline.
        out = validate_catalogue_payload(
            {"openai": {"m": self._row(input_cost_per_mtok=0.9, output_cost_per_mtok=3.6)}}
        )
        from sec_generative_search.providers.catalogue import capability_from_row

        assert capability_from_row(out["openai"]["m"]).pricing_tier is PricingTier.STANDARD


# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------


class TestNormaliseModelsDev:
    def test_maps_cost_and_limits(self) -> None:
        payload = {
            "openai": {
                "models": {
                    "gpt-5.4-mini": {
                        "cost": {"input": 0.4, "output": 1.6, "cache_read": 0.1},
                        "limit": {"context": 400000, "output": 128000},
                        "tool_call": True,
                        "modalities": {"input": ["text", "image"]},
                    }
                }
            }
        }
        out = normalise_models_dev(payload, _KNOWN)
        row = out["openai"]["gpt-5.4-mini"]
        assert row["input_cost_per_mtok"] == 0.4
        assert row["output_cost_per_mtok"] == 1.6
        assert row["context_window_tokens"] == 400000
        assert row["tool_use"] is True
        assert row["vision"] is True
        assert row["prompt_caching"] is True

    def test_provider_alias_and_filter(self) -> None:
        payload = {
            "google": {
                "models": {"gemini-3-flash-preview": {"cost": {"input": 0.1, "output": 0.4}}}
            },
            "unknownvendor": {"models": {"x": {"cost": {"input": 1, "output": 1}}}},
        }
        out = normalise_models_dev(payload, _KNOWN)
        assert "gemini" in out  # google -> gemini
        assert "unknownvendor" not in out  # not served -> dropped
        assert "google" not in out

    def test_region_mirror_never_prices_served_endpoint(self) -> None:
        # zhipuai is Z.ai's China endpoint (open.bigmodel.cn); ZaiProvider
        # targets api.z.ai, so the mirror must not price the zai key — even
        # when it comes last in the payload.
        payload = {
            "zai": {"models": {"glm-5": {"cost": {"input": 1.0, "output": 3.2}}}},
            "zhipuai": {"models": {"glm-5": {"cost": {"input": 0.6, "output": 2.7}}}},
        }
        out = normalise_models_dev(payload, _KNOWN | {"zai"})
        assert out["zai"]["glm-5"]["input_cost_per_mtok"] == 1.0
        assert out["zai"]["glm-5"]["output_cost_per_mtok"] == 3.2

    @pytest.mark.parametrize("canonical_first", [True, False])
    def test_alias_never_overrides_canonical_id(self, canonical_first: bool) -> None:
        canonical = ("gemini", {"models": {"g-1": {"cost": {"input": 1.0, "output": 2.0}}}})
        alias = (
            "google",
            {
                "models": {
                    "g-1": {"cost": {"input": 9.0, "output": 9.0}},
                    "g-2": {"cost": {"input": 0.5, "output": 0.5}},
                }
            },
        )
        payload = dict([canonical, alias] if canonical_first else [alias, canonical])
        out = normalise_models_dev(payload, _KNOWN)
        # The id that is the served key wins the clash; the alias only fills gaps.
        assert out["gemini"]["g-1"]["input_cost_per_mtok"] == 1.0
        assert out["gemini"]["g-2"]["input_cost_per_mtok"] == 0.5

    def test_non_object_payload_raises(self) -> None:
        with pytest.raises(CatalogueRefreshError, match="not a JSON object"):
            normalise_models_dev([1, 2], _KNOWN)

    def test_malformed_rows_skipped_not_raised(self) -> None:
        payload = {
            "openai": {
                "models": {
                    "good": {"cost": {"input": 1, "output": 2}},
                    "bad": "not-a-dict",
                }
            }
        }
        out = normalise_models_dev(payload, _KNOWN)
        assert "good" in out["openai"]
        assert "bad" not in out["openai"]


class TestNormaliseLiteLLM:
    def test_per_token_cost_scaled_to_mtok(self) -> None:
        payload = {
            "gpt-5.4-mini": {
                "litellm_provider": "openai",
                "input_cost_per_token": 0.0000004,
                "output_cost_per_token": 0.0000016,
                "max_input_tokens": 400000,
                "max_output_tokens": 128000,
                "supports_function_calling": True,
                "supports_vision": True,
            },
            "sample_spec": {"litellm_provider": "openai"},
        }
        out = normalise_litellm(payload, _KNOWN)
        row = out["openai"]["gpt-5.4-mini"]
        assert row["input_cost_per_mtok"] == pytest.approx(0.4)
        assert row["output_cost_per_mtok"] == pytest.approx(1.6)
        assert row["tool_use"] is True
        assert row["vision"] is True
        # sample_spec is skipped, never a model.
        assert "sample_spec" not in out["openai"]

    def test_unserved_provider_dropped(self) -> None:
        payload = {"m": {"litellm_provider": "cohere", "input_cost_per_token": 1e-6}}
        assert normalise_litellm(payload, _KNOWN) == {}


# ---------------------------------------------------------------------------
# Overlay writer
# ---------------------------------------------------------------------------


class TestWriteOverlay:
    def test_writes_meta_and_providers(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "overlay.json"
        report = write_overlay(
            {"openai": {"m": {"input_cost_per_mtok": 1.0, "output_cost_per_mtok": 2.0}}},
            target,
            source="models_dev",
            source_url="https://models.dev/api.json",
        )
        assert target.is_file()
        doc = json.loads(target.read_text())
        assert doc["_meta"]["kind"] == "overlay"
        assert doc["_meta"]["source"] == "models_dev"
        assert "generated_at" in doc["_meta"]
        assert doc["providers"]["openai"]["m"]["input_cost_per_mtok"] == 1.0
        assert report.provider_count == 1
        assert report.model_count == 1
        assert report.overlay_path == str(target)

    @pytest.mark.security
    def test_overlay_never_stores_pricing_tier(self, tmp_path: Path) -> None:
        target = tmp_path / "overlay.json"
        write_overlay(
            {"openai": {"m": {"input_cost_per_mtok": 1.0, "output_cost_per_mtok": 2.0}}},
            target,
            source="models_dev",
            source_url="https://models.dev/api.json",
        )
        doc = json.loads(target.read_text())
        for models in doc["providers"].values():
            for row in models.values():
                assert "pricing_tier" not in row

    def test_write_is_atomic_replace(self, tmp_path: Path) -> None:
        target = tmp_path / "overlay.json"
        target.write_text('{"old": true}')
        write_overlay(
            {"openai": {"m": {"input_cost_per_mtok": 1.0, "output_cost_per_mtok": 2.0}}},
            target,
            source="litellm",
            source_url="https://example.test/x.json",
        )
        doc = json.loads(target.read_text())
        assert "providers" in doc
        # No leftover temp files in the directory.
        assert sorted(p.name for p in tmp_path.iterdir()) == ["overlay.json"]


# ---------------------------------------------------------------------------
# Orchestration — refresh_overlay
# ---------------------------------------------------------------------------


class TestRefreshOverlay:
    def test_happy_path_models_dev(self, tmp_path: Path) -> None:
        payload = {"openai": {"models": {"gpt-5.4-mini": {"cost": {"input": 0.4, "output": 1.6}}}}}
        target = tmp_path / "overlay.json"
        report = refresh_overlay(
            overlay_path=target,
            source="models_dev",
            known_providers=_KNOWN,
            client_factory=_client_factory(_json_handler(payload)),
        )
        assert report.model_count == 1
        assert target.is_file()

    def test_unknown_source_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(CatalogueRefreshError, match="Unknown catalogue refresh source"):
            refresh_overlay(overlay_path=tmp_path / "o.json", source="nope")

    @pytest.mark.security
    def test_fail_closed_leaves_prior_overlay_untouched(self, tmp_path: Path) -> None:
        target = tmp_path / "overlay.json"
        target.write_text('{"_meta": {"kind": "overlay"}, "providers": {"openai": {}}}')
        prior = target.read_text()

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        with pytest.raises(CatalogueRefreshError):
            refresh_overlay(
                overlay_path=target,
                source="models_dev",
                known_providers=_KNOWN,
                client_factory=_client_factory(handler),
            )
        # The prior overlay must survive a failed refresh byte-for-byte.
        assert target.read_text() == prior

    @pytest.mark.security
    def test_url_override_must_be_https(self, tmp_path: Path) -> None:
        with pytest.raises(CatalogueRefreshError, match="https"):
            refresh_overlay(
                overlay_path=tmp_path / "o.json",
                source="models_dev",
                url="http://evil.test/api.json",
                known_providers=_KNOWN,
            )

    def test_default_known_providers_resolved_from_registry(self, tmp_path: Path) -> None:
        # When known_providers is omitted, the served LLM set is resolved from
        # the registry — openrouter (arbitrary-slug) is excluded.
        served = refresh.resolve_known_llm_providers()
        assert "openai" in served
        assert "openrouter" not in served


@pytest.mark.security
class TestAggregatorIsDataOnly:
    """Aggregators are a **data** source, never a library.

    "No LiteLLM / aggregator" was relaxed to *no aggregator library on the
    request / credential path*. LiteLLM's catalogue is consumed only as a
    pinned, validated, offline-capable JSON document over the bounded fetch
    seam — the library itself is never a dependency and is never imported.
    This lock fails closed if anyone adds ``import litellm`` (or langchain /
    any aggregator) to the catalogue or refresh modules.
    """

    _AGGREGATOR_LIBS = ("litellm", "langchain", "llama_index", "llamaindex", "haystack")

    def test_litellm_source_is_a_pinned_https_data_url(self) -> None:
        # The LiteLLM source is a static JSON document fetched over the same
        # https-only, redirect-free envelope as every other source — not an
        # import of the LiteLLM package.
        source = refresh.BUILTIN_SOURCES["litellm"]
        assert source.default_url.startswith("https://")
        assert source.default_url.endswith(".json")

    def test_no_aggregator_library_imported_by_data_path(self) -> None:
        import inspect

        from sec_generative_search.providers import catalogue as catalogue_mod

        for module in (refresh, catalogue_mod):
            src = inspect.getsource(module)
            for lib in self._AGGREGATOR_LIBS:
                assert f"import {lib}" not in src, (
                    f"{module.__name__} imports the aggregator library {lib!r}; "
                    "aggregators must stay a data-only source."
                )

    def test_no_aggregator_library_is_a_runtime_dependency(self) -> None:
        # Belt-and-suspenders: even if something transitively pulled an
        # aggregator in, none must be reachable on the request/credential
        # path. The refresh seam imports only stdlib + httpx (lazily) + the
        # project's own modules.
        import importlib.util

        for lib in ("litellm", "langchain"):
            assert importlib.util.find_spec(lib) is None, (
                f"aggregator library {lib!r} is installed; it must never be a "
                "dependency — aggregators are a data-only metadata source."
            )
