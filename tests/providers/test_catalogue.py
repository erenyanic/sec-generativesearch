"""Tests for the vendored model catalogue.

Covers:

- Baseline load from package data (``providers/data/model_catalogue.json``).
- ``capability_from_row`` field mapping: surface invariants, allow-list,
  cost validation, cost-derived pricing tier.
- Lookups (``get_llm_capability`` / ``list_llm_models``) + the
  ``set_catalogue`` / ``reset_catalogue`` / ``with_provider`` seam.
- The **overlay merge**: ``load_overlay`` re-validates the on-disk overlay as
    untrusted input (fail-closed), ``load_merged`` merges baseline and overlay
    per field, and ``model_catalogue`` wires the merge at the active-catalogue
    seam.  Security-locked: a hostile / corrupt overlay can never degrade or
    poison the baseline, never disable a curated capability, never wipe a
    curated price to ``UNKNOWN``, and never flip a row onto the embedding
    surface.
- The load-bearing **baseline invariant** (``@pytest.mark.security``): every
    vendored model carries exact cost and a non-``UNKNOWN`` derived tier. This
    stays locked for the bundled baseline even if future overlay models are
    allowed to remain ``UNKNOWN``.
"""

from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path
from typing import Any

import pytest

from sec_generative_search.core.types import PricingTier, ProviderCapability
from sec_generative_search.providers import catalogue as catalogue_mod
from sec_generative_search.providers.catalogue import (
    ModelCatalogue,
    capability_from_row,
    model_catalogue,
    reset_catalogue,
    set_catalogue,
)


def _overlay_row(**overrides: Any) -> dict[str, Any]:
    """A fully-populated, valid overlay row (mirrors the refresh writer's shape)."""
    base: dict[str, Any] = {
        "streaming": True,
        "tool_use": False,
        "structured_output": False,
        "prompt_caching": False,
        "vision": False,
        "context_window_tokens": 128_000,
        "max_output_tokens": 4096,
        "input_cost_per_mtok": 1.0,
        "output_cost_per_mtok": 2.0,
    }
    base.update(overrides)
    return base


def _write_overlay(
    path: Path,
    providers: dict[str, dict[str, dict[str, Any]]],
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write an ``{_meta, providers}`` overlay document to *path*."""
    document = {
        "_meta": meta if meta is not None else {"schema_version": 1, "kind": "overlay"},
        "providers": providers,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _reset_active_catalogue() -> None:
    """Each test starts and ends on the bundled baseline."""
    reset_catalogue()
    yield
    reset_catalogue()


# ---------------------------------------------------------------------------
# Baseline load + package data
# ---------------------------------------------------------------------------


class TestBaselineLoad:
    def test_baseline_resource_is_packaged(self) -> None:
        # The data file must ship as package data so the wheel / Docker
        # image can load it; a missing carve-out would break this.
        resource = resources.files("sec_generative_search.providers").joinpath(
            "data", "model_catalogue.json"
        )
        assert resource.is_file()

    def test_meta_block_present(self) -> None:
        text = (
            resources.files("sec_generative_search.providers")
            .joinpath("data", "model_catalogue.json")
            .read_text(encoding="utf-8")
        )
        document = json.loads(text)
        assert document["_meta"]["schema_version"] == 1
        # The tier is derived, never stored — guard against a regression
        # that re-introduces a second pricing table in the data file.
        for models in document["providers"].values():
            for row in models.values():
                assert "pricing_tier" not in row

    def test_load_baseline_has_known_providers_and_models(self) -> None:
        cat = ModelCatalogue.load_baseline()
        assert cat.has_provider("openai")
        assert cat.has_provider("anthropic")
        assert "gpt-4o" in cat.list_llm_models("openai")
        # OpenRouter is intentionally absent (arbitrary-slug meta-provider).
        assert not cat.has_provider("openrouter")
        assert cat.list_llm_models("openrouter") == []

    def test_model_catalogue_is_cached(self) -> None:
        assert model_catalogue() is model_catalogue()


# ---------------------------------------------------------------------------
# capability_from_row
# ---------------------------------------------------------------------------


class TestCapabilityFromRow:
    def test_surface_invariants_applied(self) -> None:
        cap = capability_from_row({})
        assert cap.chat is True
        assert cap.embeddings is False

    def test_full_row_maps_every_field(self) -> None:
        cap = capability_from_row(
            {
                "streaming": True,
                "tool_use": True,
                "structured_output": True,
                "prompt_caching": True,
                "vision": True,
                "context_window_tokens": 400_000,
                "max_output_tokens": 128_000,
                "input_cost_per_mtok": 0.9,
                "output_cost_per_mtok": 3.6,
            }
        )
        assert cap.streaming and cap.tool_use and cap.structured_output
        assert cap.prompt_caching and cap.vision
        assert cap.context_window_tokens == 400_000
        assert cap.max_output_tokens == 128_000
        assert cap.input_cost_per_mtok == 0.9
        assert cap.output_cost_per_mtok == 3.6
        # blended (0.9 + 3.6) / 2 = 2.25 -> STANDARD
        assert cap.pricing_tier is PricingTier.STANDARD

    def test_missing_booleans_default_false(self) -> None:
        cap = capability_from_row({"input_cost_per_mtok": 0.1, "output_cost_per_mtok": 0.1})
        assert cap.tool_use is False
        assert cap.vision is False

    def test_absent_cost_is_unknown_tier(self) -> None:
        cap = capability_from_row({"context_window_tokens": 100})
        assert cap.input_cost_per_mtok is None
        assert cap.output_cost_per_mtok is None
        assert cap.pricing_tier is PricingTier.UNKNOWN

    def test_unknown_keys_are_ignored(self) -> None:
        # The lift is an explicit allow-list: an unexpected key (e.g. one a
        # future overlay payload might smuggle) must not reach the dataclass.
        cap = capability_from_row(
            {"input_cost_per_mtok": 0.1, "output_cost_per_mtok": 0.1, "embeddings": True, "evil": 1}
        )
        # ``embeddings`` is a surface invariant, never read from the row.
        assert cap.embeddings is False

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite and non-negative"):
            capability_from_row({"input_cost_per_mtok": -1.0, "output_cost_per_mtok": 1.0})


# ---------------------------------------------------------------------------
# Lookups + swap seam
# ---------------------------------------------------------------------------


class TestLookupsAndSeam:
    def test_get_llm_capability_unknown_returns_none(self) -> None:
        assert model_catalogue().get_llm_capability("openai", "no-such-slug") is None
        assert model_catalogue().get_llm_capability("no-such-provider", "x") is None

    def test_with_provider_does_not_mutate_original(self) -> None:
        base = model_catalogue()
        extra = ProviderCapability(chat=True, streaming=True)
        layered = base.with_provider("demo-vendor", {"demo-chat": extra})
        assert layered.get_llm_capability("demo-vendor", "demo-chat") is extra
        # Original instance is untouched.
        assert base.get_llm_capability("demo-vendor", "demo-chat") is None

    def test_set_and_reset_catalogue(self) -> None:
        custom = ModelCatalogue({"demo-vendor": {"demo-chat": ProviderCapability(chat=True)}})
        set_catalogue(custom)
        assert model_catalogue() is custom
        reset_catalogue()
        # Reload yields the baseline again (a fresh instance, not ``custom``).
        assert model_catalogue() is not custom
        assert model_catalogue().has_provider("openai")


# ---------------------------------------------------------------------------
# Baseline invariant (security lock)
# ---------------------------------------------------------------------------


class TestBaselineInvariant:
    @pytest.mark.security
    def test_every_baseline_model_has_cost_and_non_unknown_tier(self) -> None:
        """Vendored baseline rows must carry exact cost + a real tier.

        Future overlay models may be ``UNKNOWN``, but the **bundled
        baseline** stays fully priced — it is the metrics ``pricing_tier``
        label + UI cheap-vs-premium source.
        """
        cat = ModelCatalogue.load_baseline()
        document = json.loads(
            resources.files("sec_generative_search.providers")
            .joinpath("data", "model_catalogue.json")
            .read_text(encoding="utf-8")
        )
        offenders: list[str] = []
        for provider, models in document["providers"].items():
            for slug in models:
                cap = cat.get_llm_capability(provider, slug)
                assert cap is not None
                if cap.input_cost_per_mtok is None or cap.output_cost_per_mtok is None:
                    offenders.append(f"{provider}/{slug}: missing cost")
                elif not (
                    math.isfinite(cap.input_cost_per_mtok)
                    and math.isfinite(cap.output_cost_per_mtok)
                ):
                    offenders.append(f"{provider}/{slug}: non-finite cost")
                elif cap.pricing_tier is PricingTier.UNKNOWN:
                    offenders.append(f"{provider}/{slug}: UNKNOWN tier")
        assert not offenders, offenders


# ---------------------------------------------------------------------------
# Overlay merge — per-field merge primitive
# ---------------------------------------------------------------------------


class TestMergeRow:
    """``_merge_row`` per-field semantics: fresh-and-valid wins, else baseline."""

    def test_boolean_overlay_enables_capability(self) -> None:
        merged = ModelCatalogue._merge_row({"vision": False}, _overlay_row(vision=True))
        assert merged["vision"] is True

    def test_boolean_overlay_false_never_disables_baseline(self) -> None:
        # A curated baseline flag survives an overlay that defaulted it to
        # False (the normalisers cannot prove a flag, so they emit False).
        merged = ModelCatalogue._merge_row({"tool_use": True}, _overlay_row(tool_use=False))
        assert merged["tool_use"] is True

    def test_token_overlay_wins_when_known(self) -> None:
        merged = ModelCatalogue._merge_row(
            {"context_window_tokens": 100}, _overlay_row(context_window_tokens=200_000)
        )
        assert merged["context_window_tokens"] == 200_000

    def test_token_baseline_survives_when_overlay_zero(self) -> None:
        # 0 means "unknown" — the baseline value is the trustworthy one.
        merged = ModelCatalogue._merge_row(
            {"context_window_tokens": 128_000}, _overlay_row(context_window_tokens=0)
        )
        assert merged["context_window_tokens"] == 128_000

    def test_cost_overlay_wins_when_known(self) -> None:
        merged = ModelCatalogue._merge_row(
            {"input_cost_per_mtok": 10.0, "output_cost_per_mtok": 40.0},
            _overlay_row(input_cost_per_mtok=0.5, output_cost_per_mtok=0.5),
        )
        assert merged["input_cost_per_mtok"] == 0.5
        assert merged["output_cost_per_mtok"] == 0.5

    def test_cost_baseline_survives_when_overlay_unknown(self) -> None:
        # None means "unknown" — a fresh overlay must never wipe a curated
        # price (this is what keeps baseline tiers non-UNKNOWN after a merge).
        merged = ModelCatalogue._merge_row(
            {"input_cost_per_mtok": 10.0, "output_cost_per_mtok": 40.0},
            _overlay_row(input_cost_per_mtok=None, output_cost_per_mtok=None),
        )
        assert merged["input_cost_per_mtok"] == 10.0
        assert merged["output_cost_per_mtok"] == 40.0

    @pytest.mark.security
    def test_merged_row_carries_only_allow_listed_keys(self) -> None:
        # A merged row is a clean allow-list projection — a smuggled field on
        # either side can never reach ``capability_from_row``.
        merged = ModelCatalogue._merge_row(
            {"evil_baseline": 1},
            {**_overlay_row(), "evil_overlay": 2, "pricing_tier": "premium"},
        )
        assert set(merged) == {
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


# ---------------------------------------------------------------------------
# Overlay load — fail-closed untrusted-input gate
# ---------------------------------------------------------------------------


class TestLoadOverlay:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        # The common "never refreshed / offline" case — silent fall-through.
        assert ModelCatalogue.load_overlay(tmp_path / "absent.json") is None

    def test_valid_overlay_returns_validated_rows(self, tmp_path: Path) -> None:
        path = _write_overlay(tmp_path / "o.json", {"openai": {"gpt-x": _overlay_row()}})
        rows = ModelCatalogue.load_overlay(path)
        assert rows is not None
        assert rows["openai"]["gpt-x"]["input_cost_per_mtok"] == 1.0

    def test_empty_providers_returns_empty_map(self, tmp_path: Path) -> None:
        path = _write_overlay(tmp_path / "o.json", {})
        assert ModelCatalogue.load_overlay(path) == {}

    @pytest.mark.security
    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "o.json"
        path.write_text("not json{{", encoding="utf-8")
        assert ModelCatalogue.load_overlay(path) is None

    @pytest.mark.security
    def test_non_object_document_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "o.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert ModelCatalogue.load_overlay(path) is None

    @pytest.mark.security
    @pytest.mark.parametrize(
        "providers",
        [
            {"Openai": {"m": _overlay_row()}},  # malformed provider slug
            {"openai": {"bad slug": _overlay_row()}},  # malformed model slug
            {"openai": {"m": _overlay_row(input_cost_per_mtok=-1.0)}},  # negative cost
            {"openai": {"m": _overlay_row(input_cost_per_mtok=2_000_000.0)}},  # out of bounds
            {"openai": {"m": _overlay_row(vision="yes")}},  # non-bool flag
        ],
    )
    def test_malicious_overlay_rejected(self, tmp_path: Path, providers: dict[str, Any]) -> None:
        path = _write_overlay(tmp_path / "o.json", providers)
        # Fail-closed: any validation failure drops the whole overlay.
        assert ModelCatalogue.load_overlay(path) is None

    @pytest.mark.security
    def test_pricing_tier_never_survives_load(self, tmp_path: Path) -> None:
        path = _write_overlay(
            tmp_path / "o.json",
            {"openai": {"m": {**_overlay_row(), "pricing_tier": "premium"}}},
        )
        rows = ModelCatalogue.load_overlay(path)
        assert rows is not None
        assert "pricing_tier" not in rows["openai"]["m"]


# ---------------------------------------------------------------------------
# Overlay merge — baseline-and-overlay integration
# ---------------------------------------------------------------------------


class TestLoadMerged:
    def test_none_path_is_pure_baseline(self) -> None:
        merged = ModelCatalogue.load_merged(None)
        baseline = ModelCatalogue.load_baseline()
        assert "gpt-4o" in merged.list_llm_models("openai")
        assert merged.get_llm_capability("openai", "gpt-4o") == baseline.get_llm_capability(
            "openai", "gpt-4o"
        )

    def test_missing_overlay_fails_closed_to_baseline(self, tmp_path: Path) -> None:
        merged = ModelCatalogue.load_merged(tmp_path / "absent.json")
        assert "gpt-4o" in merged.list_llm_models("openai")

    def test_overlay_only_model_added_to_existing_provider(self, tmp_path: Path) -> None:
        path = _write_overlay(
            tmp_path / "o.json",
            {
                "openai": {
                    "gpt-future-1": _overlay_row(input_cost_per_mtok=0.2, output_cost_per_mtok=0.2)
                }
            },
        )
        merged = ModelCatalogue.load_merged(path)
        assert "gpt-future-1" in merged.list_llm_models("openai")
        # The baseline models are still present alongside the addition.
        assert "gpt-4o" in merged.list_llm_models("openai")
        cap = merged.get_llm_capability("openai", "gpt-future-1")
        assert cap is not None and cap.pricing_tier is PricingTier.LOW

    def test_overlay_only_provider_added(self, tmp_path: Path) -> None:
        path = _write_overlay(tmp_path / "o.json", {"newvendor": {"new-chat": _overlay_row()}})
        merged = ModelCatalogue.load_merged(path)
        assert merged.has_provider("newvendor")
        assert merged.has_provider("openai")  # baseline providers survive

    def test_overlay_slug_matched_case_insensitively(self, tmp_path: Path) -> None:
        # models.dev spells MiniMax slugs with capitals; the refresh must update
        # the curated lower-case row rather than add a duplicate beside it.
        path = _write_overlay(
            tmp_path / "o.json",
            {
                "minimax": {
                    "MiniMax-M2.7": _overlay_row(input_cost_per_mtok=0.5, output_cost_per_mtok=0.5)
                }
            },
        )
        merged = ModelCatalogue.load_merged(path)
        models = merged.list_llm_models("minimax")
        assert "minimax-m2.7" in models
        assert "MiniMax-M2.7" not in models
        cap = merged.get_llm_capability("minimax", "minimax-m2.7")
        assert cap is not None and cap.input_cost_per_mtok == 0.5

    def test_baseline_only_model_survives(self, tmp_path: Path) -> None:
        # An overlay touching only one provider leaves every other untouched.
        path = _write_overlay(tmp_path / "o.json", {"openai": {"gpt-x": _overlay_row()}})
        merged = ModelCatalogue.load_merged(path)
        assert merged.has_provider("anthropic")
        assert merged.list_llm_models("anthropic")

    def test_cost_refresh_wins_and_retiers(self, tmp_path: Path) -> None:
        # gpt-4o is HIGH in the baseline (2.5 / 10); a cheaper overlay price
        # must win and re-bucket the tier from the *merged* cost.
        path = _write_overlay(
            tmp_path / "o.json",
            {"openai": {"gpt-4o": _overlay_row(input_cost_per_mtok=0.5, output_cost_per_mtok=0.5)}},
        )
        cap = ModelCatalogue.load_merged(path).get_llm_capability("openai", "gpt-4o")
        assert cap is not None
        assert cap.input_cost_per_mtok == 0.5
        assert cap.pricing_tier is PricingTier.LOW

    @pytest.mark.security
    def test_overlay_unknown_cost_never_wipes_baseline_price(self, tmp_path: Path) -> None:
        path = _write_overlay(
            tmp_path / "o.json",
            {
                "openai": {
                    "gpt-4o": _overlay_row(input_cost_per_mtok=None, output_cost_per_mtok=None)
                }
            },
        )
        cap = ModelCatalogue.load_merged(path).get_llm_capability("openai", "gpt-4o")
        baseline = ModelCatalogue.load_baseline().get_llm_capability("openai", "gpt-4o")
        assert cap is not None and baseline is not None
        assert cap.input_cost_per_mtok == baseline.input_cost_per_mtok
        assert cap.pricing_tier is baseline.pricing_tier
        assert cap.pricing_tier is not PricingTier.UNKNOWN

    @pytest.mark.security
    def test_overlay_cannot_disable_curated_capability(self, tmp_path: Path) -> None:
        # gpt-4o advertises tool_use / vision in the baseline; an overlay that
        # defaulted them to False must not strip them.
        path = _write_overlay(
            tmp_path / "o.json",
            {"openai": {"gpt-4o": _overlay_row(tool_use=False, vision=False)}},
        )
        cap = ModelCatalogue.load_merged(path).get_llm_capability("openai", "gpt-4o")
        assert cap is not None
        assert cap.tool_use is True
        assert cap.vision is True

    @pytest.mark.security
    def test_overlay_cannot_flip_row_onto_embedding_surface(self, tmp_path: Path) -> None:
        path = _write_overlay(
            tmp_path / "o.json",
            {"openai": {"sneaky": {**_overlay_row(), "embeddings": True, "chat": False}}},
        )
        cap = ModelCatalogue.load_merged(path).get_llm_capability("openai", "sneaky")
        assert cap is not None
        assert cap.chat is True
        assert cap.embeddings is False

    @pytest.mark.security
    def test_malicious_overlay_leaves_baseline_serving(self, tmp_path: Path) -> None:
        # A single bad row drops the whole overlay; the baseline is untouched.
        path = _write_overlay(
            tmp_path / "o.json",
            {"openai": {"m": _overlay_row(input_cost_per_mtok=-5.0)}},
        )
        merged = ModelCatalogue.load_merged(path)
        baseline = ModelCatalogue.load_baseline()
        assert merged.get_llm_capability("openai", "gpt-4o") == baseline.get_llm_capability(
            "openai", "gpt-4o"
        )
        assert merged.get_llm_capability("openai", "m") is None


# ---------------------------------------------------------------------------
# Active-catalogue wiring — model_catalogue() + overlay-path resolution
# ---------------------------------------------------------------------------


class TestModelCatalogueWiring:
    def test_model_catalogue_picks_up_overlay(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write_overlay(tmp_path / "o.json", {"openai": {"gpt-future-2": _overlay_row()}})
        monkeypatch.setattr(catalogue_mod, "_resolve_overlay_path", lambda: str(path))
        reset_catalogue()
        assert "gpt-future-2" in model_catalogue().list_llm_models("openai")

    def test_model_catalogue_fails_closed_without_overlay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(catalogue_mod, "_resolve_overlay_path", lambda: None)
        reset_catalogue()
        assert model_catalogue().has_provider("openai")
        assert "gpt-4o" in model_catalogue().list_llm_models("openai")

    @pytest.mark.security
    def test_model_catalogue_not_poisoned_by_malicious_overlay(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write_overlay(tmp_path / "o.json", {"openai": {"x": _overlay_row(vision="yes")}})
        monkeypatch.setattr(catalogue_mod, "_resolve_overlay_path", lambda: str(path))
        reset_catalogue()
        cat = model_catalogue()
        baseline = ModelCatalogue.load_baseline()
        # Fail-closed to baseline — the bad row's slug never appears.
        assert cat.get_llm_capability("openai", "x") is None
        assert cat.get_llm_capability("openai", "gpt-4o") == baseline.get_llm_capability(
            "openai", "gpt-4o"
        )

    def test_resolve_overlay_path_honours_setting(self, clean_env: pytest.MonkeyPatch) -> None:
        # A project-relative override is honoured (and passes the project-dir
        # guard in ProviderSettings).
        clean_env.setenv("PROVIDER_CATALOGUE_OVERLAY_PATH", "./data/custom_overlay.json")
        assert catalogue_mod._resolve_overlay_path() == "./data/custom_overlay.json"

    def test_resolve_overlay_path_fails_closed_on_bad_config(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        # An out-of-project path is rejected by ProviderSettings; resolution
        # must collapse to None rather than raise on the capability hot path.
        clean_env.setenv("PROVIDER_CATALOGUE_OVERLAY_PATH", "/etc/evil_overlay.json")
        assert catalogue_mod._resolve_overlay_path() is None
