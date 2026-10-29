"""Vendored model catalogue — single source of LLM capability + cost data.

The bundled JSON baseline (``data/model_catalogue.json``) is loaded once and
exposes O(1), credential-free, network-free lookups consumed by:

- :class:`~sec_generative_search.providers.registry.ProviderRegistry` — the
  classmethod capability probe used by the read routes / CLI, and
- the concrete LLM adapters' instance ``get_capabilities`` (e.g. the
  orchestrator's context-budgeting probe).

The pricing tier is **never** stored in the data file: it is derived from
exact cost by :class:`~sec_generative_search.core.types.ProviderCapability`
at construction (:func:`~sec_generative_search.core.types.derive_pricing_tier`).
``chat=True`` / ``embeddings=False`` are LLM-surface invariants applied by the
loader, not stored per row.

The active catalogue is a process-global swapped via :func:`set_catalogue` /
:func:`reset_catalogue`.  Tests use that seam to register fixture providers.
Nothing here touches the network or a credential — the refresh that *fetches*
an overlay is a separate, opt-in seam.

On first use the active catalogue is built as the **union of baseline and
overlay**: the bundled baseline is always the floor, and when a
validated overlay file is present on the data volume it is merged on top, per
field.  The overlay is re-treated as **untrusted input** at read time —
re-validated through the same gate the refresh seam uses at write time — and the
merge is fail-closed: a missing, unreadable, or invalid overlay leaves the
baseline serving alone.
``pricing_tier`` is never read from disk; it is derived from the *merged* cost.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Final

from sec_generative_search.core.logging import get_logger
from sec_generative_search.core.types import ProviderCapability

__all__ = [
    "ModelCatalogue",
    "capability_from_row",
    "model_catalogue",
    "reset_catalogue",
    "set_catalogue",
]

logger = get_logger(__name__)

# The baseline ships as package data under ``providers/data/``.  Referencing
# the parent package (rather than a ``data`` subpackage) keeps the data tree
# free of an ``__init__.py``; ``Traversable.joinpath`` descends into the plain
# directory.
_DATA_PACKAGE: Final = "sec_generative_search.providers"
_BASELINE_DIR: Final = "data"
_BASELINE_NAME: Final = "model_catalogue.json"

# Allow-listed capability fields a data-file row may carry.  ``chat`` /
# ``embeddings`` are LLM-surface invariants applied below, never read from the
# row, so an overlay payload cannot flip a row onto the embedding surface.
_BOOL_FIELDS: Final = (
    "streaming",
    "tool_use",
    "structured_output",
    "prompt_caching",
    "vision",
)
_INT_FIELDS: Final = ("context_window_tokens", "max_output_tokens")
_COST_FIELDS: Final = ("input_cost_per_mtok", "output_cost_per_mtok")


def capability_from_row(row: Mapping[str, Any]) -> ProviderCapability:
    """Map one catalogue row to a :class:`ProviderCapability` (LLM surface).

    ``chat=True`` / ``embeddings=False`` are applied here as surface
    invariants.  Cost validation (finite, non-negative) and pricing-tier
    derivation happen inside :meth:`ProviderCapability.__post_init__`; this
    function never assigns a tier.

    The lift is an explicit allow-list — keys outside
    ``_BOOL_FIELDS`` / ``_INT_FIELDS`` / ``_COST_FIELDS`` are ignored, so a
    future field in an overlay payload cannot smuggle state into the
    capability matrix.
    """
    kwargs: dict[str, Any] = {"chat": True}
    for field in _BOOL_FIELDS:
        kwargs[field] = bool(row.get(field, False))
    for field in _INT_FIELDS:
        value = row.get(field, 0)
        kwargs[field] = int(value) if value is not None else 0
    for field in _COST_FIELDS:
        value = row.get(field)
        kwargs[field] = float(value) if value is not None else None
    return ProviderCapability(**kwargs)


class ModelCatalogue:
    """Immutable in-memory map of LLM capabilities, keyed by (provider, slug).

    Built once from the bundled baseline.  Lookups are plain nested-dict
    access — O(1), no I/O, no credential.
    """

    __slots__ = ("_by_provider",)

    def __init__(self, by_provider: Mapping[str, Mapping[str, ProviderCapability]]) -> None:
        self._by_provider: dict[str, dict[str, ProviderCapability]] = {
            provider: dict(models) for provider, models in by_provider.items()
        }

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_llm_capability(self, provider: str, slug: str) -> ProviderCapability | None:
        """Return the capability for *(provider, slug)*, or ``None`` if absent."""
        return self._by_provider.get(provider, {}).get(slug)

    def list_llm_models(self, provider: str) -> list[str]:
        """Return *provider*'s catalogued slugs in declaration order."""
        return list(self._by_provider.get(provider, {}))

    def has_provider(self, provider: str) -> bool:
        """Whether *provider* has at least one catalogued model."""
        return provider in self._by_provider

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_rows(cls, providers: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> ModelCatalogue:
        """Build from raw data-file rows (``{provider: {slug: row}}``)."""
        built = {
            provider: {slug: capability_from_row(row) for slug, row in models.items()}
            for provider, models in providers.items()
        }
        return cls(built)

    @staticmethod
    def _read_baseline_document() -> Mapping[str, Any]:
        """Read + parse the bundled baseline JSON document (``{_meta, providers}``)."""
        text = (
            resources.files(_DATA_PACKAGE)
            .joinpath(_BASELINE_DIR, _BASELINE_NAME)
            .read_text(encoding="utf-8")
        )
        return json.loads(text)

    @classmethod
    def _baseline_rows(cls) -> dict[str, dict[str, dict[str, Any]]]:
        """Return the baseline as raw ``{provider: {slug: row}}`` dicts.

        The per-field overlay merge needs the *raw* rows (not the built
        capabilities) so the cost / token "unknown" sentinels can be compared
        field by field before the pricing tier is derived.
        """
        document = cls._read_baseline_document()
        providers = document.get("providers", {})
        return {
            provider: {slug: dict(row) for slug, row in models.items()}
            for provider, models in providers.items()
        }

    @classmethod
    def load_baseline(cls) -> ModelCatalogue:
        """Load the bundled JSON baseline shipped as package data."""
        return cls.from_rows(cls._baseline_rows())

    # ------------------------------------------------------------------
    # Overlay merge
    # ------------------------------------------------------------------

    @classmethod
    def load_overlay(
        cls, path: str | os.PathLike[str]
    ) -> dict[str, dict[str, dict[str, Any]]] | None:
        """Read + re-validate a catalogue overlay from disk as untrusted input.

        Returns the validated ``{provider: {slug: row}}`` map, or ``None`` when
        the overlay is absent, unreadable, not valid JSON, or fails the
        untrusted-input gate.  **Fail-closed**: any problem means the caller
        serves the baseline alone — a hostile, corrupt, or hand-edited overlay
        can never degrade or poison the active catalogue.

        The on-disk file is *not* a trust boundary: even though the refresh
        seam validated it at write time, it is re-validated here through the
        very same :func:`~sec_generative_search.providers.refresh.validate_catalogue_payload`
        gate.  The ``refresh`` import is deferred to keep the catalogue ⇄
        refresh dependency one-directional at import time.
        """
        target = Path(path)
        try:
            text = target.read_text(encoding="utf-8")
        except (OSError, ValueError):
            # Absent / unreadable is the common "never refreshed" case — the
            # baseline is the intended floor, so this is silent, not a warning.
            return None
        try:
            document = json.loads(text)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            logger.warning(
                "Catalogue overlay at the configured path is not valid JSON; "
                "ignoring it and serving the baseline."
            )
            return None
        if not isinstance(document, Mapping):
            logger.warning(
                "Catalogue overlay is not a JSON object; ignoring it and serving the baseline."
            )
            return None

        # Lazy import breaks the catalogue ⇄ refresh import cycle (refresh
        # imports ``capability_from_row`` from this module at its top).
        from sec_generative_search.core.exceptions import CatalogueRefreshError
        from sec_generative_search.providers.refresh import validate_catalogue_payload

        try:
            return validate_catalogue_payload(document.get("providers", {}))
        except CatalogueRefreshError:
            # Content-free by construction — the validator never echoes the
            # offending slug or value, and neither do we.
            logger.warning(
                "Catalogue overlay failed untrusted-input validation; ignoring "
                "it and serving the baseline."
            )
            return None

    @staticmethod
    def _merge_row(baseline: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
        """Per-field merge of a baseline row with a fresher overlay row.

        "Fresh-and-valid wins, else baseline survives", evaluated per field:

        - **Boolean capability flags are unioned.**  The overlay can *enable*
          a capability the baseline lacked, but a missing / ``False`` overlay
          flag never *disables* a curated baseline capability — the upstream
          normalisers default an unknown flag to ``False``, so overlay-``False``
          means "no fresh assertion", not "turn it off".  The hand-curated
          baseline is the more trustworthy source for a flag it already sets.
        - **Token-count fields** take the overlay value when it is known
          (``> 0``); ``0`` means unknown, so the baseline value survives.
        - **Cost fields** take the overlay value when it is known (not
          ``None``); ``None`` means unknown, so the baseline cost survives — a
          freshly fetched overlay can never wipe a curated baseline price to
          ``UNKNOWN``.

        Returns a clean allow-listed row ready for :func:`capability_from_row`,
        which re-derives the pricing tier from the *merged* cost.
        """
        merged: dict[str, Any] = {}
        for field in _BOOL_FIELDS:
            merged[field] = bool(baseline.get(field, False)) or bool(overlay.get(field, False))
        for field in _INT_FIELDS:
            over = overlay.get(field, 0)
            over_int = int(over) if isinstance(over, int) and not isinstance(over, bool) else 0
            merged[field] = over_int if over_int > 0 else int(baseline.get(field, 0) or 0)
        for field in _COST_FIELDS:
            over = overlay.get(field)
            merged[field] = over if over is not None else baseline.get(field)
        return merged

    @classmethod
    def load_merged(cls, overlay_path: str | os.PathLike[str] | None) -> ModelCatalogue:
        """Build the active catalogue as the **union of baseline and overlay**.

        The bundled baseline is always the floor.  When a valid overlay is
        present it is merged on top: a model only in the overlay is *added*; a
        model only in the baseline *survives*; a model in both is merged per
        field by :meth:`_merge_row`.  Slugs are matched case-insensitively
        (models.dev spells ``MiniMax-M2.7`` where the baseline curates
        ``minimax-m2.7``) and the baseline spelling stays canonical, so a
        refresh updates the curated row instead of adding a duplicate.  The
        pricing tier is re-derived from the merged cost at
        :func:`capability_from_row`, never read from disk.

        **Fail-closed**: a missing / unreadable / invalid overlay — or no path
        at all — yields the baseline alone.  Because the overlay can only
        *raise* a baseline model's cost from unknown (it never wipes a known
        baseline cost), every baseline model keeps its non-``UNKNOWN`` tier
        after the merge; only overlay-only models may carry ``UNKNOWN``.

        The result is cached on the active-catalogue slot, so a refresh that
        writes a new overlay only takes effect on the next process start or
        after :func:`reset_catalogue`.
        """
        baseline_rows = cls._baseline_rows()
        overlay_rows = cls.load_overlay(overlay_path) if overlay_path is not None else None
        if not overlay_rows:
            return cls.from_rows(baseline_rows)

        built: dict[str, dict[str, ProviderCapability]] = {}
        overlay_only_providers = [p for p in overlay_rows if p not in baseline_rows]
        for provider in (*baseline_rows, *overlay_only_providers):
            b_models = baseline_rows.get(provider, {})
            o_models = overlay_rows.get(provider, {})
            o_by_lower = {slug.lower(): o_row for slug, o_row in o_models.items()}
            models: dict[str, ProviderCapability] = {}
            # Baseline slugs first (preserving declaration order), each merged
            # with its overlay counterpart when present — an exact-case match
            # first, else a case-insensitive one.
            for slug, b_row in b_models.items():
                o_row = o_models.get(slug)
                if o_row is None:
                    o_row = o_by_lower.get(slug.lower())
                row = cls._merge_row(b_row, o_row) if o_row is not None else b_row
                models[slug] = capability_from_row(row)
            # Then overlay-only slugs, added verbatim (already validated).
            b_lower = {slug.lower() for slug in b_models}
            for slug, o_row in o_models.items():
                if slug.lower() in b_lower:
                    continue
                models[slug] = capability_from_row(o_row)
            built[provider] = models
        return cls(built)

    def with_provider(
        self, provider: str, models: Mapping[str, ProviderCapability]
    ) -> ModelCatalogue:
        """Return a new catalogue with *provider*'s models added / overridden.

        Used by the catalogue swap seam and by tests to layer additional
        providers onto the baseline without mutating the shared instance.
        """
        merged = {p: dict(m) for p, m in self._by_provider.items()}
        merged.setdefault(provider, {}).update(models)
        return ModelCatalogue(merged)


_active: ModelCatalogue | None = None


def _resolve_overlay_path() -> str | None:
    """Resolve the configured catalogue-overlay path, fail-closed.

    Reads :class:`~sec_generative_search.config.settings.ProviderSettings`
    standalone — it carries no required fields (so it never fails on a missing
    EDGAR identity) and applies the same project-dir / symlink guard as the
    database paths.  Any failure — a misconfigured path, an import problem —
    collapses to ``None`` so the catalogue always loads the baseline rather
    than raising on the capability hot path.
    """
    try:
        from sec_generative_search.config.settings import ProviderSettings

        return ProviderSettings().catalogue_overlay_path
    except Exception:
        # Config trouble (a misconfigured path, an import problem) must never
        # break the capability hot path — fall back to the baseline.
        return None


def model_catalogue() -> ModelCatalogue:
    """Return the active catalogue, lazily merging baseline and overlay on first use.

    The lazy load keeps importing this module side-effect-free; the baseline
    file (and any configured overlay) is read once, at the first capability
    probe (effectively startup), and cached for the process.  The merge is
    fail-closed: absent the overlay, this is exactly the bundled baseline.
    """
    global _active
    if _active is None:
        _active = ModelCatalogue.load_merged(_resolve_overlay_path())
    return _active


def set_catalogue(catalogue: ModelCatalogue) -> None:
    """Install *catalogue* as the active one (refresh-overlay swap / tests)."""
    global _active
    _active = catalogue


def reset_catalogue() -> None:
    """Drop the active catalogue so the next access reloads the baseline."""
    global _active
    _active = None
