"""Opt-in model-catalogue refresh seam.

This module is the **only** place that fetches model metadata over the
network.  It is deliberately decoupled from the request path: nothing here is
imported by a route, a worker, or a credential surface, and the active
catalogue (:mod:`~sec_generative_search.providers.catalogue`) never reaches
across to call it.  A refresh runs only when an operator explicitly triggers
it (a CLI command / admin route / scheduled cron hitting the trigger).

The flow is three bounded, fail-closed stages:

1. **Fetch** — a single ``httpx`` GET against a *pinned, operator-overridable*
   HTTPS URL.  TLS verification on, redirects off (SSRF defence — a pinned
   URL must not be bounced to ``http://`` or an internal host), a hard
   timeout, and a streamed response-size cap so a hostile or runaway endpoint
   cannot exhaust memory.
2. **Normalise** — map the upstream schema (models.dev primary; LiteLLM JSON
   as a data-only secondary) into the catalogue's ``{provider: {slug: row}}``
   shape, **filtered to the LLM providers this build actually serves**.  The
   filter bounds the working set and keeps an overlay focused on the curated
   catalogue rather than the upstream's full, unbounded vendor list.
3. **Validate + write** — re-validate the normalised rows as **fully
   untrusted input** through the same per-row validator the baseline uses
   (:func:`~sec_generative_search.providers.catalogue.capability_from_row`),
   plus strict slug-shape / count / type / control-character checks, then
   write an **additive** JSON overlay to the data volume atomically.

The merge of overlay over baseline into the active catalogue is a separate
concern; this module's contract ends at a validated overlay file on disk.
``pricing_tier`` is **never** written to the overlay — it is derived from
cost at load, exactly as for the baseline.

Every failure mode raises :class:`CatalogueRefreshError` with a content-free
message; the seam never echoes the offending slug or the raw upstream body.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

from sec_generative_search.core.exceptions import CatalogueRefreshError
from sec_generative_search.core.logging import get_logger
from sec_generative_search.core.types import CatalogueRefreshReport
from sec_generative_search.providers.catalogue import capability_from_row

__all__ = [
    "BUILTIN_SOURCES",
    "CatalogueSource",
    "fetch_json",
    "normalise_litellm",
    "normalise_models_dev",
    "refresh_overlay",
    "resolve_known_llm_providers",
    "validate_catalogue_payload",
    "write_overlay",
]

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Bounds — every one of these is a security control, not a tuning knob.
# ---------------------------------------------------------------------------

#: Maximum response body accepted from an upstream source (bytes).  The real
#: payloads are a few hundred KiB; the cap is generous but finite so a hostile
#: or misbehaving endpoint cannot stream an unbounded body into memory.
MAX_RESPONSE_BYTES: Final = 8 * 1024 * 1024

#: Hard per-request timeout (seconds) applied to *both* connect and read.
FETCH_TIMEOUT_SECONDS: Final = 15.0

#: Upper bound on distinct providers in a validated overlay.  After the
#: served-provider filter the real number is ~11; the bound is defence in
#: depth against a normaliser regression.
MAX_PROVIDERS: Final = 64

#: Upper bound on models per provider.  Mirrors the metric-cardinality guard
#: in :mod:`core.metrics` (``_MAX_MODELS_PER_PROVIDER``); an overlay that
#: blows past this is treated as hostile, not truncated.
MAX_MODELS_PER_PROVIDER: Final = 512

#: Maximum accepted cost (USD per million tokens).  No real model is anywhere
#: near this; a value above it signals a unit-conversion bug or hostile input.
MAX_COST_PER_MTOK: Final = 1_000_000.0

#: Maximum accepted context-window / max-output token count.  A sanity ceiling
#: against an absurd integer smuggled through ``context_window_tokens``.
MAX_TOKEN_FIELD: Final = 100_000_000

#: Provider-key shape — identical to the ``X-Provider-Key-{provider}`` suffix
#: shape enforced at the API boundary, so an overlay can never introduce a
#: provider key the rest of the system would refuse to address.
_PROVIDER_SLUG_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

#: Model-slug shape.  Broader than a provider key because real model slugs
#: carry dots, slashes, and colons (``openai/gpt-5.4-mini``, ``glm-4.5-air``,
#: ``qwen3.5-omni-plus``), but still bounded and control-char-free: the first
#: character is alphanumeric and the remainder is a closed, printable set.
_MODEL_SLUG_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")

# Allow-listed row fields, mirroring ``catalogue.capability_from_row``.  The
# overlay is re-serialised from exactly these keys, so a future upstream field
# can never smuggle state into the on-disk overlay.
_BOOL_FIELDS: Final = (
    "streaming",
    "tool_use",
    "structured_output",
    "prompt_caching",
    "vision",
)
_INT_FIELDS: Final = ("context_window_tokens", "max_output_tokens")
_COST_FIELDS: Final = ("input_cost_per_mtok", "output_cost_per_mtok")

#: Overlay file schema version — independent of the baseline's; bump on any
#: shape change so a stale overlay can be recognised and ignored.
OVERLAY_SCHEMA_VERSION: Final = 1


# ---------------------------------------------------------------------------
# Source registry — pinned URLs + their normalisers
# ---------------------------------------------------------------------------


class CatalogueSource:
    """A pinned upstream source: a default URL and its normaliser.

    Curated, not pluggable — the same posture as
    :class:`~sec_generative_search.providers.registry.ProviderRegistry`.
    Adding a source is a one-line addition to :data:`BUILTIN_SOURCES`.
    """

    __slots__ = ("default_url", "key", "normalise")

    def __init__(
        self,
        key: str,
        default_url: str,
        normalise: Callable[[Any, set[str]], dict[str, dict[str, dict[str, Any]]]],
    ) -> None:
        self.key = key
        self.default_url = default_url
        self.normalise = normalise


def resolve_known_llm_providers() -> set[str]:
    """Return the set of LLM provider keys this build serves.

    Resolved lazily from the registry so the served set is always the
    curated truth.  OpenRouter is excluded automatically: it advertises an
    empty catalogue (arbitrary-slug meta-provider), so refreshing cost data
    for it is meaningless.
    """
    # Imported here rather than at module top to keep import-time cost off the
    # critical path and avoid a heavy import for callers that only need the
    # validator (e.g. unit tests passing an explicit provider set).
    from sec_generative_search.providers.registry import (
        ProviderRegistry,
        ProviderSurface,
    )

    served: set[str] = set()
    for name in ProviderRegistry.list_providers(ProviderSurface.LLM):
        if ProviderRegistry.supports_arbitrary_models(name, ProviderSurface.LLM):
            continue
        served.add(name)
    return served


# Best-effort alias map from common upstream provider identifiers to the
# internal provider keys this build serves.  Anything that does not map to a
# served provider is dropped (not an error) — the overlay stays focused on the
# curated catalogue.  Identity entries for our own keys are added below.
#
# Region mirrors of a served vendor (models.dev ``zhipuai`` = open.bigmodel.cn,
# ``alibaba-cn``, ``moonshotai-cn``, ``minimax-cn``) are deliberately *not*
# aliased: they price a different endpoint than the one each adapter targets,
# so folding them in would overwrite the served endpoint's prices.
_PROVIDER_ALIASES: Final[dict[str, str]] = {
    "google": "gemini",
    "google-ai-studio": "gemini",
    "google_ai_studio": "gemini",
    "vertex_ai": "gemini",
    "moonshot": "kimi",
    "moonshotai": "kimi",
    "x-ai": "grok",
    "xai": "grok",
    "z-ai": "zai",
    "alibaba": "qwen",
    "dashscope": "qwen",
    "mistralai": "mistral",
    "xiaomi": "mimo",
}


def _canonical_provider(raw: str, known: set[str]) -> str | None:
    """Map an upstream provider id onto a served key, or ``None`` to drop it."""
    candidate = _PROVIDER_ALIASES.get(raw, raw)
    return candidate if candidate in known else None


# ---------------------------------------------------------------------------
# Normalisers — upstream schema -> {provider: {slug: row}}
# ---------------------------------------------------------------------------
#
# Normalisers are intentionally *lenient* at extraction (``.get`` with safe
# defaults, malformed rows skipped) because the strict gate is the validator
# that runs afterwards.  They never raise on a single odd row; they simply do
# not emit it.  What they emit is then re-checked as untrusted input.


def _coerce_cost(value: Any, *, scale: float) -> float | None:
    """Coerce an upstream numeric cost to USD-per-MTok, or ``None``.

    Returns ``None`` for absent / non-numeric / non-finite / negative input
    so the row simply carries an unknown cost rather than poisoning the
    overlay.  ``scale`` converts the upstream unit (1.0 when already per-MTok,
    1e6 when per-token).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    cost = float(value) * scale
    if not math.isfinite(cost) or cost < 0.0:
        return None
    return cost


def _coerce_int(value: Any) -> int:
    """Coerce an upstream token-count field to a non-negative int (else 0)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    ivalue = int(value)
    return ivalue if ivalue >= 0 else 0


def normalise_models_dev(payload: Any, known: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """Normalise a models.dev ``api.json`` document.

    Shape: ``{provider_id: {"models": {slug: {cost: {input, output},
    limit: {context, output}, tool_call, reasoning, modalities, ...}}}}``.
    Costs are already USD per million tokens.
    """
    if not isinstance(payload, Mapping):
        raise CatalogueRefreshError("models.dev payload is not a JSON object.")

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_provider, pdata in payload.items():
        if not isinstance(raw_provider, str) or not isinstance(pdata, Mapping):
            continue
        provider = _canonical_provider(raw_provider, known)
        if provider is None:
            continue
        models = pdata.get("models")
        if not isinstance(models, Mapping):
            continue
        rows: dict[str, dict[str, Any]] = {}
        for slug, mdata in models.items():
            if not isinstance(slug, str) or not isinstance(mdata, Mapping):
                continue
            cost = mdata.get("cost") if isinstance(mdata.get("cost"), Mapping) else {}
            limit = mdata.get("limit") if isinstance(mdata.get("limit"), Mapping) else {}
            modalities = mdata.get("modalities")
            vision = False
            if isinstance(modalities, Mapping):
                inputs = modalities.get("input")
                vision = isinstance(inputs, (list, tuple)) and "image" in inputs
            rows[slug] = {
                "streaming": True,
                "tool_use": bool(mdata.get("tool_call", False)),
                "structured_output": bool(mdata.get("structured_output", False)),
                "prompt_caching": cost.get("cache_read") is not None,
                "vision": vision,
                "context_window_tokens": _coerce_int(limit.get("context")),
                "max_output_tokens": _coerce_int(limit.get("output")),
                "input_cost_per_mtok": _coerce_cost(cost.get("input"), scale=1.0),
                "output_cost_per_mtok": _coerce_cost(cost.get("output"), scale=1.0),
            }
        if not rows:
            continue
        # Several upstream ids can alias one served key, so merge rather than
        # replace: the id that *is* the served key wins a slug clash and an
        # alias only fills slugs it lacks, independent of upstream key order.
        target = out.setdefault(provider, {})
        if raw_provider == provider:
            target.update(rows)
        else:
            for slug, row in rows.items():
                target.setdefault(slug, row)
    return out


def normalise_litellm(payload: Any, known: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """Normalise a LiteLLM ``model_prices_and_context_window.json`` document.

    Shape: a flat ``{model_name: {litellm_provider, input_cost_per_token,
    output_cost_per_token, max_input_tokens, max_output_tokens,
    supports_function_calling, supports_vision, ...}}`` map (plus a
    ``sample_spec`` key that is skipped).  Costs are USD *per token*, so they
    are scaled by 1e6.
    """
    if not isinstance(payload, Mapping):
        raise CatalogueRefreshError("LiteLLM payload is not a JSON object.")

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for slug, mdata in payload.items():
        if slug == "sample_spec" or not isinstance(slug, str):
            continue
        if not isinstance(mdata, Mapping):
            continue
        raw_provider = mdata.get("litellm_provider")
        if not isinstance(raw_provider, str):
            continue
        provider = _canonical_provider(raw_provider, known)
        if provider is None:
            continue
        row = {
            "streaming": bool(mdata.get("supports_streaming", True)),
            "tool_use": bool(mdata.get("supports_function_calling", False)),
            "structured_output": bool(mdata.get("supports_response_schema", False)),
            "prompt_caching": bool(mdata.get("supports_prompt_caching", False)),
            "vision": bool(mdata.get("supports_vision", False)),
            "context_window_tokens": _coerce_int(
                mdata.get("max_input_tokens", mdata.get("max_tokens"))
            ),
            "max_output_tokens": _coerce_int(mdata.get("max_output_tokens")),
            "input_cost_per_mtok": _coerce_cost(
                mdata.get("input_cost_per_token"), scale=1_000_000.0
            ),
            "output_cost_per_mtok": _coerce_cost(
                mdata.get("output_cost_per_token"), scale=1_000_000.0
            ),
        }
        out.setdefault(provider, {})[slug] = row
    return out


BUILTIN_SOURCES: Final[dict[str, CatalogueSource]] = {
    "models_dev": CatalogueSource(
        "models_dev", "https://models.dev/api.json", normalise_models_dev
    ),
    "litellm": CatalogueSource(
        "litellm",
        "https://raw.githubusercontent.com/BerriAI/litellm/main/"
        "model_prices_and_context_window.json",
        normalise_litellm,
    ),
}


# ---------------------------------------------------------------------------
# Bounded fetch
# ---------------------------------------------------------------------------


def fetch_json(
    url: str,
    *,
    timeout: float = FETCH_TIMEOUT_SECONDS,
    max_bytes: int = MAX_RESPONSE_BYTES,
    client_factory: Callable[[], Any] | None = None,
) -> Any:
    """Fetch and parse JSON from *url* under strict transport bounds.

    Security envelope:

    - **TLS-only** — a non-``https`` scheme is rejected before any socket
      opens.
    - **No redirects** — a pinned URL must not be bounced to ``http://`` or an
      internal host (SSRF defence).
    - **Timeout** — applied to connect and read.
    - **Response-size cap** — the body is streamed and aborted the moment it
      exceeds *max_bytes*.
    - **TLS verification on** — certificate validation is never disabled.

    ``client_factory`` is an injection seam for tests (an ``httpx.Client``
    bound to a ``MockTransport``); production passes ``None`` and a verified,
    redirect-free client is built here.  Any transport / TLS / decode failure
    is collapsed into :class:`CatalogueRefreshError` so the seam stays
    fail-closed and content-free.
    """
    if not isinstance(url, str) or not url.lower().startswith("https://"):
        raise CatalogueRefreshError("Catalogue refresh source must be an https:// URL.")

    # Local import keeps httpx off this module's import-time cost and matches
    # the project's "network libs imported at use" convention.
    import httpx

    if client_factory is not None:
        client = client_factory()
    else:
        client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=False,
            verify=True,
        )

    try:
        with client, client.stream("GET", url) as response:
            response.raise_for_status()
            buffer = bytearray()
            for chunk in response.iter_bytes():
                buffer.extend(chunk)
                if len(buffer) > max_bytes:
                    raise CatalogueRefreshError(
                        f"Catalogue refresh response exceeded the {max_bytes}-byte size cap."
                    )
        return json.loads(bytes(buffer))
    except CatalogueRefreshError:
        raise
    except httpx.HTTPError as exc:
        # Message is the exception *type* only — never the URL body or any
        # header that an upstream might reflect back.
        raise CatalogueRefreshError(
            f"Catalogue refresh fetch failed ({type(exc).__name__})."
        ) from exc
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
        raise CatalogueRefreshError("Catalogue refresh response was not valid JSON.") from exc


# ---------------------------------------------------------------------------
# Untrusted-input validation
# ---------------------------------------------------------------------------


def _validate_row(provider: str, slug: str, raw: Any) -> dict[str, Any]:
    """Validate one normalised row as untrusted input; return a clean row.

    Strict on type and range, then delegates cost validation + pricing-tier
    derivation to :func:`capability_from_row` (the same per-row validator the
    baseline uses).  The returned dict carries only allow-listed keys, so the
    on-disk overlay is a faithful, minimal projection — never the raw input.
    """
    if not isinstance(raw, Mapping):
        raise CatalogueRefreshError(f"Catalogue row for provider '{provider}' is not an object.")

    clean: dict[str, Any] = {}
    for field in _BOOL_FIELDS:
        value = raw.get(field, False)
        if not isinstance(value, bool):
            raise CatalogueRefreshError(f"Catalogue field '{field}' must be a boolean.")
        clean[field] = value
    for field in _INT_FIELDS:
        value = raw.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise CatalogueRefreshError(f"Catalogue field '{field}' must be an integer.")
        if value < 0 or value > MAX_TOKEN_FIELD:
            raise CatalogueRefreshError(f"Catalogue field '{field}' is out of range.")
        clean[field] = value
    for field in _COST_FIELDS:
        value = raw.get(field)
        if value is None:
            clean[field] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CatalogueRefreshError(f"Catalogue field '{field}' must be a number or null.")
        fvalue = float(value)
        if not math.isfinite(fvalue) or fvalue < 0.0 or fvalue > MAX_COST_PER_MTOK:
            raise CatalogueRefreshError(
                f"Catalogue field '{field}' must be finite, non-negative, and within bounds."
            )
        clean[field] = fvalue

    # Final gate: build the capability through the canonical baseline
    # validator.  This re-checks cost finiteness / sign and derives the
    # pricing tier; a failure here is a hostile or malformed row.  The
    # capability object itself is discarded — we persist only the clean row;
    # the tier is re-derived at load, never stored.
    try:
        capability_from_row(clean)
    except (ValueError, TypeError) as exc:
        raise CatalogueRefreshError(
            f"Catalogue row failed capability validation: {type(exc).__name__}."
        ) from exc

    return clean


def validate_catalogue_payload(
    payload: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Validate a normalised ``{provider: {slug: row}}`` map as untrusted input.

    Enforces, top to bottom: object shape, provider count bound, provider-slug
    shape, per-provider model-count bound, model-slug shape, and per-row
    strict typing + finite/non-negative/bounded cost (via :func:`_validate_row`
    → :func:`capability_from_row`).  Control characters are rejected implicitly
    by the anchored slug patterns.

    Returns a freshly-built, allow-listed projection safe to serialise.  Any
    violation raises :class:`CatalogueRefreshError` with a content-free
    message — the offending value is never echoed.
    """
    if not isinstance(payload, Mapping):
        raise CatalogueRefreshError("Catalogue payload is not a JSON object.")
    if len(payload) > MAX_PROVIDERS:
        raise CatalogueRefreshError(
            f"Catalogue payload exceeds the {MAX_PROVIDERS}-provider bound."
        )

    validated: dict[str, dict[str, dict[str, Any]]] = {}
    for provider, models in payload.items():
        if not isinstance(provider, str) or not _PROVIDER_SLUG_RE.match(provider):
            raise CatalogueRefreshError("Catalogue payload carries a malformed provider key.")
        if not isinstance(models, Mapping):
            raise CatalogueRefreshError(
                f"Catalogue entry for provider '{provider}' is not an object."
            )
        if len(models) > MAX_MODELS_PER_PROVIDER:
            raise CatalogueRefreshError(
                f"Provider '{provider}' exceeds the {MAX_MODELS_PER_PROVIDER}-model bound."
            )
        rows: dict[str, dict[str, Any]] = {}
        for slug, row in models.items():
            if not isinstance(slug, str) or not _MODEL_SLUG_RE.match(slug):
                raise CatalogueRefreshError(
                    f"Provider '{provider}' carries a malformed model slug."
                )
            rows[slug] = _validate_row(provider, slug, row)
        validated[provider] = rows
    return validated


# ---------------------------------------------------------------------------
# Overlay writer
# ---------------------------------------------------------------------------


def write_overlay(
    validated: Mapping[str, Mapping[str, Mapping[str, Any]]],
    path: str | os.PathLike[str],
    *,
    source: str,
    source_url: str,
) -> CatalogueRefreshReport:
    """Atomically write *validated* rows as an additive catalogue overlay.

    The file mirrors the baseline's ``{_meta, providers}`` shape so the merge
    loader can read both uniformly.  ``pricing_tier`` is **never**
    written — it is derived from cost at load, exactly as for the baseline.

    The write is atomic: a temp file in the destination directory is fully
    written and ``fsync``-ed, then ``os.replace``-d over the target, so a
    concurrent reader (or a crash mid-write) never observes a half-written
    overlay.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    provider_count = len(validated)
    model_count = sum(len(models) for models in validated.values())

    document = {
        "_meta": {
            "schema_version": OVERLAY_SCHEMA_VERSION,
            "kind": "overlay",
            "source": source,
            "source_url": source_url,
            "generated_at": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cost_units": "USD per 1,000,000 tokens; null = unknown",
        },
        "providers": {
            provider: {slug: dict(row) for slug, row in models.items()}
            for provider, models in validated.items()
        },
    }
    serialised = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False)

    # Write-to-temp-then-replace in the same directory keeps the swap atomic
    # on a POSIX filesystem (``os.replace`` is atomic within a filesystem).
    fd, tmp_name = tempfile.mkstemp(prefix=".overlay-", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialised)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except OSError:
        # Best-effort cleanup; never shadow the original failure.
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise

    return CatalogueRefreshReport(
        source=source,
        source_url=source_url,
        provider_count=provider_count,
        model_count=model_count,
        overlay_path=str(target),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def refresh_overlay(
    *,
    overlay_path: str | os.PathLike[str],
    source: str = "models_dev",
    url: str | None = None,
    known_providers: set[str] | None = None,
    timeout: float = FETCH_TIMEOUT_SECONDS,
    max_bytes: int = MAX_RESPONSE_BYTES,
    client_factory: Callable[[], Any] | None = None,
) -> CatalogueRefreshReport:
    """Fetch, validate, and write a catalogue overlay; return the audit record.

    The single public entry point a trigger (CLI / admin route / scheduled
    job) calls.  ``source`` selects a built-in
    :class:`CatalogueSource`; ``url`` overrides its pinned default (still
    subject to the https-only check in :func:`fetch_json`).

    Fail-closed: any error in fetch, normalise, or validate raises
    :class:`CatalogueRefreshError` **before** the overlay is touched, so a
    failed refresh leaves any prior overlay — and the bundled baseline —
    serving unchanged.
    """
    catalogue_source = BUILTIN_SOURCES.get(source)
    if catalogue_source is None:
        raise CatalogueRefreshError(
            f"Unknown catalogue refresh source '{source}'. "
            f"Known sources: {sorted(BUILTIN_SOURCES)}."
        )

    fetch_url = url or catalogue_source.default_url
    served = known_providers if known_providers is not None else resolve_known_llm_providers()

    raw = fetch_json(
        fetch_url,
        timeout=timeout,
        max_bytes=max_bytes,
        client_factory=client_factory,
    )
    normalised = catalogue_source.normalise(raw, served)
    validated = validate_catalogue_payload(normalised)

    report = write_overlay(validated, overlay_path, source=source, source_url=fetch_url)
    # Content-free audit line: source key + counts only, never a slug or cost.
    logger.info(
        "Catalogue overlay refreshed from source '%s': %d providers, %d models.",
        report.source,
        report.provider_count,
        report.model_count,
    )
    return report
