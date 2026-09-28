"use client";

// ModelPicker — provider + model selection plus OpenRouter
// upstream-routing hints for the RAG and Chat surfaces.
//
// Provider list comes from `GET /api/providers/` via the admin proxy.
// The same `supports_upstream_routing` capability flag the backend
// relies on for its fail-closed guard drives the UI: the routing-hint
// fields only render when the selected provider advertises the
// capability. That keeps the UI and the backend on the same side of the
// invariant — the SPA never silently sends a hint the backend will
// reject as `invalid_flag_combination`.
//
// What lives where:
//   - `provider` / `model`: free-form strings — empty `model` lets the
//     backend fall back to settings / provider default.
//   - `routing_hints.order`: comma-separated upstream slugs (lowercase,
//     `-` separator) shape-checked at the schema layer.
//   - `routing_hints.allow_fallbacks`: tri-state — `undefined` defers to
//     OpenRouter's own default, `true` / `false` overrides.
//
// The picker is intentionally side-effect-free: it owns no fetch and
// holds no per-row state — the parent owns the `ModelPickerValue` and
// passes it back to the API call. This keeps the cancellation /
// abort-controller logic in the parent (the RAG and Chat pages each
// already manage one).

import { useEffect, useId, useMemo, useState, type JSX } from "react";

import type {
  OpenRouterRoutingHintsSchema,
  ProviderInfo,
} from "@/lib/api-types";

/**
 * The shape the picker emits — a complete picker selection that the
 * parent can hand straight to `planRagQuery` / `streamRagAnswer`.
 * `routing_hints` is `undefined` when the user has not set any hints OR
 * when the resolved provider does not advertise upstream-routing
 * capability (the UI hides the fields and clears the value).
 */
export interface ModelPickerValue {
  provider: string;
  model: string;
  routing_hints?: OpenRouterRoutingHintsSchema;
}

export interface ModelPickerProps {
  providers: ProviderInfo[];
  value: ModelPickerValue;
  onChange: (next: ModelPickerValue) => void;
  /** Disables every control while a request is in flight. */
  disabled?: boolean;
}

// Slug shape mirrors `_OPENROUTER_UPSTREAM_PATTERN` in
// `api/schemas.py`. Used here only for client-side UX feedback; the
// backend re-validates and rejects anything malformed at the schema
// layer (so a buggy / out-of-date regex here is a UX issue, not a
// security issue).
const UPSTREAM_SLUG_RE = /^[a-z0-9][a-z0-9-]{0,63}$/;

function parseUpstreamOrder(raw: string): string[] {
  return raw
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
}

function upstreamOrderHasError(entries: string[]): boolean {
  return entries.some((entry) => !UPSTREAM_SLUG_RE.test(entry));
}

// Plain-text legend for the coarse `PricingTier` scale. The
// per-model tier itself is served by `GET /api/providers/{provider}/models`
// — the backend catalogue is the single source of truth. This control does
// NOT fetch or look anything up: it only EXPLAINS what the tier words mean
// so an operator who sees "premium" on a model row knows the cost band.
//
// The tier is derived from the BLENDED price — the mean of the input and
// output USD per 1M tokens — by `derive_pricing_tier` in
// `src/sec_generative_search/core/types.py`. These upper bounds MUST equal
// its `_PRICING_TIER_BLENDED_BOUNDS` (Low / Standard / High; Premium is at
// or above the last); `tests/core/test_types.py` fails the build when the
// two drift, and `tests/security/model-picker.test.tsx` pins the rendered
// numbers.
export const PRICING_TIER_BLENDED_BOUNDS_USD = [1, 4, 15] as const;

const [LOW_BELOW, STANDARD_BELOW, HIGH_BELOW] = PRICING_TIER_BLENDED_BOUNDS_USD;
const BLENDED = "per 1M tokens, blended (mean of input and output price)";

export const PRICING_TIERS: ReadonlyArray<{ tier: string; blurb: string }> = [
  { tier: "Free", blurb: "No per-token cost (e.g. a model server you host)." },
  { tier: "Low", blurb: `Under $${LOW_BELOW} ${BLENDED}.` },
  { tier: "Standard", blurb: `$${LOW_BELOW} to under $${STANDARD_BELOW} ${BLENDED}.` },
  { tier: "High", blurb: `$${STANDARD_BELOW} to under $${HIGH_BELOW} ${BLENDED}.` },
  { tier: "Premium", blurb: `$${HIGH_BELOW} or more ${BLENDED}.` },
  {
    tier: "Unknown",
    blurb:
      "Un-catalogued or free-form models (e.g. OpenRouter slugs); cost is not classified.",
  },
];

/**
 * Accessible pricing-tier help.
 *
 * A focusable `<button>` toggles a plain-text note. `aria-expanded` +
 * `aria-controls` wire the disclosure; while open the trigger also points
 * `aria-describedby` at the note so assistive tech announces the
 * explanation on focus. It is deliberately NOT a hover-only sink — click
 * or keyboard (Enter / Space) opens it, Escape closes it, so a
 * keyboard-only user can always reach the content. The tiers render as a
 * plain `<dl>`; there is no Markdown / HTML sink, so the control widens no
 * DOM-XSS surface (the SPA's `react/no-danger` lock still holds).
 *
 * This `open` flag is presentational chrome only — it is never part of
 * `ModelPickerValue`, so the picker's "purely controlled selection"
 * contract is untouched.
 */
function PricingTierHelp(): JSX.Element {
  const [open, setOpen] = useState(false);
  const noteId = useId();

  return (
    <div className="text-xs">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={noteId}
        aria-describedby={open ? noteId : undefined}
        onClick={() => {
          setOpen((prev) => !prev);
        }}
        onKeyDown={(event) => {
          if (event.key === "Escape" && open) {
            setOpen(false);
          }
        }}
        className="inline-flex items-center gap-1 rounded text-slate-600 underline decoration-dotted underline-offset-2 hover:text-slate-900"
      >
        <span aria-hidden="true">ⓘ</span>
        What do pricing tiers mean?
      </button>
      {open ? (
        <div
          id={noteId}
          role="note"
          className="mt-2 rounded border border-slate-200 bg-white p-2 text-slate-600"
        >
          <dl className="space-y-1">
            {PRICING_TIERS.map(({ tier, blurb }) => (
              <div key={tier} className="flex gap-2">
                <dt className="font-medium text-slate-700">{tier}</dt>
                <dd>{blurb}</dd>
              </div>
            ))}
          </dl>
        </div>
      ) : null}
    </div>
  );
}

export function ModelPicker({
  providers,
  value,
  onChange,
  disabled = false,
}: ModelPickerProps): JSX.Element {
  const providerId = useId();
  const modelId = useId();
  const orderId = useId();
  const fallbacksId = useId();

  // Only LLM-surface entries are eligible for the model picker. The
  // catalogue may carry embedding / reranker rows; filtering happens
  // here rather than at the API call so a future surface stays opt-in.
  const llmProviders = useMemo(
    () => providers.filter((entry) => entry.surface === "llm"),
    [providers],
  );

  const supportsRouting = useMemo(() => {
    const entry = llmProviders.find((p) => p.name === value.provider);
    return entry?.supports_upstream_routing === true;
  }, [llmProviders, value.provider]);

  // When the user switches AWAY from a routing-capable provider, clear
  // any pending hints so the next request never carries a stale hint
  // the backend would reject as `invalid_flag_combination`. The effect
  // runs once per `supportsRouting` flip, not per render.
  useEffect(() => {
    if (!supportsRouting && value.routing_hints !== undefined) {
      onChange({ ...value, routing_hints: undefined });
    }
  }, [supportsRouting, value, onChange]);

  const orderText =
    value.routing_hints?.order !== undefined
      ? value.routing_hints.order.join(", ")
      : "";
  const fallbacksValue =
    value.routing_hints?.allow_fallbacks === undefined
      ? "default"
      : value.routing_hints.allow_fallbacks
        ? "true"
        : "false";

  const parsedOrder = useMemo(() => parseUpstreamOrder(orderText), [orderText]);
  const orderInvalid = upstreamOrderHasError(parsedOrder);

  const updateHints = (
    patch: Partial<OpenRouterRoutingHintsSchema>,
  ): void => {
    const next: OpenRouterRoutingHintsSchema = {
      ...(value.routing_hints ?? {}),
      ...patch,
    };
    // A hint object with neither an order entry nor an explicit
    // fallbacks toggle is equivalent to no hint at all — collapse to
    // `undefined` so the backend's `or_hints=none` audit branch fires.
    const hasOrder = Array.isArray(next.order) && next.order.length > 0;
    const hasFallbacks = next.allow_fallbacks !== undefined && next.allow_fallbacks !== null;
    if (!hasOrder && !hasFallbacks) {
      onChange({ ...value, routing_hints: undefined });
      return;
    }
    onChange({ ...value, routing_hints: next });
  };

  return (
    <fieldset
      className="space-y-3 rounded border border-slate-200 bg-slate-50 p-3"
      aria-labelledby={`${providerId}-legend`}
      disabled={disabled}
    >
      <legend
        id={`${providerId}-legend`}
        className="text-xs font-medium uppercase tracking-wide text-slate-500"
      >
        Model
      </legend>

      <PricingTierHelp />

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="flex flex-col gap-1">
          <label
            htmlFor={providerId}
            className="text-xs font-medium text-slate-700"
          >
            Provider
          </label>
          <select
            id={providerId}
            value={value.provider}
            onChange={(event) => {
              onChange({ ...value, provider: event.target.value });
            }}
            className="rounded border border-slate-300 bg-white px-2 py-1 text-sm"
          >
            <option value="">(default)</option>
            {llmProviders.map((entry) => (
              <option key={entry.name} value={entry.name}>
                {entry.name}
                {entry.supports_upstream_routing ? " (routing)" : ""}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label
            htmlFor={modelId}
            className="text-xs font-medium text-slate-700"
          >
            Model slug
          </label>
          <input
            id={modelId}
            type="text"
            value={value.model}
            onChange={(event) => {
              onChange({ ...value, model: event.target.value });
            }}
            placeholder="(provider default)"
            maxLength={128}
            className="rounded border border-slate-300 bg-white px-2 py-1 font-mono text-sm"
            autoComplete="off"
            spellCheck={false}
          />
        </div>
      </div>

      {supportsRouting ? (
        <div className="space-y-3 rounded border border-slate-300 bg-white p-3">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            OpenRouter upstream routing
          </p>
          <div className="flex flex-col gap-1">
            <label
              htmlFor={orderId}
              className="text-xs font-medium text-slate-700"
            >
              Preferred upstream order
            </label>
            <input
              id={orderId}
              type="text"
              value={orderText}
              onChange={(event) => {
                const parsed = parseUpstreamOrder(event.target.value);
                updateHints({ order: parsed });
              }}
              placeholder="e.g. anthropic, openai"
              maxLength={512}
              autoComplete="off"
              spellCheck={false}
              aria-invalid={orderInvalid}
              aria-describedby={orderInvalid ? `${orderId}-err` : undefined}
              className={
                "rounded border bg-white px-2 py-1 font-mono text-sm " +
                (orderInvalid ? "border-red-400" : "border-slate-300")
              }
            />
            {orderInvalid ? (
              <p id={`${orderId}-err`} className="text-xs text-red-700">
                Upstream slugs must be lower-case alphanumerics with optional
                hyphens (matches the backend schema).
              </p>
            ) : (
              <p className="text-xs text-slate-500">
                Comma-separated list. First match wins; OpenRouter falls back
                in order unless &ldquo;Fallbacks&rdquo; is set to <em>No</em>.
              </p>
            )}
          </div>

          <div className="flex flex-col gap-1">
            <label
              htmlFor={fallbacksId}
              className="text-xs font-medium text-slate-700"
            >
              Allow fallbacks
            </label>
            <select
              id={fallbacksId}
              value={fallbacksValue}
              onChange={(event) => {
                const choice = event.target.value;
                if (choice === "default") {
                  updateHints({ allow_fallbacks: null });
                } else {
                  updateHints({ allow_fallbacks: choice === "true" });
                }
              }}
              className="rounded border border-slate-300 bg-white px-2 py-1 text-sm"
            >
              <option value="default">OpenRouter default</option>
              <option value="true">Yes (allow fallbacks)</option>
              <option value="false">No (pinned to order)</option>
            </select>
          </div>
        </div>
      ) : null}
    </fieldset>
  );
}
