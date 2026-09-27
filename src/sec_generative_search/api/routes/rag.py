"""RAG query-understanding and generation routes.

Three endpoints live here:

        - ``POST /api/rag/plan`` wraps
            :func:`sec_generative_search.rag.query_understanding.understand_query`
            behind the request-scoped resolver chain and returns an editable
            :class:`QueryPlanSchema`.
        - ``POST /api/rag/query`` runs
            :meth:`RAGOrchestrator.generate` against an approved plan and
            returns the generated answer with citations and traceability
            (non-streaming).
        - ``POST /api/rag/stream`` shares the same orchestrator entry point
            but consumes :meth:`RAGOrchestrator.generate_stream` and emits
            Server-Sent Events (``delta`` / ``citation`` / ``final`` /
            ``error`` event types) plus a 15s ``heartbeat`` on idle
            inter-event gaps.

Why two routes (plan vs. generate):

        - The UI lets the user inspect and edit the plan before the
            orchestrator runs. Splitting plan / generate makes that edit point
            a hard contract: generation accepts a plan, never a raw query, so
            the human-in-the-loop step is enforced by the API shape rather
            than convention.
        - Generation is high-latency; understanding is short and cheap. A
            separate route lets the UI render chips while the user reviews
            filters, and only spend the generation budget on caller approval.

Security contract (shared by both routes):

        - Query / plan travels in the body, never the URL. Audit-log carries
            metadata only; the raw query never lands in any non-redacted log
            line.
        - Both routes are read-tier (``Depends(verify_api_key)``). Generation
            is a read-only side effect on the upstream LLM (no server-state
            mutation) and every authenticated tenant must be able to run it.
        - Rate-limited under the ``rag`` category by
            :class:`RateLimitMiddleware._classify_path`.
        - LLM construction goes through
            :func:`request_scoped_resolver` → :func:`build_llm_provider`; a
            missing key is reported as 400 ``provider_key_required`` so the
            caller knows to register one rather than treating it as a server
            fault.
        - :class:`ProviderAuthError` → 401 ``provider_unauthorized``
            (caller's own key is invalid; not a server fault).
        - :class:`ProviderRateLimitError` / :class:`ProviderTimeoutError` /
            :class:`ProviderConnectionError` → 503 ``provider_unavailable``
            (transient — the ``ProviderConnectionError`` case is chiefly an
            unreachable self-hosted ``local_llm`` endpoint).
        - Every other :class:`ProviderError` → 502 ``provider_error``.
        - The unified envelope's ``message`` / ``hint`` never echoes the raw
            query, the plan body, or the provider key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, NoReturn

from fastapi import APIRouter, Depends, Request
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from sec_generative_search.api.dependencies import (
    get_retrieval_service,
    request_scoped_resolver,
    verify_api_key,
)
from sec_generative_search.api.errors import envelope, http_error
from sec_generative_search.api.schemas import (
    CitationSchema,
    ConversationTurnSchema,
    OpenRouterRoutingHintsSchema,
    QueryPlanSchema,
    RagPlanRequest,
    RagPlanResponse,
    RagQueryRequest,
    RagQueryResponse,
    TokenUsageSchema,
)
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    GenerationError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.core.types import (
    Citation,
    ConversationTurn,
    GenerationResult,
    ProviderCapability,
    TokenUsage,
    estimate_cost,
)
from sec_generative_search.providers.factory import build_llm_provider
from sec_generative_search.providers.openrouter import OpenRouterRoutingHints
from sec_generative_search.providers.registry import (
    ProviderRegistry,
    ProviderSurface,
)
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.orchestrator import RAGOrchestrator, StreamEvent
from sec_generative_search.rag.query_understanding import (
    QueryPlan,
    understand_query,
)

if TYPE_CHECKING:
    from sec_generative_search.providers.base import BaseLLMProvider
    from sec_generative_search.search import RetrievalService


# ---------------------------------------------------------------------------
# SSE constants
# ---------------------------------------------------------------------------


# Inter-event idle window after which the route emits a ``heartbeat``
# event. 15s is a balance between (a) keeping the connection visibly
# alive through proxies that sever idle TCP and (b) not flooding
# the network with no-op frames during normal generation. Override is
# intentionally omitted — every operator wants the same semantics here.
_SSE_HEARTBEAT_SECONDS: float = 15.0


# Producer-thread sentinel pushed onto the queue when the orchestrator
# generator finishes normally. ``object()`` (not a string / int) so it
# cannot accidentally collide with a payload value.
_SSE_DONE_SENTINEL = object()

__all__ = ["router"]


logger = get_logger(__name__)


router = APIRouter(dependencies=[Depends(verify_api_key)])


def _client_ip(request: Request) -> str:
    """Best-effort client IP for audit-log lines."""
    return request.client.host if request.client else "unknown"


def _resolve_provider_and_model(body: RagPlanRequest) -> tuple[str, str]:
    """Pick the provider / model for this call.

    Body fields override settings; settings defaults apply when both are
    omitted.  Returning a concrete model string (possibly empty) keeps
    the audit log uniform — the LLM provider's own default fills in the
    blank at call time when it stays empty.
    """
    settings = get_settings()
    provider = body.provider or settings.llm.default_provider
    model = body.model or (settings.llm.default_model or "")
    return provider, model


def _plan_to_schema(plan: QueryPlan) -> QueryPlanSchema:
    """Lift the dataclass plan onto the wire schema.

    Explicit field-by-field copy — no ``**asdict()`` splat — so a future
    field addition on :class:`QueryPlan` does not silently leak onto the
    API.  ``date_range`` is a tuple in the dataclass and a list on the
    wire; the conversion happens here.
    """
    return QueryPlanSchema(
        raw_query=plan.raw_query,
        detected_language=plan.detected_language,
        query_en=plan.query_en,
        tickers=list(plan.tickers),
        form_types=list(plan.form_types),
        date_range=list(plan.date_range) if plan.date_range is not None else None,
        intent=plan.intent,
        suggested_answer_mode=plan.suggested_answer_mode.value,
    )


# ---------------------------------------------------------------------------
# Shared provider-error ladder + capability probe (used by all three RAG
# surfaces: /plan, /query, /stream)
# ---------------------------------------------------------------------------
#
# The three routes must classify an upstream failure identically — a caller
# that hits ``/query`` and falls back to ``/stream`` for the same key must
# see the same ``error`` code, status, and wording.  Centralising the table
# removes drift from the earlier inlined copies.  The HTTP raise path
# (:func:`_raise_provider_http_error`) and the SSE payload builder
# (:func:`_classify_stream_exception`) both consume this one table.


@dataclass(frozen=True)
class _ProviderErrorSpec:
    """One rung of the provider/generation exception ladder.

    ``message`` may carry a single placeholder whose wording differs across
    surfaces.  ``include_details`` gates the ``{provider, kind}``
    ``details`` mapping onto the HTTP envelope only (the SSE ``error``
    event carries no ``details`` field).  ``log_level`` is ``None`` for the
    terminal auth rung (which the routes deliberately do not log) and a
    :mod:`logging` level otherwise; every emitted line carries only the
    provider slug and exception class name.
    """

    status_code: int
    error: str
    message: str
    hint: str
    log_level: int | None
    log_label: str
    include_details: bool = False


# Ordered ladder — first ``isinstance`` match wins.  Order is load-bearing:
# the transient subclasses (rate-limit / timeout / connection) and the
# terminal ``ProviderAuthError`` MUST precede their common ``ProviderError``
# base, or every subclass would collapse onto the generic 502 rung.
# ``GenerationError`` is a sibling of ``ProviderError`` (not a subclass), so
# it is matched on its own; only ``/query`` + ``/stream`` route it here
# (``understand_query`` on ``/plan`` never raises it).
_PROVIDER_ERROR_LADDER: tuple[
    tuple[type[Exception] | tuple[type[Exception], ...], _ProviderErrorSpec], ...
] = (
    (
        ProviderAuthError,
        _ProviderErrorSpec(
            status_code=401,
            error="provider_unauthorized",
            message="The upstream provider rejected the supplied API key.",
            hint="Verify or rotate the provider key (header / session / admin env).",
            log_level=None,
            log_label="",
        ),
    ),
    (
        (ProviderRateLimitError, ProviderTimeoutError),
        _ProviderErrorSpec(
            status_code=503,
            error="provider_unavailable",
            message="The upstream provider is rate-limited or timed out.",
            hint="Retry after a short backoff; do not rotate the key.",
            log_level=logging.WARNING,
            log_label="upstream transient",
        ),
    ),
    (
        ProviderConnectionError,
        _ProviderErrorSpec(
            status_code=503,
            error="provider_unavailable",
            message="The upstream provider endpoint could not be reached.",
            hint=(
                "Verify the endpoint is running and reachable (for local_llm, "
                "that the local model server is up); retry once it recovers."
            ),
            log_level=logging.WARNING,
            log_label="endpoint unreachable",
        ),
    ),
    (
        ProviderError,
        _ProviderErrorSpec(
            status_code=502,
            error="provider_error",
            message="The upstream provider returned an error during {phase}.",
            hint="Inspect the audit log; do not rotate the key on a non-auth error.",
            log_level=logging.ERROR,
            log_label="provider error",
            include_details=True,
        ),
    ),
    (
        GenerationError,
        _ProviderErrorSpec(
            status_code=502,
            error="generation_error",
            message="The orchestrator could not assemble a valid answer.",
            hint="Retry the request; if the failure persists, switch model or provider.",
            log_level=logging.ERROR,
            log_label="generation error",
        ),
    ),
)


def _match_provider_error(exc: Exception) -> _ProviderErrorSpec | None:
    """Return the ladder rung for ``exc``, or ``None`` if it is off-ladder.

    ``None`` lets each consumer pick its own fallback: the HTTP surfaces
    re-raise (the global handler maps it to a generic 500), while the SSE
    builder emits an ``internal_error`` event.
    """
    for types, spec in _PROVIDER_ERROR_LADDER:
        if isinstance(exc, types):
            return spec
    return None


def _probe_capability(provider_name: str, model: str) -> ProviderCapability:
    """Resolve the LLM capability for ``provider_name`` / ``model``, fail-fast.

    Shared by all three RAG surfaces so an unknown provider or model
    surfaces the same 400 envelope everywhere.  The probe is O(1),
    credential-free and network-free (rule **R**) — a registry lookup
    only, done before any LLM is built so a bad slug does not surface as
    an obscure ``KeyError`` from the construction call site.  ``KeyError``
    → unknown provider slug; ``ValueError`` → unknown *embedding* slug,
    which on the LLM surface would be a registry bug but is surfaced as
    400 (not 500) so a future registry edit fails caller-visibly.
    """
    try:
        return ProviderRegistry.get_capability(
            provider_name,
            ProviderSurface.LLM,
            model=model or None,
        )
    except KeyError as exc:
        raise http_error(
            status_code=400,
            error="unknown_provider",
            message=str(exc),
            hint="Use ProviderRegistry.list_providers(LLM) to see registered slugs.",
        ) from None
    except ValueError as exc:
        raise http_error(
            status_code=400,
            error="unknown_model",
            message=str(exc),
            hint="Omit the model field to use the provider default.",
        ) from None


def _raise_provider_http_error(
    exc: Exception,
    *,
    provider_name: str,
    surface: str,
    phase: str,
) -> NoReturn:
    """Map a provider/generation exception onto the shared HTTP envelope.

    Single source of truth for the ``/plan`` + ``/query`` error ladder;
    :func:`_classify_stream_exception` consumes the same table for the
    SSE surface.  ``surface`` (``"plan"`` / ``"query"``) tags the log line.
    An off-ladder exception re-raises unchanged so it reaches the global
    handler exactly as it did before the ladder was consolidated — callers
    only route ladder types here, so this branch is defensive.
    """
    spec = _match_provider_error(exc)
    if spec is None:
        raise exc
    if spec.log_level is not None:
        logger.log(
            spec.log_level,
            "rag %s %s: provider=%s kind=%s",
            surface,
            spec.log_label,
            provider_name,
            type(exc).__name__,
        )
    details = (
        {"provider": provider_name, "kind": type(exc).__name__} if spec.include_details else None
    )
    raise http_error(
        status_code=spec.status_code,
        error=spec.error,
        message=spec.message.format(phase=phase),
        details=details,
        hint=spec.hint,
    ) from exc


@router.post(
    "/plan",
    response_model=RagPlanResponse,
    tags=["rag"],
    summary="Run query-understanding and return an editable plan",
)
def plan_query(
    request: Request,
    body: RagPlanRequest,
) -> RagPlanResponse:
    """Run :func:`understand_query` and return the resulting plan.

    The route is a thin adapter — every business rule (multilingual
    handling, JSON-schema vs free-form fallback, parse-failure → minimal
    plan) lives in
    :func:`sec_generative_search.rag.query_understanding.understand_query`.
    Keeping the route this thin makes the resolver-chain seam explicit:
    a future change to credential resolution does not need to touch
    routes.
    """
    provider_name, model = _resolve_provider_and_model(body)

    # Reject unknown providers up front with a 400 — building the LLM
    # otherwise raises an obscure ``KeyError`` from the registry on the
    # same call site.  The capability probe is O(1) and credential-free.
    capability = _probe_capability(provider_name, model)

    # Build the LLM via the request-scoped resolver chain (header →
    # session → encrypted-user → admin-env).  ``ConfigurationError``
    # means no key resolved through any tier — surface as 400 so the
    # caller registers one rather than treating the absence as a server
    # fault.
    resolver = request_scoped_resolver(request)
    try:
        llm = build_llm_provider(provider_name, api_key_resolver=resolver)
    except ConfigurationError as exc:
        raise http_error(
            status_code=400,
            error="provider_key_required",
            message=f"No API key resolved for provider '{provider_name}'.",
            hint=(
                "Supply the key via the X-Provider-Key-{provider} header, "
                "register it in the active session, or configure the "
                "admin-env fallback for the named provider."
            ),
        ) from exc
    except KeyError as exc:
        # Optional-extras gating in ``ProviderRegistry.get_class``.
        raise http_error(
            status_code=400,
            error="provider_unavailable",
            message=str(exc),
            hint="Install the provider's optional extras and restart the API.",
        ) from None

    # Close the per-request SDK client on every exit path (success,
    # provider error, or an unexpected raise) so its httpx connection pool
    # and the credential-bearing transport are torn down deterministically
    # rather than at GC.
    try:
        try:
            plan = understand_query(
                body.query,
                llm=llm,
                model=model,
                structured_output_supported=capability.structured_output,
            )
        except ProviderError as exc:
            # Terminal auth (401), transient (503) and generic-provider (502)
            # failures collapse onto the shared ladder.  The auth rung is
            # distinct from the route-level 401 ``unauthorised`` (invalid
            # ``X-API-Key``).
            _raise_provider_http_error(
                exc,
                provider_name=provider_name,
                surface="plan",
                phase="query understanding",
            )

        # Audit-log MUST NOT carry the raw query — the plan IS the query
        # reformulated, so the response body has it; logs are a different
        # surface and stay metadata-only.  The query is logged via
        # ``redact_for_log`` only at debug level (inside ``understand_query``).
        audit_log(
            "rag_plan",
            client_ip=_client_ip(request),
            endpoint="POST /api/rag/plan",
            detail=(
                f"provider={provider_name} model={model or '<provider default>'} "
                f"lang={plan.detected_language} tickers={len(plan.tickers)} "
                f"forms={len(plan.form_types)} mode={plan.suggested_answer_mode.value}"
            ),
        )

        return RagPlanResponse(
            plan=_plan_to_schema(plan),
            provider=provider_name,
            model=model,
        )
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# POST /api/rag/query — non-streaming generation
# ---------------------------------------------------------------------------


def _plan_from_schema(schema: QueryPlanSchema) -> QueryPlan:
    """Lift the wire schema back into the dataclass.

    Inverse of :func:`_plan_to_schema` — explicit field-by-field copy
    (no ``**dict()`` splat) so a future field on either side is a hard
    compile-time / runtime mismatch rather than a silent leak in either
    direction.  ``date_range`` becomes a tuple again; the wire schema
    bounds the list shape so we don't need to defend against an
    arbitrary length here.

    The orchestrator and citation extractor downstream re-validate every
    field they care about; this helper trusts the Pydantic shape and
    keeps the lift minimal.
    """
    date_range: tuple[str, str] | None
    if schema.date_range is None:
        date_range = None
    else:
        # The Pydantic schema does not bound the list length directly;
        # we defensively check here and reject any other shape because
        # ``QueryPlan.date_range`` is typed as a 2-tuple.
        if len(schema.date_range) != 2:
            raise http_error(
                status_code=400,
                error="invalid_plan",
                message="plan.date_range must contain exactly two ISO dates.",
                hint="Send [start, end] (YYYY-MM-DD) or null.",
            )
        date_range = (schema.date_range[0], schema.date_range[1])
    return QueryPlan(
        raw_query=schema.raw_query,
        detected_language=schema.detected_language,
        query_en=schema.query_en,
        tickers=list(schema.tickers),
        form_types=list(schema.form_types),
        date_range=date_range,
        intent=schema.intent,
        suggested_answer_mode=AnswerMode.from_string(
            schema.suggested_answer_mode,
            default=AnswerMode.CONCISE,
        ),
    )


def _history_from_schema(
    turns: list[ConversationTurnSchema],
) -> list[ConversationTurn] | None:
    """Lift wire-tier chat history into the orchestrator's dataclass shape.

    Only ``query`` and ``answer`` survive into a synthesised
    :class:`ConversationTurn`.  The dataclass carries ``retrieval_results``
    and a full :class:`GenerationResult`, but
    :meth:`RAGOrchestrator._render_history` only ever reads
    :attr:`ConversationTurn.query` and
    :attr:`ConversationTurn.generation_result.answer` to render ``Q:/A:``
    pairs — so we splice in a minimal ``GenerationResult`` shell carrying
    just the answer string and leave retrieval results empty. Prior
    turns' ``retrieval_results`` must never re-enter a future prompt; every
    follow-up turn re-retrieves.

    ``provider`` / ``model`` / ``prompt_version`` on the synthesised
    result are deliberately ``""`` — the route does not trust the browser
    to supply meaningful trace metadata for prior turns, and the
    orchestrator does not read those fields off history entries.  The
    timestamp uses the request-handling wall clock; it is opaque to the
    orchestrator and never reaches the prompt.

    Returns ``None`` for an empty list so the orchestrator's existing
    ``history is None`` short-circuit applies.
    """
    if not turns:
        return None
    now = datetime.now(UTC)
    return [
        ConversationTurn(
            query=turn.query,
            retrieval_results=[],
            generation_result=GenerationResult(
                answer=turn.answer,
                provider="",
                model="",
                prompt_version="",
                citations=[],
                retrieved_chunks=[],
                token_usage=TokenUsage(),
                latency_seconds=0.0,
                streamed=False,
            ),
            timestamp=now,
        )
        for turn in turns
    ]


def _routing_hints_from_schema(
    schema: OpenRouterRoutingHintsSchema | None,
) -> OpenRouterRoutingHints | None:
    """Lift the wire schema into the frozen
    :class:`OpenRouterRoutingHints` dataclass.

    The dataclass is tuple-valued for hashability; the schema's lists are
    coerced here.  ``None`` propagates so the orchestrator's
    ``routing_hints is None`` branch stays the common path.
    """
    if schema is None:
        return None
    return OpenRouterRoutingHints(
        order=tuple(schema.order),
        allow_fallbacks=schema.allow_fallbacks,
        only=tuple(schema.only),
        ignore=tuple(schema.ignore),
        require_parameters=schema.require_parameters,
        data_collection=schema.data_collection,
    )


def _resolve_routing_hints_api(
    provider_name: str,
    schema: OpenRouterRoutingHintsSchema | None,
) -> OpenRouterRoutingHints | None:
    """Build :class:`OpenRouterRoutingHints` for an API call, fail-closed.

    Mirrors :func:`sec_generative_search.cli.rag._resolve_routing_hints`
    byte-for-byte on the guard semantics: the orchestrator forwards
    ``routing_hints`` verbatim into :class:`GenerationRequest`, but only
    :class:`OpenRouterProvider` consumes it.  Supplying hints against any
    other provider would silently no-op (the OpenAI-compatible base's
    empty ``_extra_request_kwargs`` default drops them) — a misleading
    UX.  We fail closed with a 400 ``invalid_flag_combination`` instead
    so the caller knows the hint did not take effect.

    The :meth:`ProviderRegistry.supports_upstream_routing` query is the
    single source of truth — currently only the ``openrouter`` LLM entry
    advertises the capability; any future meta-provider that lights it
    up would pick up the hint flow automatically.
    """
    if schema is None:
        return None

    try:
        honours = ProviderRegistry.supports_upstream_routing(
            provider_name,
            ProviderSurface.LLM,
        )
    except KeyError:
        # Unknown provider — the caller's capability probe will already
        # have raised an envelope, but be defensive here in case this
        # helper is ever invoked before that probe.
        honours = False

    if not honours:
        raise http_error(
            status_code=400,
            error="invalid_flag_combination",
            message=(
                f"routing_hints supplied but provider '{provider_name}' "
                "does not honour upstream-routing hints."
            ),
            hint=(
                "Set provider='openrouter' (the only provider that "
                "consumes these hints) or omit routing_hints."
            ),
        )

    return _routing_hints_from_schema(schema)


def _format_routing_hints_audit(hints: OpenRouterRoutingHints | None) -> str:
    """Render hints as an audit-safe metadata fragment.

    Mirrors :func:`sec_generative_search.cli.rag._format_routing_hints_audit`.
    Reports counts and toggles only — upstream-provider slugs are
    deliberately omitted so operators can pivot on "was routing pinned?"
    without leaking which vendors a given run preferred (operator-
    supplied but still a Tier-2 disclosure telemetry rarely needs).
    """
    if hints is None:
        return "none"
    fallbacks = hints.allow_fallbacks if hints.allow_fallbacks is not None else "default"
    return f"order={len(hints.order)} fallbacks={fallbacks}"


def _citation_to_schema(citation: Citation) -> CitationSchema:
    """Lift one :class:`Citation` onto the wire schema.

    Flattens :attr:`Citation.filing_id` (a frozen :class:`FilingIdentifier`)
    into the four scalar fields the wire surface carries — keeping the
    JSON flat and saving the UI a nested traversal.  ``filing_date`` is a
    :class:`datetime.date` on the dataclass and renders as ISO on the
    wire.
    """
    filing_id = citation.filing_id
    return CitationSchema(
        chunk_id=citation.chunk_id,
        ticker=filing_id.ticker,
        form_type=filing_id.form_type,
        filing_date=filing_id.filing_date.isoformat(),
        accession_number=filing_id.accession_number,
        section_path=citation.section_path,
        text_span=citation.text_span,
        similarity=citation.similarity,
        display_index=citation.display_index,
    )


def _result_to_response(
    result: GenerationResult,
    *,
    refused: bool,
    capability: ProviderCapability,
) -> RagQueryResponse:
    """Lift the orchestrator's :class:`GenerationResult` onto the wire.

    Explicit allow-list lift (no ``**asdict()`` splat) so a future field
    addition on :class:`GenerationResult` does not silently leak onto
    the API.  ``retrieved_chunks`` is intentionally dropped — citations
    are the user-visible audit trail; operators wanting the full
    candidate set use ``POST /api/search`` against the same retrieval
    primitive.

    ``estimated_cost_usd`` is derived from the call's token usage and the
    resolved model's exact per-MTok cost via the canonical
    :func:`~sec_generative_search.core.types.estimate_cost` helper —
    ``None`` when the model's cost is unknown (arbitrary-slug providers /
    overlay-only models).  The figure rides the response body only; it is
    never recorded to a metric or log line.
    """
    usage = result.token_usage
    return RagQueryResponse(
        answer=result.answer,
        citations=[_citation_to_schema(c) for c in result.citations],
        provider=result.provider,
        model=result.model,
        prompt_version=result.prompt_version,
        token_usage=TokenUsageSchema(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        ),
        estimated_cost_usd=estimate_cost(usage, capability),
        latency_seconds=result.latency_seconds,
        streamed=result.streamed,
        refused=refused,
    )


def _build_llm_for_request(
    request: Request,
    provider_name: str,
) -> BaseLLMProvider:
    """Construct the user's LLM via the request-scoped resolver chain.

    Centralised here so the same error-mapping table covers both the
    plan and query routes — a divergence between the two surfaces would
    confuse callers who hit ``provider_key_required`` from one but
    ``unknown_provider`` from the other for the same misconfiguration.
    """
    resolver = request_scoped_resolver(request)
    try:
        return build_llm_provider(provider_name, api_key_resolver=resolver)
    except ConfigurationError as exc:
        raise http_error(
            status_code=400,
            error="provider_key_required",
            message=f"No API key resolved for provider '{provider_name}'.",
            hint=(
                "Supply the key via the X-Provider-Key-{provider} header, "
                "register it in the active session, or configure the "
                "admin-env fallback for the named provider."
            ),
        ) from exc
    except KeyError as exc:
        raise http_error(
            status_code=400,
            error="provider_unavailable",
            message=str(exc),
            hint="Install the provider's optional extras and restart the API.",
        ) from None


def _resolve_provider_and_model_query(body: RagQueryRequest) -> tuple[str, str]:
    """Pick provider / model for the generation route.

    Distinct from :func:`_resolve_provider_and_model` only in the body
    type it accepts — a single helper would pull both schemas into the
    same callsite and obscure which route is being served.
    """
    settings = get_settings()
    provider = body.provider or settings.llm.default_provider
    model = body.model or (settings.llm.default_model or "")
    return provider, model


@router.post(
    "/query",
    response_model=RagQueryResponse,
    tags=["rag"],
    summary="Generate an answer for an approved QueryPlan (non-streaming)",
)
def generate_answer(
    request: Request,
    body: RagQueryRequest,
    retrieval: RetrievalService = Depends(get_retrieval_service),
) -> RagQueryResponse:
    """Run :meth:`RAGOrchestrator.generate` against the supplied plan.

    Declared **sync** (``def``) so FastAPI dispatches it to the anyio
    threadpool.  :meth:`RAGOrchestrator.generate` is a blocking chain —
    local-embedder inference (torch), a ChromaDB query, then a blocking
    LLM SDK call with ``resilient_call`` retry/backoff (10-60 s under a
    transient upstream).  Running that on the single event loop of the
    single worker would freeze ``/api/health`` liveness probes
    (risking a probe-timeout container restart mid-generation), SSE
    heartbeats, and every other tenant's request.  Off-loading to the
    threadpool bounds unrelated-endpoint latency to normal single-request
    latency; the network waits release the GIL.  The per-request SDK
    client is still closed in the ``finally`` below, and the
    correlation-ID ContextVar is copied into the worker thread so the
    audit line still carries the bound ID.

    The route is a thin adapter — every business rule (context budgeting,
    comparative fan-out, prompt assembly with the
    ``<UNTRUSTED_FILING_CONTEXT>`` delimiter contract, hybrid citation
    extraction) lives in :class:`RAGOrchestrator`.  The route's job is
    to (1) wire credentials through the resolver chain, (2) translate
    the wire schema to / from the dataclass surface, and (3) collapse
    the full provider exception ladder into the structured envelope.

    Why we accept a plan and not a raw query:

        - The UI's editable-chips review is the human-in-the-loop step
          that catches misparsed tickers, hallucinated date ranges, and
          mode misclassification *before* the generation budget is
          spent.  Accepting only a plan means a future client cannot
          accidentally bypass that review.
        - The plan carries ``query_en``; retrieval embeds the English
          rendering even when the original query is in another
          language.  A raw-query route would have to re-run
          query-understanding to get there, doubling the LLM round-trip
          on the user-facing path.
    """
    provider_name, model = _resolve_provider_and_model_query(body)

    # Reject unknown providers up front with a 400 — same contract as
    # the plan route.  The capability probe is O(1) and credential-free.
    capability = _probe_capability(provider_name, model)

    plan = _plan_from_schema(body.plan)
    llm = _build_llm_for_request(request, provider_name)

    # Close the per-request SDK client on every exit path (success, the
    # routing-hint 400, provider/generation error, or any other raise) so
    # its httpx connection pool and credential-bearing transport are torn
    # down deterministically rather than at GC. The non-streaming route
    # owns the client for its whole lifetime, so a single ``finally`` here
    # is sufficient — unlike ``/stream`` where the producer thread outlives
    # the handler and owns the close.
    try:
        orchestrator = RAGOrchestrator(retrieval=retrieval, llm=llm)

        # ``body.mode`` is already pattern-validated at the schema layer
        # (one of the four AnswerMode values).  ``from_string`` is still
        # the canonical lift — handles the explicit ``None`` case (orchestrator
        # falls back to ``plan.suggested_answer_mode``) and stays consistent
        # with the orchestrator's own internal lifts.
        effective_mode = (
            AnswerMode.from_string(body.mode, default=plan.suggested_answer_mode)
            if body.mode is not None
            else None
        )

        history = _history_from_schema(body.history)
        # Validate routing-hint compatibility before opening the orchestrator.
        # ``_resolve_routing_hints_api`` raises HTTPException(400) for the
        # invalid-combination case so the caller sees a do-not-retry error
        # rather than a silently-no-op generation.
        routing_hints = _resolve_routing_hints_api(provider_name, body.routing_hints)

        try:
            result = orchestrator.generate(
                plan,
                mode=effective_mode,
                model=model or None,
                max_output_tokens=body.max_output_tokens,
                history=history,
                prefer_structured_output=capability.structured_output,
                max_per_section=body.max_per_section,
                max_per_filing=body.max_per_filing,
                rerank_over_fetch_factor=body.rerank_over_fetch_factor,
                routing_hints=routing_hints,
            )
        except (ProviderError, GenerationError) as exc:
            # The full provider ladder plus ``GenerationError`` (citation
            # parser / orchestrator hit an unrecoverable state, e.g. a
            # malformed structured-output payload the inline fallback cannot
            # rescue → 502; the upstream output, not the caller's input, is at
            # fault) collapse onto the shared classifier.
            _raise_provider_http_error(
                exc,
                provider_name=provider_name,
                surface="query",
                phase="generation",
            )

        refused = not result.retrieved_chunks and not result.citations
        audit_mode = (
            effective_mode.value if effective_mode is not None else plan.suggested_answer_mode.value
        )

        audit_log(
            "rag_query",
            client_ip=_client_ip(request),
            endpoint="POST /api/rag/query",
            detail=(
                f"provider={provider_name} model={model or '<provider default>'} "
                f"lang={plan.detected_language} tickers={len(plan.tickers)} "
                f"forms={len(plan.form_types)} mode={audit_mode} "
                f"prompt_version={result.prompt_version} "
                f"chunks={len(result.retrieved_chunks)} citations={len(result.citations)} "
                f"history_turns={len(body.history)} "
                f"or_hints={_format_routing_hints_audit(routing_hints)} "
                f"refused={refused}"
            ),
        )

        return _result_to_response(result, refused=refused, capability=capability)
    finally:
        llm.close()


# ---------------------------------------------------------------------------
# POST /api/rag/stream — Server-Sent Events
# ---------------------------------------------------------------------------


def _sse_format(event: str, data: dict) -> str:
    """Encode one Server-Sent-Events frame.

    Per RFC the frame is ``event: <name>\\ndata: <json>\\n\\n``.  We
    deliberately serialise ``data`` on a single line — ``json.dumps``
    without ``indent`` never emits a literal newline, and a stray
    newline in the payload would terminate the frame mid-stream.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _final_payload(
    result: GenerationResult,
    *,
    refused: bool,
    capability: ProviderCapability,
) -> dict:
    """Build the JSON payload for the terminal ``final`` event.

    The shape mirrors :class:`RagQueryResponse` minus ``citations`` —
    those land on the wire as separate ``citation`` events so the UI can
    render the source panel incrementally as soon as the deltas have
    finished arriving.  ``answer`` is repeated here even though deltas
    already streamed it: clients that drop a delta packet still get the
    fully-assembled answer in one place, and the orchestrator's
    citation-extraction step may have stripped a JSON envelope from the
    streamed deltas (in structured-output mode), in which case the
    final answer differs from the concatenated deltas.

    ``estimated_cost_usd`` mirrors the non-streaming surface — derived from
    the call's token usage and the resolved model's exact per-MTok cost via
    :func:`~sec_generative_search.core.types.estimate_cost`, ``None`` for an
    unknown-cost model.  Cost rides the event body only, never a metric or
    log line.
    """
    usage = result.token_usage
    return {
        "answer": result.answer,
        "provider": result.provider,
        "model": result.model,
        "prompt_version": result.prompt_version,
        "token_usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
        },
        "estimated_cost_usd": estimate_cost(usage, capability),
        "latency_seconds": result.latency_seconds,
        "streamed": True,
        "refused": refused,
    }


def _citation_payload(citation: Citation) -> dict:
    """Build the JSON payload for one ``citation`` event.

    Same flat shape as :class:`CitationSchema` so clients can re-use the
    same parser they wrote for ``POST /api/rag/query``.  ``filing_id``
    is flattened into four scalar fields and ``filing_date`` is rendered
    as ISO.
    """
    schema = _citation_to_schema(citation)
    return schema.model_dump()


def _error_payload(*, error: str, message: str, hint: str | None = None) -> dict:
    """Build the JSON payload for an SSE ``error`` event.

    Reuses :func:`envelope` so the SSE error shape matches the unified
    HTTP error envelope — clients that already key on ``error`` /
    ``message`` / ``hint`` for the JSON surface need no second parser.
    """
    return envelope(error=error, message=message, hint=hint)


def _classify_stream_exception(exc: Exception) -> dict:
    """Map an orchestrator-side exception to an SSE error payload.

    Consumes the same :data:`_PROVIDER_ERROR_LADDER` as the HTTP raise
    path (:func:`_raise_provider_http_error`) so a caller cannot get one
    shape from ``/query`` and a different one from ``/stream`` for the
    same upstream failure.  The SSE surface mirrors ``/query``.  An
    off-ladder exception yields the generic ``internal_error`` envelope
    without echoing the exception text (internal messages routinely contain
    file paths, SQL, etc.).
    """
    spec = _match_provider_error(exc)
    if spec is None:
        return _error_payload(
            error="internal_error",
            message="The server encountered an unexpected error while streaming.",
            hint="Retry the request; if the failure persists, contact the operator.",
        )
    return _error_payload(
        error=spec.error,
        message=spec.message.format(phase="generation"),
        hint=spec.hint,
    )


class _ClosingStreamingResponse(StreamingResponse):
    """A ``StreamingResponse`` that always closes its SSE body generator.

    On a client disconnect Starlette stops iterating the body, but when the
    disconnect lands while the body is paused at a ``yield`` (mid-``send``)
    the generator is left suspended.  Its ``finally`` — which signals the
    producer thread to stop (F13) and writes the ``rag_stream_completed``
    audit line — would then run only when the generator is garbage
    collected, and the exception traceback's reference cycle defers that
    indefinitely (measured: the producer ran 153 of 200 events after the
    client had gone).  Closing it here makes that teardown deterministic on
    every exit path; on a normal exit the generator is already exhausted and
    ``aclose()`` is a no-op.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                await aclose()


def _close_event_stream(events: Iterator[StreamEvent] | None) -> None:
    """Close the orchestrator's event generator early; never raises.

    A no-op on ``None`` (the call itself raised) or an exhausted generator.
    On an abandoned one it raises ``GeneratorExit`` at the paused ``yield``
    so the orchestrator's and the adapter's ``finally`` / ``with`` blocks run
    now, on the producer thread, rather than whenever the generator is
    collected.  A failure is logged by
    exception type only (it can carry provider text).
    """
    close = getattr(events, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception as exc:
        logger.warning("rag stream: closing the event stream failed: %s", type(exc).__name__)


def _run_orchestrator_in_thread(
    orchestrator: RAGOrchestrator,
    *,
    llm: BaseLLMProvider,
    plan: QueryPlan,
    mode: AnswerMode | None,
    model: str | None,
    max_output_tokens: int | None,
    history: list[ConversationTurn] | None,
    prefer_structured_output: bool,
    max_per_section: int | None,
    max_per_filing: int | None,
    rerank_over_fetch_factor: int | None,
    routing_hints: OpenRouterRoutingHints | None,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    stop: threading.Event,
) -> None:
    """Drive the synchronous orchestrator generator from a worker thread.

    Why a thread: :meth:`RAGOrchestrator.generate_stream` is a *sync*
    generator that blocks on the LLM SDK's network calls.  Iterating it
    directly from the asyncio event loop would block the entire process
    and starve the heartbeat coroutine.  A daemon thread pumps events
    onto an :class:`asyncio.Queue` via :meth:`call_soon_threadsafe`;
    the async consumer awaits the queue and emits frames downstream.

    Sentinel discipline:

        - On normal completion the producer pushes :data:`_SSE_DONE_SENTINEL`
          so the consumer knows to close the stream.
        - On exception the producer pushes the exception itself; the
          consumer translates it into an ``error`` event via
          :func:`_classify_stream_exception`.
        - Both cases run under ``finally`` so a ``return`` from inside
          the orchestrator (refusal short-circuit) still closes the
          consumer cleanly.

    Client lifetime: the producer thread owns *llm* — the SDK
    client is used only inside ``generate_stream`` — so it is closed in
    the producer's ``finally``, i.e. exactly when the stream is exhausted
    (or fails, or is abandoned), on the same thread that used it.  Closing
    here rather than in the route handler avoids a race where a client
    disconnect tears the async consumer down while the producer is
    mid-iteration and still holding the transport open.

    Cancellation (F13): the consumer sets *stop* whenever it exits.  The
    producer checks it between events; on a set flag it stops iterating,
    closes the orchestrator generator (so the provider's SDK stream is torn
    down on this thread) and then the client, so an abandoned stream stops
    generating — and billing — within one event instead of running the
    whole completion into a queue no one reads.  Nothing is pushed onto the
    queue once stopped: the consumer is gone.
    """

    def producer() -> None:
        events: Iterator[StreamEvent] | None = None
        try:
            events = orchestrator.generate_stream(
                plan,
                mode=mode,
                model=model,
                max_output_tokens=max_output_tokens,
                history=history,
                prefer_structured_output=prefer_structured_output,
                max_per_section=max_per_section,
                max_per_filing=max_per_filing,
                rerank_over_fetch_factor=rerank_over_fetch_factor,
                routing_hints=routing_hints,
            )
            for event in events:
                if stop.is_set():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            if not stop.is_set():
                loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            # Teardown order: the orchestrator generator first (its
            # ``finally`` blocks and the adapter's SDK stream close on this
            # thread), then the client, then the sentinel — so a
            # fully-consumed stream guarantees teardown has run.  ``close()``
            # is contractually quiet, but wrap it anyway: the done-sentinel
            # MUST be pushed even if a future close() regressed and raised —
            # otherwise the consumer awaits the queue forever and the stream
            # hangs.  A stopped consumer reads nothing, so it gets no sentinel.
            try:
                _close_event_stream(events)
                llm.close()
            finally:
                if not stop.is_set():
                    loop.call_soon_threadsafe(queue.put_nowait, _SSE_DONE_SENTINEL)

    threading.Thread(
        target=producer,
        name="rag-stream-producer",
        daemon=True,
    ).start()


@router.post(
    "/stream",
    tags=["rag"],
    summary="Stream an answer for an approved QueryPlan via Server-Sent Events",
    response_class=StreamingResponse,
)
async def stream_answer(
    request: Request,
    body: RagQueryRequest,
    retrieval: RetrievalService = Depends(get_retrieval_service),
) -> StreamingResponse:
    """Stream the orchestrator output as Server-Sent Events.

    This route stays sync so blocking retrieval and generation work runs in
    the threadpool.
          they were already streamed as their own events.
        - ``error`` — emitted in place of the trailing events when the
          orchestrator raises after the SSE response is open.  Payload
          matches the unified HTTP error envelope shape so the same
          parser handles both surfaces.
        - ``heartbeat`` — emitted every 15s of inter-event silence.
        # Audit logs stay metadata-only; the raw query is returned in the
        # response body.
    Pre-stream errors (unknown provider, missing key, ProviderAuthError
    raised during LLM construction) surface as the regular HTTP error
    envelope on a 4xx / 5xx status — the SSE response only opens when
    we are committed to streaming.  Errors during streaming arrive as
    ``error`` events instead so the client knows to stop reading deltas.

    Same auth, rate-limit, and resolver-chain semantics as
    :func:`generate_answer`; both routes are read-tier and both share
    :func:`_build_llm_for_request` so a misconfiguration cannot surface
    as ``provider_key_required`` from one and ``unknown_provider`` from
    the other.

    This handler stays async because the SSE body uses asyncio primitives
    and the blocking orchestrator already runs off-loop in
    :func:`_run_orchestrator_in_thread`.
    """
    provider_name, model = _resolve_provider_and_model_query(body)

    # Pre-stream validation — these MUST raise HTTP errors (not SSE
    # error events) because the SSE response is not yet open.  A
    # 4xx response here lets the client distinguish "your request was
    # bad" (don't retry) from "the upstream blew up mid-generation"
    # (maybe retry).
    capability = _probe_capability(provider_name, model)

    plan = _plan_from_schema(body.plan)
    llm = _build_llm_for_request(request, provider_name)
    # Pre-stream setup runs synchronously (no ``await``), so the only way
    # out before the producer thread starts is a raise — chiefly the
    # routing-hint 400.  On the success path the client MUST stay open (the
    # producer thread owns it and closes it when the stream ends, see
    # ``_run_orchestrator_in_thread``); a pre-stream failure means that
    # thread never starts, so close the client here. ``except`` — not
    # ``finally`` — because ``finally`` would close it on the success path
    # too, before a single delta is streamed.
    try:
        orchestrator = RAGOrchestrator(retrieval=retrieval, llm=llm)

        effective_mode = (
            AnswerMode.from_string(body.mode, default=plan.suggested_answer_mode)
            if body.mode is not None
            else None
        )
        audit_mode = (
            effective_mode.value if effective_mode is not None else plan.suggested_answer_mode.value
        )

        # Audit-log emission happens *before* the stream opens: SSE responses
        # cannot raise after the body starts, so logging at the start makes
        # the route's intent visible even if the connection drops mid-stream.
        # Counts of chunks / citations / refused are unknown at this point;
        # the producer thread emits a follow-up ``rag_stream_completed`` line
        # carrying those once the orchestrator finishes.
        history = _history_from_schema(body.history)
        # Refuse hints against any provider that does not advertise
        # ``supports_upstream_routing`` so a mis-paired hint never opens the
        # SSE stream.
        routing_hints = _resolve_routing_hints_api(provider_name, body.routing_hints)

        audit_log(
            "rag_stream",
            client_ip=_client_ip(request),
            endpoint="POST /api/rag/stream",
            detail=(
                f"provider={provider_name} model={model or '<provider default>'} "
                f"lang={plan.detected_language} tickers={len(plan.tickers)} "
                f"forms={len(plan.form_types)} mode={audit_mode} "
                f"history_turns={len(body.history)} "
                f"or_hints={_format_routing_hints_audit(routing_hints)}"
            ),
        )
    except Exception:
        llm.close()
        raise

    async def event_stream() -> AsyncIterator[str]:
        """Async generator that interleaves orchestrator events + heartbeats."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        client_ip = _client_ip(request)

        _run_orchestrator_in_thread(
            orchestrator,
            llm=llm,
            plan=plan,
            mode=effective_mode,
            model=model or None,
            max_output_tokens=body.max_output_tokens,
            history=history,
            prefer_structured_output=capability.structured_output,
            max_per_section=body.max_per_section,
            max_per_filing=body.max_per_filing,
            rerank_over_fetch_factor=body.rerank_over_fetch_factor,
            routing_hints=routing_hints,
            queue=queue,
            loop=loop,
            stop=stop,
        )

        chunks_streamed = 0
        citations_emitted = 0
        refused = False
        completion_status = "ok"
        # Set on the two exits where the stream ran to its end (done
        # sentinel, or an in-stream error event).  Any other exit is the
        # client going away: a ``CancelledError`` from Starlette's
        # disconnect listener, or a ``GeneratorExit`` when the paused body
        # is closed — which of the two depends on where the disconnect
        # lands, so neither is caught to detect it.
        finished = False

        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=_SSE_HEARTBEAT_SECONDS,
                    )
                except TimeoutError:
                    yield _sse_format("heartbeat", {})
                    continue

                if item is _SSE_DONE_SENTINEL:
                    finished = True
                    return
                if isinstance(item, Exception):
                    finished = True
                    completion_status = type(item).__name__
                    logger.error(
                        "rag stream exception: provider=%s kind=%s",
                        provider_name,
                        completion_status,
                    )
                    yield _sse_format("error", _classify_stream_exception(item))
                    return
                if not isinstance(item, StreamEvent):
                    # Defensive — the orchestrator contract says events
                    # are StreamEvent. Anything else is a bug we want
                    # logged, not silently dropped onto the wire.
                    logger.error(
                        "rag stream producer pushed unexpected payload: %s",
                        type(item).__name__,
                    )
                    continue

                if item.delta is not None:
                    chunks_streamed += 1
                    yield _sse_format("delta", {"text": item.delta})

                if item.final is not None:
                    final_result = item.final
                    refused = not final_result.retrieved_chunks and not final_result.citations
                    for citation in final_result.citations:
                        citations_emitted += 1
                        yield _sse_format("citation", _citation_payload(citation))
                    yield _sse_format(
                        "final",
                        _final_payload(
                            final_result,
                            refused=refused,
                            capability=capability,
                        ),
                    )
        finally:
            # Always signal the producer (harmless once it has finished):
            # on a disconnect it stops at its next event and tears the
            # provider stream + client down on its own thread (F13).
            stop.set()
            if not finished:
                completion_status = "client_disconnected"
            audit_log(
                "rag_stream_completed",
                client_ip=client_ip,
                endpoint="POST /api/rag/stream",
                detail=(
                    f"provider={provider_name} model={model or '<provider default>'} "
                    f"deltas={chunks_streamed} citations={citations_emitted} "
                    f"history_turns={len(body.history)} "
                    f"or_hints={_format_routing_hints_audit(routing_hints)} "
                    f"refused={refused} status={completion_status}"
                ),
            )

    # SSE-required headers:
    #
    #   - ``Cache-Control: no-cache`` keeps caching proxies from
    #     buffering the half-open stream.
    #   - ``X-Accel-Buffering: no`` is the nginx-specific directive that
    #     disables response buffering; without it, the heartbeat would
    #     be absorbed by nginx until it filled its buffer.
    #   - ``Connection: keep-alive`` is technically the HTTP/1.1
    #     default but stating it makes the intent explicit for any
    #     intermediary that downgrades.
    return _ClosingStreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
