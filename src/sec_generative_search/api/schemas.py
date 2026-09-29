"""Pydantic v2 request and response models for the API surface.

Discipline:

    - All models use ``ConfigDict(extra="forbid")`` so unexpected fields
      are rejected loudly rather than silently dropped.
    - Response models MUST NOT carry credential-shaped fields.  A
      :pytest.mark.security regression test asserts this for the
      module's exports.
    - Models meant to be returned to the client must have stable field
      names; rename via aliases, never break.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "AdminUserCreateRequest",
    "AdminUserCreateResponse",
    "AdminUserDeleteResponse",
    "AdminUserUnlockResponse",
    "BulkDeleteRequest",
    "BulkDeleteResponse",
    "CatalogueRefreshResponse",
    "CitationSchema",
    "ClearAllResponse",
    "ConversationTurnSchema",
    "DeleteByIdsRequest",
    "DeleteByIdsResponse",
    "DeleteResponse",
    "DemoResetResponse",
    "EdgarIdentityClearResponse",
    "EdgarIdentityRegisterResponse",
    "EdgarIdentityRequest",
    "EnrolmentCompleteRequest",
    "EnrolmentCompleteResponse",
    "FilingListResponse",
    "FilingSchema",
    "GpuStatusResponse",
    "HealthResponse",
    "IngestCancelResponse",
    "IngestRequest",
    "IngestResultSchema",
    "IngestTaskResponse",
    "LoginParamsResponse",
    "LoginRequest",
    "LoginResponse",
    "ModelPricingSchema",
    "OpenRouterRoutingHintsSchema",
    "PasswordChangeRequest",
    "PasswordChangeResponse",
    "ProviderInfoSchema",
    "ProviderListResponse",
    "ProviderModelsResponse",
    "ProviderValidateRequest",
    "ProviderValidateResponse",
    "QueryPlanSchema",
    "RagPlanRequest",
    "RagPlanResponse",
    "RagQueryRequest",
    "RagQueryResponse",
    "SearchHit",
    "SearchRequest",
    "SearchResponse",
    "SessionLogoutResponse",
    "SessionResponse",
    "StatusResponse",
    "TaskListResponse",
    "TaskProgressSchema",
    "TaskStatusResponse",
    "TokenUsageSchema",
    "VaultUpdateRequest",
    "VaultUpdateResponse",
]


class _BaseModel(BaseModel):
    """Project default — strict and case-sensitive."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(_BaseModel):
    """Liveness response — deliberately tiny.

    ``version`` is *not* surfaced here so an anonymous caller cannot
    fingerprint the deployment.  Authenticated callers obtain it via
    :class:`StatusResponse`.
    """

    status: str = Field(
        default="ok",
        description="Liveness marker; always 'ok' when the API is reachable.",
    )


class StatusResponse(_BaseModel):
    """Authenticated status snapshot."""

    version: str
    deployment_profile: str
    embedding_provider: str
    embedding_model: str
    storage_filings: int
    is_admin: bool
    persist_provider_credentials: bool


class SessionResponse(_BaseModel):
    """Result of a successful ``POST /api/session``.

    The ``session_id`` itself is NEVER returned in the body — it lives
    exclusively in the ``Set-Cookie`` header so it cannot be read from
    JavaScript (the cookie is HTTP-only).  This response carries the
    operator-visible metadata only.
    """

    issued: bool
    cookie_name: str = Field(
        description="Name of the HTTP-only cookie that holds the session_id.",
    )
    expires_in_seconds: int = Field(
        description="Sliding TTL of the session as configured on the server.",
    )


class SessionLogoutResponse(_BaseModel):
    """Result of a successful ``POST /api/session/logout``."""

    cleared_credentials: int = Field(
        description="Number of provider credentials dropped from the session store.",
    )
    cleared_edgar_identity: bool = Field(
        default=False,
        description=("Whether a per-session EDGAR identity was cleared as part of logout."),
    )


class EdgarIdentityRequest(_BaseModel):
    """Body for ``POST /api/session/edgar``.

    Both fields are validated server-side before storage; the response
    NEVER echoes the values back.  The route requires an active
    server-minted ``session_id`` cookie — without one there is no scope
    in which to register the identity.
    """

    name: str = Field(
        min_length=1,
        max_length=128,
        description="Acting user's full name (sent to SEC as part of the User-Agent).",
    )
    email: str = Field(
        min_length=3,
        max_length=254,
        description="Acting user's email address (sent to SEC as part of the User-Agent).",
    )


class EdgarIdentityRegisterResponse(_BaseModel):
    """Result of a successful ``POST /api/session/edgar``.

    Body is deliberately minimal — neither name nor email is echoed back.
    """

    registered: bool


class EdgarIdentityClearResponse(_BaseModel):
    """Result of a successful ``DELETE /api/session/edgar``."""

    cleared: bool


class ProviderInfoSchema(_BaseModel):
    """One registered provider as exposed to the API.

    Deliberately narrow: only the three fields the web Provider Settings
    page needs to render the live catalogue without drift from the
    backend ``ProviderRegistry``.  No API keys, no masked tails, no
    ``requires_extras`` hint, no model catalogue — those are either
    out-of-scope secrets or catalogue details that do not belong on the
    wire. Widening this schema is a security-sensitive change and must
    be reviewed deliberately.

    ``surface`` is the lower-case :class:`ProviderSurface` value
    (``"llm"`` / ``"embedding"`` / ``"reranker"``); the route never lifts
    the enum object onto the wire so a future StrEnum value rename is
    caught as a schema mismatch rather than silently surfacing a Python
    ``repr``.
    """

    name: str
    surface: str
    supports_upstream_routing: bool


class ProviderListResponse(_BaseModel):
    """Result of ``GET /api/providers/``.

    Lifts the curated :class:`ProviderRegistry` tuple onto the wire.
    ``providers`` preserves the registry's curated order (popular vendors
    first within each surface) so the UI can render it verbatim. Only
    entries whose optional extras are importable in the running
    interpreter are surfaced — the web UI cannot meaningfully render an
    "install hint" today, and exposing un-installed entries would invite
    callers to attempt validations that would 400 at the very next step.

    ``total`` is the count of returned rows, not a registry cardinality —
    the same convention as :class:`FilingListResponse` and
    :class:`SearchResponse`.
    """

    providers: list[ProviderInfoSchema]
    total: int


class ModelPricingSchema(_BaseModel):
    """One catalogued LLM model with its coarse tier and exact per-MTok cost.

    Deliberately narrow: the model slug, its
    :class:`~sec_generative_search.core.types.PricingTier` value, and the
    exact input / output per-million-token cost — the single source of truth
    the web UI needs to surface a "cheap vs. premium" hint *and* a
    per-request cost estimate without re-deriving anything from a second
    table. Still no capability matrix (context window, tool-use, …): that
    remains out of scope. Widening this schema further is a
    security-sensitive change.

    The two cost fields are the deliberate later widening the original
    (tier-only) schema anticipated. The figures are public vendor pricing
    already shipped in the vendored catalogue, never a secret; the pricing
    tier remains derived from cost, so the two fields never contradict each
    other.

    ``pricing_tier`` is the lower-case :class:`PricingTier` value
    (``"free"`` / ``"low"`` / ``"standard"`` / ``"high"`` / ``"premium"``
    / ``"unknown"``); the route lifts the enum *value*, never the enum
    object, so a future rename is caught as a schema mismatch rather than
    silently surfacing a Python ``repr``.

    ``input_cost_per_mtok`` / ``output_cost_per_mtok`` are USD per one
    million tokens, or ``None`` when unknown (arbitrary-slug providers and
    overlay-only models) — exactly the rows whose tier is ``"unknown"``.
    """

    model: str
    pricing_tier: str
    input_cost_per_mtok: float | None = Field(
        default=None,
        ge=0.0,
        description="Exact prompt-token price, USD per 1M tokens; None when unknown.",
    )
    output_cost_per_mtok: float | None = Field(
        default=None,
        ge=0.0,
        description="Exact completion-token price, USD per 1M tokens; None when unknown.",
    )


class ProviderModelsResponse(_BaseModel):
    """Result of ``GET /api/providers/{provider}/models``.

    Lifts the LLM-surface ``MODEL_CATALOGUE`` of a single provider onto
    the wire as ``(model, pricing_tier)`` rows. The list preserves the
    catalogue's declaration order so the UI can render it verbatim.

    For meta-providers whose catalogue is intentionally empty
    (:class:`~sec_generative_search.providers.openrouter.OpenRouterProvider`),
    ``models`` is an empty list and ``supports_arbitrary_models`` is
    ``True`` — the UI renders a free-text slug input and treats the tier
    as ``UNKNOWN`` for any slug the user types.

    ``total`` is the count of returned rows, mirroring
    :class:`ProviderListResponse`.
    """

    provider: str
    surface: str
    supports_arbitrary_models: bool
    models: list[ModelPricingSchema]
    total: int


class CatalogueRefreshResponse(_BaseModel):
    """Result of ``POST /api/providers/catalogue/refresh`` (admin only).

    Content-free, allow-list lift of
    :class:`~sec_generative_search.core.types.CatalogueRefreshReport`: the
    operator-supplied ``source`` key + the public metadata ``source_url``
    that was fetched, plus the aggregate ``provider_count`` / ``model_count``
    written into the additive overlay.

    The report's ``overlay_path`` is **deliberately NOT lifted** — a
    filesystem path never belongs on the wire (same posture as
    :class:`GpuStatusResponse`).  No model slug, no per-token cost, and no
    credential ever reaches this surface: the refresh seam is credential-free
    end to end and ``pricing_tier`` is derived at load, never fetched.
    Widening this schema is a security-sensitive change.
    """

    source: str
    source_url: str
    provider_count: int
    model_count: int


class ProviderValidateRequest(_BaseModel):
    """Body for ``POST /api/providers/validate``.

    ``api_key`` is required and never echoed back in any response or
    log; the validate route hands it directly to
    :func:`validate_credential` which audit-logs only the masked tail.
    The schema does NOT support a query-string fallback for the key —
    bodies are encrypted by TLS in transit and never end up in proxy
    access logs the way query strings do.
    """

    provider: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$",
        description="Lower-case provider slug; must be registered in ProviderRegistry.",
    )
    api_key: str = Field(
        min_length=1,
        max_length=4096,
        description="The provider key to validate.  Never echoed back.",
    )
    surface: str = Field(
        default="llm",
        pattern=r"^(llm|embedding|reranker)$",
        description="Provider surface to validate against.",
    )
    model: str | None = Field(
        default=None,
        max_length=128,
        description="Optional model slug for embedding-surface validation.",
    )


# ---------------------------------------------------------------------------
# Filing management schemas
# ---------------------------------------------------------------------------


# SEC accession numbers are deterministic: 10-2-6 digits separated by
# hyphens (e.g. ``0000320193-23-000077``).  We pin the shape at the
# schema boundary so a stray identifier never reaches the registry.
_ACCESSION_PATTERN = r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$"

# Ticker symbols on US exchanges fit ``[A-Z][A-Z0-9.-]{0,9}`` but we
# upper-bound length defensively — a longer string is a sign of misuse,
# not an exotic ticker.
_TICKER_PATTERN = r"^[A-Z][A-Z0-9.\-]{0,15}$"

# Form types follow the SEC's published list; we keep validation
# permissive (any 10-character upper-case slug) so amended forms (10-K/A,
# 8-K/A) and future variants do not require a schema bump.
_FORM_TYPE_PATTERN = r"^[A-Z0-9][A-Z0-9/\-]{0,15}$"


class FilingSchema(_BaseModel):
    """Single filing row as exposed to the API.

    Strictly mirrors :class:`FilingRecord` minus the auto-increment
    ``id`` — the integer PK is an internal SQLite detail that does not
    belong on the wire.
    """

    ticker: str
    form_type: str
    filing_date: str
    accession_number: str
    chunk_count: int
    ingested_at: str


class FilingListResponse(_BaseModel):
    """Result of ``GET /api/filings/``.

    ``total`` is the count of returned rows, not the global filing
    count.  Operators that need the registry-wide total go to
    ``GET /api/status/``.
    """

    filings: list[FilingSchema]
    total: int


class DeleteResponse(_BaseModel):
    """Result of ``DELETE /api/filings/{accession}``."""

    accession_number: str
    chunks_deleted: int


class DeleteByIdsRequest(_BaseModel):
    """Body for ``POST /api/filings/delete-by-ids``.

    Bounded list length (1..500) keeps a single request from monopolising
    the SQLite write lock or generating a many-MB ChromaDB ``$in``
    clause.  Each accession is pinned to the SEC shape at schema time so
    the registry never sees a malformed identifier.
    """

    accession_numbers: list[str] = Field(
        min_length=1,
        max_length=500,
        description="Accession numbers to delete (1..500 per request).",
    )

    @field_validator("accession_numbers")
    @classmethod
    def _validate_accession_shape(cls, value: list[str]) -> list[str]:
        pattern = re.compile(_ACCESSION_PATTERN)
        for accession in value:
            if not pattern.fullmatch(accession):
                raise ValueError(
                    f"Invalid accession number shape: {accession!r}. Expected NNNNNNNNNN-NN-NNNNNN."
                )
        # Dedupe while preserving order — duplicates make the request
        # idempotent on the registry side, but waste a parameter slot in
        # the bounded ``IN (?, ?, …)`` clause.
        seen: set[str] = set()
        unique: list[str] = []
        for accession in value:
            if accession not in seen:
                seen.add(accession)
                unique.append(accession)
        return unique


class DeleteByIdsResponse(_BaseModel):
    """Result of ``POST /api/filings/delete-by-ids``."""

    filings_deleted: int
    chunks_deleted: int
    not_found: list[str]


class BulkDeleteRequest(_BaseModel):
    """Body for ``POST /api/filings/bulk-delete``.

    At least one of ``ticker`` / ``form_type`` MUST be set.  An empty
    body would expand to ``DELETE FROM filings`` and we have a separate
    confirm-gated route for that destructive shape.
    """

    ticker: str | None = Field(
        default=None,
        pattern=_TICKER_PATTERN,
        description="Filter by upper-case ticker symbol (e.g. AAPL).",
    )
    form_type: str | None = Field(
        default=None,
        pattern=_FORM_TYPE_PATTERN,
        description="Filter by upper-case form type (e.g. 10-K, 10-K/A).",
    )


class BulkDeleteResponse(_BaseModel):
    """Result of ``POST /api/filings/bulk-delete``."""

    filings_deleted: int
    chunks_deleted: int
    tickers_affected: list[str]


class ClearAllResponse(_BaseModel):
    """Result of ``DELETE /api/filings/?confirm=true``."""

    filings_deleted: int
    chunks_deleted: int


class DemoResetResponse(_BaseModel):
    """Result of ``POST /api/admin/demo-reset`` — counts only (rule L)."""

    reset: bool
    filings_removed: int
    chunks_removed: int


class ProviderValidateResponse(_BaseModel):
    """Verdict from a key-validation attempt.

    ``valid=True`` means the provider accepted the key.  ``valid=False``
    is reserved for the explicit ``ProviderAuthError`` case — every
    other ``ProviderError`` (rate limit, timeout, content filter,
    transport) propagates as a 502 / 503 envelope so the caller does
    not interpret a network blip as a "wrong key".
    """

    valid: bool
    provider: str
    surface: str


class ProviderHealthSchema(_BaseModel):
    """One provider's passive liveness health.

    An explicit allow-list lift of
    :class:`~sec_generative_search.core.provider_health.ProviderHealthSnapshot`
    — never an ``**asdict()`` splat, so the snapshot dataclass cannot grow
    a field that silently surfaces here. Every field is content-free: a
    curated provider name, a breaker state string, integer counters, and
    derived scalars.

    ``state`` is the lower-case
    :class:`~sec_generative_search.core.resilience.CircuitState` value
    (``"closed"`` / ``"open"`` / ``"half_open"``); the route lifts the
    enum *value*, never the enum object.

    ``last_error_type`` is the exception *class name* of the most recent
    failure (``"ProviderTimeoutError"``) — never a message that could
    carry a ticker / query / accession number. ``last_failure_seconds_ago``
    is a derived elapsed scalar, not a wall-clock or monotonic timestamp.
    """

    provider: str = Field(description="Curated LLM provider slug.")
    state: str = Field(
        description="Circuit-breaker state: 'closed' / 'open' / 'half_open'.",
    )
    consecutive_failures: int = Field(
        ge=0,
        description="Failures since the last success (resets to 0 on success).",
    )
    total_failures: int = Field(ge=0, description="Lifetime failed-call count for this provider.")
    total_successes: int = Field(
        ge=0, description="Lifetime successful-call count for this provider."
    )
    last_error_type: str | None = Field(
        default=None,
        description=(
            "Exception class name of the most recent failure (content-free); "
            "None when no failure has been recorded."
        ),
    )
    last_failure_seconds_ago: float | None = Field(
        default=None,
        ge=0,
        description="Seconds since the last failure; None when none recorded.",
    )
    last_latency_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Latency of the most recent successful call; None when none recorded.",
    )


class ProviderHealthResponse(_BaseModel):
    """Result of ``GET /api/providers/health`` (admin-gated).

    Lifts the process-global
    :class:`~sec_generative_search.core.provider_health.ProviderHealthRegistry`
    snapshot onto the wire. ``providers`` lists only providers that have
    seen at least one call this process — a provider absent from the list
    has simply not been exercised yet, not "unknown". The endpoint makes
    no upstream call and reads no credential; it reflects only the
    content-free outcomes the generation path already recorded.

    ``total`` is the count of returned rows, mirroring
    :class:`ProviderListResponse`.
    """

    providers: list[ProviderHealthSchema]
    total: int


# ---------------------------------------------------------------------------
# Retrieval (POST /api/search) schemas
# ---------------------------------------------------------------------------


# ISO date shape is pinned at the schema boundary so a malformed date
# never reaches ``RetrievalService._validate_iso_date``. The retrieval
# service itself validates with ``date.fromisoformat``; this regex is
# the cheap syntactic gate.
_ISO_DATE_PATTERN = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"


class SearchRequest(_BaseModel):
    """Body for ``POST /api/search``.

        Retrieval-only surface over :class:`RetrievalService`. The query
        travels in the body, never the URL, so it does not land in proxy
        access logs.

    Filter fields accept either a single value or a list; the underlying
    :meth:`RetrievalService.retrieve` already handles both shapes.  Each
    list is bounded so a malicious caller cannot spend the embedder
    budget by passing thousands of tickers in one request.

    Numeric bounds are deliberately tight:

                - ``top_k`` is capped at 50. Without a cap a single request
                    could pin Chroma's connection on a giant scan.
        - ``min_similarity`` is in [0.0, 1.0] — anything else is a
          caller bug, not a configuration knob.
        - ``context_token_budget`` is bounded at 100k tokens, matching
          the largest realistic context window across supported
          providers; values of 0 disable packing.
    """

    query: str = Field(
        min_length=1,
        max_length=1024,
        description=(
            "Natural language query.  Echoed nowhere in the response or any non-redacted log line."
        ),
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description=(
            "Maximum number of hits to return.  Defaults to ``settings.search.top_k`` when omitted."
        ),
    )
    min_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity threshold; results below are dropped before any further processing."
        ),
    )
    ticker: str | list[str] | None = Field(
        default=None,
        description="Single ticker symbol or a list of tickers.",
    )
    form_type: str | list[str] | None = Field(
        default=None,
        description="Single SEC form type (e.g. '10-K') or a list.",
    )
    accession_number: str | list[str] | None = Field(
        default=None,
        description="Single SEC accession number or a list.",
    )
    start_date: str | None = Field(
        default=None,
        pattern=_ISO_DATE_PATTERN,
        description="Inclusive lower bound, YYYY-MM-DD.",
    )
    end_date: str | None = Field(
        default=None,
        pattern=_ISO_DATE_PATTERN,
        description="Inclusive upper bound, YYYY-MM-DD.",
    )
    max_per_section: int | None = Field(
        default=None,
        ge=0,
        le=50,
        description=(
            "Maximum chunks per section path; 0 disables the cap.  Defaults "
            "to ``settings.search.max_per_section`` when omitted."
        ),
    )
    max_per_filing: int | None = Field(
        default=None,
        ge=0,
        le=50,
        description=(
            "Maximum chunks per accession number; 0 disables the cap.  "
            "Defaults to ``settings.search.max_per_filing`` when omitted."
        ),
    )
    rerank_over_fetch_factor: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description=(
            "Reranker over-fetch multiplier (top_k * factor); 1 disables.  "
            "Only active when a reranker is bound.  Bounded at 10 so a "
            "single request cannot explode the candidate fetch.  Defaults "
            "to ``settings.search.rerank_over_fetch_factor`` when omitted."
        ),
    )
    context_token_budget: int | None = Field(
        default=None,
        ge=0,
        le=100_000,
        description=(
            "Cumulative token budget for the returned hits; 0 disables "
            "packing.  Defaults to ``settings.rag.context_token_budget``."
        ),
    )

    @field_validator("ticker", "form_type", "accession_number")
    @classmethod
    def _bounded_list(cls, value: str | list[str] | None) -> str | list[str] | None:
        # Bound list length so a malicious payload cannot expand into a
        # huge ChromaDB ``$in`` clause.  Single strings pass through
        # unchanged — the underlying retrieval helpers shape-check those.
        if isinstance(value, list):
            if len(value) == 0:
                raise ValueError("Filter list must not be empty; omit the field instead.")
            if len(value) > 50:
                raise ValueError("Filter list is bounded to 50 entries per request.")
        return value


class SearchHit(_BaseModel):
    """One retrieval hit on the wire.

    Mirrors :class:`RetrievalResult` minus internal-only fields.  The
    fields chosen for the wire are the union of "needed by the UI" and
    "auditable" — ``content`` is Tier-1 public SEC data and is safe to
    return to authenticated callers.

    Notes:

                - ``rerank_score`` is ``None`` when no reranker is bound. UI
                    code should prefer ``rerank_score`` when present and fall back
                    to ``similarity``; the two scales are not commensurable.
        - ``token_count`` is exposed because operators tuning the budget
          want to see it; it is NOT a credential and reveals nothing
          about the underlying corpus shape that ``content`` does not
          already.
        - ``RetrievalResult.truncated`` is deliberately **not** lifted:
          the packer is drop-tail and never sets it, so it was a constant
          ``false`` on every hit (F36).  Add it back — a widening, so a
          reviewed rule-L change — only with a mid-chunk clipping strategy.
    """

    chunk_id: str | None
    content: str
    path: str
    content_type: str
    ticker: str
    form_type: str
    filing_date: str | None
    accession_number: str | None
    similarity: float
    rerank_score: float | None = None
    token_count: int = 0
    section_boundaries: list[str] = Field(default_factory=list)


class SearchResponse(_BaseModel):
    """Result of ``POST /api/search``.

    ``query`` is intentionally NOT echoed back — the request body is
    sufficient round-trip context for any client (clients keep the body
    they sent), and echoing it would land queries in any caller-side log
    or browser dev-tools history that captures responses but redacts
    requests.  ``total`` is the count of returned hits, not a corpus
    cardinality.
    """

    hits: list[SearchHit]
    total: int


# ---------------------------------------------------------------------------
# RAG query-understanding (POST /api/rag/plan) schemas
# ---------------------------------------------------------------------------


class RagPlanRequest(_BaseModel):
    """Body for ``POST /api/rag/plan``.

    The query travels in the body — never the URL — for the same reason
    as :class:`SearchRequest`: queries are Tier-3 user-generated data and
    a query-string transport would land them in proxy access logs.  The
    optional ``provider`` / ``model`` fields let callers pin which LLM
    runs the understanding step; both default to ``settings.llm`` when
    omitted.
    """

    query: str = Field(
        min_length=1,
        max_length=1024,
        description=(
            "Natural-language query. Tier-3 user data — never echoed in any "
            "non-redacted log line; the audit log carries metadata only."
        ),
    )
    provider: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$",
        description=(
            "Optional registered provider slug.  Defaults to ``settings.llm.default_provider``."
        ),
    )
    model: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Optional model slug.  Defaults to ``settings.llm.default_model`` "
            "or the provider's own default when both are unset."
        ),
    )


class QueryPlanSchema(_BaseModel):
    """Wire shape of :class:`~sec_generative_search.rag.query_understanding.QueryPlan`.

    Mirrors the dataclass field-for-field, with ``date_range`` lifted
    onto the wire as a fixed-length list (``[start, end]``) instead of a
    Python tuple.  ``suggested_answer_mode`` is the lower-case enum value.

    The plan IS the query reformulated for the editable-chips UI; the
    raw query unavoidably appears here (``raw_query`` and possibly
    ``query_en``) — that is the documented contract.  The route never
    embeds the query in any other surface (audit log, error envelope).
    """

    raw_query: str
    detected_language: str
    query_en: str
    tickers: list[str]
    form_types: list[str]
    date_range: list[str] | None
    intent: str
    suggested_answer_mode: str


class RagPlanResponse(_BaseModel):
    """Result of ``POST /api/rag/plan``.

    Carries the planned :class:`QueryPlanSchema` plus the resolved
    ``provider`` / ``model`` so the UI can show "ran on X / Y" without
    re-deriving them from settings.
    """

    plan: QueryPlanSchema
    provider: str
    model: str


# ---------------------------------------------------------------------------
# RAG generation (POST /api/rag/query) schemas
# ---------------------------------------------------------------------------


# Mode strings accepted on the wire — mirrors the lower-case values of
# :class:`AnswerMode`.  Validating the alphabet at the schema boundary
# means an unknown mode is rejected as 422 before the route runs, rather
# than silently coerced to the default by ``AnswerMode.from_string``.
_ANSWER_MODE_PATTERN = r"^(concise|analytical|extractive|comparative)$"


class TokenUsageSchema(_BaseModel):
    """Wire shape of :class:`~sec_generative_search.core.types.TokenUsage`.

    ``total_tokens`` is computed by the dataclass; we surface it
    explicitly on the wire so a client does not have to re-derive it
    from the two component counts.
    """

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class CitationSchema(_BaseModel):
    """Wire shape of :class:`~sec_generative_search.core.types.Citation`.

    The citation is the audit trail of *which* retrieved chunk the model
    actually leaned on.  Every field maps one-for-one onto :class:`Citation`,
    with :attr:`Citation.filing_id` flattened into the four-tuple
    ``(ticker, form_type, filing_date, accession_number)`` so the wire
    schema stays JSON-flat and the UI does not have to walk a nested
    object.

    The ``text_span`` is Tier 1 (public SEC filing) content — safe to
    surface to authenticated callers.  ``display_index`` is the 1-based
    ordinal that maps to inline ``[N]`` markers in the answer.
    """

    chunk_id: str
    ticker: str
    form_type: str
    filing_date: str = Field(
        description="ISO date the filing was submitted to SEC (YYYY-MM-DD).",
    )
    accession_number: str
    section_path: str
    text_span: str
    similarity: float
    display_index: int = Field(
        ge=0,
        description=("1-based ordinal mapped to inline [N] markers; 0 means 'not assigned'."),
    )


class ConversationTurnSchema(_BaseModel):
    """Wire shape of a single prior chat turn supplied by the browser.

    The browser-tab chat surface keeps conversation history client-side and
    replays it on every follow-up turn so the orchestrator can build the
    ``Q:/A:`` history block. The schema deliberately carries **only** the
    user query and the model answer — never the prior turn's retrieved
    chunks or citations. Prior turn's ``retrieval_results`` must never
    re-enter a future prompt; every follow-up turn re-retrieves.

    Both strings are user-derived (the query is Tier-3 user input; the
    answer is upstream-LLM output that surfaced to the user) so length
    bounds defend memory ahead of the orchestrator's own budgeting
    step.  ``query`` shares the 1024-char cap of :class:`RagPlanRequest`;
    ``answer`` is bounded at 32 KiB which comfortably exceeds the
    largest budget-allowed answer slice (8192 tokens) while still
    refusing pathological payloads at the boundary.
    """

    query: str = Field(
        min_length=1,
        max_length=1024,
        description=(
            "The prior turn's user query. Tier-3 user data — never echoed in"
            " any non-redacted log line."
        ),
    )
    answer: str = Field(
        min_length=1,
        max_length=4096,
        description=(
            "The prior turn's assistant answer (plain text).  Bounded at "
            "4096 chars to keep a 10-turn history comfortably under the "
            "64 KiB body cap on ``/api/rag/{query,stream}``."
        ),
    )


# Slug shape mirrors OpenRouter's documented upstream-provider slug
# alphabet (lower-case alphanumerics + ``-`` separator). The bound is
# intentionally permissive: OpenRouter gains and loses upstreams almost
# weekly, so this regex defends against control characters and length
# bombs without baking the catalogue into the API surface. Anything
# OpenRouter rejects at call time surfaces as their own error message
# (which the orchestrator's exception mapping translates to 502).
_OPENROUTER_UPSTREAM_PATTERN = r"^[a-z0-9][a-z0-9-]{0,63}$"


class OpenRouterRoutingHintsSchema(_BaseModel):
    """Wire shape of
    :class:`~sec_generative_search.providers.openrouter.OpenRouterRoutingHints`.

    Only :class:`OpenRouterProvider` consumes this — every other adapter
    silently ignores the field via the OpenAI-compatible base's empty
    ``_extra_request_kwargs`` default. The RAG route fails closed with
    ``invalid_flag_combination`` (HTTP 400) when hints are supplied but
    the resolved LLM provider does not advertise
    ``ProviderRegistry.supports_upstream_routing``. Allowing the hint to
    silently no-op would be misleading UX — a caller who pinned
    ``order=["anthropic"]`` against ``provider=openai`` would expect the
    routing to be honoured.

    Fields mirror the dataclass exactly so the schema does not have to be
    revised when OpenRouter adds new ``provider``-block keys we have not
    surfaced in the UI yet — extending the dataclass alone is the
    canonical way forward (the schema's ``extra="forbid"`` discipline
    still rejects unknown fields, so a new key requires a deliberate
    schema edit).

    Carries no credential-shaped fields by design — a
    ``@pytest.mark.security`` regression test pins the field set against
    the same credential-hint list used for the registry rows.
    """

    order: list[str] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "Preferred upstream-provider order (first match wins). "
            "Bounded at 16 entries — OpenRouter's effective ceiling is "
            "lower; this is a memory-side defence against payload bombs."
        ),
    )
    allow_fallbacks: bool | None = Field(
        default=None,
        description=(
            "When ``False``, OpenRouter refuses to fall back to upstreams "
            "outside ``order`` / ``only``. ``None`` defers to OpenRouter's "
            "own default."
        ),
    )
    only: list[str] = Field(
        default_factory=list,
        max_length=16,
        description="Allowlist of upstream slugs.",
    )
    ignore: list[str] = Field(
        default_factory=list,
        max_length=16,
        description="Blocklist of upstream slugs.",
    )
    require_parameters: bool | None = Field(
        default=None,
        description=(
            "When ``True``, only route to upstreams that honour every "
            "request parameter (e.g. reasoning, tools)."
        ),
    )
    data_collection: str | None = Field(
        default=None,
        pattern=r"^(allow|deny)$",
        description=(
            "``'allow'`` or ``'deny'`` — refuse upstreams that log "
            "prompt/response content. Anything else rejected at the "
            "schema layer."
        ),
    )

    @field_validator("order", "only", "ignore")
    @classmethod
    def _validate_upstream_slugs(cls, value: list[str]) -> list[str]:
        """Shape-check each slug at the schema boundary.

        The dataclass downstream is pass-through (frozen, no validation),
        so this is the only place malformed slugs are caught before they
        reach OpenRouter. Control characters / oversize entries are
        rejected here so the upstream API receives well-formed payloads
        only.
        """
        slug_re = re.compile(_OPENROUTER_UPSTREAM_PATTERN)
        for slug in value:
            if not slug_re.fullmatch(slug):
                raise ValueError("Upstream slug must match ^[a-z0-9][a-z0-9-]{0,63}$")
        return value


class RagQueryRequest(_BaseModel):
    """Body for ``POST /api/rag/query``.

    Accepts a :class:`QueryPlanSchema` — never a raw query string.  This
    is load-bearing: splitting plan / generate makes the human-in-the-
    loop edit step a hard contract enforced by the API shape, so the UI
    cannot accidentally bypass the editable-chips review by sending a
    raw query straight at generation.

    All other fields are optional overrides:

    - ``provider`` / ``model`` default to ``settings.llm`` (same defaults
      as :class:`RagPlanRequest`); the route resolves the LLM through
      the request-scoped resolver chain so the generation lands on the
      caller's key, not on admin env.
    - ``mode`` overrides the plan's ``suggested_answer_mode``.  Useful
      when the user edited the chip in the UI but did not regenerate
      the plan.
    - ``max_output_tokens`` caps the answer slice.  Defaults to
      ``settings.llm.max_output_tokens``.
    - ``history`` carries prior ``Q:/A:`` turns from the browser-tab
      chat surface so the orchestrator's history block renders
      continuity.  Conversation history is **never persisted server-
      side** (the SPA holds it in tab memory; chat-history persistence is
      out of scope by design), and the orchestrator strips retrieved
      chunks / citations from prior turns — only the rendered ``Q:``
      and ``A:`` strings survive into the follow-up prompt.

    """

    plan: QueryPlanSchema
    provider: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$",
        description=(
            "Optional registered provider slug.  Defaults to ``settings.llm.default_provider``."
        ),
    )
    model: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Optional model slug.  Defaults to ``settings.llm.default_model`` "
            "or the provider's own default when both are unset."
        ),
    )
    mode: str | None = Field(
        default=None,
        pattern=_ANSWER_MODE_PATTERN,
        description=(
            "Optional override of ``plan.suggested_answer_mode``.  "
            "One of: concise, analytical, extractive, comparative."
        ),
    )
    max_output_tokens: int | None = Field(
        default=None,
        ge=1,
        le=8192,
        description=(
            "Optional cap on the answer slice.  Defaults to "
            "``settings.llm.max_output_tokens``.  Bounded at 8192 to "
            "keep a single request from burning a giant generation budget."
        ),
    )
    max_per_section: int | None = Field(
        default=None,
        ge=0,
        le=50,
        description=(
            "Retrieval diversity cap: maximum chunks per section path; 0 "
            "disables.  Defaults to ``settings.search.max_per_section``."
        ),
    )
    max_per_filing: int | None = Field(
        default=None,
        ge=0,
        le=50,
        description=(
            "Retrieval diversity cap: maximum chunks per filing; 0 "
            "disables.  Defaults to ``settings.search.max_per_filing``."
        ),
    )
    rerank_over_fetch_factor: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description=(
            "Reranker over-fetch multiplier (top_k * factor); 1 disables.  "
            "Only active when a reranker is bound.  Bounded at 10.  "
            "Defaults to ``settings.search.rerank_over_fetch_factor``."
        ),
    )
    history: list[ConversationTurnSchema] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Optional prior ``Q:/A:`` turns for chat-mode continuity.  "
            "Bounded at 10 turns to mirror the ``RAG_CHAT_HISTORY_TURNS`` "
            "default cap the orchestrator would honour anyway, and to "
            "keep the request body under ``/api/rag/{query,stream}``'s "
            "64 KiB cap.  The orchestrator drops prior-turn retrieval "
            "/ citations — only the rendered Q/A pairs reach the prompt."
        ),
    )
    routing_hints: OpenRouterRoutingHintsSchema | None = Field(
        default=None,
        description=(
            "Optional upstream-provider routing hints. Forwarded only "
            "when the resolved provider advertises "
            "``ProviderRegistry.supports_upstream_routing`` (currently "
            "OpenRouter only); supplied against any other provider "
            "raises 400 ``invalid_flag_combination``. Mirrors the CLI's "
            "``--openrouter-provider`` / ``--openrouter-fallbacks`` "
            "surface — the API never silently no-ops the hint."
        ),
    )


class RagQueryResponse(_BaseModel):
    """Result of ``POST /api/rag/query``.

    Carries the generated answer plus full traceability — provider,
    model, prompt-template version, citations, token usage, and wall-
    clock latency — so the UI can render "Generated by X / Y in N
    seconds, used Z chunks" without re-deriving anything.

    ``refused`` is true when the orchestrator short-circuited the LLM
    call because retrieval returned nothing; it lets the UI show a
    distinct "no sources" state without parsing the answer text for the
    refusal phrase.

    ``retrieved_chunks`` is intentionally NOT on the wire in v1 — the
    citations are the user-visible audit trail and the full retrieved
    set would balloon the payload.  When debugging is needed, operators
    use the existing ``POST /api/search`` route (same retrieval primitive)
    to inspect the candidate set.
    """

    answer: str
    citations: list[CitationSchema]
    provider: str
    model: str
    prompt_version: str
    token_usage: TokenUsageSchema
    estimated_cost_usd: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Estimated USD cost of this call, derived from token usage and the "
            "model's exact per-MTok cost via "
            ":func:`~sec_generative_search.core.types.estimate_cost`. ``None`` "
            "when the model's cost is unknown (arbitrary-slug providers / "
            "overlay-only models) — the UI renders that as '—'. An estimate, "
            "not a final bill: prompt-caching discounts and tiered pricing are "
            "not modelled, and token counts for some providers are an offline "
            "approximation."
        ),
    )
    latency_seconds: float = Field(ge=0.0)
    streamed: bool = Field(
        default=False,
        description="Always False on this non-streaming surface.",
    )
    refused: bool = Field(
        default=False,
        description=(
            "True when the orchestrator short-circuited the LLM call "
            "because retrieval returned no chunks."
        ),
    )


# ---------------------------------------------------------------------------
# Ingestion (POST /api/ingest/*) schemas
# ---------------------------------------------------------------------------


# The work-list builder in ``TaskManager._build_work_list`` interprets
# three count modes; pinning the enum at the schema boundary surfaces an
# unknown value as 422 before the worker thread starts.
_COUNT_MODE_PATTERN = r"^(latest|per_form|total)$"


class IngestRequest(_BaseModel):
    """Body for ``POST /api/ingest/{add,batch}``.

    The route uses the *same* request shape for both endpoints — the
    ``/add`` route additionally enforces a single-ticker invariant at the
    handler so a future schema change cannot accidentally relax it. List
    bounds defend the worker thread before any EDGAR network call.

    ``year`` / ``start_date`` / ``end_date`` are SEC filter knobs passed
    through to ``FilingFetcher.list_available`` verbatim; the ISO date
    pattern is the cheap syntactic gate (the fetcher revalidates with
    ``date.fromisoformat``).
    """

    tickers: list[str] = Field(
        min_length=1,
        max_length=50,
        description="Ticker symbols to ingest. Length-bounded so a payload cannot OOM the worker.",
    )
    form_types: list[str] = Field(
        min_length=1,
        max_length=20,
        description="SEC form types (e.g. '10-K', '10-Q/A'). At least one required.",
    )
    count_mode: str = Field(
        default="latest",
        pattern=_COUNT_MODE_PATTERN,
        description=(
            "How to interpret ``count``: 'latest' (filter-aware default), "
            "'per_form' (count per (ticker, form_type)), or 'total' "
            "(count across forms per ticker)."
        ),
    )
    count: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description=(
            "Maximum filings per (ticker, form_type) pair (or total per "
            "ticker in 'total' mode). Bounded at 500 to keep a single "
            "request from monopolising the GPU semaphore."
        ),
    )
    year: int | None = Field(
        default=None,
        ge=1990,
        le=2100,
        description="Filter by filing year. Mutually compatible with date-range filters.",
    )
    start_date: str | None = Field(
        default=None,
        pattern=_ISO_DATE_PATTERN,
        description="Inclusive lower bound, YYYY-MM-DD.",
    )
    end_date: str | None = Field(
        default=None,
        pattern=_ISO_DATE_PATTERN,
        description="Inclusive upper bound, YYYY-MM-DD.",
    )

    @field_validator("tickers")
    @classmethod
    def _validate_tickers(cls, value: list[str]) -> list[str]:
        pattern = re.compile(_TICKER_PATTERN)
        # Upper-case at the schema boundary so the registry and fetcher
        # never see a mixed-case symbol; deduplicate preserving order to
        # avoid a needlessly fat work-list when the caller submitted
        # duplicates.
        seen: set[str] = set()
        unique: list[str] = []
        for ticker in value:
            upper = ticker.strip().upper()
            if not pattern.fullmatch(upper):
                raise ValueError(
                    f"Invalid ticker symbol: {ticker!r}. Expected [A-Z][A-Z0-9.-]{{0,15}}."
                )
            if upper not in seen:
                seen.add(upper)
                unique.append(upper)
        return unique

    @field_validator("form_types")
    @classmethod
    def _validate_form_types(cls, value: list[str]) -> list[str]:
        pattern = re.compile(_FORM_TYPE_PATTERN)
        seen: set[str] = set()
        unique: list[str] = []
        for form in value:
            upper = form.strip().upper()
            if not pattern.fullmatch(upper):
                raise ValueError(
                    f"Invalid form type: {form!r}. "
                    "Expected upper-case slug like '10-K' or '10-K/A'."
                )
            if upper not in seen:
                seen.add(upper)
                unique.append(upper)
        return unique


class IngestTaskResponse(_BaseModel):
    """Result of ``POST /api/ingest/{add,batch}``.

    Returns the opaque ``task_id`` (a 32-char hex uuid) plus a hint at
    where to poll progress.  The WebSocket URL is documented here so the
    UI does not need to hard-code the path shape; the route is gated on
    the *same* session/ownership checks as the polling surface.
    """

    task_id: str
    status: str = Field(
        description="Initial state ('pending'); transitions are observed via polling or WebSocket.",
    )
    websocket_url: str = Field(
        description="Relative URL of the per-task progress WebSocket.",
    )


class IngestResultSchema(_BaseModel):
    """Per-filing outcome echoed back to the caller.

    Mirrors :class:`sec_generative_search.api.tasks.FilingResult` minus
    the worker-internal serialisation helpers.  Every field is public SEC
    metadata (Tier 1) — no PII or credential ever lands here.
    """

    ticker: str
    form_type: str
    filing_date: str
    accession_number: str
    segment_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0.0)


class TaskProgressSchema(_BaseModel):
    """Mutable progress snapshot exposed by the polling surface.

    Mirrors :class:`sec_generative_search.api.tasks.TaskProgress`; the
    fields are individually scalar so a partial update on the worker side
    does not need any cross-field coordination at the schema layer.
    """

    current_ticker: str | None = None
    current_form_type: str | None = None
    step_label: str = ""
    step_index: int = 0
    step_total: int = 5
    filings_done: int = 0
    filings_total: int = 0
    filings_skipped: int = 0
    filings_failed: int = 0


class TaskStatusResponse(_BaseModel):
    """Result of ``GET /api/ingest/tasks/{task_id}``.

    Carries everything a polling client needs to render the live progress
    surface.  The fields mirror :class:`TaskInfo` minus the ownership
    (`session_id`) and the worker-internal fields (`cancel_event`,
    `_stored_accessions`, `_duration_timer`, `_message_queue`); the
    session id is intentionally not echoed back because exposing it would
    let a same-origin script read the cookie indirectly through the
    polling response body.
    """

    task_id: str
    status: str
    tickers: list[str]
    form_types: list[str]
    progress: TaskProgressSchema
    results: list[IngestResultSchema]
    error: str | None
    started_at: str | None
    completed_at: str | None


class TaskListResponse(_BaseModel):
    """Result of ``GET /api/ingest/tasks``.

    Carries the *session-scoped* task list — the route filters by the
    server-minted cookie before lifting to this schema. ``total`` is the
    count of returned rows; there is no cross-session aggregate exposed
    on this surface.
    """

    tasks: list[TaskStatusResponse]
    total: int


class IngestCancelResponse(_BaseModel):
    """Result of ``DELETE /api/ingest/tasks/{task_id}``.

    Cancellation is cooperative — the worker observes ``cancel_event``
    between pipeline steps. The route returns immediately on signal
    delivery; ``status='cancelling'`` distinguishes the in-flight transit
    from the terminal ``cancelled`` state surfaced by the polling
    response.
    """

    task_id: str
    status: str = Field(
        default="cancelling",
        description="Always 'cancelling'; the terminal state is observable via polling.",
    )


# ---------------------------------------------------------------------------
# GPU / embedder resource schema
# ---------------------------------------------------------------------------


class GpuStatusResponse(_BaseModel):
    """Result of ``GET /api/resources/gpu``.

    Read-tier operator surface that reports the current state of the
    embedder. The schema is intentionally provider-neutral: hosted
    embedders are stateless HTTP clients and report ``is_local=False``
    with the load-state fields collapsed to a no-op shape, while the
    on-device :class:`LocalEmbeddingProvider` populates ``device`` /
    ``idle_seconds`` from its in-memory load handle.

    Privacy contract:

        - No file system paths. Model identifiers are slugs only.
        - No credentials. The HF token / API key never reach this
          surface — :class:`_ProviderBase` keeps them off ``repr`` and
          this schema has no field that could accept one.
        - ``idle_seconds`` is a derived elapsed-time scalar, not the
          internal monotonic ``last_used`` timestamp; surfacing the
          monotonic value would leak process start-time information
          without operator value.
    """

    provider: str = Field(
        description="Embedder provider slug (e.g. 'local', 'openai', 'gemini').",
    )
    model: str = Field(
        description="Embedder model slug (vendor-defined identifier; never a path).",
    )
    is_local: bool = Field(
        description=(
            "True when the embedder runs on-device (LocalEmbeddingProvider). "
            "Hosted embedders report False and collapse the load-state fields."
        ),
    )
    is_loaded: bool = Field(
        description=(
            "True when the on-device model handle is resident. Hosted "
            "embedders always report True — there is nothing to load."
        ),
    )
    device: str | None = Field(
        default=None,
        description=(
            "Resolved device for an on-device load (e.g. 'cuda', 'cuda:0', 'cpu'). "
            "None when not loaded or when the embedder is hosted."
        ),
    )
    idle_seconds: float | None = Field(
        default=None,
        description=(
            "Seconds since the last successful encode call. None when the "
            "model is not currently loaded or when the embedder is hosted."
        ),
    )
    idle_timeout_minutes: int = Field(
        default=0,
        ge=0,
        description=(
            "Operator-configured idle-unload threshold; 0 disables "
            "auto-unload. Always 0 for hosted embedders."
        ),
    )


# ---------------------------------------------------------------------------
# User-tier authentication schemas
# ---------------------------------------------------------------------------


# Username shape: alphanumeric + `._-`, length 1-64, must start with an
# alnum.  Permissive enough for an email-as-username deployment while
# excluding control characters, whitespace, and shell metacharacters
# that the wire layer cannot trust.
_USERNAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"


# KDF algorithm rotation seam — initial value is the only currently
# supported algorithm; a future ``argon2id-wasm`` lands by adding to
# this set, never removing.  Code paths reading a vault MUST honour the
# row's recorded algorithm; this set bounds what we accept at the wire.
_KDF_ALGO_PATTERN = r"^(pbkdf2-sha256)$"


# Base64url encoding pattern (no padding) — opaque binary fields ride
# the JSON layer as base64url strings.  Length bounds defend against
# absurdly-large declared payloads; the route layer further validates
# decoded-length where the shape is fixed (16-byte salts, 32-byte
# proofs, 12-byte IVs).
_BASE64URL_PATTERN = r"^[A-Za-z0-9_\-]+$"


class LoginParamsResponse(_BaseModel):
    """Response envelope for ``GET /api/auth/login-params``.

    Carries the parameters a browser needs to derive ``auth_proof`` from
    a user-supplied password. The route returns the same shape — and
    the same length / KDF parameters — for unknown usernames as for
    real ones, using :func:`decoy_salt` so an attacker cannot
    distinguish "no such user" from "yes, here is the salt".

    Fields:
        salt_m: 16-byte salt as base64url. Real for enrolled users,
            deterministic decoy for unknown ones.
        kdf_algo: Algorithm slug; client-side dispatches on this for
            forward compatibility.
        pbkdf2_iterations: Work-factor for the PBKDF2 stretch.
    """

    salt_m: str = Field(min_length=22, max_length=24, pattern=_BASE64URL_PATTERN)
    kdf_algo: str = Field(pattern=_KDF_ALGO_PATTERN)
    pbkdf2_iterations: int = Field(ge=100_000, le=10_000_000)


class LoginRequest(_BaseModel):
    """Body for ``POST /api/auth/login``.

    ``auth_proof`` is the 32-byte HKDF-SHA256 output the browser
    derived from the user's password (the ``"sec-gs/auth/v1"`` context).
    NEVER echoed back. The password itself never crosses the wire.
    """

    username: str = Field(min_length=1, max_length=64, pattern=_USERNAME_PATTERN)
    auth_proof: str = Field(
        min_length=43,
        max_length=44,
        pattern=_BASE64URL_PATTERN,
        description="32-byte HKDF output, base64url. Never echoed back.",
    )


class LoginResponse(_BaseModel):
    """Response envelope for a successful ``POST /api/auth/login``.

    Carries the ciphertext + IV the browser needs to decrypt the vault
    locally. The server cannot decrypt — the KEK was derived
    client-side and never crossed the wire.
    """

    user_id: int
    username: str
    ciphertext_vault: str = Field(pattern=_BASE64URL_PATTERN)
    vault_iv: str = Field(min_length=16, max_length=16, pattern=_BASE64URL_PATTERN)


class EnrolmentCompleteRequest(_BaseModel):
    """Body for ``POST /api/auth/enrol``.

    Closes the enrolment loop: the user has visited
    ``/enrol?token=…``, typed their password twice, derived
    ``salt_M``/``auth_proof``/``kek`` client-side, encrypted an empty
    vault, and now ships the artefacts. The route validates the token
    + flips ``must_enrol`` + creates the row.
    """

    token: str = Field(min_length=1, max_length=512)
    salt_m: str = Field(min_length=22, max_length=24, pattern=_BASE64URL_PATTERN)
    auth_proof: str = Field(min_length=43, max_length=44, pattern=_BASE64URL_PATTERN)
    ciphertext_vault: str = Field(min_length=1, max_length=65536, pattern=_BASE64URL_PATTERN)
    vault_iv: str = Field(min_length=16, max_length=16, pattern=_BASE64URL_PATTERN)
    kdf_algo: str = Field(pattern=_KDF_ALGO_PATTERN)
    pbkdf2_iterations: int = Field(ge=100_000, le=10_000_000)


class EnrolmentCompleteResponse(_BaseModel):
    """Response envelope for a successful ``POST /api/auth/enrol``.

    Minimal by design — the user redirects to the login surface and
    completes a normal login round-trip on their next request. Echoing
    the username back is acceptable because the token already bound it
    server-side; an attacker forging the response cannot promote the
    enrolment they could not complete.
    """

    enrolled: bool
    user_id: int
    username: str


class PasswordChangeRequest(_BaseModel):
    """Body for ``POST /api/auth/password``.

    Atomic: validates ``auth_proof_old`` against the current
    ``auth_hash``, then replaces ``salt_M`` / ``auth_hash`` / vault in a
    single transaction. ``kdf_algo`` rotation is forward-only; the
    route enforces algo monotonicity.
    """

    auth_proof_old: str = Field(min_length=43, max_length=44, pattern=_BASE64URL_PATTERN)
    salt_m: str = Field(min_length=22, max_length=24, pattern=_BASE64URL_PATTERN)
    auth_proof_new: str = Field(min_length=43, max_length=44, pattern=_BASE64URL_PATTERN)
    ciphertext_vault: str = Field(min_length=1, max_length=65536, pattern=_BASE64URL_PATTERN)
    vault_iv: str = Field(min_length=16, max_length=16, pattern=_BASE64URL_PATTERN)
    kdf_algo: str = Field(pattern=_KDF_ALGO_PATTERN)
    pbkdf2_iterations: int = Field(ge=100_000, le=10_000_000)


class PasswordChangeResponse(_BaseModel):
    """Response envelope for a successful password change."""

    rotated: bool


class VaultUpdateRequest(_BaseModel):
    """Body for ``POST /api/auth/vault`` — re-upload an updated ciphertext.

    Used by the provider-key write path and the EDGAR-identity update
    path. The IV is fresh per call (the client never reuses an IV
    against a given key — load-bearing for AES-GCM security).
    """

    ciphertext_vault: str = Field(min_length=1, max_length=65536, pattern=_BASE64URL_PATTERN)
    vault_iv: str = Field(min_length=16, max_length=16, pattern=_BASE64URL_PATTERN)


class VaultUpdateResponse(_BaseModel):
    """Response envelope for a successful vault re-upload."""

    updated: bool


# ---------------------------------------------------------------------------
# Admin user-management schemas
# ---------------------------------------------------------------------------


class AdminUserCreateRequest(_BaseModel):
    """Body for ``POST /api/admin/users``.

    Admin mints a single-use enrolment token bound to ``username``;
    the response carries the token verbatim so the admin can share the
    enrolment link with the user out-of-band. The admin NEVER sees the
    user password.
    """

    username: str = Field(min_length=1, max_length=64, pattern=_USERNAME_PATTERN)


class AdminUserCreateResponse(_BaseModel):
    """Result of a successful ``POST /api/admin/users``.

    ``enrolment_token`` is the signed token the admin shares with the
    user. ``expires_at`` is epoch seconds. ``enrol_url`` is a
    convenience for the SPA — the route returns it verbatim with no
    extra parameters baked in.
    """

    username: str
    enrolment_token: str
    expires_at: int
    enrol_url: str


class AdminUserDeleteResponse(_BaseModel):
    """Result of a successful ``DELETE /api/admin/users/{id}``."""

    deleted: bool
    user_id: int


class AdminUserUnlockResponse(_BaseModel):
    """Result of a successful ``POST /api/admin/users/{id}/unlock``."""

    unlocked: bool
    user_id: int
