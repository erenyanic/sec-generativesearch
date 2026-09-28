"""Synchronous CLI wrappers over :class:`RAGOrchestrator`.

Two operator-facing subcommands live here:

- ``sec-rag rag query QUESTION`` — run the full understand → retrieve →
  generate pipeline non-streaming and render the answer + citations +
  traceability.
- ``sec-rag rag chat`` — interactive REPL using
  :meth:`RAGOrchestrator.generate_stream`.  Conversation history is
  kept in process memory only (``ConversationTurn`` — never persisted);
  ``/clear`` drops it, ``/exit`` (or ``/quit``) closes the loop.  One
  Ctrl-C cancels the in-flight stream and returns to the prompt; a
  second Ctrl-C (or Ctrl-D / EOF) at the prompt exits 130.

Trust model and load-bearing rules:

1. The embedder is built via :func:`build_embedder`; the LLM via
   :func:`build_llm_provider` — both *sole construction seams*.  Direct
   adapter instantiation is forbidden at this surface so the resolver
   chain stays the only credential path.
2. The CLI runs at operator scope: no auth gate, no per-session EDGAR
   identity, no rate limit, no access-log redaction.  Distributing CLI
   access to team users in Scenario B/C is unsupported — the web is
   the team-user surface.
3. The CLI resolver chain is ``encrypted-user (ADMIN_USER_ID="__admin__")
   → admin-env (default_api_key_resolver)`` — no header tier, no
   session tier (neither exists at the terminal).  The encrypted tier
   is only added when both ``DB_PERSIST_PROVIDER_CREDENTIALS=true`` and
   SQLCipher are configured; otherwise the chain collapses to admin-env
   alone.  No ``--api-key`` flag on any command — the shell history
   and ``/proc/<pid>/environ`` footgun is intentional.
4. ``rag query`` accepts a positional QUESTION and synthesises the
   ``QueryPlan`` itself.  The wire-tier ``/api/rag/query`` enforces a
   plan-only shape because the web UI's chip-edit review is the
   human-in-the-loop step; at the operator terminal that gate is the
   operator themselves, and we offer two equivalent affordances:

   - ``--show-plan`` runs ``understand_query``, prints the plan, exits
     0.  The operator reads the plan, then re-runs the command after
     applying any overrides via the flags below.  This is the CLI
     analogue of the web's chip-edit step.
   - ``--ticker`` / ``--form`` / ``--since`` / ``--until`` / ``--mode``
     override individual plan fields after the auto-plan.  The
     overrides shadow the model's extraction whenever supplied.
   - ``--skip-plan`` bypasses ``understand_query`` entirely; the
     orchestrator embeds the raw query.  Useful when the operator
     knows the question is plain English and does not want to spend
     the structured-output round-trip.
   - ``--provider`` / ``--model`` override ``LLM_PROVIDER`` /
     ``LLM_DEFAULT_MODEL`` for a single invocation (``query``) or for
     the lifetime of the REPL (``chat``).
   - ``--openrouter-provider`` (repeatable) and
     ``--openrouter-fallbacks/--no-openrouter-fallbacks`` populate
     :class:`OpenRouterRoutingHints` on the generation request.  The
     hint is consumed only by :class:`OpenRouterProvider`; the CLI
     hard-fails when either flag is supplied while the resolved
     provider is not OpenRouter, so the operator never silently loses
     a routing decision (every non-OpenRouter adapter drops the hint
     via the OpenAI-compatible base's empty ``_extra_request_kwargs``
     default).

Logging discipline:

- The raw question is Tier 3 user-generated content.  ``audit_log``
  emits a single ``cli_rag_query`` line carrying only metadata
  (provider, model, mode, language, counts) — never the question, the
  plan body, or the resolved provider key.
- :func:`redact_for_log` already wraps every internal query reference
  inside :class:`RAGOrchestrator` / :class:`RetrievalService`; this
  module never re-emits the raw question at any log level.

Output discipline mirrors the rest of the adapted CLI surface: every
operator-facing string flows through :func:`rich.markup.escape` so
retrieved text containing literal ``[red]...[/red]`` cannot repaint
operator output — the same hostile-markup defence ``cli.search`` ships.
"""

from __future__ import annotations

import queue as _queue
import threading
from datetime import UTC, datetime
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from sec_generative_search.cli._json import (
    OutputFormat,
    coerce_output_format,
    error_envelope,
    is_json,
    print_json,
)
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    DatabaseError,
    GenerationError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    SearchError,
)
from sec_generative_search.core.logging import audit_log
from sec_generative_search.core.types import (
    Citation,
    ConversationTurn,
    EmbedderStamp,
    GenerationResult,
    ProviderCapability,
    estimate_cost,
)
from sec_generative_search.database import ChromaDBClient, MetadataRegistry
from sec_generative_search.providers.factory import (
    build_embedder,
    build_llm_provider,
    default_api_key_resolver,
)
from sec_generative_search.providers.registry import (
    ProviderRegistry,
    ProviderSurface,
)
from sec_generative_search.providers.routing_hints import OpenRouterRoutingHints
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.orchestrator import RAGOrchestrator
from sec_generative_search.rag.query_understanding import (
    QueryPlan,
    understand_query,
)
from sec_generative_search.search import RetrievalService

__all__ = ["rag_app"]


console = Console()

rag_app = typer.Typer(
    name="rag",
    help="Ask grounded questions over ingested SEC filings.",
    no_args_is_help=True,
)


# Permanent admin-user id for the encrypted store.  Mirrors
# ``api.dependencies.ADMIN_USER_ID``; duplicated here so the CLI does not
# import from ``api/`` (the two surfaces are deliberately decoupled).
_ADMIN_USER_ID = "__admin__"

# Shared operator hint for an unreachable provider endpoint
# (``ProviderConnectionError``).  The dominant trigger on the CLI — the
# operator-on-host surface — is a self-hosted ``local_llm`` server that is
# not running or a mis-pointed ``LOCAL_LLM_BASE_URL``.
_LOCAL_ENDPOINT_HINT = (
    "Verify the endpoint is running and reachable (for local_llm, that the "
    "local model server is up, e.g. `ollama serve`, and LOCAL_LLM_BASE_URL is "
    "correct); retry once it recovers."
)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _print_error(
    label: str,
    message: str,
    *,
    details: str | None = None,
    hint: str | None = None,
    output: OutputFormat = OutputFormat.TEXT,
    error_code: str | None = None,
) -> None:
    """Render an error with optional details and a single hint line.

    When ``output == OutputFormat.JSON`` the document is an
    :func:`error_envelope` instead of the Rich text.  ``error_code``
    drives the machine-readable ``error`` slug (mirrors the API
    envelope discipline).  ``rag chat`` deliberately keeps ``output``
    defaulted to TEXT — the REPL surface is interactive and does not
    expose the JSON flag.
    """
    if is_json(output):
        slug = error_code or label.lower().replace(" ", "_")
        print_json(error_envelope(slug, message, hint=hint, details=details))
        return
    console.print(f"[red]{escape(label)}:[/red] {escape(message)}")
    if details:
        console.print(f"  [dim]{escape(details)}[/dim]")
    if hint:
        console.print(f"  [dim italic]Hint: {escape(hint)}[/dim italic]")


def _validate_date(value: str | None, param_name: str) -> str | None:
    """Validate ``YYYY-MM-DD`` strings at the CLI boundary.

    Same shape as ``cli.search`` / ``cli.ingest``.  Surfacing a
    :class:`typer.BadParameter` keeps the error consistent with the rest
    of the CLI.
    """
    if value is None:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise typer.BadParameter(
            f"Invalid date format for {param_name}: {value!r}. Expected YYYY-MM-DD."
        ) from None
    return value


def _coerce_mode(value: str | None) -> AnswerMode | None:
    """Lift the ``--mode`` flag onto :class:`AnswerMode` strictly.

    Unlike :meth:`AnswerMode.from_string` (which falls back to
    ``CONCISE`` on unknown values to keep the orchestrator tolerant of
    LLM output), the CLI fails closed: an unknown ``--mode`` value
    must surface as a :class:`typer.BadParameter`.  Otherwise an
    operator typing ``--mode analyitcal`` would silently receive
    ``concise`` and never see the typo.
    """
    if value is None:
        return None
    normalised = value.strip().lower()
    try:
        return AnswerMode(normalised)
    except ValueError as exc:
        valid = ", ".join(m.value for m in AnswerMode)
        raise typer.BadParameter(f"Invalid --mode: {value!r}. Expected one of: {valid}.") from exc


# ---------------------------------------------------------------------------
# Resolver chain (operator scope)
# ---------------------------------------------------------------------------


def _build_api_key_resolver(registry: MetadataRegistry):  # type: ignore[no-untyped-def]
    """Compose the operator-scope resolver chain.

    The CLI uses ``encrypted-user (ADMIN_USER_ID="__admin__") →
    admin-env (default_api_key_resolver)`` — no header tier, no session
    tier (neither exists at the terminal).

    The encrypted-user tier is only added when both
    ``DB_PERSIST_PROVIDER_CREDENTIALS=true`` *and* SQLCipher are
    configured: :class:`EncryptedCredentialStore` refuses construction
    otherwise.  A misconfigured environment is silently degraded to
    admin-env alone rather than failing closed — ``rag query`` should
    still work in Scenario A where the operator passes
    ``OPENAI_API_KEY`` directly.  The ``provider set`` command in
    the provider-management flow hard-fails when the encrypted tier is missing because
    *writing* a credential under those conditions would silently fall
    back to plaintext; *reading* through admin-env is the documented
    Scenario A path.

    Returns:
        A first-hit-wins callable matching the ``ApiKeyResolver``
        shape — plugs straight into :func:`build_llm_provider`.
    """
    # Local imports keep this module's top-level import surface
    # decoupled from the encrypted-store code path; operators on
    # admin-env only never pay the import cost.
    from sec_generative_search.core.credentials import (
        chain_resolvers,
        encrypted_user_resolver,
    )

    settings = get_settings()
    chain = []
    if settings.database.persist_provider_credentials and registry.encrypted:
        # Local import to avoid pulling sqlcipher symbols when the
        # encrypted tier is not in play.
        from sec_generative_search.database.credentials import (
            EncryptedCredentialStore,
        )

        try:
            store = EncryptedCredentialStore(registry)
            chain.append(encrypted_user_resolver(store, _ADMIN_USER_ID))
        except ConfigurationError:
            # Defensive: settings validation should make this
            # unreachable, but a hand-crafted registry could land
            # here.  Fall through to admin-env silently.
            pass

    chain.append(default_api_key_resolver)
    return chain_resolvers(*chain)


# ---------------------------------------------------------------------------
# Stack construction
# ---------------------------------------------------------------------------


def _resolve_stamp(*, output: OutputFormat = OutputFormat.TEXT) -> EmbedderStamp:
    """Compose the embedder stamp from settings + registry.

    Mirrors :mod:`cli.manage`'s ``_resolve_stamp`` so the failure
    envelope is uniform across the adapted CLI surface.
    """
    settings = get_settings()
    embedding = settings.embedding
    try:
        dim = ProviderRegistry.get_dimension(embedding.provider, embedding.model_name)
    except (KeyError, ValueError) as exc:
        _print_error(
            "Embedder configuration invalid",
            f"Cannot resolve dimension for {embedding.provider}/{embedding.model_name}.",
            details=str(exc),
            hint=(
                "Check EMBEDDING_PROVIDER and EMBEDDING_MODEL_NAME against "
                "the registry — defaults live in providers/registry.py."
            ),
            output=output,
            error_code="embedder_configuration_invalid",
        )
        raise typer.Exit(code=1) from None
    return EmbedderStamp(
        provider=embedding.provider,
        model=embedding.model_name,
        dimension=dim,
    )


def _build_retrieval(
    *,
    output: OutputFormat = OutputFormat.TEXT,
) -> tuple[RetrievalService, MetadataRegistry]:
    """Build the retrieval primitive + open the registry handle.

    The registry handle is returned so the caller can compose the
    encrypted-user resolver tier (which reuses the registry's
    connection) without re-opening SQLite/SQLCipher.
    """
    settings = get_settings()
    embedding = settings.embedding

    try:
        embedder = build_embedder(embedding)
    except ConfigurationError as exc:
        _print_error(
            "Embedder construction failed",
            exc.message,
            hint="Set the expected API-key env var for the embedding provider.",
            output=output,
            error_code="embedder_construction_failed",
        )
        raise typer.Exit(code=1) from None
    except KeyError as exc:
        _print_error(
            "Embedder unavailable",
            f"Provider {embedding.provider!r} requires additional packages.",
            details=str(exc),
            hint=(
                "Install the matching extra, e.g. "
                "`uv pip install -e '.[local-embeddings]'` for the local provider."
            ),
            output=output,
            error_code="embedder_unavailable",
        )
        raise typer.Exit(code=1) from None

    stamp = _resolve_stamp(output=output)

    try:
        chroma = ChromaDBClient(stamp)
        registry = MetadataRegistry()
    except DatabaseError as exc:
        _print_error(
            "Storage initialisation failed",
            exc.message,
            details=exc.details,
            hint=(
                "Check DB_CHROMA_PATH / DB_METADATA_DB_PATH are accessible and "
                "that the existing collection's embedder stamp matches EMBEDDING_*."
            ),
            output=output,
            error_code="storage_initialisation_failed",
        )
        raise typer.Exit(code=1) from None

    return RetrievalService(embedder=embedder, chroma_client=chroma), registry


def _build_llm(  # type: ignore[no-untyped-def]
    registry: MetadataRegistry,
    provider_name: str,
    *,
    output: OutputFormat = OutputFormat.TEXT,
):
    """Build the LLM via the operator-scope resolver chain.

    Failures map onto the same operator-facing envelopes the API uses
    so the operator never sees a stack trace for a credential / extras
    misconfiguration.
    """
    resolver = _build_api_key_resolver(registry)
    try:
        return build_llm_provider(provider_name, api_key_resolver=resolver)
    except ConfigurationError as exc:
        _print_error(
            "LLM provider key required",
            f"No API key resolved for provider '{provider_name}'.",
            details=exc.message,
            hint=(
                "Set the provider's admin-env var "
                "(e.g. OPENAI_API_KEY / ANTHROPIC_API_KEY) or register it via "
                "'sec-rag provider set' once that command lands."
            ),
            output=output,
            error_code="provider_key_required",
        )
        raise typer.Exit(code=1) from None
    except KeyError as exc:
        _print_error(
            "LLM provider unavailable",
            f"Provider {provider_name!r} requires additional packages.",
            details=str(exc),
            hint="Install the provider's optional extras and try again.",
            output=output,
            error_code="provider_unavailable",
        )
        raise typer.Exit(code=1) from None


# ---------------------------------------------------------------------------
# Plan handling
# ---------------------------------------------------------------------------


def _apply_overrides(
    plan: QueryPlan,
    *,
    ticker: list[str] | None,
    form: list[str] | None,
    since: str | None,
    until: str | None,
    mode: AnswerMode | None,
) -> QueryPlan:
    """Apply CLI flag overrides onto the plan in place.

    Each override shadows the model's extraction whenever supplied.
    Date overrides honour partial input: ``--since`` alone overrides
    the lower bound only when ``until`` is also provided (the
    dataclass field is a 2-tuple, not a pair of optionals).  Supplying
    only one of ``--since`` / ``--until`` while the plan has no
    ``date_range`` raises a clear :class:`typer.BadParameter`.
    """
    if ticker is not None:
        plan.tickers = [t.upper() for t in ticker]
    if form is not None:
        plan.form_types = [f.upper() for f in form]
    if since is not None or until is not None:
        existing = plan.date_range
        start = since if since is not None else (existing[0] if existing else None)
        end = until if until is not None else (existing[1] if existing else None)
        if start is None or end is None:
            raise typer.BadParameter(
                "--since and --until must be supplied together when no plan "
                "date_range is present.  Supply both, or omit both."
            )
        plan.date_range = (start, end)
    if mode is not None:
        plan.suggested_answer_mode = mode
    return plan


def _plan_to_dict(plan: QueryPlan) -> dict[str, Any]:
    """Lift a :class:`QueryPlan` onto the JSON wire shape.

    Mirrors :class:`api.schemas.QueryPlanSchema` exactly (allow-list
    lift, never a ``**asdict()`` splat).  ``date_range`` is a 2-tuple
    on the dataclass but lifted as a fixed-length list on the wire
    because JSON has no tuple type; ``None`` stays ``None``.
    ``suggested_answer_mode`` is the enum's lower-case value (same as
    the API surface).
    """
    return {
        "raw_query": plan.raw_query,
        "detected_language": plan.detected_language,
        "query_en": plan.query_en,
        "tickers": list(plan.tickers),
        "form_types": list(plan.form_types),
        "date_range": (
            [plan.date_range[0], plan.date_range[1]] if plan.date_range is not None else None
        ),
        "intent": plan.intent,
        "suggested_answer_mode": plan.suggested_answer_mode.value,
    }


def _citation_to_dict(citation: Citation) -> dict[str, Any]:
    """Lift one :class:`Citation` onto the JSON wire shape.

    Mirrors :class:`api.schemas.CitationSchema` exactly:
    :attr:`Citation.filing_id` is flattened into the four scalars the
    wire surface carries (``ticker`` / ``form_type`` / ``filing_date``
    / ``accession_number``), and the date is serialised as ISO.
    """
    fid = citation.filing_id
    return {
        "chunk_id": citation.chunk_id,
        "ticker": fid.ticker,
        "form_type": fid.form_type,
        "filing_date": fid.filing_date.isoformat(),
        "accession_number": fid.accession_number,
        "section_path": citation.section_path,
        "text_span": citation.text_span,
        "similarity": citation.similarity,
        "display_index": citation.display_index,
    }


def _result_to_dict(
    result: GenerationResult,
    *,
    refused: bool,
    capability: ProviderCapability,
) -> dict[str, Any]:
    """Lift a :class:`GenerationResult` onto the JSON wire shape.

    Mirrors :class:`api.schemas.RagQueryResponse` exactly, including the
    ``estimated_cost_usd`` per-request estimate (``None`` for an
    unknown-cost model). ``retrieved_chunks`` is intentionally NOT echoed
    back (same discipline as the API surface — the citations are the audit
    trail; operators wanting the full candidate set use ``sec-rag search``).
    """
    usage = result.token_usage
    return {
        "answer": result.answer,
        "citations": [_citation_to_dict(c) for c in result.citations],
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
        "streamed": result.streamed,
        "refused": refused,
    }


def _render_plan(plan: QueryPlan) -> None:
    """Render the plan as a Rich panel, mirroring the web's chip UI.

    Mode names, dates, tickers, form types render verbatim — none of
    these are user-controlled at this point (the model produces them,
    or the operator typed them as CLI flags).  ``intent`` and
    ``query_en`` flow through :func:`rich.markup.escape` defensively
    because the model could in principle emit literal brackets.
    """
    detail = Table(show_header=False, box=None, padding=(0, 2))
    detail.add_column("Key", style="bold")
    detail.add_column("Value")
    detail.add_row("Query (raw)", escape(plan.raw_query))
    detail.add_row("Query (en)", escape(plan.query_en))
    detail.add_row("Language", escape(plan.detected_language))
    detail.add_row("Mode", f"[cyan]{escape(plan.suggested_answer_mode.value)}[/cyan]")
    detail.add_row(
        "Tickers",
        f"[cyan]{escape(', '.join(plan.tickers) or '—')}[/cyan]",
    )
    detail.add_row(
        "Form types",
        f"[green]{escape(', '.join(plan.form_types) or '—')}[/green]",
    )
    if plan.date_range is not None:
        detail.add_row(
            "Date range",
            f"{escape(plan.date_range[0])} → {escape(plan.date_range[1])}",
        )
    else:
        detail.add_row("Date range", "[dim]any[/dim]")
    if plan.intent:
        detail.add_row("Intent", escape(plan.intent))
    console.print(Panel(detail, title="[bold]Query Plan[/bold]", expand=False))


# ---------------------------------------------------------------------------
# Generation result rendering
# ---------------------------------------------------------------------------


def _render_citations(citations: list[Citation]) -> None:
    """Render citations as a Rich table.

    The flat shape mirrors the API's :class:`CitationSchema`
    (``ticker / form_type / filing_date / accession_number /
    section_path / display_index / similarity``) so the operator
    visual contract aligns with the wire contract.  Every cell flows
    through :func:`rich.markup.escape` so a hostile section path or
    text span cannot repaint operator output.
    """
    if not citations:
        console.print("[yellow]No citations.[/yellow]")
        return
    table = Table(
        title="[bold]Citations[/bold]",
        border_style="dim",
        header_style="bold",
        expand=True,
    )
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Source", style="cyan", width=20, no_wrap=True)
    table.add_column("Section", style="dim", max_width=40)
    table.add_column("Similarity", width=10, justify="right")
    table.add_column("Excerpt")

    for c in citations:
        fid = c.filing_id
        source = f"{fid.ticker} {fid.form_type}\n{fid.date_str}"
        idx = str(c.display_index) if c.display_index > 0 else "—"
        excerpt = c.text_span
        if len(excerpt) > 400:
            excerpt = excerpt[:400] + "..."
        table.add_row(
            idx,
            escape(source),
            escape(c.section_path),
            f"{c.similarity:.1%}",
            escape(excerpt),
        )
    console.print(table)


def _render_result(
    result: GenerationResult,
    *,
    refused: bool,
    capability: ProviderCapability,
) -> None:
    """Render the answer panel + citations + traceability footer.

    Footer mirrors the API's :class:`RagQueryResponse` traceability
    fields (``provider``, ``model``, ``prompt_version``,
    ``token_usage``, ``estimated_cost_usd``, ``latency_seconds``) so the
    operator sees the same evidence the web UI would. The cost is an
    estimate (``—`` when the model's cost is unknown), never a final bill.
    """
    title = "[bold red]Refused[/bold red]" if refused else "[bold]Answer[/bold]"
    console.print(Panel(escape(result.answer), title=title, expand=True))

    _render_citations(result.citations)

    usage = result.token_usage
    cost = estimate_cost(usage, capability)
    cost_label = f"~${cost:.6f}" if cost is not None else "—"
    console.print(
        f"\n[dim]Provider:[/dim] [cyan]{escape(result.provider)}[/cyan]  "
        f"[dim]Model:[/dim] [green]{escape(result.model or '—')}[/green]  "
        f"[dim]Prompt:[/dim] {escape(result.prompt_version)}  "
        f"[dim]Tokens:[/dim] {usage.input_tokens}+{usage.output_tokens}"
        f"={usage.total_tokens}  "
        f"[dim]Est. cost:[/dim] {cost_label}  "
        f"[dim]Latency:[/dim] {result.latency_seconds:.2f}s"
    )


# ---------------------------------------------------------------------------
# Provider / model resolution
# ---------------------------------------------------------------------------


def _resolve_provider_and_model(
    provider_override: str | None,
    model_override: str | None,
) -> tuple[str, str]:
    """Pick provider + model from CLI overrides, with settings as fallback.

    Mirrors :func:`api.routes.rag._resolve_provider_and_model_query`'s
    precedence (request body wins, settings is the fallback) so the CLI
    and API surfaces describe the same operator mental model: an
    explicit override always beats ``LLM_PROVIDER`` / ``LLM_DEFAULT_MODEL``
    from the environment.  Empty strings on either flag fall back to the
    settings value — Typer renders ``--model ""`` for empty input, and
    silently treating that as "use the default" matches the API
    schema's ``model: str | None`` field semantics.
    """
    settings = get_settings()
    provider = (provider_override or "").strip() or settings.llm.default_provider
    model = (model_override or "").strip() or (settings.llm.default_model or "")
    return provider, model


def _resolve_routing_hints(
    provider_name: str,
    *,
    openrouter_provider: list[str] | None,
    openrouter_fallbacks: bool | None,
    output: OutputFormat = OutputFormat.TEXT,
) -> OpenRouterRoutingHints | None:
    """Build :class:`OpenRouterRoutingHints` from CLI flags, with a strict guard.

    The orchestrator forwards the hint object verbatim into
    :class:`GenerationRequest.routing_hints`; only
    :class:`OpenRouterProvider` consumes it.  Allowing the flags to land
    on a non-OpenRouter run would silently no-op (the OpenAI-compatible
    base's empty ``_extra_request_kwargs`` default drops them), which is
    a misleading UX: an operator who supplied ``--openrouter-provider
    anthropic`` against ``--provider openai`` would expect the routing
    to be honoured.  We fail closed instead.

    The :meth:`ProviderRegistry.supports_upstream_routing` query is the
    single source of truth — currently only the ``openrouter`` LLM entry
    advertises the capability, and any future meta-provider that lights
    it up would pick up the flags automatically.

    Returns ``None`` when no openrouter flags are supplied; otherwise a
    frozen :class:`OpenRouterRoutingHints` populated from the flag values.
    """
    has_any = bool(openrouter_provider) or openrouter_fallbacks is not None
    if not has_any:
        return None

    try:
        honours = ProviderRegistry.supports_upstream_routing(provider_name, ProviderSurface.LLM)
    except KeyError:
        # Unknown provider — the caller's capability probe will already
        # have raised an envelope, but be defensive here in case this
        # helper is ever invoked before that probe.
        honours = False

    if not honours:
        _print_error(
            "Invalid flag combination",
            (
                f"--openrouter-provider / --openrouter-fallbacks were supplied "
                f"but the resolved LLM provider is {provider_name!r}, which "
                "does not honour upstream-routing hints."
            ),
            hint=(
                "Re-run with --provider openrouter (the only provider that "
                "consumes these flags), or drop the --openrouter-* flags."
            ),
            output=output,
            error_code="invalid_flag_combination",
        )
        raise typer.Exit(code=1)

    # OpenRouterRoutingHints is frozen and tuple-valued; coerce the
    # mutable list[str] from Typer into a tuple for hashability.  Pass
    # the explicit None bool through verbatim so OpenRouter's own
    # default for ``allow_fallbacks`` applies when the operator did
    # not toggle ``--openrouter-fallbacks/--no-openrouter-fallbacks``.
    return OpenRouterRoutingHints(
        order=tuple(openrouter_provider or ()),
        allow_fallbacks=openrouter_fallbacks,
    )


def _format_routing_hints_audit(hints: OpenRouterRoutingHints | None) -> str:
    """Render routing hints as an audit-safe metadata fragment.

    Reports only the count of upstream slugs and the boolean toggle —
    the upstream-provider names themselves are deliberately omitted so
    operators can pivot on "was routing pinned?" without leaking which
    vendors a given run preferred (the slug list is operator-supplied
    but still a Tier-2 disclosure that telemetry pipelines rarely need).
    """
    if hints is None:
        return "none"
    return (
        f"order={len(hints.order)} "
        f"fallbacks={hints.allow_fallbacks if hints.allow_fallbacks is not None else 'default'}"
    )


# ---------------------------------------------------------------------------
# Public command — `sec-rag rag query QUESTION`
# ---------------------------------------------------------------------------


@rag_app.command("query")
def query(
    question: Annotated[
        str,
        typer.Argument(help="Natural language question to answer over ingested filings."),
    ],
    show_plan: Annotated[
        bool,
        typer.Option(
            "--show-plan",
            "-p",
            help="Run query-understanding, print the plan, and exit without generating.",
        ),
    ] = False,
    skip_plan: Annotated[
        bool,
        typer.Option(
            "--skip-plan",
            "-s",
            help=(
                "Bypass query-understanding entirely; the raw question is "
                "embedded directly.  Overrides may still be applied via "
                "--ticker / --form / --since / --until / --mode."
            ),
        ),
    ] = False,
    ticker: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-k",
            help="Override plan tickers.  Repeatable.",
        ),
    ] = None,
    form: Annotated[
        list[str] | None,
        typer.Option(
            "--form",
            "-f",
            help="Override plan form types (e.g. 10-K).  Repeatable.",
        ),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option("--since", help="Override plan date range start (YYYY-MM-DD)."),
    ] = None,
    until: Annotated[
        str | None,
        typer.Option("--until", help="Override plan date range end (YYYY-MM-DD)."),
    ] = None,
    mode: Annotated[
        str | None,
        typer.Option(
            "--mode",
            "-m",
            help="Override plan answer mode (concise / analytical / extractive / comparative).",
        ),
    ] = None,
    max_output_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-output-tokens",
            help="Cap on the answer slice (defaults to LLM_MAX_OUTPUT_TOKENS).",
            min=1,
            max=8192,
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-P",
            help=(
                "Override the LLM provider (e.g. openai / anthropic / openrouter). "
                "Defaults to LLM_PROVIDER."
            ),
        ),
    ] = None,
    llm_model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-M",
            help=(
                "Override the LLM model slug (e.g. gpt-4o, claude-haiku-4-5, "
                "openai/gpt-4o on openrouter).  Defaults to LLM_DEFAULT_MODEL "
                "or the provider's own default."
            ),
        ),
    ] = None,
    openrouter_provider: Annotated[
        list[str] | None,
        typer.Option(
            "--openrouter-provider",
            help=(
                "Preferred upstream-provider slug for OpenRouter routing "
                "(e.g. anthropic, openai).  Repeatable — earlier entries "
                "take priority.  Rejected when --provider != openrouter."
            ),
        ),
    ] = None,
    openrouter_fallbacks: Annotated[
        bool | None,
        typer.Option(
            "--openrouter-fallbacks/--no-openrouter-fallbacks",
            help=(
                "Allow OpenRouter to fall back to upstreams outside the "
                "--openrouter-provider order.  Defaults to OpenRouter's "
                "own default when unset.  Rejected when --provider != openrouter."
            ),
        ),
    ] = None,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output format: 'text' (default Rich panels) or 'json'.  "
                "JSON mode emits an allow-list lift of "
                "api.schemas.RagQueryResponse (or RagPlanResponse with "
                "--show-plan); failures render as {error, message, hint} "
                "envelopes.  The raw question is NEVER part of any envelope."
            ),
        ),
    ] = "text",
) -> None:
    """Run the full RAG pipeline non-streaming and render the answer.

    Examples:

        sec-rag rag query "What are Apple's main revenue segments?"

        sec-rag rag query "Compare AAPL and MSFT cloud strategy" \\
            --mode comparative --ticker AAPL --ticker MSFT

        sec-rag rag query "Risk factors related to AI" --show-plan

        sec-rag rag query "What is revenue?" --skip-plan --ticker AAPL

        sec-rag rag query "Risk factors" \\
            --provider openrouter --model anthropic/claude-sonnet-4-6 \\
            --openrouter-provider anthropic --no-openrouter-fallbacks

        sec-rag rag query "What is revenue?" --output json | jq '.answer'
    """
    output_format = coerce_output_format(output)

    if show_plan and skip_plan:
        _print_error(
            "Invalid flag combination",
            "--show-plan and --skip-plan are mutually exclusive.",
            output=output_format,
            error_code="invalid_flag_combination",
        )
        raise typer.Exit(code=1)

    if not question or not question.strip():
        _print_error(
            "Invalid question",
            "Question must not be empty.",
            output=output_format,
            error_code="invalid_question",
        )
        raise typer.Exit(code=1)

    _validate_date(since, "--since")
    _validate_date(until, "--until")
    effective_mode = _coerce_mode(mode)

    provider_name, model_name = _resolve_provider_and_model(provider, llm_model)

    # Reject unknown providers up front (same shape as POST /api/rag/query).
    try:
        capability = ProviderRegistry.get_capability(
            provider_name,
            ProviderSurface.LLM,
            model=model_name or None,
        )
    except KeyError as exc:
        _print_error(
            "Unknown LLM provider",
            f"{provider_name!r} is not a registered LLM provider.",
            details=str(exc),
            hint="Set LLM_PROVIDER to a registered slug (openai / anthropic / gemini / ...).",
            output=output_format,
            error_code="unknown_provider",
        )
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        _print_error(
            "Unknown LLM model",
            f"{model_name!r} is not registered for provider {provider_name!r}.",
            details=str(exc),
            hint="Unset LLM_DEFAULT_MODEL to use the provider default.",
            output=output_format,
            error_code="unknown_model",
        )
        raise typer.Exit(code=1) from None

    # Build routing hints (or hard-fail on a provider mismatch) BEFORE
    # touching the storage layer — a misconfigured flag combination is
    # a boundary error, not a runtime error, and surfacing it before any
    # ChromaDB / SQLite open keeps the failure cheap and obvious.
    routing_hints = _resolve_routing_hints(
        provider_name,
        openrouter_provider=openrouter_provider,
        openrouter_fallbacks=openrouter_fallbacks,
        output=output_format,
    )

    retrieval, registry = _build_retrieval(output=output_format)
    llm = _build_llm(registry, provider_name, output=output_format)

    # --- Plan resolution ------------------------------------------------------
    if skip_plan:
        plan = QueryPlan(raw_query=question)
    else:
        try:
            plan = understand_query(
                question,
                llm=llm,
                model=model_name,
                structured_output_supported=capability.structured_output,
            )
        except ProviderAuthError:
            _print_error(
                "Provider unauthorised",
                "The upstream provider rejected the supplied API key.",
                hint=(
                    "Verify or rotate the provider key for "
                    f"{provider_name!r}; do not retry until corrected."
                ),
                output=output_format,
                error_code="provider_unauthorized",
            )
            raise typer.Exit(code=1) from None
        except (ProviderRateLimitError, ProviderTimeoutError):
            _print_error(
                "Provider unavailable",
                "The upstream provider is rate-limited or timed out.",
                hint="Retry after a short backoff; do not rotate the key.",
                output=output_format,
                error_code="provider_unavailable",
            )
            raise typer.Exit(code=1) from None
        except ProviderConnectionError:
            _print_error(
                "Provider unavailable",
                "The upstream provider endpoint could not be reached.",
                hint=_LOCAL_ENDPOINT_HINT,
                output=output_format,
                error_code="provider_unavailable",
            )
            raise typer.Exit(code=1) from None
        except ProviderError as exc:
            _print_error(
                "Provider error",
                "The upstream provider returned an error during query understanding.",
                details=type(exc).__name__,
                hint="Inspect the audit log; do not rotate the key on a non-auth error.",
                output=output_format,
                error_code="provider_error",
            )
            raise typer.Exit(code=1) from None

    plan = _apply_overrides(
        plan,
        ticker=ticker,
        form=form,
        since=since,
        until=until,
        mode=effective_mode,
    )

    if show_plan:
        if is_json(output_format):
            # Mirror ``api.schemas.RagPlanResponse``: ``{plan, provider,
            # model}``.  ``model`` is the resolved slug (empty string
            # falls back to the provider default) so the JSON consumer
            # can pin to the exact run later.
            print_json(
                {
                    "plan": _plan_to_dict(plan),
                    "provider": provider_name,
                    "model": model_name or "",
                }
            )
            return
        _render_plan(plan)
        return

    # --- Generation ----------------------------------------------------------
    orchestrator = RAGOrchestrator(retrieval=retrieval, llm=llm)
    try:
        result = orchestrator.generate(
            plan,
            mode=effective_mode,
            model=model_name or None,
            max_output_tokens=max_output_tokens,
            prefer_structured_output=capability.structured_output,
            routing_hints=routing_hints,
        )
    except ProviderAuthError:
        _print_error(
            "Provider unauthorised",
            "The upstream provider rejected the supplied API key during generation.",
            hint=(
                "Verify or rotate the provider key for "
                f"{provider_name!r}; do not retry until corrected."
            ),
            output=output_format,
            error_code="provider_unauthorized",
        )
        raise typer.Exit(code=1) from None
    except (ProviderRateLimitError, ProviderTimeoutError):
        _print_error(
            "Provider unavailable",
            "The upstream provider is rate-limited or timed out.",
            hint="Retry after a short backoff; do not rotate the key.",
            output=output_format,
            error_code="provider_unavailable",
        )
        raise typer.Exit(code=1) from None
    except ProviderConnectionError:
        _print_error(
            "Provider unavailable",
            "The upstream provider endpoint could not be reached.",
            hint=_LOCAL_ENDPOINT_HINT,
            output=output_format,
            error_code="provider_unavailable",
        )
        raise typer.Exit(code=1) from None
    except ProviderError as exc:
        _print_error(
            "Provider error",
            "The upstream provider returned an error during generation.",
            details=type(exc).__name__,
            hint="Inspect the audit log; do not rotate the key on a non-auth error.",
            output=output_format,
            error_code="provider_error",
        )
        raise typer.Exit(code=1) from None
    except GenerationError as exc:
        _print_error(
            "Generation failed",
            "The orchestrator could not assemble a valid answer.",
            details=exc.message,
            hint="Retry the request; if it persists, switch model or provider.",
            output=output_format,
            error_code="generation_failed",
        )
        raise typer.Exit(code=1) from None
    except SearchError as exc:
        _print_error(
            "Retrieval failed",
            exc.message,
            details=exc.details,
            hint=(
                "Check that filings have been ingested (`sec-rag manage status`) "
                "and that --since / --until are valid YYYY-MM-DD values."
            ),
            output=output_format,
            error_code="retrieval_failed",
        )
        raise typer.Exit(code=1) from None
    except DatabaseError as exc:
        _print_error(
            "Database failure",
            exc.message,
            details=exc.details,
            hint="Check DB_CHROMA_PATH is readable and the collection is intact.",
            output=output_format,
            error_code="database_error",
        )
        raise typer.Exit(code=1) from None

    refused = not result.retrieved_chunks and not result.citations

    # Metadata-only audit line — never the question, plan, or key.
    # ``or_hints`` reports only counts / the boolean toggle so the
    # routing decision is auditable without leaking the upstream slugs
    # (operators can pivot on whether routing was active without seeing
    # which vendors a given run preferred).
    audit_log(
        "cli_rag_query",
        endpoint="cli rag query",
        detail=(
            f"provider={provider_name} model={model_name or '<provider default>'} "
            f"lang={plan.detected_language} tickers={len(plan.tickers)} "
            f"forms={len(plan.form_types)} "
            f"mode={(effective_mode or plan.suggested_answer_mode).value} "
            f"prompt_version={result.prompt_version} "
            f"chunks={len(result.retrieved_chunks)} "
            f"citations={len(result.citations)} refused={refused} "
            f"or_hints={_format_routing_hints_audit(routing_hints)}"
        ),
    )

    if is_json(output_format):
        # Mirror ``api.schemas.RagQueryResponse`` exactly.  The audit
        # log above carries the same metadata footprint as the API
        # surface (provider / model / counts only); the JSON document
        # adds the answer + citations because the wire response on
        # ``/api/rag/query`` does the same.
        print_json(_result_to_dict(result, refused=refused, capability=capability))
        return

    _render_result(result, refused=refused, capability=capability)


# ---------------------------------------------------------------------------
# `sec-rag rag chat` — interactive REPL
# ---------------------------------------------------------------------------


# Sentinel terminator pushed onto the stream queue by the producer thread
# in :func:`_run_chat_stream`.  Avoids confusing a legitimate ``None``
# delta with the end-of-stream marker.
_STREAM_SENTINEL = object()

# Poll interval for the consumer's blocking queue read.  Short enough to
# pick up a SIGINT promptly (CPython only delivers signal handlers on
# bytecode boundaries — a long ``queue.get(timeout=…)`` would defer
# Ctrl-C cancellation to the next timeout fire), long enough that the
# busy loop does not eat measurable CPU.
_STREAM_POLL_SECONDS = 0.1


# Map a streaming-side exception to ``(label, message, hint)``.  Same
# ladder as the non-streaming ``rag query`` exception map so an operator
# sees the same envelope shape whether the failure happens during
# ``generate`` or ``generate_stream``.  Unmapped exceptions surface as
# a generic ``internal_error`` envelope — never echo the exception text
# (it routinely carries provider URLs / SQL / paths).
def _classify_stream_error(exc: BaseException, provider_name: str) -> tuple[str, str, str | None]:
    """Return ``(label, message, hint)`` for the REPL error envelope.

    Mirrors :func:`api.routes.rag._classify_stream_exception` so the CLI
    and the SSE route emit byte-identical operator-facing shapes for
    the same upstream failure.
    """
    if isinstance(exc, ProviderAuthError):
        return (
            "Provider unauthorised",
            "The upstream provider rejected the supplied API key.",
            (
                "Verify or rotate the provider key for "
                f"{provider_name!r}; do not retry until corrected."
            ),
        )
    if isinstance(exc, (ProviderRateLimitError, ProviderTimeoutError)):
        return (
            "Provider unavailable",
            "The upstream provider is rate-limited or timed out.",
            "Retry after a short backoff; do not rotate the key.",
        )
    if isinstance(exc, ProviderConnectionError):
        return (
            "Provider unavailable",
            "The upstream provider endpoint could not be reached.",
            _LOCAL_ENDPOINT_HINT,
        )
    if isinstance(exc, ProviderError):
        return (
            "Provider error",
            "The upstream provider returned an error during generation.",
            "Inspect the audit log; do not rotate the key on a non-auth error.",
        )
    if isinstance(exc, GenerationError):
        return (
            "Generation failed",
            "The orchestrator could not assemble a valid answer.",
            "Retry the question; if the failure persists, switch model or provider.",
        )
    if isinstance(exc, SearchError):
        return (
            "Retrieval failed",
            "Retrieval could not service the question.",
            "Check `sec-rag manage status` and the --since / --until values.",
        )
    if isinstance(exc, DatabaseError):
        return (
            "Database failure",
            "The storage layer is unavailable.",
            "Check DB_CHROMA_PATH is readable and the collection is intact.",
        )
    return (
        "Stream error",
        "An unexpected error occurred while streaming the answer.",
        "Retry the question; if the failure persists, contact the operator.",
    )


def _run_chat_stream(
    orchestrator: Any,
    plan: QueryPlan,
    *,
    mode: AnswerMode | None,
    model: str | None,
    max_output_tokens: int | None,
    history: list[ConversationTurn] | None,
    prefer_structured_output: bool,
    provider_name: str,
    routing_hints: OpenRouterRoutingHints | None = None,
) -> tuple[GenerationResult | None, bool]:
    """Consume ``orchestrator.generate_stream`` interactively.

    Producer / consumer split:

    - The orchestrator's ``generate_stream`` is a *synchronous*
      generator that blocks on the LLM SDK.  A daemon producer thread
      iterates it and pushes :class:`StreamEvent` instances (or an
      exception) onto a :class:`queue.Queue`.
    - The main thread (consumer) polls the queue with a short timeout
      so that a Python signal handler running on the main thread
      (``SIGINT`` → :class:`KeyboardInterrupt`) can interrupt the wait
      promptly and cancel the in-flight stream.

    Cancellation semantics:

    - On the first Ctrl-C the consumer sets ``cancel_event`` (the
      producer checks it before pushing the next event), prints a
      single cancellation notice, and returns ``(None, True)``.  The
      caller (the REPL loop) then returns to the prompt without
      committing the cancelled turn to history.
    - The producer thread is a daemon: if it is mid-LLM-call when the
      consumer abandons it, the SDK call eventually returns and the
      thread exits naturally.  We do not try to forcibly terminate
      Python threads — there is no safe primitive for that in CPython.

    Error semantics:

    - A producer-side exception is forwarded onto the queue.  The
      consumer reads it, drains the queue to the sentinel, and renders
      a single operator-facing envelope via
      :func:`_classify_stream_error`.  Returns ``(None, False)``.
    - The exception text is never echoed verbatim — the envelope text
      is hand-tuned per exception class (same discipline as the SSE
      route's ``_classify_stream_exception``).

    Returns:
        ``(result, interrupted)`` where ``result`` is the final
        :class:`GenerationResult` on success, ``None`` on cancel or
        error; ``interrupted`` is True iff the user hit Ctrl-C during
        streaming.
    """
    stream_queue: _queue.Queue = _queue.Queue()
    cancel_event = threading.Event()

    def producer() -> None:
        try:
            for event in orchestrator.generate_stream(
                plan,
                mode=mode,
                model=model,
                max_output_tokens=max_output_tokens,
                history=history,
                prefer_structured_output=prefer_structured_output,
                routing_hints=routing_hints,
            ):
                if cancel_event.is_set():
                    # Stop pushing further events; the consumer has
                    # already returned to the REPL.  The thread itself
                    # continues to drain the generator naturally so the
                    # provider connection / token counter can finalise.
                    break
                stream_queue.put(event)
        except BaseException as exc:
            stream_queue.put(exc)
        finally:
            stream_queue.put(_STREAM_SENTINEL)

    threading.Thread(
        target=producer,
        name="rag-chat-producer",
        daemon=True,
    ).start()

    final: GenerationResult | None = None
    error: BaseException | None = None
    streamed_any = False

    try:
        while True:
            try:
                item = stream_queue.get(timeout=_STREAM_POLL_SECONDS)
            except _queue.Empty:
                continue
            if item is _STREAM_SENTINEL:
                break
            if isinstance(item, BaseException):
                error = item
                # Drain any trailing events so the producer can exit.
                while True:
                    nxt = stream_queue.get()
                    if nxt is _STREAM_SENTINEL:
                        break
                break
            # StreamEvent: delta and/or final.
            if getattr(item, "delta", None):
                # Stream deltas as plain text.  ``escape`` neutralises
                # any literal ``[red]...[/red]`` in the model output so
                # hostile retrieved text cannot repaint operator
                # output — same discipline as the rest of the CLI.
                console.print(escape(item.delta), end="", markup=True, highlight=False)
                streamed_any = True
            if getattr(item, "final", None) is not None:
                final = item.final
    except KeyboardInterrupt:
        cancel_event.set()
        if streamed_any:
            console.print()  # newline after partial stream
        console.print("[yellow]Cancelled. Type your next question, /clear, or /exit.[/yellow]")
        return None, True

    if streamed_any:
        console.print()  # newline after the last delta

    if error is not None:
        label, message, hint = _classify_stream_error(error, provider_name)
        _print_error(label, message, hint=hint)
        return None, False

    return final, False


def _render_chat_traceability(result: GenerationResult) -> None:
    """Render the per-turn traceability footer.

    Mirrors :func:`_render_result`'s footer but skipping the answer
    panel — the answer was already streamed.  Citations are still
    rendered here so the operator sees them attached to the streamed
    answer rather than only in the final-event payload.
    """
    _render_citations(result.citations)

    usage = result.token_usage
    console.print(
        f"[dim]Provider:[/dim] [cyan]{escape(result.provider)}[/cyan]  "
        f"[dim]Model:[/dim] [green]{escape(result.model or '—')}[/green]  "
        f"[dim]Prompt:[/dim] {escape(result.prompt_version)}  "
        f"[dim]Tokens:[/dim] {usage.input_tokens}+{usage.output_tokens}"
        f"={usage.total_tokens}  "
        f"[dim]Latency:[/dim] {result.latency_seconds:.2f}s"
    )


# REPL slash commands.  Kept as a small fixed set; ``rag chat`` is not a
# scripting surface — operators with complex flows should use
# ``rag query`` or the HTTP API.
_REPL_EXIT_COMMANDS = frozenset({"/exit", "/quit"})
_REPL_CLEAR_COMMAND = "/clear"
_REPL_HELP_COMMAND = "/help"


def _print_chat_banner() -> None:
    console.print("[bold]sec-rag rag chat[/bold] — ask questions over ingested SEC filings.")
    console.print(
        "  [dim]Commands:[/dim] [cyan]/clear[/cyan] (drop history)  "
        "[cyan]/help[/cyan]  [cyan]/exit[/cyan]  "
        "[dim](Ctrl-C cancels in-flight answer; Ctrl-D or second Ctrl-C exits.)[/dim]"
    )


def _print_chat_help() -> None:
    console.print(
        "[bold]Slash commands:[/bold] "
        "[cyan]/clear[/cyan] drop conversation history  |  "
        "[cyan]/help[/cyan] this message  |  "
        "[cyan]/exit[/cyan] / [cyan]/quit[/cyan] leave the REPL"
    )


@rag_app.command("chat")
def chat(
    show_plan: Annotated[
        bool,
        typer.Option(
            "--show-plan",
            "-p",
            help=(
                "Print the resolved QueryPlan before each turn's stream "
                "(generation still runs — unlike `rag query --show-plan` "
                "which exits)."
            ),
        ),
    ] = False,
    skip_plan: Annotated[
        bool,
        typer.Option(
            "--skip-plan",
            "-s",
            help=(
                "Bypass query-understanding for every turn; the raw "
                "question is embedded directly.  Session-scope overrides "
                "(--ticker / --form / --since / --until / --mode) still apply."
            ),
        ),
    ] = False,
    ticker: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-k",
            help="Session-wide ticker override.  Repeatable.",
        ),
    ] = None,
    form: Annotated[
        list[str] | None,
        typer.Option(
            "--form",
            "-f",
            help="Session-wide form-type override (e.g. 10-K).  Repeatable.",
        ),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option("--since", help="Session-wide date range start (YYYY-MM-DD)."),
    ] = None,
    until: Annotated[
        str | None,
        typer.Option("--until", help="Session-wide date range end (YYYY-MM-DD)."),
    ] = None,
    mode: Annotated[
        str | None,
        typer.Option(
            "--mode",
            "-m",
            help=(
                "Session-wide answer mode override "
                "(concise / analytical / extractive / comparative)."
            ),
        ),
    ] = None,
    max_output_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-output-tokens",
            help="Cap on the answer slice for every turn.",
            min=1,
            max=8192,
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-P",
            help=(
                "Override the LLM provider for every turn in this session. "
                "Defaults to LLM_PROVIDER."
            ),
        ),
    ] = None,
    llm_model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-M",
            help=(
                "Override the LLM model slug for every turn in this session. "
                "Defaults to LLM_DEFAULT_MODEL or the provider's own default."
            ),
        ),
    ] = None,
    openrouter_provider: Annotated[
        list[str] | None,
        typer.Option(
            "--openrouter-provider",
            help=(
                "Preferred upstream-provider slug for OpenRouter routing. "
                "Repeatable.  Rejected when --provider != openrouter."
            ),
        ),
    ] = None,
    openrouter_fallbacks: Annotated[
        bool | None,
        typer.Option(
            "--openrouter-fallbacks/--no-openrouter-fallbacks",
            help=(
                "Allow OpenRouter to fall back to upstreams outside the "
                "--openrouter-provider order.  Rejected when "
                "--provider != openrouter."
            ),
        ),
    ] = None,
) -> None:
    """Open an interactive RAG chat over ingested SEC filings.

    Each turn streams the answer via
    :meth:`RAGOrchestrator.generate_stream` and renders citations
    afterwards.  Conversation history is kept **in process memory only**
    (``ConversationTurn`` is never persisted — chat history persistence
    is out of scope by design).

    Examples:

        sec-rag rag chat

        sec-rag rag chat --ticker AAPL --mode analytical

        sec-rag rag chat --skip-plan
    """
    if show_plan and skip_plan:
        _print_error(
            "Invalid flag combination",
            "--show-plan and --skip-plan are mutually exclusive.",
        )
        raise typer.Exit(code=1)

    _validate_date(since, "--since")
    _validate_date(until, "--until")
    effective_mode = _coerce_mode(mode)

    provider_name, model_name = _resolve_provider_and_model(provider, llm_model)

    # Pre-stack validation mirrors `rag query`: reject unknown
    # providers / models before we open ChromaDB or the registry so
    # misconfiguration is one envelope instead of a cascade.
    try:
        capability = ProviderRegistry.get_capability(
            provider_name,
            ProviderSurface.LLM,
            model=model_name or None,
        )
    except KeyError as exc:
        _print_error(
            "Unknown LLM provider",
            f"{provider_name!r} is not a registered LLM provider.",
            details=str(exc),
            hint="Set LLM_PROVIDER to a registered slug (openai / anthropic / gemini / ...).",
        )
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        _print_error(
            "Unknown LLM model",
            f"{model_name!r} is not registered for provider {provider_name!r}.",
            details=str(exc),
            hint="Unset LLM_DEFAULT_MODEL to use the provider default.",
        )
        raise typer.Exit(code=1) from None

    # Session-scope routing hints — built once, applied to every turn.
    # The hard-fail on provider mismatch fires before any storage open
    # so the operator never opens an SQLite handle on a doomed session.
    routing_hints = _resolve_routing_hints(
        provider_name,
        openrouter_provider=openrouter_provider,
        openrouter_fallbacks=openrouter_fallbacks,
    )

    retrieval, registry = _build_retrieval()
    llm = _build_llm(registry, provider_name)
    orchestrator = RAGOrchestrator(retrieval=retrieval, llm=llm)

    # In-memory history.  Never persisted, never logged, cleared on
    # ``/clear`` and dropped entirely when the process exits.
    history: list[ConversationTurn] = []

    _print_chat_banner()

    while True:
        # Read one prompt line.  Ctrl-D (EOF) and Ctrl-C at the prompt
        # both exit cleanly — the second-Ctrl-C-exits-130 contract is
        # encoded by Click's outer ``main()`` wrapper which converts
        # the raised :class:`KeyboardInterrupt` to ``sys.exit(130)``.
        try:
            question = console.input("[bold cyan]>[/bold cyan] ")
        except EOFError:
            console.print()
            return
        except KeyboardInterrupt:
            # Re-raise so the outer ``main()`` wrapper converts to
            # POSIX exit 130 (cli/main.py).
            console.print()
            raise

        question = question.strip()
        if not question:
            continue

        if question in _REPL_EXIT_COMMANDS:
            console.print("[dim]Bye.[/dim]")
            return
        if question == _REPL_CLEAR_COMMAND:
            cleared = len(history)
            history.clear()
            console.print(
                f"[dim]History cleared ({cleared} turn{'s' if cleared != 1 else ''}).[/dim]"
            )
            continue
        if question == _REPL_HELP_COMMAND:
            _print_chat_help()
            continue
        # Any other ``/...`` token is rejected — typing a literal slash
        # word as a real question is exceedingly unusual, and silently
        # passing it through risks confusing operators who fat-fingered
        # a slash command.
        if question.startswith("/"):
            console.print(
                f"[yellow]Unknown command:[/yellow] {escape(question)}.  "
                f"Type [cyan]/help[/cyan] for the command list."
            )
            continue

        # ---- Plan resolution per turn ------------------------------------
        if skip_plan:
            plan = QueryPlan(raw_query=question)
        else:
            try:
                plan = understand_query(
                    question,
                    llm=llm,
                    model=model_name,
                    structured_output_supported=capability.structured_output,
                )
            except ProviderAuthError:
                _print_error(
                    "Provider unauthorised",
                    "The upstream provider rejected the supplied API key.",
                    hint=(
                        "Verify or rotate the provider key for "
                        f"{provider_name!r}; do not retry until corrected."
                    ),
                )
                continue
            except (ProviderRateLimitError, ProviderTimeoutError):
                _print_error(
                    "Provider unavailable",
                    "The upstream provider is rate-limited or timed out.",
                    hint="Retry after a short backoff; do not rotate the key.",
                )
                continue
            except ProviderConnectionError:
                _print_error(
                    "Provider unavailable",
                    "The upstream provider endpoint could not be reached.",
                    hint=_LOCAL_ENDPOINT_HINT,
                )
                continue
            except ProviderError as exc:
                _print_error(
                    "Provider error",
                    "The upstream provider returned an error during query understanding.",
                    details=type(exc).__name__,
                    hint="Inspect the audit log; do not rotate the key on a non-auth error.",
                )
                continue

        try:
            plan = _apply_overrides(
                plan,
                ticker=ticker,
                form=form,
                since=since,
                until=until,
                mode=effective_mode,
            )
        except typer.BadParameter as exc:
            # Per-turn override violation — surface, do not exit the
            # REPL.  The flag combination cannot change between turns
            # in this session, so the operator must Ctrl-C out and
            # re-launch with corrected flags; the diagnostic is the
            # same one ``rag query`` raises.
            _print_error("Invalid override", str(exc))
            continue

        if show_plan:
            _render_plan(plan)

        # ---- Streaming generation ---------------------------------------
        result, interrupted = _run_chat_stream(
            orchestrator,
            plan,
            mode=effective_mode,
            model=model_name or None,
            max_output_tokens=max_output_tokens,
            history=history or None,
            prefer_structured_output=capability.structured_output,
            provider_name=provider_name,
            routing_hints=routing_hints,
        )

        if interrupted or result is None:
            # Cancelled or upstream error.  Do NOT commit the turn to
            # conversation history — a half-completed turn would
            # pollute the next prompt's history block.
            continue

        _render_chat_traceability(result)

        refused = not result.retrieved_chunks and not result.citations

        # Metadata-only audit line.  Never the question, the plan body,
        # or the resolved provider key.  Mirrors ``cli_rag_query`` so
        # downstream log shipping can treat both surfaces identically.
        audit_log(
            "cli_rag_chat",
            endpoint="cli rag chat",
            detail=(
                f"provider={provider_name} model={model_name or '<provider default>'} "
                f"lang={plan.detected_language} tickers={len(plan.tickers)} "
                f"forms={len(plan.form_types)} "
                f"mode={(effective_mode or plan.suggested_answer_mode).value} "
                f"prompt_version={result.prompt_version} "
                f"chunks={len(result.retrieved_chunks)} "
                f"citations={len(result.citations)} "
                f"history_turns={len(history)} refused={refused} "
                f"or_hints={_format_routing_hints_audit(routing_hints)}"
            ),
        )

        # Commit the turn.  ``retrieval_results`` are kept for
        # traceability on the audit record; they MUST NOT re-enter a
        # later prompt — :meth:`RAGOrchestrator._render_history` only
        # renders ``Q:/A:`` pairs, never chunks.
        history.append(
            ConversationTurn(
                query=question,
                retrieval_results=list(result.retrieved_chunks),
                generation_result=result,
                timestamp=datetime.now(UTC),
            )
        )
