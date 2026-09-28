"""Synchronous CLI wrapper over :class:`RetrievalService`.

Single operator-facing subcommand:

- ``sec-rag search QUERY`` — run a single-query retrieval and render the
  ranked hits in a Rich table.

Trust model and load-bearing rules:

1. The embedder is built via :func:`build_embedder` (the sole
   construction seam).  Direct adapter instantiation is forbidden here
   for the same reason it is forbidden in :mod:`cli.ingest` — every
   wiring site must route through the resolver so the credential seam
   stays consistent across surfaces.
2. The :class:`ChromaDBClient` is sealed with the same
   ``(provider, model, dimension)`` triple resolved from
   :class:`ProviderRegistry`.  Opening a stamped collection is the only
   way :class:`RetrievalService` will accept the client — the seal is
   the embedding-identity invariant.
3. The CLI runs at operator scope: no auth gate, no per-session EDGAR
    identity, no rate limit, no access-log redaction.  Distributing CLI
    access to team users is unsupported.
4. ISO date filters validate at the boundary so a malformed value
   (e.g. ``"2024-13-99"``) does not slip into ChromaDB's integer
   ``filing_date_int`` coercion path.  :class:`RetrievalService`
   re-validates defensively; failing fast here keeps the error
   attributable to the caller.

Output discipline mirrors the rest of the adapted CLI surface:
operator-facing strings flow through :func:`rich.markup.escape` so
incidental square brackets (section paths like ``"[redacted]"``,
ticker punctuation, install hints) survive Rich's markup parser.

Logging discipline: the query is Tier 3 user-generated data and never
reaches the operator log unredacted — :class:`RetrievalService` honours
``LOG_REDACT_QUERIES`` via :func:`redact_for_log`.  This module never
calls :func:`logger.info` / :func:`audit_log` directly with the query.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from sec_generative_search.cli._common import print_error, resolve_stamp, validate_date
from sec_generative_search.cli._json import (
    OutputFormat,
    coerce_output_format,
    is_json,
    print_json,
)
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    DatabaseError,
    ProviderError,
    SearchError,
)
from sec_generative_search.core.types import RetrievalResult
from sec_generative_search.database import ChromaDBClient
from sec_generative_search.providers.factory import build_embedder
from sec_generative_search.search import RetrievalService

__all__ = ["search"]


console = Console()

# Display caps for the rendered table.  Mirrors the legacy ergonomics —
# section paths and content previews are heavily truncated so a single
# wide chunk never blows up the operator terminal.
_CONTENT_PREVIEW_LIMIT = 1000
_SECTION_PATH_LIMIT = 500


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _similarity_text(similarity: float) -> Text:
    """Colour-coded similarity percentage.

    Green for >= 40%, yellow for >= 25%, dim otherwise.  These bands
    reflect typical cosine ranges from sentence-level embeddings on SEC
    text — exact tuning is operator-visual; absolute scores are not
    commensurable across embedders so do not over-fit.
    """
    pct = f"{similarity:.1%}"
    if similarity >= 0.40:
        return Text(pct, style="bold green")
    if similarity >= 0.25:
        return Text(pct, style="yellow")
    return Text(pct, style="dim")


# ---------------------------------------------------------------------------
# Service construction
# ---------------------------------------------------------------------------


def _hit_to_dict(result: RetrievalResult) -> dict[str, Any]:
    """Lift one :class:`RetrievalResult` onto the JSON wire shape.

    Field selection mirrors :class:`api.schemas.SearchHit` exactly
    (allow-list lift, never a ``**asdict()`` splat) so a future field
    addition on :class:`RetrievalResult` does not silently leak onto
    the operator surface.  ``content_type`` is the enum's lower-case
    value to match the API; ``section_boundaries`` is materialised as
    a list (the dataclass field is a sequence).
    """
    return {
        "chunk_id": result.chunk_id,
        "content": result.content,
        "path": result.path,
        "content_type": result.content_type.value,
        "ticker": result.ticker,
        "form_type": result.form_type,
        "filing_date": result.filing_date,
        "accession_number": result.accession_number,
        "similarity": result.similarity,
        "rerank_score": result.rerank_score,
        "token_count": result.token_count,
        "section_boundaries": list(result.section_boundaries),
    }


def _build_service(*, output: OutputFormat = OutputFormat.TEXT) -> RetrievalService:
    """Construct a stamp-sealed :class:`RetrievalService`.

    Failures surface as single operator-facing envelopes — the caller
    never sees a stack trace for a misconfigured environment.  The
    chain is identical to :mod:`cli.ingest`'s ``_build_pipeline``:

    1. Resolve ``(provider, model, dimension)`` from the registry.
    2. Build the embedder through :func:`build_embedder` (sole seam).
    3. Open :class:`ChromaDBClient` under the resulting stamp; the
       client's ``_verify_stamp`` refuses any mismatch.

    The :class:`MetadataRegistry` is not opened — search is a read-only
    path over ChromaDB and never touches SQLite directly.
    """
    settings = get_settings()
    embedding = settings.embedding

    stamp = resolve_stamp(embedding.provider, embedding.model_name, output=output)

    try:
        embedder = build_embedder(embedding)
    except ConfigurationError as exc:
        print_error(
            "Embedder construction failed",
            exc.message,
            hint="Set the expected API-key env var for this provider.",
            output=output,
            error_code="embedder_construction_failed",
        )
        raise typer.Exit(code=1) from None
    except KeyError as exc:
        print_error(
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

    try:
        chroma = ChromaDBClient(stamp)
    except DatabaseError as exc:
        print_error(
            "Storage initialisation failed",
            exc.message,
            details=exc.details,
            hint=(
                "Check DB_CHROMA_PATH is readable and that the existing "
                "collection's embedder stamp matches EMBEDDING_*."
            ),
            output=output,
            error_code="storage_initialisation_failed",
        )
        raise typer.Exit(code=1) from None

    return RetrievalService(embedder=embedder, chroma_client=chroma)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_results(results: list[RetrievalResult], query: str) -> None:
    """Render the ranked hits in a Rich table.

    Mirrors the legacy presentation (similarity badge, source column,
    section path, truncated preview) so the operator visual contract is
    preserved across the rewrite.  ``rerank_score`` is intentionally
    omitted from the table — no reranker is bound in this CLI surface
    and surfacing a permanently-empty column would mislead operators.
    """
    console.print(
        f"\n[bold]Found {len(results)} result(s)[/bold] for: [italic]{escape(query)}[/italic]\n"
    )

    table = Table(show_lines=True, expand=True, border_style="dim")
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Similarity", width=10, justify="right")
    table.add_column("Source", style="cyan", width=20, no_wrap=True)
    table.add_column("Section", style="dim", max_width=30)
    table.add_column("Content")

    for i, result in enumerate(results, 1):
        source = f"{result.ticker} {result.form_type}"
        if result.filing_date:
            source += f"\n{result.filing_date}"

        content = result.content
        if len(content) > _CONTENT_PREVIEW_LIMIT:
            content = content[:_CONTENT_PREVIEW_LIMIT] + "..."

        section = result.path
        if len(section) > _SECTION_PATH_LIMIT:
            section = section[:_SECTION_PATH_LIMIT] + "..."

        table.add_row(
            str(i),
            _similarity_text(result.similarity),
            escape(source),
            escape(section),
            escape(content),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Public command
# ---------------------------------------------------------------------------


def search(
    query: Annotated[str, typer.Argument(help="Natural language search query.")],
    top: Annotated[
        int | None,
        typer.Option("--top", "-t", help="Number of results to return.", min=1),
    ] = None,
    ticker: Annotated[
        list[str] | None,
        typer.Option("--ticker", "-k", help="Filter by ticker symbol(s). Repeat for multiple."),
    ] = None,
    form: Annotated[
        list[str] | None,
        typer.Option("--form", "-f", help="Filter by form type(s). Repeat for multiple."),
    ] = None,
    accession: Annotated[
        list[str] | None,
        typer.Option(
            "--accession",
            "-a",
            help="Restrict search to specific filing(s) by accession number.",
        ),
    ] = None,
    start_date: Annotated[
        str | None,
        typer.Option(
            "--start-date",
            help="Filter results to filings on or after this date (YYYY-MM-DD).",
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(
            "--end-date",
            help="Filter results to filings on or before this date (YYYY-MM-DD).",
        ),
    ] = None,
    max_per_section: Annotated[
        int | None,
        typer.Option(
            "--max-per-section",
            help=("Cap chunks per section path; 0 disables. Omit to use SEARCH_MAX_PER_SECTION."),
            min=0,
        ),
    ] = None,
    max_per_filing: Annotated[
        int | None,
        typer.Option(
            "--max-per-filing",
            help=(
                "Cap chunks per filing (accession number); 0 disables. "
                "Omit to use SEARCH_MAX_PER_FILING."
            ),
            min=0,
        ),
    ] = None,
    rerank_over_fetch: Annotated[
        int | None,
        typer.Option(
            "--rerank-over-fetch",
            help=(
                "Reranker over-fetch multiplier (top_k * factor); 1 disables. "
                "Only active when a reranker is bound. "
                "Omit to use SEARCH_RERANK_OVER_FETCH_FACTOR."
            ),
            min=1,
        ),
    ] = None,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output format: 'text' (default Rich table) or 'json' "
                "(single-document allow-list lift of api.schemas.SearchResponse). "
                "JSON mode suppresses status spinners and renders failures as "
                "{error, message, hint} envelopes."
            ),
        ),
    ] = "text",
) -> None:
    """Search ingested SEC filings with a natural-language query.

    Examples:

        sec-rag search "risk factors related to supply chain"

        sec-rag search "revenue recognition" -t 10 -k AAPL

        sec-rag search "revenue" -k AAPL -k MSFT

        sec-rag search "liquidity" -f 10-Q

        sec-rag search "debt covenants" -a 0000320193-23-000106

        sec-rag search "revenue" --start-date 2023-01-01 --end-date 2023-12-31

        sec-rag search "supply chain risk" --output json | jq '.hits[0]'
    """
    output_format = coerce_output_format(output)

    # Reject empty / whitespace-only queries before constructing any
    # storage — the service would raise SearchError anyway, but failing
    # here saves an embedder build for a known-bad invocation.
    if not query or not query.strip():
        print_error(
            "Invalid query",
            "Query must not be empty.",
            output=output_format,
            error_code="invalid_query",
        )
        raise typer.Exit(code=1)

    validate_date(start_date, "--start-date")
    validate_date(end_date, "--end-date")

    # Normalise filters to uppercase to match the metadata stored on
    # the ChromaDB collection (tickers and form types are uppercased at
    # ingest time).  Accession numbers are case-stable and pass through.
    ticker_filter = [t.upper() for t in ticker] if ticker else None
    form_filter = [f.upper() for f in form] if form else None
    accession_filter = list(accession) if accession else None

    service = _build_service(output=output_format)

    # The Rich status spinner emits its own ANSI redraw frames; in
    # JSON mode we suppress it so a tee'd ``stdout`` does not pick up
    # animation bytes before the JSON document.
    def _run_retrieve() -> list[RetrievalResult]:
        return service.retrieve(
            query,
            top_k=top,
            ticker=ticker_filter,
            form_type=form_filter,
            accession_number=accession_filter,
            start_date=start_date,
            end_date=end_date,
            max_per_section=max_per_section,
            max_per_filing=max_per_filing,
            rerank_over_fetch_factor=rerank_over_fetch,
        )

    try:
        if is_json(output_format):
            results = _run_retrieve()
        else:
            with console.status("Searching..."):
                results = _run_retrieve()
    except SearchError as exc:
        print_error(
            "Search failed",
            exc.message,
            details=exc.details,
            hint=(
                "Ensure filings have been ingested with "
                "'sec-rag ingest add' and that --start-date / --end-date "
                "are valid YYYY-MM-DD values."
            ),
            output=output_format,
            error_code="search_failed",
        )
        raise typer.Exit(code=1) from None
    except ProviderError as exc:
        # Embedding-side failure — the corpus and storage layer are
        # fine, the embedder upstream is not.  Mirror the
        # ``/api/search`` mapping (502 there) with a single
        # operator-facing line here.
        print_error(
            "Embedding provider failure",
            "The embedding provider failed while processing the query.",
            details=exc.message,
            hint=(
                "Retry after a short backoff; if the failure persists, "
                "check the embedder API-key env var."
            ),
            output=output_format,
            error_code="provider_error",
        )
        raise typer.Exit(code=1) from None
    except DatabaseError as exc:
        print_error(
            "Database failure",
            exc.message,
            details=exc.details,
            hint="Check DB_CHROMA_PATH is readable and the collection is intact.",
            output=output_format,
            error_code="database_error",
        )
        raise typer.Exit(code=1) from None

    if is_json(output_format):
        # Mirror ``api.schemas.SearchResponse``: ``{hits, total}``.  The
        # raw query is deliberately NOT echoed (same discipline as the
        # API route) — operators piping JSON into a log shipper should
        # not find the query inlined into a per-result document.
        print_json(
            {
                "hits": [_hit_to_dict(r) for r in results],
                "total": len(results),
            }
        )
        return

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        console.print(
            "[dim italic]Hint: Try a broader query, or check that filings are "
            "ingested with 'sec-rag manage status'.[/dim italic]"
        )
        return

    _render_results(results, query)
