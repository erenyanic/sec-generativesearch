"""sec-rag evaluate retrieval — live retrieval-quality evaluation.

Single subcommand within the ``evaluate`` sub-Typer:

    sec-rag evaluate retrieval --cases CASES.json

Loads :class:`~sec_generative_search.search.evaluation.EvaluationCase`
objects from a JSON file, runs them through the live
:class:`~sec_generative_search.search.retrieval.RetrievalService`, and
reports aggregate precision@k plus recall@k with a per-case breakdown.

Trust model:
    Operator-scope only — same as :mod:`cli.search` (no auth gate, no
    rate limit, no access-log redaction).  The CLI is for on-host use;
    distributing it to team users is unsupported.

Query string / data discipline:
    Case queries are Tier-3 data.  :class:`RetrievalService` emits them
    through :func:`~sec_generative_search.core.logging.redact_for_log`
    internally (honoring ``LOG_REDACT_QUERIES``).  This module NEVER logs
    raw case queries or any retrieved chunk content.

Output discipline:
    Every report carries ``case_id`` + numeric metrics only — never
    query text, chunk content, or answer text.  Same invariant as the
    offline harnesses in :mod:`search.evaluation` and
    :mod:`rag.evaluation`.  The ``--output json`` shape is content-free
    by construction and safe to pipe into a log shipper.

JSON output shape (``--output json``)::

    {
        "top_k": 5,
        "case_count": 3,
        "precision_at_k": 0.667,
        "recall_at_k": 0.800,
        "per_case": [
            {"case_id": "...", "precision": 1.0, "recall": 1.0,
             "hits": 1, "expected": 1}
        ]
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

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
from sec_generative_search.search.evaluation import (
    EvaluationReport,
    evaluate_retrieval,
    load_cases_from_json,
)

__all__ = ["evaluate_app"]


console = Console()

evaluate_app = typer.Typer(
    name="evaluate",
    help="Offline and live evaluation harnesses for retrieval and generation quality.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Service construction (mirrors cli.search._build_service)
# ---------------------------------------------------------------------------


def _build_service(*, output: OutputFormat = OutputFormat.TEXT) -> RetrievalService:
    """Construct a stamp-sealed :class:`RetrievalService`.

    Identical seam order to :func:`cli.search._build_service`:
    registry → :func:`build_embedder` → :class:`ChromaDBClient`.
    Duplicated here so ``cli.evaluate`` is independently patchable in
    tests without coupling to the ``cli.search`` import site.
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


def _render_report(report: EvaluationReport) -> None:
    """Print aggregate metrics and a per-case breakdown table.

    Content-free: only ``case_id`` and numeric metrics appear — query
    text and chunk text are never printed.
    """
    console.print(
        f"\n[bold]Evaluation Report[/bold]  "
        f"top_k=[cyan]{report.top_k}[/cyan]  "
        f"cases=[cyan]{report.case_count}[/cyan]\n"
    )

    console.print(f"  Precision@{report.top_k:<4}  [green]{report.precision_at_k:.3f}[/green]")
    console.print(f"  Recall@{report.top_k:<4}     [green]{report.recall_at_k:.3f}[/green]\n")

    if not report.per_case:
        return

    table = Table(show_lines=False, border_style="dim", show_header=True, header_style="bold")
    table.add_column("Case ID", style="cyan", no_wrap=False)
    table.add_column("Precision", justify="right", width=10)
    table.add_column("Recall", justify="right", width=10)
    table.add_column("Hits", justify="right", width=6)
    table.add_column("Expected", justify="right", width=9)

    for case_id, precision, recall, hits, expected in report.per_case:
        table.add_row(
            escape(case_id),
            f"{precision:.3f}",
            f"{recall:.3f}",
            str(hits),
            str(expected),
        )

    console.print(table)


def _report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    """Lift :class:`EvaluationReport` onto the JSON wire shape.

    Allow-list lift — never ``**asdict()``.  Content-free: only
    ``case_id`` + numeric fields.
    """
    return {
        "top_k": report.top_k,
        "case_count": report.case_count,
        "precision_at_k": report.precision_at_k,
        "recall_at_k": report.recall_at_k,
        "per_case": [
            {
                "case_id": row[0],
                "precision": row[1],
                "recall": row[2],
                "hits": row[3],
                "expected": row[4],
            }
            for row in report.per_case
        ],
    }


# ---------------------------------------------------------------------------
# Public command
# ---------------------------------------------------------------------------


@evaluate_app.command(name="retrieval")
def retrieval(
    cases: Annotated[
        Path,
        typer.Option(
            "--cases",
            "-c",
            help=(
                "Path to a JSON file of evaluation cases. "
                "Each case must be a JSON object matching "
                "sec_generative_search.search.evaluation.EvaluationCase."
            ),
        ),
    ],
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            "-k",
            help="Number of results to request per case (precision denominator).",
            min=1,
        ),
    ] = 5,
    ticker: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-t",
            help="Restrict all case queries to this ticker. Repeat for multiple.",
        ),
    ] = None,
    form: Annotated[
        list[str] | None,
        typer.Option(
            "--form",
            "-f",
            help="Restrict all case queries to this form type. Repeat for multiple.",
        ),
    ] = None,
    start_date: Annotated[
        str | None,
        typer.Option(
            "--start-date",
            help="Restrict retrieval to filings on or after this date (YYYY-MM-DD).",
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option(
            "--end-date",
            help="Restrict retrieval to filings on or before this date (YYYY-MM-DD).",
        ),
    ] = None,
    max_per_section: Annotated[
        int | None,
        typer.Option(
            "--max-per-section",
            help="Cap chunks per section path; 0 disables. Omit to use SEARCH_MAX_PER_SECTION.",
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
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output format: 'text' (default Rich table) or 'json' "
                "(content-free allow-list lift of EvaluationReport). "
                "JSON mode suppresses status spinners and renders failures as "
                "{error, message, hint} envelopes."
            ),
        ),
    ] = "text",
) -> None:
    """Run the retrieval evaluation harness against the live collection.

    Loads evaluation cases from ``--cases``, runs each query through the
    :class:`RetrievalService`, and reports aggregate precision@k and
    recall@k plus a per-case breakdown.

    The ``--ticker``, ``--form``, ``--start-date``, ``--end-date``,
    ``--max-per-section``, and ``--max-per-filing`` flags are applied to
    every case uniformly — useful for scoping evaluation to a specific
    company or date window.

    Examples::

        sec-rag evaluate retrieval --cases tests/fixtures/retrieval_eval_cases.json

        sec-rag evaluate retrieval --cases my_cases.json --top-k 10 --ticker AAPL

        sec-rag evaluate retrieval --cases cases.json --output json | jq '.precision_at_k'
    """
    output_format = coerce_output_format(output)

    validate_date(start_date, "--start-date")
    validate_date(end_date, "--end-date")

    # Validate cases file exists before constructing storage.
    if not cases.exists():
        print_error(
            "Cases file not found",
            f"No file at {cases}.",
            hint=(
                "Provide a valid path to a JSON evaluation-case file. "
                "See tests/fixtures/retrieval_eval_cases.json for the expected format."
            ),
            output=output_format,
            error_code="cases_file_not_found",
        )
        raise typer.Exit(code=1)

    try:
        eval_cases = load_cases_from_json(cases)
    except (ValueError, OSError) as exc:
        print_error(
            "Cases file invalid",
            "Could not load evaluation cases.",
            details=str(exc),
            hint="Check that the file is valid JSON matching the EvaluationCase schema.",
            output=output_format,
            error_code="cases_file_invalid",
        )
        raise typer.Exit(code=1) from None

    ticker_filter = [t.upper() for t in ticker] if ticker else None
    form_filter = [f.upper() for f in form] if form else None

    service = _build_service(output=output_format)

    def _run_retrieve(query: str, k: int) -> list[RetrievalResult]:
        return service.retrieve(
            query,
            top_k=k,
            ticker=ticker_filter,
            form_type=form_filter,
            start_date=start_date,
            end_date=end_date,
            max_per_section=max_per_section,
            max_per_filing=max_per_filing,
        )

    def _do_evaluate() -> EvaluationReport:
        return evaluate_retrieval(eval_cases, _run_retrieve, top_k=top_k)

    try:
        if is_json(output_format):
            report = _do_evaluate()
        else:
            with console.status(f"Evaluating {len(eval_cases)} case(s) at top_k={top_k}..."):
                report = _do_evaluate()
    except SearchError as exc:
        print_error(
            "Search failed",
            exc.message,
            details=exc.details,
            hint=(
                "Ensure filings have been ingested with "
                "'sec-rag ingest add' and that date filters are valid YYYY-MM-DD values."
            ),
            output=output_format,
            error_code="search_failed",
        )
        raise typer.Exit(code=1) from None
    except ProviderError as exc:
        print_error(
            "Embedding provider failure",
            "The embedding provider failed while processing a query.",
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
        print_json(_report_to_dict(report))
        return

    _render_report(report)
