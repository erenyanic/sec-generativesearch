"""Synchronous CLI wrappers over the management surface.

Four operator-facing subcommands live here:

- ``sec-rag manage status`` — snapshot of filing / chunk counts and
  per-form breakdown.
- ``sec-rag manage list``   — table of ingested filings with optional
  ticker / form filters.
- ``sec-rag manage remove`` — delete one filing by accession number, or
  every filing matching ``--ticker`` and/or ``--form``.
- ``sec-rag manage clear``  — drop everything from both backing stores.

Operator trust model:

1. **Reads** go directly through :class:`MetadataRegistry` (no
   dual-store invariant on the read side; mirrors the API filings GETs).
2. **Writes** go exclusively through :class:`FilingStore` — never a
   direct :class:`ChromaDBClient` / :class:`MetadataRegistry` mutation
   from this surface.  The store owns the ChromaDB-first ordering and
   the best-effort rollback semantics.
3. The CLI does **not** honour ``API_DEMO_MODE`` — it is an
   operator-scope tool that bypasses every API control.  ``--yes`` is a "skip prompt"
   ergonomic knob, never a demo-mode bypass.
4. Like the other adapted CLIs, every user-facing string flows through
   :func:`rich.markup.escape` so accession numbers / ticker lists with
   incidental square brackets render verbatim.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sec_generative_search.cli._common import print_error, resolve_stamp
from sec_generative_search.cli._json import (
    OutputFormat,
    coerce_output_format,
    is_json,
    print_json,
)
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import DatabaseError
from sec_generative_search.database import (
    ChromaDBClient,
    FilingRecord,
    FilingStore,
    MetadataRegistry,
)

__all__ = ["manage_app"]


console = Console()

manage_app = typer.Typer(
    name="manage",
    help="Inspect and prune the local ChromaDB + SQLite filing store.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _record_to_dict(record: FilingRecord) -> dict[str, Any]:
    """Lift a :class:`FilingRecord` onto the JSON wire shape.

    Mirrors :class:`api.schemas.FilingSchema` exactly (allow-list lift)
    — the auto-increment ``id`` is dropped because it is an internal
    SQLite detail.  Used by ``manage list --output json``.
    """
    return {
        "ticker": record.ticker,
        "form_type": record.form_type,
        "filing_date": record.filing_date,
        "accession_number": record.accession_number,
        "chunk_count": record.chunk_count,
        "ingested_at": record.ingested_at,
    }


# ---------------------------------------------------------------------------
# Storage construction (shared by the four subcommands)
# ---------------------------------------------------------------------------


def _open_registry_only(*, output: OutputFormat = OutputFormat.TEXT) -> MetadataRegistry:
    """Open the metadata registry for read-only queries.

    Read paths (``status`` filing-count read, ``list``, ``remove``
    detail lookup) do not need a stamped ChromaDB client.  Keeping the
    happy-path open minimal avoids spurious
    :class:`EmbeddingCollectionMismatchError` surfaces from the seal in
    case a future flag flips the embedder stamp out from under an
    operator that only wants to *read*.
    """
    try:
        return MetadataRegistry()
    except DatabaseError as exc:
        print_error(
            "Registry initialisation failed",
            exc.message,
            details=exc.details,
            hint="Check DB_METADATA_DB_PATH is readable and SQLCipher is set up if encrypted.",
            output=output,
            error_code="registry_initialisation_failed",
        )
        raise typer.Exit(code=1) from None


def _open_store(
    *,
    output: OutputFormat = OutputFormat.TEXT,
) -> tuple[ChromaDBClient, MetadataRegistry, FilingStore]:
    """Open both backing stores + the dual-store coordinator.

    Required by every write path (``remove``, ``clear``) and by
    ``status`` because the chunk count lives on the ChromaDB collection.
    Failures surface as a single operator-facing envelope — the caller
    never sees a stack trace.
    """
    embedding = get_settings().embedding
    stamp = resolve_stamp(embedding.provider, embedding.model_name, output=output)
    try:
        chroma = ChromaDBClient(stamp)
        registry = MetadataRegistry()
    except DatabaseError as exc:
        print_error(
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

    return chroma, registry, FilingStore(chroma, registry)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@manage_app.command("status")
def status(
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output format: 'text' (default Rich panel) or 'json'.  "
                "JSON shape is an allow-list lift of the operator-relevant "
                "fields ({filing_count, max_filings, chunk_count, tickers, "
                "form_breakdown}); failures render as {error, message, hint} "
                "envelopes."
            ),
        ),
    ] = "text",
) -> None:
    """Show filing / chunk counts, ticker list, and form-type breakdown.

    Examples:

        sec-rag manage status

        sec-rag manage status --output json | jq '.filing_count'
    """
    output_format = coerce_output_format(output)

    chroma, registry, _ = _open_store(output=output_format)
    settings = get_settings()

    try:
        stats = registry.get_statistics()
        chunk_count = chroma.collection_count()
    except DatabaseError as exc:
        print_error(
            "Status query failed",
            exc.message,
            details=exc.details,
            output=output_format,
            error_code="status_query_failed",
        )
        raise typer.Exit(code=1) from None

    max_filings = settings.database.max_filings

    if is_json(output_format):
        # Operator-relevant snapshot — does NOT mirror
        # ``api.schemas.StatusResponse`` because the CLI surface
        # exposes contents (tickers, form breakdown) instead of
        # deployment / auth metadata (``is_admin``,
        # ``deployment_profile``).  Field names are stable and
        # explicit so a future schema bump on the dataclass does not
        # silently leak.
        print_json(
            {
                "filing_count": stats.filing_count,
                "max_filings": max_filings,
                "chunk_count": chunk_count,
                "tickers": list(stats.tickers),
                "form_breakdown": dict(stats.form_breakdown),
            }
        )
        return

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    filing_style = "green" if stats.filing_count > 0 else "dim"
    table.add_row(
        "Filings",
        Text(f"{stats.filing_count}/{max_filings}", style=filing_style),
    )

    chunk_style = "green" if chunk_count > 0 else "dim"
    table.add_row("Chunks", Text(str(chunk_count), style=chunk_style))

    if stats.filing_count > 0:
        ticker_list = ", ".join(stats.tickers)
        table.add_row(
            "Tickers",
            Text(f"{len(stats.tickers)} ({ticker_list})", style="cyan"),
        )
        breakdown = "  |  ".join(f"{form}: {count}" for form, count in stats.form_breakdown.items())
        table.add_row("Forms", Text(breakdown))
    else:
        table.add_row("Tickers", Text("—", style="dim"))
        table.add_row("Forms", Text("—", style="dim"))

    console.print(Panel(table, title="[bold]Database Status[/bold]", expand=False))


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@manage_app.command("list")
def list_filings(
    ticker: Annotated[
        str | None,
        typer.Option("--ticker", "-k", help="Filter by ticker symbol."),
    ] = None,
    form: Annotated[
        str | None,
        typer.Option("--form", "-f", help="Filter by form type."),
    ] = None,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output format: 'text' (default Rich table) or 'json' "
                "(single-document allow-list lift of api.schemas.FilingListResponse)."
            ),
        ),
    ] = "text",
) -> None:
    """List ingested filings, optionally filtered by ticker / form.

    Reads :class:`MetadataRegistry` directly — listing does not require
    a stamped ChromaDB client (no dual-store invariant on the read
    side).

    Examples:

        sec-rag manage list

        sec-rag manage list -k AAPL

        sec-rag manage list -f 10-K

        sec-rag manage list --output json | jq '.filings[].accession_number'
    """
    output_format = coerce_output_format(output)

    registry = _open_registry_only(output=output_format)

    try:
        filings = registry.list_filings(
            ticker=ticker.upper() if ticker else None,
            form_type=form.upper() if form else None,
        )
    except DatabaseError as exc:
        print_error(
            "List failed",
            exc.message,
            details=exc.details,
            output=output_format,
            error_code="list_failed",
        )
        raise typer.Exit(code=1) from None

    if is_json(output_format):
        # Mirror ``api.schemas.FilingListResponse``: ``{filings, total}``.
        # An empty list is a valid result — the JSON shape is uniform
        # so a downstream parser does not need a "no filings" sentinel.
        print_json(
            {
                "filings": [_record_to_dict(f) for f in filings],
                "total": len(filings),
            }
        )
        return

    if not filings:
        console.print("[yellow]No filings found.[/yellow]")
        return

    table = Table(
        title="[bold]Ingested Filings[/bold]",
        border_style="dim",
        header_style="bold",
    )
    table.add_column("Ticker", style="cyan")
    table.add_column("Form", style="green")
    table.add_column("Filing Date")
    table.add_column("Accession Number", style="dim")
    table.add_column("Chunks", justify="right", style="bold")
    table.add_column("Ingested At", style="dim")

    for f in filings:
        table.add_row(
            escape(f.ticker),
            escape(f.form_type),
            escape(f.filing_date),
            escape(f.accession_number),
            str(f.chunk_count),
            escape(f.ingested_at),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


def _render_filing_detail(filing: FilingRecord, *, title: str) -> None:
    """Render a single-filing detail panel for the remove-confirmation prompt."""
    detail = Table(show_header=False, box=None, padding=(0, 2))
    detail.add_column("Key", style="bold")
    detail.add_column("Value")
    detail.add_row(
        "Filing",
        Text(f"{filing.ticker} {filing.form_type}", style="cyan"),
    )
    detail.add_row("Date", Text(filing.filing_date))
    detail.add_row("Chunks", Text(str(filing.chunk_count), style="bold"))
    detail.add_row("Accession", Text(filing.accession_number, style="dim"))
    console.print(Panel(detail, title=title, expand=False))


@manage_app.command("remove")
def remove(
    accession_number: Annotated[
        str | None,
        typer.Argument(help="Accession number of the filing to remove."),
    ] = None,
    ticker: Annotated[
        str | None,
        typer.Option("--ticker", "-k", help="Remove all filings for this ticker."),
    ] = None,
    form: Annotated[
        str | None,
        typer.Option("--form", "-f", help="Remove all filings of this form type."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the destructive-action confirmation prompt.",
        ),
    ] = False,
) -> None:
    """Remove filing(s) from both backing stores.

    Two modes:

    - **Single**: positional accession number — deletes one filing.
    - **Bulk**: ``--ticker`` and/or ``--form`` — deletes every filing
      matching the filter combination.

    Every write flows through :class:`FilingStore` — never a direct
    :class:`ChromaDBClient` / :class:`MetadataRegistry` mutation.  The
    coordinator owns the ChromaDB-first delete order and rolls back
    SQLite-side on a ChromaDB failure.

    Examples:

        sec-rag manage remove 0000320193-24-000123

        sec-rag manage remove --ticker AAPL

        sec-rag manage remove --form 10-K

        sec-rag manage remove --ticker AAPL --form 10-K

        sec-rag manage remove --ticker MSFT -y
    """
    has_filters = ticker is not None or form is not None

    if accession_number is None and not has_filters:
        print_error(
            "Missing target",
            "Provide an accession number or use --ticker/--form to select filings.",
        )
        raise typer.Exit(code=1)

    if accession_number is not None and has_filters:
        print_error(
            "Invalid flag combination",
            "Cannot combine an accession number with --ticker/--form filters.",
        )
        raise typer.Exit(code=1)

    _chroma, registry, store = _open_store()

    # --- Single filing by accession number ----------------------------------
    if accession_number is not None:
        try:
            filing = registry.get_filing(accession_number)
        except DatabaseError as exc:
            print_error("Lookup failed", exc.message, details=exc.details)
            raise typer.Exit(code=1) from None

        if filing is None:
            print_error(
                "Filing not found",
                escape(accession_number),
                hint="Run 'sec-rag manage list' to see available accession numbers.",
            )
            raise typer.Exit(code=1)

        _render_filing_detail(filing, title="[bold yellow]Remove Filing[/bold yellow]")

        if not yes:
            confirmed = typer.confirm("Remove this filing?")
            if not confirmed:
                console.print("[dim]Cancelled.[/dim]")
                raise typer.Exit(code=0)

        try:
            store.delete_filing(filing.accession_number)
        except DatabaseError as exc:
            print_error(
                "Removal failed",
                exc.message,
                details=exc.details,
                hint="Check that the data directory is writable.",
            )
            raise typer.Exit(code=1) from None

        console.print(
            f"[green]Removed:[/green] {escape(filing.ticker)} "
            f"{escape(filing.form_type)} ({escape(filing.filing_date)}) — "
            f"{filing.chunk_count} chunks deleted"
        )
        return

    # --- Bulk removal by --ticker and/or --form -----------------------------
    try:
        filings = registry.list_filings(
            ticker=ticker.upper() if ticker else None,
            form_type=form.upper() if form else None,
        )
    except DatabaseError as exc:
        print_error("Lookup failed", exc.message, details=exc.details)
        raise typer.Exit(code=1) from None

    if not filings:
        filter_desc = " and ".join(
            part
            for part in (
                f"ticker={ticker.upper()}" if ticker else None,
                f"form={form.upper()}" if form else None,
            )
            if part
        )
        console.print(f"[yellow]No filings found matching {escape(filter_desc)}.[/yellow]")
        return

    total_chunks = sum(f.chunk_count for f in filings)
    filter_parts: list[str] = []
    if ticker:
        filter_parts.append(f"ticker=[cyan]{escape(ticker.upper())}[/cyan]")
    if form:
        filter_parts.append(f"form=[green]{escape(form.upper())}[/green]")
    filter_desc = ", ".join(filter_parts)

    console.print(
        f"\n[bold yellow]Bulk Remove[/bold yellow]  ({filter_desc})\n"
        f"  {len(filings)} filing(s), {total_chunks} chunks total\n"
    )

    for f in filings:
        console.print(
            f"  [dim]•[/dim] {escape(f.ticker)} {escape(f.form_type)} "
            f"({escape(f.filing_date)}) — {f.chunk_count} chunks"
        )

    console.print()

    if not yes:
        confirmed = typer.confirm(f"{len(filings)} filing(s) will be deleted. Are you sure?")
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)

    try:
        rows_removed = store.delete_filings_batch([f.accession_number for f in filings])
    except DatabaseError as exc:
        print_error(
            "Removal failed",
            exc.message,
            details=exc.details,
            hint="Check that the data directory is writable.",
        )
        raise typer.Exit(code=1) from None

    console.print(
        f"\n[green]Done:[/green] {rows_removed} filing(s) removed, {total_chunks} chunks deleted"
    )


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


@manage_app.command("clear")
def clear(
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the destructive-action confirmation prompt.",
        ),
    ] = False,
) -> None:
    """Drop every filing from both backing stores.

    Delegates to :meth:`FilingStore.clear_all`, which truncates the
    SQLite ``filings`` table (``schema_version`` is preserved) and
    re-seals the ChromaDB ``sec_filings`` collection.

    The CLI is operator-scope and does **not** honour
    ``API_DEMO_MODE`` — that flag exists to lock the API surface against
    web users in cloud / demo profiles, not to gate an admin's local
    tooling.  ``--yes`` is a "skip prompt" ergonomic knob, never a
    demo-mode bypass.

    Examples:

        sec-rag manage clear

        sec-rag manage clear -y
    """
    _chroma, registry, store = _open_store()

    try:
        filings = registry.list_filings()
    except DatabaseError as exc:
        print_error("Lookup failed", exc.message, details=exc.details)
        raise typer.Exit(code=1) from None

    if not filings:
        console.print("[yellow]Database is already empty.[/yellow]")
        return

    total_chunks = sum(f.chunk_count for f in filings)
    unique_tickers = sorted({f.ticker for f in filings})

    console.print(
        f"\n[bold red]Clear Database[/bold red]\n"
        f"  {len(filings)} filing(s), {total_chunks} chunks, "
        f"{len(unique_tickers)} ticker(s): "
        f"{escape(', '.join(unique_tickers))}\n"
    )

    if not yes:
        confirmed = typer.confirm(f"ALL {len(filings)} filing(s) will be deleted. Are you sure?")
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)

    try:
        chunks_removed, filings_removed = store.clear_all()
    except DatabaseError as exc:
        print_error(
            "Clear failed",
            exc.message,
            details=exc.details,
            hint="Check that the data directory is writable.",
        )
        raise typer.Exit(code=1) from None

    console.print(
        f"\n[green]Database cleared:[/green] {filings_removed} filing(s) removed, "
        f"{chunks_removed} chunks deleted"
    )
