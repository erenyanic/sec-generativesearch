"""Synchronous CLI wrappers over the ingest pipeline.

Two operator-facing subcommands live here:

- ``sec-rag ingest add TICKER ...`` — fetch and ingest filings for one
  company across one or more form types.
- ``sec-rag ingest batch TICKER ...`` — repeat the same flow over a list
  of tickers.

Both commands share three load-bearing rules:

1. Every dual-store write flows through :class:`FilingStore` — never a
   direct :class:`ChromaDBClient` / :class:`MetadataRegistry` mutation.
   The CLI is single-process and pre-checks duplicates in a single SQL
   batch, so the carry-over ChromaDB-first path is safe; the atomic
   path would only add round-trips with no concurrency to defend
   against.
2. The embedder is built via :func:`build_embedder` (the sole
   construction seam).  The same ``(provider, model, dimension)``
   triple seals the :class:`ChromaDBClient` collection.
3. No per-IP cooldown — the CLI has no IP and bypasses every API
   control by design.  Rate limiting is an API-tier concern.

Output discipline mirrors the rest of the adapted CLI surface:
operator-facing strings flow through :func:`rich.markup.escape` so
literal square brackets (env-var names, accession numbers, install
hints) survive Rich's markup parser.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from sec_generative_search.config.constants import DEFAULT_FORM_TYPES, parse_form_types
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    DatabaseError,
    FetchError,
    FilingLimitExceededError,
    SECGenerativeSearchError,
)
from sec_generative_search.core.types import EmbedderStamp, FilingIdentifier
from sec_generative_search.database import (
    ChromaDBClient,
    FilingStore,
    MetadataRegistry,
)
from sec_generative_search.pipeline import (
    FilingFetcher,
    FilingInfo,
    PipelineOrchestrator,
    ProcessedFiling,
)
from sec_generative_search.pipeline.prefetch import OneAheadFetches
from sec_generative_search.providers.factory import build_embedder
from sec_generative_search.providers.registry import ProviderRegistry

__all__ = ["ingest_app"]


console = Console()

ingest_app = typer.Typer(
    name="ingest",
    help="Fetch and ingest SEC filings into ChromaDB + SQLite.",
    no_args_is_help=True,
)

# Step labels used in the progress display for ingestion.
_STEPS = ("Fetching", "Parsing", "Chunking", "Embedding", "Storing")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _print_error(
    label: str,
    message: str,
    *,
    details: str | None = None,
    hint: str | None = None,
) -> None:
    """Render an error with optional details and a single hint line.

    All operator-facing strings flow through :func:`rich.markup.escape`
    so hints / accession numbers / install snippets carrying literal
    square brackets render verbatim instead of being silently stripped
    by Rich's markup parser.
    """
    console.print(f"[red]{escape(label)}:[/red] {escape(message)}")
    if details:
        console.print(f"  [dim]{escape(details)}[/dim]")
    if hint:
        console.print(f"  [dim italic]Hint: {escape(hint)}[/dim italic]")


def _make_progress() -> Progress:
    """Build a Rich :class:`Progress` instance with ingest-specific columns."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("{task.completed}/{task.total} steps"),
        TimeElapsedColumn(),
        console=console,
    )


def _validate_date(value: str | None, param_name: str) -> str | None:
    """Validate ``YYYY-MM-DD`` strings at the CLI boundary.

    Same shape Typer would reject internally if it knew about ISO
    dates; surfacing a :class:`typer.BadParameter` here means the
    error renders consistently with the rest of the CLI.
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


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------


def _build_pipeline() -> tuple[FilingFetcher, PipelineOrchestrator, MetadataRegistry, FilingStore]:
    """Construct the fetch + orchestrate + registry + store quartet.

    Resolves the embedder via :func:`build_embedder` (the sole
    construction seam) and seals the ChromaDB collection with the
    matching :class:`EmbedderStamp`.  Failures bubble up as Typer exits
    with a single operator-facing message — the caller never sees a
    stack trace for a misconfigured environment.

    The registry is returned alongside the store so callers can drive
    read-only queries (``get_existing_accessions``, ``check_filing_limit``)
    without reaching past the store seam.  Writes still flow through
    :class:`FilingStore` exclusively.
    """
    settings = get_settings()
    embedding = settings.embedding

    try:
        target_dim = ProviderRegistry.get_dimension(embedding.provider, embedding.model_name)
    except (KeyError, ValueError) as exc:
        _print_error(
            "Embedder configuration invalid",
            f"Cannot resolve dimension for {embedding.provider}/{embedding.model_name}.",
            details=str(exc),
            hint=(
                "Check EMBEDDING_PROVIDER and EMBEDDING_MODEL_NAME against "
                "the registry — defaults live in providers/registry.py."
            ),
        )
        raise typer.Exit(code=1) from None

    try:
        embedder = build_embedder(embedding)
    except ConfigurationError as exc:
        _print_error(
            "Embedder construction failed",
            exc.message,
            hint="Set the expected API-key env var for this provider.",
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
        )
        raise typer.Exit(code=1) from None

    stamp = EmbedderStamp(
        provider=embedding.provider,
        model=embedding.model_name,
        dimension=target_dim,
    )

    try:
        chroma = ChromaDBClient(stamp)
        registry = MetadataRegistry()
    except DatabaseError as exc:
        _print_error(
            "Storage initialisation failed",
            exc.message,
            details=exc.details,
            hint=(
                "Check DB_CHROMA_PATH / DB_METADATA_DB_PATH are writable and that "
                "the existing collection's embedder stamp matches EMBEDDING_*."
            ),
        )
        raise typer.Exit(code=1) from None

    fetcher = FilingFetcher()
    orchestrator = PipelineOrchestrator(embedder=embedder)
    store = FilingStore(chroma, registry)
    return fetcher, orchestrator, registry, store


# ---------------------------------------------------------------------------
# Fetch dispatcher
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Single-form ingestion (used by both ``add`` and ``batch``)
# ---------------------------------------------------------------------------


def _process_and_store(
    processed: ProcessedFiling,
    store: FilingStore,
) -> None:
    """Persist a processed filing via :class:`FilingStore`.

    The dual-store coordinator owns the ChromaDB-first ordering + the
    SQLite rollback semantics — the CLI never touches either backend
    directly.  ``register_if_new=False`` matches the carry-over path
    used for pre-checked callers.
    """
    store.store_filing(processed, register_if_new=False)


def _ingest_one_form(
    ticker: str,
    form_type: str,
    *,
    count: int | None,
    year: int | None,
    start_date: str | None,
    end_date: str | None,
    fetcher: FilingFetcher,
    orchestrator: PipelineOrchestrator,
    registry: MetadataRegistry,
    store: FilingStore,
    progress: Progress,
    step_task_id: int,
    filing_task_id: int | None = None,
    form_label: str = "",
) -> tuple[int, int, int]:
    """Ingest filings for one ticker + one form type.

    Pipeline: list the matching filings' metadata (no HTML) → batch
    duplicate check → per new filing: fetch → process (parse + chunk +
    embed) → store via :class:`FilingStore`.  HTML is streamed — the next
    new filing is fetched one ahead while the current one is processed
    (F21), so at most two filings are resident and a duplicate is never
    downloaded.  ``list_available`` selects exactly what the old
    ``fetch_latest`` / ``fetch_one`` / ``fetch`` routing did (the newest
    *count* non-amendment filings).  Per-filing failure isolation mirrors
    the API-side worker: a single bad filing emits a row error and the
    loop continues; only a listing failure aborts the form.

    Returns:
        ``(succeeded, skipped, failed)`` counts.
    """
    multi = count is None or count > 1

    progress.update(
        step_task_id,
        description=f"Fetching {ticker} {form_type}{form_label}...",
    )
    try:
        filings = fetcher.list_available(
            ticker,
            form_type,
            count=count,
            year=year,
            start_date=start_date,
            end_date=end_date,
        )
    except FetchError as exc:
        progress.stop()
        _print_error(
            f"Fetch failed for {ticker} {form_type}",
            exc.message,
            details=exc.details,
            hint="Check the ticker symbol is valid and you have an internet connection.",
        )
        return 0, 0, 1

    if not filings:
        progress.stop()
        console.print(
            f"[yellow]No filings found[/yellow] for {escape(ticker)} "
            f"{escape(form_type)} with the given filters."
        )
        return 0, 0, 0

    if filing_task_id is not None:
        progress.update(filing_task_id, total=len(filings))

    progress.advance(step_task_id)

    # Single SQL batch in place of N per-accession lookups.
    existing = registry.get_existing_accessions([fi.accession_number for fi in filings])
    with OneAheadFetches(
        fetcher.fetch_filing_content,
        [fi for fi in filings if fi.accession_number not in existing],
        name="ingest-prefetch",
    ) as fetches:
        return _ingest_listed_form(
            ticker,
            form_type,
            filings,
            existing,
            fetches=fetches,
            multi=multi,
            orchestrator=orchestrator,
            registry=registry,
            store=store,
            progress=progress,
            step_task_id=step_task_id,
            filing_task_id=filing_task_id,
            form_label=form_label,
        )


def _ingest_listed_form(
    ticker: str,
    form_type: str,
    filings: list[FilingInfo],
    existing: set[str],
    *,
    fetches: OneAheadFetches[FilingInfo, tuple[FilingIdentifier, str]],
    multi: bool,
    orchestrator: PipelineOrchestrator,
    registry: MetadataRegistry,
    store: FilingStore,
    progress: Progress,
    step_task_id: int,
    filing_task_id: int | None,
    form_label: str,
) -> tuple[int, int, int]:
    """Stream *filings* (already listed and dup-checked) through the pipeline."""
    succeeded = 0
    skipped = 0
    failed = 0

    for filing_idx, filing_info in enumerate(filings):
        filing_num = f" [{filing_idx + 1}/{len(filings)}]" if multi else ""
        filing_id = filing_info.to_identifier()

        if filing_idx > 0:
            try:
                registry.check_filing_limit()
            except FilingLimitExceededError:
                progress.stop()
                console.print(
                    f"[yellow]Filing limit reached[/yellow] after "
                    f"{succeeded} ingestion(s) — stopping."
                )
                break
            # Reset step bar between filings.
            progress.update(step_task_id, completed=1)

        if filing_id.accession_number in existing:
            line = (
                f"  [yellow]Already ingested{filing_num}:[/yellow] "
                f"{escape(ticker)} {escape(form_type)} ({filing_id.date_str})"
            )
            if multi:
                progress.console.print(line)
            else:
                progress.stop()
                console.print(
                    f"[yellow]Already ingested:[/yellow] {escape(ticker)} "
                    f"{escape(form_type)} ({filing_id.date_str}, "
                    f"{escape(filing_id.accession_number)})"
                )
            skipped += 1
            if filing_task_id is not None:
                progress.advance(filing_task_id)
            continue

        try:
            filing_id, html_content = fetches.fetch(filing_info)
        except FetchError as exc:
            if multi:
                progress.console.print(
                    f"  [red]Fetch failed{filing_num}:[/red] {escape(exc.message)}"
                )
            else:
                progress.stop()
                _print_error(
                    f"Fetch failed for {ticker} {form_type}",
                    exc.message,
                    details=exc.details,
                    hint="Check you have an internet connection, then retry.",
                )
            failed += 1
            if filing_task_id is not None:
                progress.advance(filing_task_id)
            continue

        # Fetch the next new filing while this one is processed (F21).
        fetches.start_after(filing_info)

        def _on_progress(
            step: str,
            _current: int,
            _total: int,
            _fnum: str = filing_num,
        ) -> None:
            if step != "Complete":
                progress.update(
                    step_task_id,
                    description=f"{step} {ticker} {form_type}{form_label}{_fnum}...",
                )
                progress.advance(step_task_id)

        try:
            result = orchestrator.process_filing(
                filing_id,
                html_content,
                progress_callback=_on_progress,
            )
        except SECGenerativeSearchError as exc:
            if multi:
                progress.console.print(
                    f"  [red]Processing failed{filing_num}:[/red] {escape(exc.message)}"
                )
            else:
                progress.stop()
                _print_error(
                    "Processing failed",
                    exc.message,
                    details=exc.details,
                    hint="If this is a memory error, try lowering EMBEDDING_BATCH_SIZE in .env.",
                )
            failed += 1
            if filing_task_id is not None:
                progress.advance(filing_task_id)
            continue

        progress.update(
            step_task_id,
            description=f"Storing {ticker} {form_type}{form_label}{filing_num}...",
        )
        try:
            _process_and_store(result, store)
        except DatabaseError as exc:
            if multi:
                progress.console.print(
                    f"  [red]Storage failed{filing_num}:[/red] {escape(exc.message)}"
                )
            else:
                progress.stop()
                _print_error(
                    "Storage failed",
                    exc.message,
                    hint="Check disk space and that the data directory is writable.",
                )
            failed += 1
            if filing_task_id is not None:
                progress.advance(filing_task_id)
            continue

        progress.advance(step_task_id)

        stats = result.ingest_result
        if multi:
            progress.console.print(
                f"  [green]Ingested{filing_num}:[/green] {escape(ticker)} "
                f"{escape(form_type)} ({filing_id.date_str})  |  "
                f"Chunks: {stats.chunk_count}  |  "
                f"Time: {stats.duration_seconds:.1f}s"
            )
        else:
            progress.stop()
            console.print(
                f"[green]Ingested:[/green] {escape(ticker)} {escape(form_type)} "
                f"({filing_id.date_str})\n"
                f"  Segments: {stats.segment_count}  |  "
                f"Chunks: {stats.chunk_count}  |  "
                f"Time: {stats.duration_seconds:.1f}s"
            )
        succeeded += 1
        if filing_task_id is not None:
            progress.advance(filing_task_id)

    return succeeded, skipped, failed


# ---------------------------------------------------------------------------
# Cross-form ingestion (``-t/--total``)
# ---------------------------------------------------------------------------


def _ingest_across_forms(
    ticker: str,
    form_types: tuple[str, ...],
    *,
    count: int,
    year: int | None,
    start_date: str | None,
    end_date: str | None,
    fetcher: FilingFetcher,
    orchestrator: PipelineOrchestrator,
    registry: MetadataRegistry,
    store: FilingStore,
) -> tuple[int, int, int]:
    """Ingest the *count* newest filings across all *form_types*.

    Uses :meth:`FilingFetcher.list_available_across_forms` to merge
    candidates by date without downloading any HTML, then drives the
    standard pipeline per selected filing.
    """
    console.print(
        f"Listing available {escape(ticker)} filings across {escape(', '.join(form_types))}..."
    )
    try:
        selected = fetcher.list_available_across_forms(
            ticker,
            form_types,
            count=count,
            year=year,
            start_date=start_date,
            end_date=end_date,
        )
    except FetchError as exc:
        _print_error(
            f"Listing failed for {ticker}",
            exc.message,
            details=exc.details,
            hint="Check the ticker symbol is valid and you have an internet connection.",
        )
        return 0, 0, 1

    if not selected:
        console.print(
            f"[yellow]No filings found[/yellow] for {escape(ticker)} with the given filters."
        )
        return 0, 0, 0

    console.print(
        f"Found {len(selected)} filing(s): "
        + escape(", ".join(f"{fi.form_type} ({fi.filing_date})" for fi in selected))
    )

    succeeded = 0
    skipped = 0
    failed = 0

    existing = registry.get_existing_accessions([fi.accession_number for fi in selected])

    with (
        _make_progress() as progress,
        OneAheadFetches(
            fetcher.fetch_filing_content,
            [fi for fi in selected if fi.accession_number not in existing],
            name="ingest-prefetch",
        ) as fetches,
    ):
        filing_task = progress.add_task(
            f"{escape(ticker)}: 0/{len(selected)} filings",
            total=len(selected),
        )
        step_task = progress.add_task(
            "Fetching...",
            total=len(_STEPS),
        )

        for filing_idx, fi in enumerate(selected):
            filing_num = f" [{filing_idx + 1}/{len(selected)}]"
            label = f"{ticker} {fi.form_type}"

            try:
                registry.check_filing_limit()
            except FilingLimitExceededError:
                progress.stop()
                console.print(
                    f"[yellow]Filing limit reached[/yellow] after "
                    f"{succeeded} ingestion(s) — stopping."
                )
                break

            progress.update(
                step_task,
                completed=0,
                description=f"Fetching {label}{filing_num}...",
            )

            if fi.accession_number in existing:
                progress.console.print(
                    f"  [yellow]Already ingested{filing_num}:[/yellow] "
                    f"{escape(label)} ({fi.filing_date})"
                )
                skipped += 1
                progress.advance(filing_task)
                continue

            try:
                filing_id, html_content = fetches.fetch(fi)
            except FetchError as exc:
                progress.console.print(
                    f"  [red]Fetch failed{filing_num}:[/red] {escape(exc.message)}"
                )
                failed += 1
                progress.advance(filing_task)
                continue

            # Fetch the next new filing while this one is processed (F21).
            fetches.start_after(fi)
            progress.advance(step_task)

            def _on_progress(
                step: str,
                _current: int,
                _total: int,
                _label: str = label,
                _fnum: str = filing_num,
            ) -> None:
                if step != "Complete":
                    progress.update(
                        step_task,
                        description=f"{step} {_label}{_fnum}...",
                    )
                    progress.advance(step_task)

            try:
                result = orchestrator.process_filing(
                    filing_id,
                    html_content,
                    progress_callback=_on_progress,
                )
            except SECGenerativeSearchError as exc:
                progress.console.print(
                    f"  [red]Processing failed{filing_num}:[/red] {escape(exc.message)}"
                )
                failed += 1
                progress.advance(filing_task)
                continue

            progress.update(step_task, description=f"Storing {label}{filing_num}...")
            try:
                _process_and_store(result, store)
            except DatabaseError as exc:
                progress.console.print(
                    f"  [red]Storage failed{filing_num}:[/red] {escape(exc.message)}"
                )
                failed += 1
                progress.advance(filing_task)
                continue

            progress.advance(step_task)

            stats = result.ingest_result
            progress.console.print(
                f"  [green]Ingested{filing_num}:[/green] {escape(label)} "
                f"({filing_id.date_str})  |  "
                f"Chunks: {stats.chunk_count}  |  "
                f"Time: {stats.duration_seconds:.1f}s"
            )
            succeeded += 1
            progress.advance(filing_task)

    return succeeded, skipped, failed


# ---------------------------------------------------------------------------
# Common flag-resolution
# ---------------------------------------------------------------------------


def _resolve_per_form_count(
    *,
    number: int | None,
    year: int | None,
    start_date: str | None,
    end_date: str | None,
) -> int | None:
    """Pick the effective per-form count from the flag combination.

    - explicit ``-n N`` wins.
    - any time filter (``-y`` / ``--start-date`` / ``--end-date``) without
      an explicit count means "all matching" (``None``, capped by
      ``max_filings`` inside the fetcher).
    - otherwise the default is one filing (the latest).
    """
    if number is not None:
        return number
    if year is not None or start_date is not None or end_date is not None:
        return None
    return 1


def _print_summary(succeeded: int, skipped: int, failed: int, *, header: str) -> None:
    """Render the trailing summary line shared by both commands."""
    console.print(
        f"\n[bold]{escape(header)}[/bold] "
        f"[green]{succeeded} ingested[/green], "
        f"[yellow]{skipped} skipped[/yellow], "
        f"[red]{failed} failed[/red]"
    )


# ---------------------------------------------------------------------------
# Public commands
# ---------------------------------------------------------------------------


@ingest_app.command("add")
def add(
    ticker: Annotated[str, typer.Argument(help="Stock ticker symbol (e.g. AAPL).")],
    form: Annotated[
        str,
        typer.Option(
            "--form",
            "-f",
            help="SEC form type(s), comma-separated (e.g. 8-K, 10-K, 10-Q).",
        ),
    ] = DEFAULT_FORM_TYPES,
    total: Annotated[
        int | None,
        typer.Option(
            "--total",
            "-t",
            help="Total number of filings to ingest (across all form types, newest first).",
            min=1,
        ),
    ] = None,
    number: Annotated[
        int | None,
        typer.Option(
            "--number",
            "-n",
            help="Number of filings to ingest per form type.",
            min=1,
        ),
    ] = None,
    year: Annotated[
        int | None,
        typer.Option("--year", "-y", help="Filter by filing year (e.g. 2023)."),
    ] = None,
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", help="Start date filter (YYYY-MM-DD)."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option("--end-date", help="End date filter (YYYY-MM-DD)."),
    ] = None,
) -> None:
    """Fetch and ingest SEC filing(s) for a company.

    Examples:

        sec-rag ingest add AAPL

        sec-rag ingest add AAPL -f 10-K

        sec-rag ingest add AAPL -t 3

        sec-rag ingest add AAPL -n 2 -f 10-K

        sec-rag ingest add AAPL -y 2023

        sec-rag ingest add AAPL --start-date 2022-01-01 --end-date 2023-12-31
    """
    ticker = ticker.upper()

    if total is not None and number is not None:
        _print_error(
            "Invalid flag combination",
            "--total and --number are mutually exclusive.",
        )
        raise typer.Exit(code=1)

    try:
        form_types = parse_form_types(form)
    except ValueError as exc:
        _print_error("Invalid form type", str(exc))
        raise typer.Exit(code=1) from None

    _validate_date(start_date, "--start-date")
    _validate_date(end_date, "--end-date")

    fetcher, orchestrator, registry, store = _build_pipeline()

    # --- Cross-form mode: -t (total across form types) -----------------------
    if total is not None:
        succeeded, skipped, failed = _ingest_across_forms(
            ticker,
            form_types,
            count=total,
            year=year,
            start_date=start_date,
            end_date=end_date,
            fetcher=fetcher,
            orchestrator=orchestrator,
            registry=registry,
            store=store,
        )
        if total > 1:
            _print_summary(succeeded, skipped, failed, header="Summary:")
        if failed > 0 and succeeded == 0 and skipped == 0:
            raise typer.Exit(code=1)
        return

    # --- Per-form mode: -n or default ---------------------------------------
    effective_per_form = _resolve_per_form_count(
        number=number, year=year, start_date=start_date, end_date=end_date
    )

    succeeded = 0
    skipped = 0
    failed = 0

    # One edgartools ``Company`` for all of this ticker's forms (F22).
    with fetcher.company_cache():
        for idx, form_type in enumerate(form_types):
            try:
                registry.check_filing_limit()
            except FilingLimitExceededError as exc:
                _print_error(
                    "Filing limit reached",
                    exc.message,
                    hint=(
                        "Remove filings with the management command or raise the "
                        "limit via DB_MAX_FILINGS."
                    ),
                )
                raise typer.Exit(code=1) from None

            form_label = f" ({idx + 1}/{len(form_types)})" if len(form_types) > 1 else ""

            with _make_progress() as progress:
                if effective_per_form == 1:
                    step_task = progress.add_task(
                        f"Fetching {ticker} {form_type}{form_label}...",
                        total=len(_STEPS),
                    )
                    s, sk, f = _ingest_one_form(
                        ticker,
                        form_type,
                        count=1,
                        year=year,
                        start_date=start_date,
                        end_date=end_date,
                        fetcher=fetcher,
                        orchestrator=orchestrator,
                        registry=registry,
                        store=store,
                        progress=progress,
                        step_task_id=step_task,
                        form_label=form_label,
                    )
                else:
                    estimated = effective_per_form or 0
                    filing_task = progress.add_task(
                        f"{ticker} {form_type}{form_label}: filings",
                        total=estimated or None,
                    )
                    step_task = progress.add_task(
                        f"Fetching {ticker} {form_type}{form_label}...",
                        total=len(_STEPS),
                    )
                    s, sk, f = _ingest_one_form(
                        ticker,
                        form_type,
                        count=effective_per_form,
                        year=year,
                        start_date=start_date,
                        end_date=end_date,
                        fetcher=fetcher,
                        orchestrator=orchestrator,
                        registry=registry,
                        store=store,
                        progress=progress,
                        step_task_id=step_task,
                        filing_task_id=filing_task,
                        form_label=form_label,
                    )

            succeeded += s
            skipped += sk
            failed += f

    if len(form_types) > 1 or effective_per_form != 1:
        _print_summary(succeeded, skipped, failed, header="Summary:")

    if failed > 0 and succeeded == 0 and skipped == 0:
        raise typer.Exit(code=1)


@ingest_app.command("batch")
def batch(
    tickers: Annotated[
        list[str],
        typer.Argument(help="Stock ticker symbols (e.g. AAPL MSFT GOOGL)."),
    ],
    form: Annotated[
        str,
        typer.Option(
            "--form",
            "-f",
            help="SEC form type(s), comma-separated (e.g. 8-K, 10-K, 10-Q).",
        ),
    ] = DEFAULT_FORM_TYPES,
    total: Annotated[
        int | None,
        typer.Option(
            "--total",
            "-t",
            help="Total filings per ticker (across form types, newest first).",
            min=1,
        ),
    ] = None,
    number: Annotated[
        int | None,
        typer.Option(
            "--number",
            "-n",
            help="Number of filings per ticker per form type.",
            min=1,
        ),
    ] = None,
    year: Annotated[
        int | None,
        typer.Option("--year", "-y", help="Filter by filing year (e.g. 2023)."),
    ] = None,
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", help="Start date filter (YYYY-MM-DD)."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option("--end-date", help="End date filter (YYYY-MM-DD)."),
    ] = None,
) -> None:
    """Fetch and ingest filings for multiple companies.

    Examples:

        sec-rag ingest batch AAPL MSFT GOOGL

        sec-rag ingest batch AAPL MSFT -f 10-K

        sec-rag ingest batch AAPL MSFT -t 3

        sec-rag ingest batch AAPL MSFT GOOGL -n 2 -y 2023
    """
    tickers = [t.upper() for t in tickers]

    if total is not None and number is not None:
        _print_error(
            "Invalid flag combination",
            "--total and --number are mutually exclusive.",
        )
        raise typer.Exit(code=1)

    try:
        form_types = parse_form_types(form)
    except ValueError as exc:
        _print_error("Invalid form type", str(exc))
        raise typer.Exit(code=1) from None

    _validate_date(start_date, "--start-date")
    _validate_date(end_date, "--end-date")

    fetcher, orchestrator, registry, store = _build_pipeline()

    total_succeeded = 0
    total_skipped = 0
    total_failed = 0

    # --- Cross-form mode: -t (total per ticker across form types) ------------
    if total is not None:
        for ticker in tickers:
            console.print(f"\n[bold]{escape(ticker)}[/bold]")
            s, sk, f = _ingest_across_forms(
                ticker,
                form_types,
                count=total,
                year=year,
                start_date=start_date,
                end_date=end_date,
                fetcher=fetcher,
                orchestrator=orchestrator,
                registry=registry,
                store=store,
            )
            total_succeeded += s
            total_skipped += sk
            total_failed += f

        _print_summary(
            total_succeeded,
            total_skipped,
            total_failed,
            header="Batch complete:",
        )
        if total_failed > 0 and total_succeeded == 0 and total_skipped == 0:
            raise typer.Exit(code=1)
        return

    # --- Per-form mode: -n or default ---------------------------------------
    effective_per_form = _resolve_per_form_count(
        number=number, year=year, start_date=start_date, end_date=end_date
    )

    for ticker in tickers:
        # One edgartools ``Company`` per ticker across its forms (F22).
        with fetcher.company_cache():
            for idx, form_type in enumerate(form_types):
                try:
                    registry.check_filing_limit()
                except FilingLimitExceededError as exc:
                    _print_error(
                        "Filing limit reached",
                        exc.message,
                        hint=(
                            "Remove filings with the management command or raise the "
                            "limit via DB_MAX_FILINGS."
                        ),
                    )
                    raise typer.Exit(code=1) from None

                form_label = f" ({idx + 1}/{len(form_types)})" if len(form_types) > 1 else ""

                with _make_progress() as progress:
                    if effective_per_form == 1:
                        step_task = progress.add_task(
                            f"Fetching {ticker} {form_type}{form_label}...",
                            total=len(_STEPS),
                        )
                        s, sk, f = _ingest_one_form(
                            ticker,
                            form_type,
                            count=1,
                            year=year,
                            start_date=start_date,
                            end_date=end_date,
                            fetcher=fetcher,
                            orchestrator=orchestrator,
                            registry=registry,
                            store=store,
                            progress=progress,
                            step_task_id=step_task,
                            form_label=form_label,
                        )
                    else:
                        estimated = effective_per_form or 0
                        filing_task = progress.add_task(
                            f"{ticker} {form_type}{form_label}: filings",
                            total=estimated or None,
                        )
                        step_task = progress.add_task(
                            f"Fetching {ticker} {form_type}{form_label}...",
                            total=len(_STEPS),
                        )
                        s, sk, f = _ingest_one_form(
                            ticker,
                            form_type,
                            count=effective_per_form,
                            year=year,
                            start_date=start_date,
                            end_date=end_date,
                            fetcher=fetcher,
                            orchestrator=orchestrator,
                            registry=registry,
                            store=store,
                            progress=progress,
                            step_task_id=step_task,
                            filing_task_id=filing_task,
                            form_label=form_label,
                        )

                total_succeeded += s
                total_skipped += sk
                total_failed += f

    _print_summary(
        total_succeeded,
        total_skipped,
        total_failed,
        header="Batch complete:",
    )
    if total_failed > 0 and total_succeeded == 0 and total_skipped == 0:
        raise typer.Exit(code=1)
