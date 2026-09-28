"""Synchronous CLI wrappers over the portable export / import services.

Two bare functions:

- :func:`export` — wraps :meth:`PortableExportService.export`.
  Read-only; constructs no embedder.
- :func:`import_` — wraps :meth:`PortableImportService.import_`.
  Builds the host's embedder via :func:`providers.factory.build_embedder`
    (the sole legal embedder construction site outside the factory's own
    tests) and composes a :class:`FilingStore` over
  the configured paths so the import path inherits the dual-store
  ordering and rollback semantics from the rest of the project.

``import_`` is exported with the trailing underscore because ``import``
is a Python keyword; Typer's ``command(name="import")`` attaches it
under the operator-facing ``import`` name.
"""

from __future__ import annotations

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

from sec_generative_search.cli._common import print_error, resolve_stamp
from sec_generative_search.config.settings import EmbeddingSettings, get_settings
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    DatabaseError,
)
from sec_generative_search.database import (
    ChromaDBClient,
    FilingStore,
    MetadataRegistry,
    PortableExportService,
    PortableImportService,
)
from sec_generative_search.providers.factory import build_embedder

__all__ = ["export", "import_"]


console = Console()


def _build_import_embedding_settings(provider: str, model_name: str) -> EmbeddingSettings:
    """Build :class:`EmbeddingSettings` honouring the host's configured target.

    Hosted providers reject non-default ``device`` / ``batch_size`` /
    ``idle_timeout_minutes`` via the model validator, which would trip
    when an operator has ``EMBEDDING_DEVICE=cuda`` configured for the
    live local embedder but is now importing into a hosted target.
    Pin the local-only knobs to their defaults whenever the provider
    is not ``local`` — same pattern as :func:`cli.reindex._build_embedding_settings`.
    """
    if provider == "local":
        return EmbeddingSettings(provider=provider, model_name=model_name)
    return EmbeddingSettings(
        provider=provider,
        model_name=model_name,
        device="auto",
        batch_size=32,
        idle_timeout_minutes=0,
    )


def export(
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Destination directory (will be created if missing).",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite an existing non-empty directory.",
        ),
    ] = False,
    tickers: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-t",
            help="Filter to filings whose ticker is in this list (repeatable).",
        ),
    ] = None,
    form_types: Annotated[
        list[str] | None,
        typer.Option(
            "--form-type",
            "-f",
            help="Filter to filings of this form type (repeatable, e.g. '10-K').",
        ),
    ] = None,
    accessions: Annotated[
        list[str] | None,
        typer.Option(
            "--accession",
            "-a",
            help="Filter to filings with this accession number (repeatable).",
        ),
    ] = None,
) -> None:
    """
    Export filings as portable JSONL chunks plus a manifest.

    The artefact is *not* a backup — it carries chunk text and filing
    metadata but no embeddings.  Imports re-embed through the host's
    configured embedder, so the export is portable across hosts that
    run different providers.

    Output layout:

        <output>/
            manifest.json
            chunks.jsonl

    Examples:

        sec-rag manage export -o ./exports/all/

        sec-rag manage export -o ./exports/aapl/ -t AAPL -f 10-K
    """
    service = PortableExportService()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("{task.completed} chunks"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Exporting filings...", total=None)

        def _on_progress(step: str, current: int, total: int) -> None:
            description = {
                "export-list-filings": "Listing filings",
                "export-chunks": "Writing chunks",
                "export-manifest": "Writing manifest",
            }.get(step, step)
            progress.update(
                task_id,
                description=description,
                completed=current,
                total=total,
            )

        try:
            report = service.export(
                output,
                force=force,
                tickers=tickers,
                form_types=form_types,
                accessions=accessions,
                progress_callback=_on_progress,
            )
        except DatabaseError as exc:
            progress.stop()
            print_error("Export failed", exc.message, details=exc.details)
            raise typer.Exit(code=1) from None
        except KeyboardInterrupt:
            progress.stop()
            console.print(
                "\n[yellow]Interrupted.[/yellow] The output directory at "
                f"'{escape(output)}' may be partial — delete it and re-run "
                "'sec-rag manage export'."
            )
            raise typer.Exit(code=130) from None

    console.print(
        "\n[green]Export complete:[/green] "
        f"{report.filing_count} filing(s), {report.chunk_count} chunk(s) "
        f"written to {escape(report.output_dir)} "
        f"(source embedder={report.source_embedder_stamp.provider}/"
        f"{report.source_embedder_stamp.model}, "
        f"{report.duration_seconds:.1f}s)"
    )


def import_(
    input_path: Annotated[
        str,
        typer.Option(
            "--input",
            "-i",
            help="Source directory produced by 'sec-rag manage export'.",
        ),
    ],
    tickers: Annotated[
        list[str] | None,
        typer.Option(
            "--ticker",
            "-t",
            help="Filter to filings whose ticker is in this list (repeatable).",
        ),
    ] = None,
    form_types: Annotated[
        list[str] | None,
        typer.Option(
            "--form-type",
            "-f",
            help="Filter to filings of this form type (repeatable).",
        ),
    ] = None,
    accessions: Annotated[
        list[str] | None,
        typer.Option(
            "--accession",
            "-a",
            help="Filter to filings with this accession number (repeatable).",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the confirmation prompt before writing.",
        ),
    ] = False,
) -> None:
    """
    Re-embed and load a portable export under the host's configured embedder.

    Constructs the host embedder via the factory seam, composes a
    :class:`FilingStore` over the configured ChromaDB + SQLite paths,
    and writes via :meth:`FilingStore.store_filing`
    ``register_if_new=True`` — duplicate accessions on the host are
    skipped (not overwritten).

    Examples:

        sec-rag manage import -i ./exports/all/

        sec-rag manage import -i ./exports/aapl/ -t AAPL -y
    """
    settings = get_settings()
    provider_name = settings.embedding.provider
    model_name = settings.embedding.model_name

    # Resolve the host stamp via the registry — O(1), credential-free.
    target_stamp = resolve_stamp(provider_name, model_name)

    # Build the embedder through the factory seam.  Hosted providers
    # need their local-only knobs pinned to defaults to avoid tripping
    # the EmbeddingSettings validator on a stray ``EMBEDDING_DEVICE=cuda``.
    try:
        embedding_settings = _build_import_embedding_settings(provider_name, model_name)
        embedder = build_embedder(embedding_settings)
    except ConfigurationError as exc:
        print_error(
            "Embedder construction failed",
            exc.message,
            hint="Set the expected API-key env var for this provider.",
        )
        raise typer.Exit(code=1) from None
    except KeyError as exc:
        print_error(
            "Embedder unavailable",
            f"Provider {provider_name!r} requires additional packages.",
            details=str(exc),
            hint=(
                "Install the matching extra, e.g. "
                "`uv pip install -e '.[local-embeddings]'` for the local provider."
            ),
        )
        raise typer.Exit(code=1) from None

    console.print(
        "\n[bold yellow]Import portable export[/bold yellow]\n"
        f"  Input: [cyan]{escape(input_path)}[/cyan]\n"
        f"  Re-embedding under: [cyan]{provider_name}[/cyan] / "
        f"[green]{model_name}[/green] (dim={target_stamp.dimension})\n"
        "  [dim italic]Duplicate accessions on the host are skipped, not "
        "overwritten.[/dim italic]\n"
    )

    if not yes:
        confirmed = typer.confirm("Proceed with import?")
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)

    # Compose the FilingStore over the live storage paths.  Both the
    # ChromaDBClient stamp seal and the registry's WAL connection are
    # established at construction; failures here are configuration
    # errors the operator must resolve before importing.
    try:
        chroma = ChromaDBClient(target_stamp)
        registry = MetadataRegistry()
        store = FilingStore(chroma, registry)
    except DatabaseError as exc:
        print_error(
            "Storage open failed",
            exc.message,
            details=exc.details,
        )
        raise typer.Exit(code=1) from None

    service = PortableImportService(store, embedder)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("{task.completed}/{task.total} filings"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Importing filings...", total=None)

        def _on_progress(step: str, current: int, total: int) -> None:
            description = {
                "import-read": "Reading JSONL",
                "import-embed": "Embedding chunks",
                "import-write": "Writing to storage",
            }.get(step, step)
            progress.update(
                task_id,
                description=description,
                completed=current,
                total=total,
            )

        try:
            report = service.import_(
                input_path,
                tickers=tickers,
                form_types=form_types,
                accessions=accessions,
                progress_callback=_on_progress,
            )
        except DatabaseError as exc:
            progress.stop()
            print_error("Import failed", exc.message, details=exc.details)
            raise typer.Exit(code=1) from None
        except KeyboardInterrupt:
            progress.stop()
            console.print(
                "\n[yellow]Interrupted.[/yellow] Import may be partial — "
                "duplicate accessions on a re-run are skipped, so it is safe "
                "to re-run 'sec-rag manage import' on the same input."
            )
            raise typer.Exit(code=130) from None

    console.print(
        "\n[green]Import complete:[/green] "
        f"{report.filings_imported} imported, {report.filings_skipped} skipped "
        f"({report.chunks_imported} chunk(s)) in {report.duration_seconds:.1f}s"
    )
