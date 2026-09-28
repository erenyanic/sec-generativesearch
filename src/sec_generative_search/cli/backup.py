"""Synchronous CLI wrappers over :class:`BackupService`.

Two bare functions:

- :func:`backup` — wraps :meth:`BackupService.backup` with a Rich
  progress bar and the destructive-action confirmation pattern shared
  with :mod:`cli.reindex` / :mod:`cli.evict`.
- :func:`restore` — wraps :meth:`BackupService.restore`; resolves the
  host's currently-configured embedder stamp from settings + registry
  (no embedder construction needed, eviction-style probe) and hands it
  to the service for refuse-on-mismatch validation.

The functions are exported bare so the CLI can attach them directly
without extra wrapper code.
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
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import (
    DatabaseError,
    EmbeddingCollectionMismatchError,
)
from sec_generative_search.database import BackupService

__all__ = ["backup", "restore"]


console = Console()


def backup(
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help="Destination tarball path (e.g. /backups/2026-04-28.tar.gz).",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite an existing file at the output path.",
        ),
    ] = False,
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
    Write a byte-faithful tarball backup of ChromaDB + SQLite.

    The artefact contains a top-level ``MANIFEST.json``
    (``format_version``, ``created_at_utc``, ``embedder_stamp``,
    ``schema_version``, ``sqlcipher_encrypted``), a live SQLite
    snapshot taken via the DB-API ``Connection.backup()`` API, and a
    recursive copy of the ChromaDB persistence directory.  The output
    file is written with mode ``0600`` so only the operator can read
    it.

    The operator must quiesce writers (stop the API and any
    long-running ingest jobs) before running this command — Chroma
    exposes no atomic-snapshot primitive, mirroring the
    ``sec-rag manage reindex`` operator-scope contract.

    Examples:

        sec-rag manage backup -o /backups/$(date +%F).tar.gz

        sec-rag manage backup -o backup.tar.gz --force -y
    """
    console.print(
        "\n[bold yellow]Backup storage layer[/bold yellow]\n"
        f"  Output: [cyan]{escape(output)}[/cyan]\n"
        "  [dim italic]Hint: stop the API and any in-flight ingest "
        "before running so the snapshot is consistent.[/dim italic]\n"
    )

    if not yes:
        confirmed = typer.confirm("Proceed with backup?")
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)

    service = BackupService()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Total is unknown up-front; use ``total=None`` to render an
        # indeterminate task that completes when the service returns.
        task_id = progress.add_task("Backing up storage...", total=None)

        def _on_progress(step: str, current: int, total: int) -> None:
            description = {
                "backup-sqlite": "Snapshotting SQLite",
                "backup-chroma": "Copying ChromaDB",
                "backup-archive": "Writing tarball",
            }.get(step, step)
            progress.update(
                task_id,
                description=description,
                completed=current,
                total=total,
            )

        try:
            report = service.backup(
                output,
                force=force,
                progress_callback=_on_progress,
            )
        except DatabaseError as exc:
            progress.stop()
            print_error(
                "Backup failed",
                exc.message,
                details=exc.details,
            )
            raise typer.Exit(code=1) from None
        except KeyboardInterrupt:
            progress.stop()
            console.print(
                "\n[yellow]Interrupted.[/yellow] The output tarball at "
                f"'{escape(output)}' may be partial — delete it and re-run "
                "'sec-rag manage backup'."
            )
            raise typer.Exit(code=130) from None

    console.print(
        "\n[green]Backup complete:[/green] "
        f"{report.size_bytes} bytes at "
        f"{escape(report.output_path)} "
        f"(embedder={report.embedder_stamp.provider}/{report.embedder_stamp.model}, "
        f"schema={report.schema_version}, "
        f"encrypted={report.sqlcipher_encrypted}, "
        f"{report.duration_seconds:.1f}s)"
    )


def restore(
    input_path: Annotated[
        str,
        typer.Option(
            "--input",
            "-i",
            help="Source tarball path produced by 'sec-rag manage backup'.",
        ),
    ],
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the confirmation prompt before replacing live state.",
        ),
    ] = False,
) -> None:
    """
    Restore a backup tarball into the configured ChromaDB + SQLite paths.

    Validates the manifest against the host's currently-configured
    embedder stamp (from ``EMBEDDING_PROVIDER`` + ``EMBEDDING_MODEL_NAME``)
    and the host's encryption state (``DB_ENCRYPTION_KEY{,_FILE}``)
    *before any filesystem mutation*.  Refuses with a typed exception
    on mismatch — the live state is never touched on the refusal path.

    The operator must quiesce writers (stop the API and any running
    ingest jobs) before running this command.  After restore, the next
    :class:`MetadataRegistry` open will run any pending migrations on
    the artefact's SQLite, so a ``schema_version`` lower than the
    host's latest available is the lossless-upgrade case rather than
    an error.

    Examples:

        sec-rag manage restore -i /backups/2026-04-28.tar.gz

        sec-rag manage restore -i backup.tar.gz -y
    """
    # The host's expected stamp — registry probe only, no embedder built.
    embedding = get_settings().embedding
    expected_stamp = resolve_stamp(embedding.provider, embedding.model_name)

    console.print(
        "\n[bold yellow]Restore storage layer[/bold yellow]\n"
        f"  Input: [cyan]{escape(input_path)}[/cyan]\n"
        f"  Expected stamp: [cyan]{expected_stamp.provider}[/cyan] / "
        f"[green]{expected_stamp.model}[/green] (dim={expected_stamp.dimension})\n"
        "  [red]This replaces the live ChromaDB + SQLite paths.[/red]\n"
        "  [dim italic]Hint: stop the API and any in-flight ingest "
        "before running.[/dim italic]\n"
    )

    if not yes:
        confirmed = typer.confirm("Proceed with restore?")
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)

    service = BackupService()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Restoring storage...", total=None)

        def _on_progress(step: str, current: int, total: int) -> None:
            description = {
                "restore-extract": "Extracting tarball",
                "restore-chroma": "Replacing ChromaDB",
                "restore-sqlite": "Replacing SQLite",
            }.get(step, step)
            progress.update(
                task_id,
                description=description,
                completed=current,
                total=total,
            )

        try:
            report = service.restore(
                input_path,
                expected_stamp=expected_stamp,
                progress_callback=_on_progress,
            )
        except EmbeddingCollectionMismatchError as exc:
            progress.stop()
            print_error(
                "Restore refused — embedder stamp mismatch",
                exc.message,
                details=(
                    f"Artefact stamp: {exc.actual.provider}/{exc.actual.model} "
                    f"(dim={exc.actual.dimension}); "
                    f"host stamp: {exc.expected.provider}/{exc.expected.model} "
                    f"(dim={exc.expected.dimension})."
                ),
                hint=exc.hint,
            )
            raise typer.Exit(code=1) from None
        except DatabaseError as exc:
            progress.stop()
            print_error(
                "Restore failed",
                exc.message,
                details=exc.details,
            )
            raise typer.Exit(code=1) from None
        except KeyboardInterrupt:
            progress.stop()
            console.print(
                "\n[yellow]Interrupted.[/yellow] Restore may be partial — "
                "the live ChromaDB or SQLite path may be in an inconsistent "
                "state. Re-run 'sec-rag manage restore' from the same tarball."
            )
            raise typer.Exit(code=130) from None

    console.print(
        "\n[green]Restore complete:[/green] "
        f"embedder={report.embedder_stamp.provider}/{report.embedder_stamp.model} "
        f"(dim={report.embedder_stamp.dimension}), "
        f"schema={report.schema_version}, "
        f"encrypted={report.sqlcipher_encrypted}, "
        f"{report.duration_seconds:.1f}s"
    )
