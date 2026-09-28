"""Synchronous CLI wrapper over :meth:`FilingStore.evict_expired`.

Reads the configured embedder stamp from settings + registry (no factory
call — eviction performs no embeddings, only deletes by accession),
opens both backing stores, and delegates to the dual-store evictor.
The cutoff defaults to ``DB_RETENTION_MAX_AGE_DAYS`` and can be
overridden per-invocation with ``--max-age-days``.

The function is exported bare so the CLI can attach it directly
without extra wrapper code.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from sec_generative_search.cli._common import print_error, resolve_stamp
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import DatabaseError
from sec_generative_search.database import (
    ChromaDBClient,
    FilingStore,
    MetadataRegistry,
)

__all__ = ["evict"]


console = Console()


def evict(
    max_age_days: Annotated[
        int | None,
        typer.Option(
            "--max-age-days",
            "-d",
            help=(
                "Cutoff age in days. Filings whose ingested_at is older "
                "than this are dropped from both stores. Defaults to "
                "DB_RETENTION_MAX_AGE_DAYS."
            ),
        ),
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
    """
    Drop filings older than the cutoff from ChromaDB and SQLite.

    Composes :meth:`FilingStore.evict_expired` over the configured
    storage layer.  The cutoff comes from ``DB_RETENTION_MAX_AGE_DAYS``
    by default, which itself defaults to the deployment profile's
    baseline (local=0/disabled, team=90, cloud=30); pass
    ``--max-age-days N`` to override per-invocation.

    A cutoff of ``0`` disables eviction entirely; the command refuses
    to run rather than silently no-op so an operator who expected
    a sweep gets a clear hint instead of confused logs.

    Examples:

        sec-rag manage evict                  # uses DB_RETENTION_MAX_AGE_DAYS

        sec-rag manage evict --max-age-days 30 -y

        sec-rag manage evict -d 90            # ad-hoc 90-day sweep
    """
    settings = get_settings()

    # Resolve the cutoff: explicit flag wins; otherwise fall through to
    # settings.  Both surfaces validate on entry — settings rejects
    # negatives at load, the registry rejects non-positives at the
    # primitive — so a non-positive value here is exclusively a
    # zero-from-disabled-settings case.
    cutoff = max_age_days if max_age_days is not None else settings.database.retention_max_age_days

    if cutoff <= 0:
        print_error(
            "Eviction disabled",
            f"Cutoff is {cutoff} day(s); nothing to do.",
            hint=(
                "Pass --max-age-days N for an ad-hoc sweep, or set "
                "DB_RETENTION_MAX_AGE_DAYS to a positive value (or pick a "
                "non-local profile via DB_DEPLOYMENT_PROFILE)."
            ),
        )
        raise typer.Exit(code=1)

    # Compose the embedder stamp from settings + registry.  No factory
    # call is needed — eviction only deletes by accession, so we never
    # construct a real embedder.  The registry's get_dimension probe is
    # O(1) and credential-free.
    stamp = resolve_stamp(settings.embedding.provider, settings.embedding.model_name)

    # Confirmation (destructive action).
    console.print(
        "\n[bold yellow]Evict expired filings[/bold yellow]\n"
        f"  Cutoff: filings older than [cyan]{cutoff}[/cyan] day(s) "
        "(by ingested_at)\n"
        f"  Profile: [cyan]{settings.database.deployment_profile}[/cyan] "
        f"(DB_RETENTION_MAX_AGE_DAYS="
        f"{settings.database.retention_max_age_days})\n"
        "  [red]This deletes the expired filings from both ChromaDB "
        "and SQLite.[/red]\n"
    )

    if not yes:
        confirmed = typer.confirm("Proceed with eviction?")
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)

    # Open both stores and delegate.  ChromaDBClient construction will
    # refuse loudly if the configured stamp does not match the live
    # collection — the same seal the rest of the storage layer enforces.
    try:
        chroma = ChromaDBClient(stamp)
        registry = MetadataRegistry()
        store = FilingStore(chroma, registry)
        report = store.evict_expired(cutoff)
    except DatabaseError as exc:
        print_error(
            "Eviction failed",
            exc.message,
            details=exc.details,
        )
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        # Best-effort message; the SQLite connection (if opened) will
        # close on interpreter shutdown.  The ChromaDB delete is
        # idempotent on a partial run, so a re-invocation is safe.
        console.print(
            "\n[yellow]Interrupted.[/yellow] Eviction may be partial — "
            "re-run 'sec-rag manage evict' to complete the sweep."
        )
        raise typer.Exit(code=130) from None

    if report.filings_evicted == 0:
        console.print(
            f"[dim]Nothing to evict — no filings older than {report.max_age_days} day(s).[/dim]"
        )
    else:
        console.print(
            f"\n[green]Eviction complete:[/green] "
            f"{report.filings_evicted} filing(s) "
            f"({report.chunks_evicted} chunk(s)) older than "
            f"{report.max_age_days} day(s) removed."
        )
