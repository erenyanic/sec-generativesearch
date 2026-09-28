"""Synchronous CLI wrapper over :class:`ReindexService` with a Rich progress bar.

The command resolves the target embedder through the factory seam, keeps
``--dimension`` off the surface, and reports interrupt / failure states with
operator-facing messages.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from sec_generative_search.cli._common import print_error
from sec_generative_search.config.settings import EmbeddingSettings
from sec_generative_search.core.exceptions import ConfigurationError, DatabaseError
from sec_generative_search.core.types import EmbedderStamp
from sec_generative_search.database import ReindexService
from sec_generative_search.providers.factory import build_embedder
from sec_generative_search.providers.registry import ProviderRegistry, ProviderSurface

__all__ = ["reindex"]


console = Console()


def _build_embedding_settings(provider: str, model_name: str) -> EmbeddingSettings:
    """Build an :class:`EmbeddingSettings` that honours the CLI flags.

    Two concerns are fused here:

    - Constructor kwargs override env vars in pydantic-settings, so the
      ``--provider`` / ``--model`` flags always win.
    - ``EmbeddingSettings`` rejects non-default ``device`` / ``batch_size``
      / ``idle_timeout_minutes`` whenever ``provider != "local"``.  For a
      hosted target we must pin those knobs to their defaults so a stray
      ``EMBEDDING_DEVICE=cuda`` in the environment (common when the live
      embedder is local) does not trip the model validator during a
      reindex into a hosted embedder.  For ``local`` targets we let env
      vars flow through so the operator can still tune the reindex run.
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


def reindex(
    provider: Annotated[
        str,
        typer.Option(
            "--provider",
            "-p",
            help="Target embedding provider (registry key, e.g. openai, gemini, local).",
        ),
    ],
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Target embedding model slug. Omit to use the provider's default model.",
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
    Re-embed every stored chunk through a different embedder.

    Drops and rebuilds the ``sec_filings`` ChromaDB collection against the
    target ``(provider, model, dimension)`` without re-fetching anything
    from SEC EDGAR.  The live collection is replaced only after the new
    vectors are fully written to a staging collection; on any pre-swap
    failure the staging is dropped and the live collection is untouched.

    The operator is expected to take a filesystem-level backup of the
    Chroma directory before running this command.  Mid-swap crashes are
    operator-scope.

    Examples:

        sec-rag manage reindex --provider openai --model text-embedding-3-small

        sec-rag manage reindex -p gemini -m text-embedding-004 -y

        sec-rag manage reindex -p local  # uses LocalEmbeddingProvider default model
    """
    # Resolve the target model slug.
    # The registry is the single source of truth for both the default
    # model and the dimension — we never derive either from user input.
    # ``get_class`` raises ``KeyError`` for unknown providers or providers
    # whose optional extras are not installed; both surfaces carry
    # actionable messages we re-render as operator errors.
    try:
        provider_cls = ProviderRegistry.get_class(provider, ProviderSurface.EMBEDDING)
    except KeyError as exc:
        print_error(
            "Unknown embedding provider",
            f"{provider!r} is not a registered embedding provider.",
            details=str(exc),
            hint="Run 'sec-rag provider list --surface embedding' to see registered providers.",
        )
        raise typer.Exit(code=1) from None

    resolved_model = model or provider_cls.default_model

    # Resolve the target dimension (O(1), no network).
    try:
        target_dim = ProviderRegistry.get_dimension(provider, resolved_model)
    except ValueError as exc:
        print_error(
            "Unknown embedding model",
            f"{resolved_model!r} is not registered for provider {provider!r}.",
            details=str(exc),
            hint="Omit --model to use the provider's default, or pick a slug from the "
            "MODEL_DIMENSIONS catalogue for this provider.",
        )
        raise typer.Exit(code=1) from None

    # Build the embedder through the factory seam.
    try:
        settings = _build_embedding_settings(provider, resolved_model)
        embedder = build_embedder(settings)
    except ConfigurationError as exc:
        print_error(
            "Embedder construction failed",
            exc.message,
            hint="Set the expected API-key env var for this provider.",
        )
        raise typer.Exit(code=1) from None
    except KeyError as exc:
        # Raised by ``ProviderRegistry.get_class`` re-entry inside the
        # factory when a provider requiring an optional extra is picked
        # without the extra installed (e.g. ``local`` without
        # ``[local-embeddings]``).
        print_error(
            "Embedder unavailable",
            f"Provider {provider!r} requires additional packages.",
            details=str(exc),
            hint="Install the matching extra, e.g. "
            "`uv pip install -e '.[local-embeddings]'` for the local provider.",
        )
        raise typer.Exit(code=1) from None

    target_stamp = EmbedderStamp(
        provider=provider,
        model=resolved_model,
        dimension=target_dim,
    )

    # Confirmation (destructive action).
    # The body deliberately surfaces only what the operator asked for
    # plus the derived dimension.  The source stamp is not probed here —
    # the service will refuse loudly on a no-op / corrupt / unstamped
    # source, and re-probing it in the CLI would duplicate storage-layer
    # logic the service owns.
    console.print(
        "\n[bold yellow]Reindex Database[/bold yellow]\n"
        f"  Target: [cyan]{target_stamp.provider}[/cyan] / "
        f"[green]{target_stamp.model}[/green] (dim={target_stamp.dimension})\n"
        "  [red]This drops and rebuilds the sec_filings collection.[/red]\n"
        "  [dim italic]Hint: take a filesystem-level backup of the Chroma "
        "directory before proceeding.[/dim italic]\n"
    )

    if not yes:
        confirmed = typer.confirm("Proceed with reindex?")
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)

    # Drive the service behind a single relabelling progress bar.
    service = ReindexService()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("{task.completed}/{task.total} chunks"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Total is unknown until the first callback fires; Rich accepts
        # ``total=None`` as an indeterminate task.
        task_id = progress.add_task("Re-embedding chunks...", total=None)

        def _on_progress(step: str, current: int, total: int) -> None:
            if step == "reindex":
                description = "Re-embedding chunks"
            elif step == "reindex-swap":
                description = "Copying to live collection"
            else:
                # Unknown step names are forwarded verbatim so future
                # service additions surface in the bar without a silent
                # swallow here.
                description = step
            progress.update(
                task_id,
                description=description,
                completed=current,
                total=total,
            )

        try:
            report = service.run(
                target_stamp,
                embedder,
                progress_callback=_on_progress,
            )
        except DatabaseError as exc:
            progress.stop()
            # All refuse-early messages already carry operator-actionable
            # text; we re-render rather than re-wording, so a future
            # service-level change in hint is visible without edits here.
            print_error(
                "Reindex failed",
                exc.message,
                details=exc.details,
            )
            raise typer.Exit(code=1) from None
        except KeyboardInterrupt:
            progress.stop()
            console.print(
                "\n[yellow]Interrupted.[/yellow] The staging collection may still exist — "
                "re-run 'sec-rag manage reindex' to drop it and retry."
            )
            raise typer.Exit(code=130) from None

    console.print(
        "\n[green]Reindex complete:[/green] "
        f"{report.chunks_copied} chunk(s) re-embedded from "
        f"{report.source_stamp.provider}/{report.source_stamp.model} "
        f"→ {report.target_stamp.provider}/{report.target_stamp.model} "
        f"(dim={report.target_stamp.dimension}) in {report.duration_seconds:.1f}s"
    )
