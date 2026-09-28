"""Helpers shared by the ``sec-rag`` sub-command modules.

One copy each (OPTIMIZATIONS F32) of what every sub-command module used to
carry: the operator error renderer (ten copies in two variants), the
``YYYY-MM-DD`` flag validator (four) and the embedder-stamp resolution
(eight, some with a stale hint).

Test seams stay where the tests patch them: :func:`resolve_stamp` takes the
provider and model as arguments, so each command keeps reading its *own*
module's ``get_settings`` (which the CLI tests monkeypatch per module).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.markup import escape

from sec_generative_search.cli._json import OutputFormat, error_envelope, is_json, print_json
from sec_generative_search.core.types import EmbedderStamp
from sec_generative_search.providers.registry import ProviderRegistry

if TYPE_CHECKING:
    from sec_generative_search.cli._errors import CliError

__all__ = ["print_cli_error", "print_error", "resolve_stamp", "validate_date"]

# Rich resolves ``sys.stdout`` at print time, so this captures under
# ``CliRunner`` exactly like each module's own console does.
_console = Console()


def print_error(
    label: str,
    message: str,
    *,
    details: str | None = None,
    hint: str | None = None,
    output: OutputFormat = OutputFormat.TEXT,
    error_code: str | None = None,
) -> None:
    """Render an operator-facing error: Rich text, or a JSON envelope.

    Every string goes through :func:`rich.markup.escape` — hints carry
    literal ``[...]`` (e.g. ``'.[local-embeddings]'``) that Rich would
    otherwise silently strip as markup.  Under ``--output json`` the
    document is :func:`error_envelope` on stdout instead; ``error_code``
    is its stable slug (derived from ``label`` when omitted).
    """
    if is_json(output):
        slug = error_code or label.lower().replace(" ", "_")
        print_json(error_envelope(slug, message, hint=hint, details=details))
        return
    _console.print(f"[red]{escape(label)}:[/red] {escape(message)}")
    if details:
        _console.print(f"  [dim]{escape(details)}[/dim]")
    if hint:
        _console.print(f"  [dim italic]Hint: {escape(hint)}[/dim italic]")


def print_cli_error(error: CliError, *, output: OutputFormat = OutputFormat.TEXT) -> None:
    """Render a classified :class:`~sec_generative_search.cli._errors.CliError`."""
    print_error(
        error.label,
        error.message,
        details=error.details,
        hint=error.hint,
        output=output,
        error_code=error.error_code,
    )


def validate_date(value: str | None, param_name: str) -> str | None:
    """Validate a ``YYYY-MM-DD`` flag at the CLI boundary.

    Raising :class:`typer.BadParameter` renders the error like any other
    flag error (``RetrievalService`` would also reject a malformed date,
    but later and less clearly).  ``datetime`` is imported here, not at
    module level, to keep ``--help`` / ``--version`` start-up lean.
    """
    if value is None:
        return None
    from datetime import datetime

    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise typer.BadParameter(
            f"Invalid date format for {param_name}: {value!r}. Expected YYYY-MM-DD."
        ) from None
    return value


def resolve_stamp(
    provider: str,
    model: str,
    *,
    output: OutputFormat = OutputFormat.TEXT,
) -> EmbedderStamp:
    """Compose the collection's expected :class:`EmbedderStamp` for the host.

    ``ProviderRegistry.get_dimension`` is O(1) and credential-free — no
    embedder is built here.  An unresolvable provider/model prints the
    ``embedder_configuration_invalid`` error and exits 1.
    """
    try:
        dimension = ProviderRegistry.get_dimension(provider, model)
    except (KeyError, ValueError) as exc:
        print_error(
            "Embedder configuration invalid",
            f"Cannot resolve dimension for {provider}/{model}.",
            details=str(exc),
            hint=(
                "Check EMBEDDING_PROVIDER and EMBEDDING_MODEL_NAME — "
                "`sec-rag provider list --surface embedding` shows the valid combinations."
            ),
            output=output,
            error_code="embedder_configuration_invalid",
        )
        raise typer.Exit(code=1) from None
    return EmbedderStamp(provider=provider, model=model, dimension=dimension)
