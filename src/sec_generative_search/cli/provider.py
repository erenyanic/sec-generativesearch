"""Operator-scope provider-credential management.

Three subcommands live here:

- ``sec-rag provider list`` — enumerate every registered provider
  (LLM + embedding surfaces) with its default model, pricing tier of
  that model, admin-env var name, and a flag indicating whether a key
  currently resolves through the operator-scope chain.  Read-only;
  never instantiates an adapter.
- ``sec-rag provider validate`` — wrap
  :func:`core.credentials.validate_credential` to round-trip a key
  against the upstream provider.  Resolves the key through the same
  operator-scope chain ``rag query`` / ``rag chat`` use
  (``encrypted-user (ADMIN_USER_ID="__admin__") → admin-env``).
- ``sec-rag provider set`` — write a credential into the
  :class:`EncryptedCredentialStore`.  **Hard-fails** with a single
  hint when any of the three opt-in steps is missing (install the
  ``[encryption]`` extra, set ``DB_ENCRYPTION_KEY`` /
  ``DB_ENCRYPTION_KEY_FILE``, set
  ``DB_PERSIST_PROVIDER_CREDENTIALS=true``).  Never falls back to
  plaintext on disk.

Trust model and load-bearing rules:

1. ``set`` MUST hard-fail when the encrypted-credential store is not
   available — silently writing to a plaintext sqlite3 file would
   defeat the entire point of opting in.  Contrast with
   ``cli/rag.py``'s read-path resolver chain, which silently degrades
   to admin-env when the encrypted tier is absent (settings validation
   should make that unreachable in practice; reading admin-env is the
   documented Scenario A path).
2. No ``--api-key`` flag on ``set`` (shell-history /
   ``/proc/<pid>/environ`` footgun).  The key is prompted on stdin via
   :func:`typer.prompt` with ``hide_input=True``; piped invocations
   (``echo sk-... | sec-rag provider set openai``) work too.
3. ``ADMIN_USER_ID = "__admin__"`` is duplicated locally rather than
   imported from ``api/dependencies.py`` — the two surfaces are
   deliberately decoupled (same rationale as ``cli/rag.py``).
4. Every credential touch already audit-logs via
   :class:`EncryptedCredentialStore` / :func:`validate_credential` —
   no additional audit lines here, but the rendered output NEVER
   echoes the raw key (only its masked tail).
"""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from sec_generative_search.cli._json import (
    OutputFormat,
    coerce_output_format,
    error_envelope,
    is_json,
    print_json,
)
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.credentials import (
    chain_resolvers,
    encrypted_user_resolver,
    validate_credential,
)
from sec_generative_search.core.exceptions import (
    CatalogueRefreshError,
    ConfigurationError,
    DatabaseError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.security import mask_secret
from sec_generative_search.database import MetadataRegistry
from sec_generative_search.providers.factory import default_api_key_resolver
from sec_generative_search.providers.registry import (
    ProviderRegistry,
    ProviderSurface,
)

__all__ = ["provider_app"]


console = Console()

provider_app = typer.Typer(
    name="provider",
    help="Manage provider credentials at operator scope (admin-default keys).",
    no_args_is_help=True,
)


# Mirrors ``api.dependencies.ADMIN_USER_ID`` and ``cli.rag._ADMIN_USER_ID``.
# Duplicated rather than imported so this module has no dependency on
# either surface.
_ADMIN_USER_ID = "__admin__"


# Single hint string used by every ``provider set`` hard-fail.  Names
# all three opt-in steps so the operator does not have to guess which
# one is missing — the failure mode is "any one of three is missing",
# and we refuse to leak which one because the most useful diagnostic
# is the full list every time.
_HARD_FAIL_SET_HINT = (
    "Encrypted-credential storage requires all three opt-in steps: "
    "(1) install the `[encryption]` extra (`uv pip install -e '.[encryption]'`), "
    "(2) set DB_ENCRYPTION_KEY or DB_ENCRYPTION_KEY_FILE, "
    "(3) set DB_PERSIST_PROVIDER_CREDENTIALS=true.  "
    "Refusing to fall back to plaintext on disk."
)


# ---------------------------------------------------------------------------
# Rendering helpers (shared with cli.rag / cli.search)
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

    Mirrors the same shape as ``cli.rag._print_error`` so operator
    output stays uniform across the adapted CLI surface.

    When ``output == OutputFormat.JSON`` the document is an
    :func:`error_envelope` instead of the Rich text.  ``error_code``
    is the machine-readable ``error`` slug (mirrors the API envelope
    discipline; defaults to a slugified ``label``).  ``provider set``
    deliberately keeps ``output`` defaulted to TEXT because the JSON
    flag is scoped to the read paths (``list`` / ``validate``).
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


# ---------------------------------------------------------------------------
# Surface coercion
# ---------------------------------------------------------------------------


def _coerce_surface(value: str) -> ProviderSurface:
    """Lift a ``--surface`` flag onto :class:`ProviderSurface` strictly.

    Fails closed on unknown values via :class:`typer.BadParameter` —
    silently coercing a typo to a default surface would route the
    validate / set against the wrong adapter and produce confusing
    "key works for X but not Y" reports.
    """
    normalised = value.strip().lower()
    try:
        return ProviderSurface(normalised)
    except ValueError as exc:
        valid = ", ".join(s.value for s in ProviderSurface)
        raise typer.BadParameter(
            f"Invalid --surface: {value!r}. Expected one of: {valid}."
        ) from exc


# ---------------------------------------------------------------------------
# Read-path resolver chain (operator scope) — shared by `list` + `validate`
# ---------------------------------------------------------------------------


def _build_read_resolver(
    registry: MetadataRegistry | None,
) -> Any:
    """Compose ``encrypted-user → admin-env`` for read paths only.

    Read paths (``list`` shows ``key resolves?``; ``validate`` resolves
    before round-trip) silently degrade to admin-env alone when the
    encrypted tier is unavailable — same behaviour as ``cli.rag``'s
    read-path resolver.  Contrast with ``provider set`` below which
    hard-fails under the same conditions because *writing* to a
    non-encrypted store would silently land on plaintext.
    """
    settings = get_settings()
    chain: list = []
    if (
        registry is not None
        and settings.database.persist_provider_credentials
        and registry.encrypted
    ):
        # Local import keeps the encrypted-store dependency off the
        # admin-env-only operator's import path.
        from sec_generative_search.database.credentials import (
            EncryptedCredentialStore,
        )

        try:
            store = EncryptedCredentialStore(registry)
            chain.append(encrypted_user_resolver(store, _ADMIN_USER_ID))
        except ConfigurationError:
            # Defensive: settings validation should make this
            # unreachable, but a hand-crafted registry could land
            # here.  Read paths silently fall through to admin-env.
            pass
    chain.append(default_api_key_resolver)
    return chain_resolvers(*chain)


def _open_registry_or_none() -> MetadataRegistry | None:
    """Open the registry handle, or return ``None`` for the admin-env-only path.

    ``provider list`` runs even when the database is unavailable —
    enumerating registered adapters and admin-env vars is useful for
    bootstrapping a fresh deployment where the SQLite file does not
    yet exist.  We swallow :class:`DatabaseError` here and the read
    resolver collapses to admin-env alone.
    """
    try:
        return MetadataRegistry()
    except DatabaseError:
        return None


# ---------------------------------------------------------------------------
# `provider list`
# ---------------------------------------------------------------------------


# Provider name → admin-default env var.  Mirrors
# :data:`providers.factory._DEFAULT_ENV_VAR_BY_PROVIDER`; duplicated
# only at the rendering boundary so the CLI does not reach into a
# private module attribute.  A consistency test pairs the two tables.
_ENV_VAR_BY_PROVIDER: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "local": "HF_TOKEN",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "zai": "ZAI_API_KEY",
    "grok": "XAI_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "mimo": "MIMO_API_KEY",
}


def _pricing_label(name: str, surface: ProviderSurface, slug: str | None) -> str:
    """Pricing tier of the default model, with a permissive fallback.

    Resolves the tier through :meth:`ProviderRegistry.get_capability`,
    which reads the vendored model catalogue (LLM surface).  Returns
    ``"unknown"`` for embedding / reranker providers (no pricing surface)
    and for slugs absent from the catalogue (a freshly released model not
    yet added).  Never instantiates the provider.
    """
    if not slug or surface is not ProviderSurface.LLM:
        return "unknown"
    return ProviderRegistry.get_capability(name, surface, slug).pricing_tier.value


@provider_app.command("list")
def list_providers(
    surface: Annotated[
        str | None,
        typer.Option(
            "--surface",
            "-s",
            help="Filter by surface (llm / embedding / reranker).",
        ),
    ] = None,
    include_unavailable: Annotated[
        bool,
        typer.Option(
            "--include-unavailable",
            help=(
                "Show providers whose optional extras are not installed "
                "(default: hide them; an un-installed `[local-embeddings]` "
                "would otherwise pollute the listing)."
            ),
        ),
    ] = False,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output format: 'text' (default Rich table) or 'json'.  "
                "JSON shape is an allow-list lift exposing only the "
                "boolean 'key_resolves' flag — the masked tail rendered "
                "in text mode is NEVER serialised, so piping JSON into "
                "a log shipper cannot leak credential bytes."
            ),
        ),
    ] = "text",
) -> None:
    """Enumerate registered providers with admin-env and resolution status.

    Output columns mirror what ``provider set`` will accept as a name
    and what ``provider validate`` will round-trip against.  The
    ``Key resolves?`` column reports the operator-scope chain
    (encrypted-user → admin-env), so a green checkmark means
    ``provider validate`` would have something to send.
    """
    output_format = coerce_output_format(output)
    surface_filter = _coerce_surface(surface) if surface is not None else None

    registry = _open_registry_or_none()
    resolver = _build_read_resolver(registry)

    try:
        entries = ProviderRegistry.all_entries(
            surface_filter,
            include_unavailable=include_unavailable,
        )

        if is_json(output_format):
            providers_payload: list[dict[str, Any]] = []
            for entry in entries:
                # The entry records it — never import the adapter (and SDK) here.
                default_model = entry.default_model_slug or None
                pricing = _pricing_label(entry.name, entry.surface, default_model)
                env_var = _ENV_VAR_BY_PROVIDER.get(entry.name)
                # ``key_resolves`` is a boolean only — never include
                # the masked tail.  Rich-mode rendering surfaces the
                # tail for *visual* operator feedback (a glance
                # confirms which slot matched); piping that tail into
                # JSON would put it into log files that the operator
                # likely did not intend to keep credential metadata in.
                providers_payload.append(
                    {
                        "name": entry.name,
                        "surface": entry.surface.value,
                        "default_model": default_model,
                        "pricing_tier": pricing,
                        "admin_env_var": env_var,
                        "key_resolves": resolver(entry.name) is not None,
                        "requires_extras": list(entry.requires_extras),
                        "supports_arbitrary_models": entry.supports_arbitrary_models,
                        "supports_upstream_routing": entry.supports_upstream_routing,
                    }
                )
            print_json(
                {
                    "providers": providers_payload,
                    "total": len(providers_payload),
                }
            )
            return

        if not entries:
            console.print("[yellow]No providers registered for that surface.[/yellow]")
            return

        table = Table(
            title="[bold]Registered providers[/bold]",
            border_style="dim",
            header_style="bold",
            expand=True,
        )
        table.add_column("Provider", style="cyan", no_wrap=True)
        table.add_column("Surface", style="magenta", no_wrap=True)
        table.add_column("Default model", style="green")
        table.add_column("Pricing", style="yellow", no_wrap=True)
        table.add_column("Admin env var", style="dim", no_wrap=True)
        table.add_column("Key resolves?", justify="center", no_wrap=True)
        table.add_column("Notes", style="dim")

        for entry in entries:
            default_model = entry.default_model_slug or "—"
            pricing = _pricing_label(
                entry.name, entry.surface, None if default_model == "—" else default_model
            )
            env_var = _ENV_VAR_BY_PROVIDER.get(entry.name, "—")
            # Resolver call — returns a string when ANY tier holds a
            # value.  Render only the masked tail (mask_secret), never
            # the value itself.
            resolved = resolver(entry.name)
            if resolved is not None:
                key_cell = f"[green]✓[/green] [dim]{escape(mask_secret(resolved))}[/dim]"
            else:
                key_cell = "[red]·[/red]"
            notes_bits: list[str] = []
            if entry.requires_extras:
                notes_bits.append("needs " + "+".join(entry.requires_extras))
            if entry.supports_arbitrary_models:
                notes_bits.append("arbitrary models")
            if entry.supports_upstream_routing:
                notes_bits.append("upstream routing")
            table.add_row(
                escape(entry.name),
                escape(entry.surface.value),
                escape(default_model),
                escape(pricing),
                escape(env_var),
                key_cell,
                escape(", ".join(notes_bits) or "—"),
            )
        console.print(table)
    finally:
        if registry is not None:
            registry.close()


# ---------------------------------------------------------------------------
# `provider validate`
# ---------------------------------------------------------------------------


@provider_app.command("validate")
def validate(
    provider: Annotated[
        str,
        typer.Argument(help="Registered provider name (e.g. openai, anthropic, local)."),
    ],
    surface: Annotated[
        str,
        typer.Option(
            "--surface",
            "-s",
            help="Provider surface to validate against (default: llm).",
        ),
    ] = "llm",
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help=(
                "Model slug to bind during validation.  Required for "
                "embedding providers; ignored for LLM providers."
            ),
        ),
    ] = None,
    output: Annotated[
        str,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Output format: 'text' (default Rich line) or 'json' "
                "(allow-list lift of api.schemas.ProviderValidateResponse). "
                "Failure paths render as {error, message, hint} envelopes; "
                "the raw key is NEVER part of any envelope."
            ),
        ),
    ] = "text",
) -> None:
    """Round-trip a stored credential against the upstream provider.

    Exit codes:

    - ``0`` — provider accepted the key.
    - ``1`` — provider rejected the key (auth failure), or the key did
      not resolve at all, or the provider / model is unregistered.
    - ``2`` — transient upstream failure (rate limit, timeout, network).
      The key is *not* a verdict here; do not rotate.
    """
    output_format = coerce_output_format(output)
    surface_enum = _coerce_surface(surface)

    # Validate provider + model exist in the registry up front so the
    # operator sees a single envelope rather than a chain of failures.
    try:
        ProviderRegistry.get_capability(provider, surface_enum, model=model)
    except KeyError as exc:
        _print_error(
            "Unknown provider",
            f"{provider!r} is not a registered provider on the {surface_enum.value} surface.",
            details=str(exc),
            hint="Run `sec-rag provider list` to see registered names.",
            output=output_format,
            error_code="unknown_provider",
        )
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        _print_error(
            "Unknown model",
            f"{model!r} is not registered for provider {provider!r}.",
            details=str(exc),
            hint="Omit --model to use the provider default.",
            output=output_format,
            error_code="unknown_model",
        )
        raise typer.Exit(code=1) from None

    registry = _open_registry_or_none()
    try:
        resolver = _build_read_resolver(registry)
        api_key = resolver(provider)
        if api_key is None:
            env_var = _ENV_VAR_BY_PROVIDER.get(provider, "<unknown>")
            _print_error(
                "No credential",
                f"No API key resolves for provider {provider!r}.",
                hint=(
                    f"Set {env_var} in the environment, or persist one via "
                    "`sec-rag provider set` (requires the encrypted store)."
                ),
                output=output_format,
                error_code="no_credential",
            )
            raise typer.Exit(code=1)

        try:
            ok = validate_credential(provider, surface_enum, api_key, model=model)
        except (ProviderRateLimitError, ProviderTimeoutError) as exc:
            _print_error(
                "Provider unavailable",
                "The upstream provider is rate-limited or timed out.",
                details=type(exc).__name__,
                hint="Retry after a short backoff; do not rotate the key.",
                output=output_format,
                error_code="provider_unavailable",
            )
            raise typer.Exit(code=2) from None
        except ProviderError as exc:
            _print_error(
                "Provider error",
                "The upstream provider returned an error during validation.",
                details=type(exc).__name__,
                hint="Inspect the audit log; do not rotate the key on a non-auth error.",
                output=output_format,
                error_code="provider_error",
            )
            raise typer.Exit(code=2) from None
    finally:
        if registry is not None:
            registry.close()

    if is_json(output_format):
        # Mirror ``api.schemas.ProviderValidateResponse`` —
        # ``{valid, provider, surface}``.  The masked tail rendered in
        # text mode is NEVER serialised: this document goes to stdout
        # which operators routinely pipe into log shippers.
        print_json(
            {
                "valid": ok,
                "provider": provider,
                "surface": surface_enum.value,
            }
        )
        if not ok:
            raise typer.Exit(code=1)
        return

    if ok:
        console.print(
            f"[green]✓[/green] {escape(provider)} ({escape(surface_enum.value)}): "
            f"key accepted ({escape(mask_secret(api_key))})"
        )
        return

    console.print(
        f"[red]✗[/red] {escape(provider)} ({escape(surface_enum.value)}): "
        f"key rejected by upstream ({escape(mask_secret(api_key))})"
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# `provider set`
# ---------------------------------------------------------------------------


def _refuse_set_unless_encrypted_ready() -> MetadataRegistry:
    """Hard-fail when the encrypted-credential store is not available.

    Returns an *open* :class:`MetadataRegistry` on success; the caller
    owns closing it.

    The single hint names all three opt-in steps every time (install
    extra, set encryption key, set persist toggle) so the operator does
    not have to play guess-the-missing-step — diagnosing which one is
    missing requires probing the import state and reading the live
    settings, which is exactly the kind of multi-step diagnosis we
    flatten into one error.
    """
    settings = get_settings()
    if not settings.database.persist_provider_credentials:
        _print_error(
            "Encrypted credential storage disabled",
            "`provider set` refuses to write a credential without encrypted-at-rest storage.",
            hint=_HARD_FAIL_SET_HINT,
        )
        raise typer.Exit(code=1)

    # Settings load already rejects ``persist=true`` without an
    # encryption key, so reaching this point implies the key is set.
    # The next failure mode is ``pysqlcipher3`` missing — open the
    # registry and check ``encrypted`` to catch it.
    try:
        registry = MetadataRegistry()
    except DatabaseError as exc:
        _print_error(
            "Storage initialisation failed",
            exc.message,
            details=exc.details,
            hint="Check DB_METADATA_DB_PATH is writable.",
        )
        raise typer.Exit(code=1) from None

    if not registry.encrypted:
        # Either pysqlcipher3 is not installed (settings let through
        # because the import probe runs at registry construction, not
        # at settings load) or the key was supplied but somehow the
        # registry opened without it.  Either way, refuse — writing
        # plaintext is not an acceptable fallback.
        registry.close()
        _print_error(
            "Encrypted credential storage unavailable",
            "MetadataRegistry opened without SQLCipher — `provider set` "
            "refuses to write a credential to plaintext SQLite.",
            hint=_HARD_FAIL_SET_HINT,
        )
        raise typer.Exit(code=1)

    return registry


def _read_api_key_from_stdin(provider: str) -> str:
    """Prompt for the API key, hiding terminal echo.

    No ``--api-key`` flag is offered (shell-history /
    ``/proc/<pid>/environ`` footgun).  :func:`typer.prompt` with
    ``hide_input=True`` reads from a TTY without echo; for piped
    invocations (``echo sk-... | sec-rag provider set openai``) the
    same call falls back to plain stdin so automation still works.
    Empty input is rejected — an empty string would silently shadow a
    working downstream resolver (see
    :meth:`InMemorySessionCredentialStore.set`).
    """
    api_key = typer.prompt(
        f"API key for {provider!r}",
        hide_input=True,
        prompt_suffix=": ",
    ).strip()
    if not api_key:
        _print_error(
            "Empty key rejected",
            "An empty API key is never a valid credential.",
            hint="Re-run the command and paste the actual key.",
        )
        raise typer.Exit(code=1)
    return api_key


@provider_app.command("set")
def set_key(
    provider: Annotated[
        str,
        typer.Argument(
            help=(
                "Registered provider name (e.g. openai, anthropic).  The "
                "credential is stored under (user_id=`__admin__`, "
                "provider=<name>) in the encrypted store."
            )
        ),
    ],
    validate_after_set: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate",
            help=(
                "Round-trip the key against the upstream provider after "
                "writing.  Default on — a stored-but-rejected key is "
                "operationally the worst-of-both-worlds, so flip this off "
                "only when network is intentionally unavailable."
            ),
        ),
    ] = True,
    surface: Annotated[
        str,
        typer.Option(
            "--surface",
            "-s",
            help=(
                "Surface to round-trip against when --validate is on "
                "(default: llm).  Ignored when --no-validate is passed."
            ),
        ),
    ] = "llm",
) -> None:
    """Store an admin-default provider credential in the encrypted store.

    Hard-fails (exit 1) when any of the three opt-in steps is missing.
    Never silently falls back to a plaintext SQLite path — that is the
    failure mode operating this command exists to prevent.

    The API key is prompted on stdin (hidden echo); there is no
    ``--api-key`` flag.  Pipe-friendly: ``echo sk-... | sec-rag
    provider set openai --no-validate``.
    """
    # Up-front guard so we do not even prompt for a key in a misconfigured
    # environment — refusing after the operator already pasted the key
    # would be a worse UX.
    surface_enum = _coerce_surface(surface)

    # Validate provider name early — refusing after key entry would be
    # disrespectful of the operator's effort.
    try:
        ProviderRegistry.get_entry(provider, ProviderSurface.LLM)
    except KeyError:
        # Try embedding surface — many providers ship only one surface.
        try:
            ProviderRegistry.get_entry(provider, ProviderSurface.EMBEDDING)
        except KeyError as exc:
            _print_error(
                "Unknown provider",
                f"{provider!r} is not a registered provider name.",
                details=str(exc),
                hint="Run `sec-rag provider list` to see registered names.",
            )
            raise typer.Exit(code=1) from None

    registry = _refuse_set_unless_encrypted_ready()
    try:
        api_key = _read_api_key_from_stdin(provider)

        # Local import keeps the encrypted-store dependency off the
        # import path of the read commands.
        from sec_generative_search.database.credentials import (
            EncryptedCredentialStore,
        )

        try:
            store = EncryptedCredentialStore(registry)
        except ConfigurationError as exc:
            # Defensive: settings + registry checks above should make
            # this unreachable, but a torn-down config could still land
            # here.  Refuse — never fall back.
            _print_error(
                "Encrypted credential storage unavailable",
                exc.message,
                hint=_HARD_FAIL_SET_HINT,
            )
            raise typer.Exit(code=1) from None

        try:
            store.set(_ADMIN_USER_ID, provider, api_key)
        except (DatabaseError, ValueError) as exc:
            _print_error(
                "Could not store credential",
                getattr(exc, "message", str(exc)),
                hint="Inspect the operator logs for the underlying database error.",
            )
            raise typer.Exit(code=1) from None

        console.print(
            f"[green]✓[/green] Stored credential for {escape(provider)} "
            f"({escape(mask_secret(api_key))})"
        )

        if validate_after_set:
            try:
                ok = validate_credential(provider, surface_enum, api_key)
            except (ProviderRateLimitError, ProviderTimeoutError) as exc:
                console.print(
                    "[yellow]![/yellow] Credential stored but post-write "
                    f"validation failed transiently: {escape(type(exc).__name__)}.  "
                    "Retry `sec-rag provider validate` after a short backoff."
                )
                return
            except ProviderError as exc:
                console.print(
                    "[yellow]![/yellow] Credential stored but post-write "
                    "validation returned a non-auth provider error: "
                    f"{escape(type(exc).__name__)}.  "
                    "Inspect the audit log; do not rotate the key on a non-auth error."
                )
                return

            if ok:
                console.print(
                    f"  [dim]Validated against {escape(surface_enum.value)} surface.[/dim]"
                )
            else:
                console.print(
                    "[red]✗[/red] The upstream provider rejected the key.  "
                    "The credential is still stored — remove with "
                    "`sec-rag provider clear` once that command lands, or "
                    "overwrite via another `sec-rag provider set`."
                )
                raise typer.Exit(code=1)
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# `provider refresh-catalogue`
# ---------------------------------------------------------------------------


@provider_app.command("refresh-catalogue")
def refresh_catalogue() -> None:
    """Refresh the model-catalogue overlay from the configured upstream.

    The operator trigger for the opt-in refresh seam
    (:mod:`providers.refresh`).  Performs a single bounded, fail-closed
    HTTPS fetch of the configured source
    (``PROVIDER_CATALOGUE_REFRESH_SOURCE`` / ``_URL``), re-validates it as
    untrusted input, and writes an additive overlay to
    ``PROVIDER_CATALOGUE_OVERLAY_PATH`` — never the request path, never a
    credential.  A running API picks the overlay up on its next start (or on
    the admin route's in-process reload).

    Exit codes:

    - ``0`` — overlay written.
    - ``1`` — fetch / validation / write failed (fail-closed: any prior
      overlay and the bundled baseline keep serving untouched).
    """
    provider_settings = get_settings().provider

    # Local import keeps the refresh module (and its lazy httpx) off the
    # CLI's ``--help`` / ``--version`` start-up path.
    from sec_generative_search.providers.refresh import refresh_overlay

    try:
        report = refresh_overlay(
            overlay_path=provider_settings.catalogue_overlay_path,
            source=provider_settings.catalogue_refresh_source,
            url=provider_settings.catalogue_refresh_url,
        )
    except CatalogueRefreshError as exc:
        # The message is content-free by construction — never an upstream body.
        _print_error(
            "Catalogue refresh failed",
            str(exc),
            hint=(
                "Fail-closed: any prior overlay and the bundled baseline are "
                "untouched.  Check PROVIDER_CATALOGUE_REFRESH_SOURCE / _URL is "
                "reachable and returns the expected schema, then retry."
            ),
        )
        raise typer.Exit(code=1) from None
    except OSError as exc:
        # Disk failure writing the overlay — render the error *type* only, never
        # ``str(exc)`` (it can carry the configured filesystem path).
        _print_error(
            "Catalogue overlay write failed",
            f"Could not write the overlay to disk ({type(exc).__name__}).",
            hint="Verify the overlay path is writable and the volume has free space.",
        )
        raise typer.Exit(code=1) from None

    console.print(
        f"[green]✓[/green] Catalogue overlay refreshed from "
        f"[cyan]{escape(report.source)}[/cyan]: "
        f"{report.provider_count} providers, {report.model_count} models."
    )
    console.print(f"  [dim]Source: {escape(report.source_url)}[/dim]")
    console.print(f"  [dim]Overlay: {escape(report.overlay_path)}[/dim]")
    console.print(
        "  [dim italic]Takes effect on the next API start, or immediately via "
        "the admin refresh route.[/dim italic]"
    )
