"""Configuration management using Pydantic Settings v2.

Hierarchical settings pattern: each domain area is a separate nested
``BaseSettings`` subclass with its own ``env_prefix``.  The root
``Settings`` class composes them all and provides a singleton accessor.

Environment variable mapping (examples):
    EDGAR_IDENTITY_NAME     -> settings.edgar.identity_name
    EMBEDDING_MODEL_NAME    -> settings.embedding.model_name
    LLM_DEFAULT_PROVIDER    -> settings.llm.default_provider
    PROVIDER_TIMEOUT         -> settings.provider.timeout
    RAG_CONTEXT_TOKEN_BUDGET -> settings.rag.context_token_budget
    DB_ENCRYPTION_KEY       -> settings.database.encryption_key
    API_KEY                 -> settings.api.key
"""

from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sec_generative_search.core.types import DeploymentProfile

# Load .env into os.environ BEFORE nested BaseSettings classes are
# instantiated as default values in the Settings class body.  Without
# this, EdgarSettings() (which has required fields and no defaults)
# fails because it only searches os.environ — it has no env_file of
# its own.
load_dotenv()


class EdgarSettings(BaseSettings):
    """SEC EDGAR API credentials.

    In web deployments (Scenarios B/C) where ``EDGAR_SESSION_REQUIRED=true``,
    ``identity_name`` and ``identity_email`` may be unset — each user provides
    their own credentials per session via HTTP headers.  The CLI still requires
    them.

    EDGAR rate limiting is handled by edgartools internally (``pyrate_limiter``
    token bucket at 9 req/s by default, configurable via the
    ``EDGAR_RATE_LIMIT_PER_SEC`` env var that edgartools reads directly).
    """

    identity_name: str | None = None
    identity_email: str | None = None

    model_config = SettingsConfigDict(env_prefix="EDGAR_")


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration.

    ``provider`` selects the embedding backend and is validated against
    :class:`~sec_generative_search.providers.registry.ProviderRegistry` so
    typos surface at settings load rather than at first embed call.  The
    local-only knobs (``device``, ``batch_size``, ``idle_timeout_minutes``)
    have no meaning for hosted providers; a ``model_validator`` rejects
    non-default values whenever ``provider != "local"``.

    Credentials never live here — :mod:`sec_generative_search.providers.factory`
    resolves them at construction time via an injected ``api_key_resolver``.
    """

    provider: str = "local"
    model_name: str = "google/embeddinggemma-300m"
    device: str = "auto"  # "cuda", "cpu", or "auto"
    batch_size: int = 32
    idle_timeout_minutes: int = 0  # 0 = disabled; auto-unload model after idle

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        """Reject names not registered on the embedding surface.

        ``include_unavailable=True`` accepts providers whose optional
        extras are not installed (e.g. ``"local"`` without
        ``[local-embeddings]``) — settings load must not depend on the
        extras being present.  The factory raises a clear install-hint
        error at construction time if the extra is still missing when an
        embedder is actually built.

        The import is deferred to keep ``config`` free of a top-level
        dependency on ``providers`` — loading every adapter at settings
        import would be wasteful and couples two otherwise-independent
        layers.
        """
        from sec_generative_search.providers.registry import (
            ProviderRegistry,
            ProviderSurface,
        )

        known = {
            entry.name
            for entry in ProviderRegistry.all_entries(
                ProviderSurface.EMBEDDING,
                include_unavailable=True,
            )
        }
        if value not in known:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER '{value}'. Known embedding providers: {sorted(known)}."
            )
        return value

    @model_validator(mode="after")
    def _validate_local_only_knobs(self) -> "EmbeddingSettings":
        """Reject non-default local-only knobs when the provider is hosted.

        ``device``, ``batch_size``, and ``idle_timeout_minutes`` are only
        meaningful for :class:`LocalEmbeddingProvider`.  Silently
        accepting them with a hosted provider would invite
        misconfiguration where an operator thinks they have tuned a
        hosted embedder — fail loudly and name every offending field.
        """
        if self.provider == "local":
            return self

        offenders: list[str] = []
        if self.device != "auto":
            offenders.append(f"device={self.device!r}")
        if self.batch_size != 32:
            offenders.append(f"batch_size={self.batch_size!r}")
        if self.idle_timeout_minutes != 0:
            offenders.append(f"idle_timeout_minutes={self.idle_timeout_minutes!r}")

        if offenders:
            raise ValueError(
                f"EMBEDDING_PROVIDER='{self.provider}' is a hosted provider "
                f"but local-only knobs were set: {', '.join(offenders)}. "
                f"Remove these env vars or switch to EMBEDDING_PROVIDER=local."
            )
        return self


class ChunkingSettings(BaseSettings):
    """Text chunking configuration.

    ``token_limit`` is the target chunk size; ``tolerance`` is the
    bidirectional (``±``) band around that target inside which the
    chunker is free to cut at a sentence boundary.  Concretely, chunks
    are expected to land in ``[token_limit - tolerance,
    token_limit + tolerance]`` whenever the sentence structure permits;
    the active-centring cut in :class:`TextChunker` prefers boundaries
    near ``token_limit`` rather than dragging on to the upper edge.
    """

    token_limit: int = 1000
    tolerance: int = 150

    model_config = SettingsConfigDict(env_prefix="CHUNKING_")


def resolve_secret_from_value_or_file(
    value: str | None,
    file_path: str | None,
    *,
    value_env_name: str,
    file_env_name: str,
) -> str | None:
    """Resolve a secret from a direct value or a file path.

    Enforces mutual exclusion between the two sources and validates the
    file when ``file_path`` is used. Returns the resolved secret, or
    ``None`` if neither source is set. The env-var names are passed
    explicitly so error messages name the operator-facing knob rather
    than the internal field.

    This is the single canonical secret-file resolver. Besides the
    encryption key and auth pepper wrappers below, it backs the
    admin-default provider-key ``*_FILE`` indirection consumed by
    :func:`sec_generative_search.providers.factory.default_api_key_resolver`,
    so a Secret-Manager file mount works for provider keys exactly as it
    does for the SQLCipher key and the pepper.
    """
    if value and file_path:
        raise ValueError(
            f"{value_env_name} and {file_env_name} are mutually exclusive. Set only one."
        )

    if value:
        return value

    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"{file_env_name} '{file_path}' does not exist.")
        if not path.is_file():
            raise ValueError(f"{file_env_name} '{file_path}' is not a file.")
        content = path.read_text().strip()
        if not content:
            raise ValueError(f"{file_env_name} '{file_path}' is empty.")
        return content

    return None


def resolve_encryption_key_from_values(key: str | None, key_file: str | None) -> str | None:
    """Resolve an encryption key from a direct value or file path.

    Enforces mutual exclusion between the two sources and validates the
    file when ``key_file`` is used. Returns the resolved key string, or
    ``None`` if neither source is set.

    Used by both ``DatabaseSettings`` (Pydantic validation) and
    ``MetadataRegistry`` (runtime resolution without re-instantiating
    settings).
    """
    return resolve_secret_from_value_or_file(
        key,
        key_file,
        value_env_name="DB_ENCRYPTION_KEY",
        file_env_name="DB_ENCRYPTION_KEY_FILE",
    )


def resolve_auth_pepper_from_values(
    pepper: str | None,
    pepper_file: str | None,
) -> str | None:
    """Resolve the HMAC auth pepper from a direct value or file path.

    Mirrors :func:`resolve_encryption_key_from_values` byte-for-byte —
    the pepper for ``auth_hash`` shares the deployment story of the
    SQLCipher key: an env-var inline in Scenario A, a file mount in
    Scenario B/C secrets managers. The two sources are mutually
    exclusive.

    Returns ``None`` when neither source is set; the runtime caller
    (``UserStore`` construction) is responsible for refusing to operate
    a non-empty ``users`` table with a missing pepper.
    """
    return resolve_secret_from_value_or_file(
        pepper,
        pepper_file,
        value_env_name="API_AUTH_PEPPER",
        file_env_name="API_AUTH_PEPPER_FILE",
    )


_DEPLOYMENT_PROFILE_DEFAULTS: dict[str, tuple[int, int]] = {
    DeploymentProfile.LOCAL.value: (2500, 0),
    DeploymentProfile.TEAM.value: (10000, 90),
    DeploymentProfile.CLOUD.value: (10000, 30),
}
"""Profile → (max_filings, retention_max_age_days) defaults.

Local matches the historical static defaults (2500 filings, no
time-based eviction) so an existing operator who never sets
``DB_DEPLOYMENT_PROFILE`` sees zero behavioural change.  Team and cloud
ship eviction enabled by default — operators who want eviction off can
override with ``DB_RETENTION_MAX_AGE_DAYS=0``.  Both knobs are
individually overridable; the profile only fills *unset* fields.
"""


_DEPLOYMENT_PROFILE_PERSIST_CREDENTIALS: dict[str, bool] = {
    DeploymentProfile.LOCAL.value: False,
    DeploymentProfile.TEAM.value: True,
    DeploymentProfile.CLOUD.value: True,
}
"""Profile → default for ``persist_provider_credentials``.

Local profile defaults to *off* — single-user dev rarely benefits from
the encrypted credential table and most local installs do not configure
SQLCipher.  Team / cloud default to *on* because the table is the
ergonomic seam that lets returning users avoid re-entering keys; both
profiles already require SQLCipher for the rest of the metadata
registry, so the requirement adds no new operational burden.

Persistence enabled without SQLCipher is rejected at settings load (see
``DatabaseSettings._validate_credential_persistence``) — the contract is
that a stored provider key is *always* encrypted at rest.
"""


class DatabaseSettings(BaseSettings):
    """Database configuration.

    ``deployment_profile`` selects between local / team / cloud
    presets that drive the default ceilings on filing count and
    retention age.  Explicit env vars (``DB_MAX_FILINGS``,
    ``DB_RETENTION_MAX_AGE_DAYS``) always win — the profile only
    fills in fields the operator did not set.
    """

    deployment_profile: str = DeploymentProfile.LOCAL.value
    chroma_path: str = "./data/chroma_db"
    metadata_db_path: str = "./data/metadata.sqlite"
    max_filings: int = 2500

    # Time-based retention.  ``0`` disables eviction; the value is the
    # ``ingested_at`` cutoff applied by ``MetadataRegistry.list_expired_filings``
    # and ``FilingStore.evict_expired``.  Defaults are profile-driven; see
    # ``_DEPLOYMENT_PROFILE_DEFAULTS``.
    retention_max_age_days: int = 0

    # SQLCipher encryption key; unset = plain sqlite3 (local dev).
    encryption_key: str | None = None

    # Path to a file containing the SQLCipher encryption key (e.g. Docker
    # secrets at ``/run/secrets/db_encryption_key``). Mutually exclusive with
    # ``encryption_key``. Preferred in production — file contents are not
    # visible in ``/proc/<pid>/environ``.
    encryption_key_file: str | None = None

    # Task history privacy settings.
    task_history_retention_days: int = 0  # 0 = keep indefinitely
    task_history_persist_tickers: bool = False

    # Provider-credential persistence.
    #
    # When ``True``, ``EncryptedCredentialStore`` may persist user-supplied
    # provider API keys into the SQLCipher-encrypted ``provider_credentials``
    # table.  When ``False``, the encrypted store refuses to construct and
    # the resolver chain falls back to in-memory + admin-env only.  The
    # default is profile-driven (local=False, team/cloud=True).  Persistence
    # without SQLCipher is rejected at load — a stored provider key must
    # always be encrypted at rest.
    persist_provider_credentials: bool | None = None

    model_config = SettingsConfigDict(env_prefix="DB_")

    @field_validator("deployment_profile")
    @classmethod
    def _validate_deployment_profile(cls, value: str) -> str:
        """Reject unknown profile names at settings load.

        Values must match one of :class:`DeploymentProfile`'s members.
        Case is preserved verbatim to keep the env var unambiguous.
        """
        valid = {profile.value for profile in DeploymentProfile}
        if value not in valid:
            raise ValueError(
                f"Unknown DB_DEPLOYMENT_PROFILE '{value}'. Valid profiles: {sorted(valid)}."
            )
        return value

    @field_validator("retention_max_age_days")
    @classmethod
    def _validate_retention_non_negative(cls, value: int) -> int:
        """Reject negative retention values.

        A negative cutoff would invert the SQL WHERE clause and delete
        recent filings — defence-in-depth against an operator typo.
        ``0`` (disabled) is the only special-cased non-positive value.
        """
        if value < 0:
            raise ValueError(
                f"DB_RETENTION_MAX_AGE_DAYS must be >= 0; got {value}. "
                f"Use 0 to disable time-based eviction."
            )
        return value

    @model_validator(mode="after")
    def _apply_profile_defaults(self) -> "DatabaseSettings":
        """Fill ``max_filings`` / ``retention_max_age_days`` from the profile.

        Runs only against fields the operator did not set explicitly
        (probed via ``model_fields_set``).  Pydantic-settings includes
        env-var sourced values in that set, so an explicit
        ``DB_MAX_FILINGS=20000`` always wins over the profile baseline.

        Local profile preserves the historical static defaults; the
        method is a no-op for ``LOCAL`` operators who never opt in.
        """
        defaults = _DEPLOYMENT_PROFILE_DEFAULTS.get(self.deployment_profile)
        if defaults is None:
            # Field validator already rejected unknown profiles; this is
            # defensive against future enum additions without table updates.
            return self

        default_max, default_retention = defaults
        if "max_filings" not in self.model_fields_set:
            self.max_filings = default_max
        if "retention_max_age_days" not in self.model_fields_set:
            self.retention_max_age_days = default_retention

        # Credential-persistence default is also profile-driven.  ``None``
        # is the sentinel for "operator did not set it" — Pydantic's
        # ``model_fields_set`` treats env-supplied values as set, so the
        # ``None`` check covers both "absent from env" and "absent from
        # constructor kwargs".  An explicit ``DB_PERSIST_PROVIDER_CREDENTIALS=true``
        # always wins.
        if self.persist_provider_credentials is None:
            self.persist_provider_credentials = _DEPLOYMENT_PROFILE_PERSIST_CREDENTIALS.get(
                self.deployment_profile, False
            )
        return self

    @model_validator(mode="after")
    def _resolve_encryption_key(self) -> "DatabaseSettings":
        """Resolve ``encryption_key`` from ``encryption_key_file`` if set.

        Delegates to :func:`resolve_encryption_key_from_values` for the
        actual validation and file reading. See that function's docstring
        for the mutual-exclusion and file-validation rules.
        """
        self.encryption_key = resolve_encryption_key_from_values(
            self.encryption_key, self.encryption_key_file
        )
        return self

    @model_validator(mode="after")
    def _validate_credential_persistence(self) -> "DatabaseSettings":
        """Refuse to persist provider credentials without SQLCipher.

        The encrypted-credential store is layered on top of SQLCipher's
        whole-database encryption — that is the load-bearing control.
        Allowing the toggle to be ``True`` while ``encryption_key`` is
        unset would write provider keys into a plain-text SQLite file,
        defeating the entire point of the table.  Fail at load with a
        message that names both knobs the operator can fix.

        Must run *after* :meth:`_resolve_encryption_key` so an operator
        who sets ``DB_ENCRYPTION_KEY_FILE`` (rather than the inline key)
        is not falsely rejected before the file has been read.
        """
        if self.persist_provider_credentials and not self.encryption_key:
            raise ValueError(
                "DB_PERSIST_PROVIDER_CREDENTIALS=true requires SQLCipher "
                "encryption.  Configure DB_ENCRYPTION_KEY (or "
                "DB_ENCRYPTION_KEY_FILE), or set "
                "DB_PERSIST_PROVIDER_CREDENTIALS=false to disable the "
                "encrypted credential store."
            )
        return self

    @model_validator(mode="after")
    def _validate_paths(self) -> "DatabaseSettings":
        """Validate that database paths resolve within the working directory.

        Prevents path traversal attacks where an attacker controls environment
        variables (e.g. ``DB_METADATA_DB_PATH=../../sensitive/data.sqlite``)
        to write files outside the project directory.

        Checks:
        - Resolved path must be relative to ``Path.cwd()``.
        - No symlinks in any lexical parent directory.  The walk is
          intentionally done over the *lexical* (non-resolved) path — a
          post-``resolve()`` walk never sees symlinks because they have
          already been followed, which would silently pass a symlink
          whose target happens to live inside cwd.
        """
        base_dir = Path.cwd().resolve()
        for field_name in ("chroma_path", "metadata_db_path"):
            raw_value = getattr(self, field_name)
            lexical = Path(raw_value).absolute()
            resolved = Path(raw_value).resolve()

            # Check the resolved path stays within the working directory.
            if not resolved.is_relative_to(base_dir):
                raise ValueError(
                    f"Database path '{field_name}' resolves to "
                    f"'{resolved}' which is outside the project "
                    f"directory '{base_dir}'. Use a relative path "
                    f"within the project directory."
                )

            # Walk up the lexical path and refuse if any existing parent
            # is a symlink.  Stops at cwd once it is reached.
            check = lexical
            while check != check.parent:  # stop at filesystem root
                if check.exists() and check.is_symlink():
                    raise ValueError(
                        f"Database path '{field_name}' contains a "
                        f"symlink at '{check}'. Symlinks are not "
                        f"permitted in database paths for security."
                    )
                if check == base_dir:
                    break
                check = check.parent

        return self


class LLMSettings(BaseSettings):
    """LLM model selection and generation defaults.

    These control which model is used and how it generates responses.
    Provider-level network policy (timeouts, retries) lives in
    ``ProviderSettings``; this class covers model behaviour.
    """

    default_provider: str = "openai"  # provider key registered in ProviderRegistry
    default_model: str | None = None  # None = use the provider's own default
    temperature: float = 0.1  # low temperature for factual SEC analysis
    max_output_tokens: int = 2048
    streaming: bool = True  # prefer streaming responses by default

    model_config = SettingsConfigDict(env_prefix="LLM_")


def _is_loopback_host(host: str) -> bool:
    """Whether *host* is a loopback address or the literal ``localhost``.

    ``127.0.0.0/8``, ``::1``, and the case-insensitive name ``localhost``
    are loopback; every other IP or hostname (private, public, or
    DNS-resolvable) is not.  No DNS resolution is performed — the policy is
    purely lexical, so it cannot be subverted by a hosts-file entry that
    points a decoy name at a loopback address while the real traffic leaves
    the host.
    """
    if host.lower() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        # Not an IP literal — a hostname.  Hostnames are never treated as
        # loopback (see docstring); reaching a hostname requires the opt-in.
        return False


class LocalLLMSettings(BaseSettings):
    """Self-hosted (local) LLM endpoint configuration.

    Backs the ``local_llm`` provider — a model server the operator runs
    themselves (Ollama by default, or llama.cpp-server / vLLM / LM Studio)
    that speaks the OpenAI Chat Completions wire protocol.  Unlike the
    hosted vendors, the endpoint URL is operator-owned infrastructure, so
    it is a *deployment* setting rather than a per-request / per-user one:
    it is read from here at provider construction, never through the
    credential resolver chain.

    Host policy (the load-bearing security control):

    - ``base_url`` scheme must be ``http`` or ``https``.  Plain ``http`` is
      deliberately permitted because a loopback endpoint carries no TLS and
      needs none — the traffic never leaves the host.
    - The host must be loopback (``127.0.0.0/8``, ``::1``, or the literal
      ``localhost``) unless ``allow_non_local`` is set.  Pointing the
      provider at a private IP, a hostname, or a public IP — anywhere the
      prompt (Tier-3 data) would leave the machine — is an explicit,
      operator-acknowledged decision guarded by
      ``LOCAL_LLM_ALLOW_NON_LOCAL=true``.

    Validation runs at settings load so a misconfiguration fails fast,
    before the app can serve a request against an unintended endpoint.  The
    provider additionally re-reads these settings standalone at
    construction and fails *closed* to the loopback default, so even a
    torn-down config never silently sends the prompt off-box.
    """

    base_url: str = "http://127.0.0.1:11434/v1"
    default_model: str = "llama3.2"
    allow_non_local: bool = False

    model_config = SettingsConfigDict(env_prefix="LOCAL_LLM_")

    @model_validator(mode="after")
    def _validate_base_url_host_policy(self) -> "LocalLLMSettings":
        """Enforce scheme + loopback-only host policy on ``base_url``.

        Runs after the whole model is built because the host check depends
        on ``allow_non_local``.  Rejects any non-``http(s)`` scheme, a URL
        with no host, and — unless the operator opted in — any host that is
        not loopback.  Every message names the offending value and the knob
        the operator can set, so a misconfiguration is self-diagnosing at
        load time.
        """
        parsed = urlparse(self.base_url)
        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https"):
            raise ValueError(
                f"LOCAL_LLM_BASE_URL must use the http or https scheme; got {self.base_url!r}."
            )
        host = parsed.hostname
        if not host:
            raise ValueError(f"LOCAL_LLM_BASE_URL must include a host; got {self.base_url!r}.")
        if self.allow_non_local:
            # Operator has explicitly acknowledged that prompts may leave
            # the host — accept any well-formed http(s) URL.
            return self
        if not _is_loopback_host(host):
            raise ValueError(
                f"LOCAL_LLM_BASE_URL host {host!r} is not loopback. The local LLM "
                f"endpoint must be loopback (127.0.0.0/8, ::1, or localhost) so the "
                f"prompt never leaves the host. To target a non-local endpoint "
                f"(a private or hosted server), set LOCAL_LLM_ALLOW_NON_LOCAL=true to "
                f"acknowledge that prompts will be sent off-box."
            )
        return self


class ProviderSettings(BaseSettings):
    """Provider-level network and resilience policy.

    Applies to all external LLM/embedding API calls (OpenAI, Anthropic,
    Gemini, etc.).  Per-provider overrides will be supported in the
    provider registry; these are the global defaults.
    """

    timeout: int = 60  # seconds per API call
    max_retries: int = 3
    retry_backoff_base: float = 2.0  # exponential backoff base (seconds)
    circuit_breaker_threshold: int = 5  # consecutive failures before circuit opens
    circuit_breaker_reset: int = 60  # seconds before half-open retry
    cost_tracking_enabled: bool = True  # track token usage and estimated cost

    # --- Opt-in model-catalogue refresh seam ----------------------------
    #
    # Which built-in upstream source the refresh trigger fetches from.  The
    # refresh is never in the request path; it is driven by an explicit
    # operator trigger (CLI / admin route / external scheduler).
    catalogue_refresh_source: str = "models_dev"  # models_dev | litellm

    # Optional operator override of the source's pinned default URL.  Must be
    # https:// (re-checked at fetch time).  ``None`` = use the built-in pin.
    catalogue_refresh_url: str | None = None

    # Where the additive, validated catalogue overlay is written.  Lives in
    # the data volume alongside the ChromaDB / SQLite stores; constrained to
    # the project directory (no traversal, no parent symlink) just like the
    # database paths.
    catalogue_overlay_path: str = "./data/model_catalogue_overlay.json"

    model_config = SettingsConfigDict(env_prefix="PROVIDER_")

    @field_validator("catalogue_refresh_source")
    @classmethod
    def _validate_catalogue_source(cls, value: str) -> str:
        """Reject an unknown refresh source name at settings load."""
        valid = {"models_dev", "litellm"}
        if value not in valid:
            raise ValueError(
                f"Unknown PROVIDER_CATALOGUE_REFRESH_SOURCE '{value}'. "
                f"Valid sources: {sorted(valid)}."
            )
        return value

    @field_validator("catalogue_refresh_url")
    @classmethod
    def _validate_catalogue_url(cls, value: str | None) -> str | None:
        """Require any operator-supplied refresh URL to be https://.

        Defence in depth — :func:`providers.refresh.fetch_json` re-checks the
        scheme, but failing here surfaces a misconfiguration at load rather
        than only when a refresh is first triggered.
        """
        if value is None:
            return None
        if not value.lower().startswith("https://"):
            raise ValueError("PROVIDER_CATALOGUE_REFRESH_URL must be an https:// URL.")
        return value

    @model_validator(mode="after")
    def _validate_overlay_path(self) -> "ProviderSettings":
        """Constrain the overlay path to the project directory.

        Mirrors ``DatabaseSettings._validate_paths``: the resolved path must
        stay within ``cwd`` and no lexical parent may be a symlink, so an
        attacker-controlled env var cannot redirect the overlay write outside
        the data volume.  The walk is over the *lexical* path on purpose — a
        post-``resolve()`` walk would have already followed any symlink.
        """
        base_dir = Path.cwd().resolve()
        raw_value = self.catalogue_overlay_path
        lexical = Path(raw_value).absolute()
        resolved = Path(raw_value).resolve()

        if not resolved.is_relative_to(base_dir):
            raise ValueError(
                f"PROVIDER_CATALOGUE_OVERLAY_PATH resolves to '{resolved}' "
                f"which is outside the project directory '{base_dir}'. Use a "
                f"relative path within the project directory."
            )

        check = lexical
        while check != check.parent:
            if check.exists() and check.is_symlink():
                raise ValueError(
                    f"PROVIDER_CATALOGUE_OVERLAY_PATH contains a symlink at "
                    f"'{check}'. Symlinks are not permitted for security."
                )
            if check == base_dir:
                break
            check = check.parent
        return self


class RAGSettings(BaseSettings):
    """RAG orchestration configuration.

    Controls how retrieval results are assembled into a prompt context
    and how the generation pipeline behaves.  ``SearchSettings`` covers
    the vector-search layer; this class covers the generation layer
    on top of it.
    """

    context_token_budget: int = 6000  # max tokens allocated to retrieved context
    citation_mode: str = "inline"  # "inline" or "footnote"
    default_answer_mode: str = "concise"  # "concise", "analytical", "extractive", "comparative"
    refusal_enabled: bool = True  # refuse when context is insufficient
    # Sentence-level overlap between adjacent chunks (≈ 15 % of the
    # 1000-token default chunk size).  Carries the last whole sentence
    # of the previous chunk into the next so referent words like
    # "the Company" / "such filing" survive embedding boundaries.  See
    # :meth:`TextChunker._tail_for_overlap` for the boundary policy.
    chunk_overlap_tokens: int = 150
    chat_history_enabled: bool = False  # session-scoped conversation memory (off by default)
    chat_history_max_turns: int = 10  # max turns retained in session memory

    # Operator override for the conversation-history slice of the four-way
    # context budget.  ``0`` means "use the default fraction of total
    # context window (15%)" — see :class:`ContextBudget` in
    # :mod:`sec_generative_search.rag.context`.  Negative values are
    # rejected at load.
    history_token_budget: int = 0

    # BCP-47 code (e.g. ``"en"``, ``"tr"``) or ``"auto"``.  ``"auto"``
    # makes the orchestrator answer in whatever language the
    # query-understanding step detected; an explicit BCP-47 code locks
    # the output language regardless of the input language.  Operators
    # set this once per deployment.
    output_language: str = "auto"

    model_config = SettingsConfigDict(env_prefix="RAG_")

    @field_validator("history_token_budget")
    @classmethod
    def _validate_history_budget(cls, value: int) -> int:
        """Reject negative budgets — would over-allocate other slices."""
        if value < 0:
            raise ValueError(
                f"RAG_HISTORY_TOKEN_BUDGET must be >= 0; got {value}. "
                f"Use 0 to fall back to the default fraction of total context."
            )
        return value


class SearchSettings(BaseSettings):
    """Vector search configuration (retrieval layer)."""

    top_k: int = 5
    min_similarity: float = 0.0

    # Diversity caps bound how many chunks sharing a ``section_path`` /
    # ``accession_number`` may appear in one result set, so a single
    # verbose section or filing cannot crowd out corpus coverage.
    # ``0`` disables a cap. These are the operator *defaults*;
    # :meth:`RetrievalService.retrieve` accepts a per-call override and
    # the ``POST /api/search`` wire re-bounds the untrusted per-request
    # value at ``<= 50``.
    max_per_section: int = 0
    max_per_filing: int = 0

    # Rerank over-fetch multiplier: when a reranker is bound, retrieval
    # fetches ``top_k * factor`` candidates so the reranker has a pool
    # to re-order, then slices back to ``top_k``. ``1`` disables
    # over-fetch. Inert until a concrete reranker is wired — no
    # reranker ships today — so this knob is forward-looking but
    # validated now so a deployment can pre-set it.
    rerank_over_fetch_factor: int = 4

    model_config = SettingsConfigDict(env_prefix="SEARCH_")

    @field_validator("max_per_section", "max_per_filing")
    @classmethod
    def _validate_diversity_cap(cls, value: int, info: ValidationInfo) -> int:
        """Reject negative caps — only ``0`` (disabled) or a positive cap."""
        if value < 0:
            env_var = f"SEARCH_{info.field_name.upper()}"
            raise ValueError(f"{env_var} must be >= 0; got {value}. Use 0 to disable the cap.")
        return value

    @field_validator("rerank_over_fetch_factor")
    @classmethod
    def _validate_over_fetch_factor(cls, value: int) -> int:
        """Reject factors below 1 — a 0/negative multiplier would zero out
        the candidate fetch and starve retrieval. ``1`` disables over-fetch."""
        if value < 1:
            raise ValueError(
                f"SEARCH_RERANK_OVER_FETCH_FACTOR must be >= 1; got {value}. "
                f"Use 1 to disable over-fetch."
            )
        return value


class LoggingSettings(BaseSettings):
    """Logging configuration for optional file logging."""

    # Optional file logging (in addition to stdout).
    # Env vars: LOG_FILE_PATH, LOG_FILE_MAX_BYTES, LOG_FILE_BACKUP_COUNT
    path: str | None = None  # unset = stdout only
    max_bytes: int = 10_485_760  # 10 MB
    backup_count: int = 3

    model_config = SettingsConfigDict(env_prefix="LOG_FILE_")


class HuggingFaceSettings(BaseSettings):
    """Hugging Face configuration."""

    token: str | None = None

    model_config = SettingsConfigDict(env_prefix="HUGGING_FACE_")


class ApiSettings(BaseSettings):
    """API server configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]
    key: str | None = None  # API key; None = auth disabled (local dev)

    # Rate limiting (requests per minute; 0 = disabled)
    rate_limit_search: int = 60
    rate_limit_ingest: int = 10
    rate_limit_delete: int = 30
    rate_limit_general: int = 120
    rate_limit_rag: int = 60
    # Provider-validation rate limits: per-IP and per-session_id are
    # separate sliding windows, and both must allow.
    rate_limit_validate: int = 10
    rate_limit_validate_per_session: int = 5
    rate_limit_session: int = 20
    # User-tier login rate limits. Same pattern as ``validate``:
    # per-IP + per-username sliding windows; both must allow the
    # request. ``login`` floors are aggressively low because the route
    # is the brute-force surface; tune higher only in closed-network
    # deployments. ``0`` disables (Scenario A).
    rate_limit_login: int = 5
    rate_limit_login_per_username: int = 3

    # Admin key for destructive operations; unset = unrestricted (Scenario A).
    admin_key: str | None = None

    # HMAC pepper for the ``auth_hash`` column. Mutually exclusive with
    # ``auth_pepper_file``. Settings load coerces empty string to
    # ``None`` (same pattern as the API keys); the runtime ``UserStore``
    # refuses to operate a non-empty ``users`` table when both sources
    # are unset. The pepper turns a leaked ``auth_hash`` column into a
    # useless artefact for offline attack — without the
    # pepper, ``HMAC(pepper, auth_proof)`` collapses to ``SHA256``-shaped
    # output an attacker could brute-force from a stolen DB.
    auth_pepper: str | None = None
    auth_pepper_file: str | None = None

    # Per-session EDGAR credentials requirement.
    edgar_session_required: bool = False

    # Sliding TTL for the server-minted ``session_id`` cookie.  Mirrors
    # the in-memory credential-store default; lowering this reduces the
    # window in which a stolen cookie can resolve user-supplied keys.
    session_ttl_seconds: int = 60 * 60  # one hour

    # Demo mode — FIFO eviction, nightly reset banner, "clear all" disabled.
    demo_mode: bool = False
    demo_eviction_buffer: int = 500

    # Task queue size (maximum concurrent + pending ingest tasks).
    max_task_queue_size: int = 5

    # Abuse prevention caps (0 = unlimited/disabled).
    max_tickers_per_request: int = 0
    max_filings_per_request: int = 0
    max_task_duration_minutes: int = 0

    @field_validator("key", "admin_key", "auth_pepper", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: str | None) -> str | None:
        return v or None

    @model_validator(mode="after")
    def _reject_wildcard_cors_origin(self) -> "ApiSettings":
        """Refuse a wildcard entry in ``cors_origins``.

        Starlette's ``CORSMiddleware`` special-cases ``allow_origins=["*"]``
        when ``allow_credentials=True`` (which ``create_app`` always sets):
        instead of emitting a literal ``*``, it reflects the requesting
        ``Origin`` back and sets ``Access-Control-Allow-Credentials:
        true``, so **any** site could read credentialed cross-origin
        responses.  The shipped default is safe — this guard is here so
        the "just make it work" wildcard fails at load rather than
        silently widening the browser trust boundary.

        Unconditional, not gated on whether auth is enabled: the same
        list is the WebSocket ``Origin`` allow-list
        (``api/websocket.py::_is_origin_allowed``), where auth-
        conditionality would be meaningless — and a wildcard there never
        matches a real origin anyway, so it can only ever misconfigure.

        Matched per entry against the stripped value: a whitespace-padded
        ``" * "`` is refused too, while an origin that merely *contains*
        a star is left alone (this is not a substring ban).  A bare
        ``API_CORS_ORIGINS=*`` never reaches this validator — the env
        source decodes ``list[str]`` as JSON and raises before validation
        — but that path is fail-closed as well.
        """
        if any(origin.strip() == "*" for origin in self.cors_origins):
            raise ValueError(
                "API_CORS_ORIGINS must not contain '*' — the API is served with "
                "allow_credentials=True, under which a wildcard makes Starlette "
                "reflect any requesting Origin. List your exact origins instead, "
                "e.g. API_CORS_ORIGINS='[\"https://app.example.com\"]'."
            )
        return self

    @model_validator(mode="after")
    def _resolve_auth_pepper(self) -> "ApiSettings":
        """Resolve ``auth_pepper`` from ``auth_pepper_file`` if set.

        Mutual-exclusion + file validation lives in
        :func:`resolve_auth_pepper_from_values`. Runs unconditionally —
        an unset pair simply leaves ``auth_pepper`` as ``None``; the
        runtime ``UserStore`` is the load-bearing refuser when the
        ``users`` table needs the pepper.
        """
        self.auth_pepper = resolve_auth_pepper_from_values(self.auth_pepper, self.auth_pepper_file)
        return self

    model_config = SettingsConfigDict(env_prefix="API_")


class Settings(BaseSettings):
    """Root settings class combining all sections.

    Nested fields use ``Field(default_factory=...)`` rather than a
    cached default instance.  Each ``Settings()`` call rebuilds the
    nested models so they observe the *current* ``os.environ`` —
    important for ``reload_settings()`` and for tests that mutate the
    environment between assertions.  ``load_dotenv()`` still fires at
    module top so ``.env`` is in ``os.environ`` before any factory call.
    """

    edgar: EdgarSettings = Field(default_factory=EdgarSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    local_llm: LocalLLMSettings = Field(default_factory=LocalLLMSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    log_file: LoggingSettings = Field(default_factory=LoggingSettings)
    hugging_face: HuggingFaceSettings = Field(default_factory=HuggingFaceSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore prefixed env vars handled by nested classes
    )


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Return the global Settings instance (singleton pattern)."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def reload_settings() -> Settings:
    """Reload settings from environment (mainly for testing)."""
    global _settings_instance
    _settings_instance = Settings()
    return _settings_instance
