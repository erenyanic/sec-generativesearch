"""Tests for ``sec-rag provider {list,validate,set}``.

The provider registry, credential validation, encrypted-credential
store, and metadata registry are stubbed at the ``cli.provider`` import
site so the CLI's wire-up assertions stay independent of the real
adapters.

Goals:

- ``provider list`` enumerates registered adapters (incl. surface,
  default model, pricing tier of default, admin env var, key-resolves
  flag) and never echoes a raw key.
- ``provider validate`` resolves the key through the operator-scope
  chain (``encrypted-user → admin-env``), calls ``validate_credential``,
  and maps exception types to exit codes:
  - 0 — accepted, 1 — rejected / no key / unknown provider,
  - 2 — transient upstream (rate limit / timeout / other ProviderError).
- ``provider set`` **hard-fails** with the single three-step hint when
  any of (a) ``DB_PERSIST_PROVIDER_CREDENTIALS=true``, (b) SQLCipher
  available (``registry.encrypted=True``) is missing; never falls back
  to plaintext.  The hard-fail fires **before** the key prompt so the
  operator never pastes a credential into a doomed flow.
- ``provider set`` happy path stores via ``EncryptedCredentialStore``,
  validates post-write, and prints only the masked tail.
- Security: no raw key ever appears in any rendered output, regardless
  of which code path runs (list / validate / set, happy / sad).
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import typer
from typer.testing import CliRunner

import sec_generative_search.cli.provider as provider_module
from sec_generative_search.cli.provider import provider_app
from sec_generative_search.config.settings import (
    DatabaseSettings,
    EmbeddingSettings,
    LLMSettings,
    RAGSettings,
    SearchSettings,
)
from sec_generative_search.core.exceptions import (
    CatalogueRefreshError,
    DatabaseError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.types import CatalogueRefreshReport
from sec_generative_search.providers.registry import (
    ProviderEntry,
    ProviderSurface,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _stripped(output: str) -> str:
    return re.sub(r"\s+", " ", _ANSI_RE.sub("", output))


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeRegistry:
    """Stand-in for :class:`MetadataRegistry`.

    Exposes the ``encrypted`` flag the provider CLI consults and a
    ``close`` no-op so ``finally`` blocks do not blow up.
    """

    instances: ClassVar[list[_FakeRegistry]] = []
    encrypted_default: ClassVar[bool] = False
    raise_on_construct: ClassVar[BaseException | None] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self).raise_on_construct is not None:
            raise type(self).raise_on_construct
        self.encrypted = type(self).encrypted_default
        self.closed = False
        _FakeRegistry.instances.append(self)

    def close(self) -> None:
        self.closed = True


class _FakeStore:
    """Recording :class:`EncryptedCredentialStore` stand-in."""

    instances: ClassVar[list[_FakeStore]] = []
    raise_on_construct: ClassVar[BaseException | None] = None

    def __init__(self, registry: Any) -> None:
        if type(self).raise_on_construct is not None:
            raise type(self).raise_on_construct
        self.registry = registry
        self.set_calls: list[tuple[str, str, str]] = []
        _FakeStore.instances.append(self)

    def set(self, user_id: str, provider: str, api_key: str) -> None:
        self.set_calls.append((user_id, provider, api_key))


class _StubSettings:
    """Hand-built settings stand-in matching the seams ``cli.provider`` reads."""

    def __init__(self, *, persist_provider_credentials: bool = False) -> None:
        self.embedding = EmbeddingSettings.model_construct(
            provider="openai",
            model_name="text-embedding-3-small",
            device="auto",
            batch_size=32,
            idle_timeout_minutes=0,
        )
        self.database = DatabaseSettings.model_construct(
            deployment_profile="local",
            retention_max_age_days=0,
            chroma_path="./data/chroma_db",
            metadata_db_path="./data/metadata.sqlite",
            max_filings=10000,
            encryption_key=None,
            encryption_key_file=None,
            task_history_retention_days=0,
            task_history_persist_tickers=False,
            persist_provider_credentials=persist_provider_credentials,
        )
        self.llm = LLMSettings.model_construct(
            default_provider="openai",
            default_model=None,
            max_output_tokens=2048,
        )
        self.rag = RAGSettings.model_construct(
            context_token_budget=8000,
            refusal_enabled=True,
        )
        self.search = SearchSettings.model_construct(top_k=5, min_similarity=0.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def app() -> typer.Typer:
    test_app = typer.Typer()
    test_app.add_typer(provider_app, name="provider")
    return test_app


@pytest.fixture(autouse=True)
def _reset_doubles() -> Any:
    _FakeRegistry.instances.clear()
    _FakeRegistry.encrypted_default = False
    _FakeRegistry.raise_on_construct = None
    _FakeStore.instances.clear()
    _FakeStore.raise_on_construct = None
    yield


@pytest.fixture
def patch_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_module, "MetadataRegistry", _FakeRegistry)


@pytest.fixture
def patch_settings_persist_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provider_module, "get_settings", lambda: _StubSettings(persist_provider_credentials=False)
    )


@pytest.fixture
def patch_settings_persist_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provider_module, "get_settings", lambda: _StubSettings(persist_provider_credentials=True)
    )


@pytest.fixture
def patch_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``EncryptedCredentialStore`` at its real import path.

    ``cli.provider`` does a *local* import inside the command body, so
    we patch the source module that the local import will resolve to.
    """
    import sec_generative_search.database.credentials as creds_module

    monkeypatch.setattr(creds_module, "EncryptedCredentialStore", _FakeStore)


# ---------------------------------------------------------------------------
# `provider list`
# ---------------------------------------------------------------------------


class TestProviderList:
    def test_lists_registered_providers(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_off: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """At least the openai LLM row renders with the env-var column."""
        # No env keys set — every row should show the "no key" glyph.
        for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        result = runner.invoke(app, ["provider", "list"])
        assert result.exit_code == 0, result.output
        flat = _stripped(result.output)
        assert "openai" in flat
        assert "OPENAI_API_KEY" in flat
        assert "anthropic" in flat
        assert "ANTHROPIC_API_KEY" in flat

    def test_env_var_resolved_key_renders_masked(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_off: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A resolvable admin-env key renders only its masked tail —
        never the raw value."""
        canary = "sk-canary-list-key-must-never-render"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", canary)
        result = runner.invoke(app, ["provider", "list"])
        assert result.exit_code == 0, result.output
        assert canary not in result.output

    def test_surface_filter(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_off: None,
    ) -> None:
        """``--surface embedding`` shows only embedding providers."""
        result = runner.invoke(app, ["provider", "list", "--surface", "embedding"])
        assert result.exit_code == 0, result.output
        flat = _stripped(result.output)
        # OpenAI ships both surfaces; the embedding row must be present.
        assert "openai" in flat
        # LLM-only providers MUST NOT appear when filtering by embedding.
        assert "deepseek" not in flat
        assert "anthropic" not in flat

    def test_unknown_surface_rejected(
        self,
        runner: CliRunner,
        app: typer.Typer,
    ) -> None:
        result = runner.invoke(app, ["provider", "list", "--surface", "reranker-typo"])
        assert result.exit_code == 2
        assert "Invalid --surface" in _stripped(result.output)

    def test_registry_unavailable_still_lists(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_settings_persist_off: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``provider list`` MUST work on a fresh deployment where the
        SQLite file does not yet exist."""
        _FakeRegistry.raise_on_construct = DatabaseError("no db yet")
        monkeypatch.setattr(provider_module, "MetadataRegistry", _FakeRegistry)
        result = runner.invoke(app, ["provider", "list"])
        assert result.exit_code == 0, result.output
        assert "openai" in _stripped(result.output)


# ---------------------------------------------------------------------------
# `provider validate`
# ---------------------------------------------------------------------------


class TestProviderValidate:
    def test_validate_accepted(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_off: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        canary = "sk-test-accept"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", canary)
        monkeypatch.setattr(
            provider_module,
            "validate_credential",
            lambda provider, surface, api_key, model=None: True,
        )
        result = runner.invoke(app, ["provider", "validate", "openai"])
        assert result.exit_code == 0, result.output
        assert "accepted" in result.output
        assert canary not in result.output

    def test_validate_rejected(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_off: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        canary = "sk-test-reject"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", canary)
        monkeypatch.setattr(
            provider_module,
            "validate_credential",
            lambda provider, surface, api_key, model=None: False,
        )
        result = runner.invoke(app, ["provider", "validate", "openai"])
        assert result.exit_code == 1, result.output
        assert "rejected" in result.output
        assert canary not in result.output

    def test_validate_no_credential(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_off: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = runner.invoke(app, ["provider", "validate", "openai"])
        assert result.exit_code == 1
        assert "No credential" in result.output
        assert "OPENAI_API_KEY" in result.output

    def test_validate_unknown_provider(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_off: None,
    ) -> None:
        result = runner.invoke(app, ["provider", "validate", "doesnotexist"])
        assert result.exit_code == 1
        assert "Unknown provider" in result.output

    def test_validate_rate_limit_is_exit_2(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_off: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Transient upstream failure MUST NOT be reported as a verdict."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-anything")  # pragma: allowlist secret

        def _raise(*_a: Any, **_kw: Any) -> bool:
            raise ProviderRateLimitError("slow down")

        monkeypatch.setattr(provider_module, "validate_credential", _raise)
        result = runner.invoke(app, ["provider", "validate", "openai"])
        assert result.exit_code == 2
        assert "Provider unavailable" in result.output

    def test_validate_other_provider_error_is_exit_2(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_off: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-anything")  # pragma: allowlist secret

        def _raise(*_a: Any, **_kw: Any) -> bool:
            raise ProviderError("transport down")

        monkeypatch.setattr(provider_module, "validate_credential", _raise)
        result = runner.invoke(app, ["provider", "validate", "openai"])
        assert result.exit_code == 2
        assert "Provider error" in result.output

    def test_validate_embedding_surface_requires_model(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_off: None,
    ) -> None:
        """The embedding surface requires a known model slug — registry
        raises ``ValueError`` for unknown slugs, which we surface as an
        unknown-model envelope (exit 1)."""
        result = runner.invoke(
            app,
            [
                "provider",
                "validate",
                "openai",
                "--surface",
                "embedding",
                "--model",
                "not-a-real-model",
            ],
        )
        assert result.exit_code == 1
        assert "Unknown model" in result.output


# ---------------------------------------------------------------------------
# `provider set` — hard-fail discipline
# ---------------------------------------------------------------------------


class TestProviderSetHardFail:
    """When the encrypted store is unavailable ``set`` MUST refuse —
    never write to plaintext SQLite.

    The hint must name **all three** opt-in steps (extra / key /
    persist toggle) so the operator does not have to guess which one
    is missing.  We assert all three keywords appear in the hint
    string.
    """

    def _assert_three_step_hint(self, output: str) -> None:
        flat = _stripped(output)
        assert "[encryption]" in flat
        assert "DB_ENCRYPTION_KEY" in flat
        assert "DB_PERSIST_PROVIDER_CREDENTIALS" in flat

    def test_persist_off_hard_fails(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_settings_persist_off: None,
    ) -> None:
        """Persistence disabled — refuse without prompting for a key."""
        result = runner.invoke(app, ["provider", "set", "openai"], input="sk-must-never-be-read\n")
        assert result.exit_code == 1
        self._assert_three_step_hint(result.output)
        # No FakeStore was ever constructed.
        assert _FakeStore.instances == []

    def test_persist_on_but_registry_unencrypted_hard_fails(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_on: None,
    ) -> None:
        """``persist=true`` but SQLCipher missing (registry.encrypted=False)
        MUST refuse — writing plaintext is the trap."""
        _FakeRegistry.encrypted_default = False
        result = runner.invoke(app, ["provider", "set", "openai"], input="sk-must-never-be-read\n")
        assert result.exit_code == 1
        self._assert_three_step_hint(result.output)
        assert _FakeStore.instances == []

    def test_hard_fail_runs_before_key_prompt(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_settings_persist_off: None,
    ) -> None:
        """The operator MUST NOT be asked for a key when the store is
        unavailable — leaking a paste into shell history under a doomed
        flow is exactly what hard-failing exists to prevent."""
        # No stdin supplied; if the prompt fired the CliRunner would
        # error on EOF before printing the hard-fail envelope.
        result = runner.invoke(app, ["provider", "set", "openai"], input="")
        assert result.exit_code == 1
        self._assert_three_step_hint(result.output)

    def test_unknown_provider_rejected_before_anything(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_settings_persist_off: None,
    ) -> None:
        result = runner.invoke(app, ["provider", "set", "doesnotexist"], input="")
        assert result.exit_code == 1
        assert "Unknown provider" in result.output


# ---------------------------------------------------------------------------
# `provider set` — happy path
# ---------------------------------------------------------------------------


class TestProviderSetHappy:
    def test_writes_and_validates(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_on: None,
        patch_store: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _FakeRegistry.encrypted_default = True
        monkeypatch.setattr(
            provider_module,
            "validate_credential",
            lambda provider, surface, api_key, model=None: True,
        )
        result = runner.invoke(
            app,
            ["provider", "set", "openai"],
            input="sk-fresh-key-paste\n",  # pragma: allowlist secret
        )
        assert result.exit_code == 0, result.output
        assert len(_FakeStore.instances) == 1
        store = _FakeStore.instances[0]
        assert len(store.set_calls) == 1
        user_id, provider, key = store.set_calls[0]
        assert user_id == "__admin__"
        assert provider == "openai"
        assert key == "sk-fresh-key-paste"  # pragma: allowlist secret
        # Output shows masked tail only, not the raw key.
        assert "sk-fresh-key-paste" not in result.output
        assert "Stored credential" in result.output

    def test_no_validate_flag_skips_round_trip(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_on: None,
        patch_store: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _FakeRegistry.encrypted_default = True

        called: list[tuple] = []

        def _validate(*args: Any, **kwargs: Any) -> bool:
            called.append(args)
            return True

        monkeypatch.setattr(provider_module, "validate_credential", _validate)
        result = runner.invoke(
            app,
            ["provider", "set", "openai", "--no-validate"],
            input="sk-paste\n",  # pragma: allowlist secret
        )
        assert result.exit_code == 0, result.output
        # validate_credential never called.
        assert called == []

    def test_post_write_rejection_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_on: None,
        patch_store: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the upstream rejects the freshly-stored key, exit 1 so
        automation can react.  The credential is still stored (the
        rejection is a verdict on the *key*, not on the write); the
        message says so."""
        _FakeRegistry.encrypted_default = True
        monkeypatch.setattr(
            provider_module,
            "validate_credential",
            lambda provider, surface, api_key, model=None: False,
        )
        result = runner.invoke(
            app,
            ["provider", "set", "openai"],
            input="sk-bad\n",  # pragma: allowlist secret
        )
        assert result.exit_code == 1
        assert "rejected" in result.output

    def test_post_write_transient_does_not_fail_store(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_on: None,
        patch_store: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A transient upstream failure during post-write validation
        MUST NOT exit non-zero — the credential is stored, the
        validation just happened to be inconclusive."""
        _FakeRegistry.encrypted_default = True

        def _raise(*_a: Any, **_kw: Any) -> bool:
            raise ProviderTimeoutError("ten seconds")

        monkeypatch.setattr(provider_module, "validate_credential", _raise)
        result = runner.invoke(
            app,
            ["provider", "set", "openai"],
            input="sk-paste\n",  # pragma: allowlist secret
        )
        assert result.exit_code == 0, result.output
        assert "transiently" in result.output

    def test_empty_key_rejected(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_on: None,
        patch_store: None,
    ) -> None:
        _FakeRegistry.encrypted_default = True
        result = runner.invoke(
            app,
            ["provider", "set", "openai"],
            input="\n",
        )
        # typer.prompt re-prompts on empty input, which CliRunner sees
        # as EOF and surfaces as a non-zero exit.  The result MUST NOT
        # be a successful store.
        assert result.exit_code != 0
        assert _FakeStore.instances == [] or not any(s.set_calls for s in _FakeStore.instances)


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestProviderSecurity:
    def test_list_never_echoes_resolved_key(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_off: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        canary = "sk-list-canary-must-never-leak"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", canary)
        result = runner.invoke(app, ["provider", "list"])
        assert result.exit_code == 0, result.output
        assert canary not in result.output

    def test_validate_never_echoes_resolved_key(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_off: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        canary = "sk-validate-canary-must-never-leak"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", canary)
        monkeypatch.setattr(
            provider_module,
            "validate_credential",
            lambda provider, surface, api_key, model=None: True,
        )
        result = runner.invoke(app, ["provider", "validate", "openai"])
        assert canary not in result.output

    def test_set_never_echoes_pasted_key(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_on: None,
        patch_store: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _FakeRegistry.encrypted_default = True
        canary = "sk-set-canary-paste-must-never-leak"  # pragma: allowlist secret
        monkeypatch.setattr(
            provider_module,
            "validate_credential",
            lambda provider, surface, api_key, model=None: True,
        )
        result = runner.invoke(app, ["provider", "set", "openai"], input=f"{canary}\n")
        assert result.exit_code == 0, result.output
        assert canary not in result.output

    def test_set_hard_fail_never_prompts_for_key(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_settings_persist_off: None,
    ) -> None:
        """Misconfigured environment MUST NOT cause the CLI to ask for a
        key — pasting a credential into a doomed flow would land in
        terminal scrollback / shell history under exactly the wrong
        circumstances.  We provide no stdin; the command MUST hard-fail
        before any prompt fires."""
        result = runner.invoke(app, ["provider", "set", "openai"], input="")
        assert result.exit_code == 1
        # Hint mentions all three steps — operator does not have to
        # guess which one is missing.
        flat = _stripped(result.output)
        assert "[encryption]" in flat
        assert "DB_ENCRYPTION_KEY" in flat
        assert "DB_PERSIST_PROVIDER_CREDENTIALS" in flat

    def test_provider_cli_does_not_honour_demo_mode(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patch_registry: None,
        patch_settings_persist_on: None,
        patch_store: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``API_DEMO_MODE`` is an API-tier control; the operator CLI
        does not consult it (mirrors ``cli.manage``'s same discipline)."""
        _FakeRegistry.encrypted_default = True
        monkeypatch.setenv("API_DEMO_MODE", "true")
        monkeypatch.setattr(
            provider_module,
            "validate_credential",
            lambda provider, surface, api_key, model=None: True,
        )
        result = runner.invoke(
            app,
            ["provider", "set", "openai"],
            input="sk-paste\n",  # pragma: allowlist secret
        )
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Registry-table consistency
# ---------------------------------------------------------------------------


class TestEnvVarTableConsistency:
    def test_env_var_table_matches_factory(self) -> None:
        """The local ``_ENV_VAR_BY_PROVIDER`` table is duplicated from
        ``providers.factory`` to keep the CLI free of a private import.
        A drift test pins them together — they must stay byte-identical."""
        from sec_generative_search.providers.factory import (
            _DEFAULT_ENV_VAR_BY_PROVIDER,
        )

        assert provider_module._ENV_VAR_BY_PROVIDER == _DEFAULT_ENV_VAR_BY_PROVIDER

    def test_every_registry_name_has_env_var(self) -> None:
        """Every provider in the registry MUST have an admin-env entry —
        otherwise ``provider list``'s 'Admin env var' column would be
        empty for that row, and ``provider validate`` could not tell the
        operator which env var to set.  Skip ``local`` (still entered
        as HF_TOKEN) and ``local_llm`` (a self-hosted endpoint with no
        credential — it is reached via a sentinel, not an admin-default
        key), the surfaces that fundamentally have no admin-default."""
        from sec_generative_search.providers.registry import ProviderRegistry

        # Credential-free surfaces: a self-hosted ``local_llm`` endpoint
        # carries no API key, so it has no admin-env var by design.
        no_admin_default = {"local_llm"}

        names = {entry.name for entry in ProviderRegistry.all_entries(include_unavailable=True)}
        for name in names - no_admin_default:
            assert name in provider_module._ENV_VAR_BY_PROVIDER, (
                f"{name!r} has no admin env var entry"
            )


# ---------------------------------------------------------------------------
# Pricing-tier rendering
# ---------------------------------------------------------------------------


class TestPricingLabel:
    def test_known_model_returns_tier(self) -> None:
        """A catalogued LLM model returns its cost-derived tier value."""
        assert provider_module._pricing_label("openai", ProviderSurface.LLM, "gpt-4o") == "high"

    def test_unknown_slug_returns_unknown(self) -> None:
        """OpenRouter's arbitrary-slug surface has no catalogued tier."""
        assert (
            provider_module._pricing_label("openrouter", ProviderSurface.LLM, "vendor/x")
            == "unknown"
        )

    def test_embedding_surface_returns_unknown(self) -> None:
        """Embedding providers have no pricing surface."""
        assert (
            provider_module._pricing_label(
                "openai", ProviderSurface.EMBEDDING, "text-embedding-3-small"
            )
            == "unknown"
        )

    def test_missing_slug_returns_unknown(self) -> None:
        assert provider_module._pricing_label("openai", ProviderSurface.LLM, None) == "unknown"


# ---------------------------------------------------------------------------
# `provider refresh-catalogue`
# ---------------------------------------------------------------------------


class _StubProviderRefreshSettings:
    """Minimal settings stub exposing only the catalogue-refresh fields."""

    def __init__(self) -> None:
        self.provider = SimpleNamespace(
            catalogue_refresh_source="models_dev",
            catalogue_refresh_url=None,
            catalogue_overlay_path="./data/model_catalogue_overlay.json",
        )


class TestProviderRefreshCatalogue:
    """``provider refresh-catalogue`` drives the opt-in refresh seam."""

    _OVERLAY = "/srv/data/SENTINEL-overlay.json"

    @pytest.fixture
    def patch_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(provider_module, "get_settings", lambda: _StubProviderRefreshSettings())

    def _patch_refresh(self, monkeypatch: pytest.MonkeyPatch, fn: Any) -> None:
        # ``cli.provider`` local-imports ``refresh_overlay`` inside the command,
        # so patch the source module the import resolves to.
        import sec_generative_search.providers.refresh as refresh_module

        monkeypatch.setattr(refresh_module, "refresh_overlay", fn)

    def test_happy_path_reports_counts(
        self, runner: CliRunner, app: typer.Typer, monkeypatch: pytest.MonkeyPatch, patch_settings
    ) -> None:
        def _fake(**_: Any) -> CatalogueRefreshReport:
            return CatalogueRefreshReport(
                source="models_dev",
                source_url="https://models.dev/api.json",
                provider_count=7,
                model_count=42,
                overlay_path=self._OVERLAY,
            )

        self._patch_refresh(monkeypatch, _fake)
        result = runner.invoke(app, ["provider", "refresh-catalogue"])
        assert result.exit_code == 0
        out = _stripped(result.stdout)
        assert "7 providers, 42 models" in out
        assert "models_dev" in out

    def test_forwards_configured_args(
        self, runner: CliRunner, app: typer.Typer, monkeypatch: pytest.MonkeyPatch, patch_settings
    ) -> None:
        captured: dict[str, Any] = {}

        def _spy(**kwargs: Any) -> CatalogueRefreshReport:
            captured.update(kwargs)
            return CatalogueRefreshReport(
                source="models_dev",
                source_url="https://models.dev/api.json",
                provider_count=1,
                model_count=1,
                overlay_path=self._OVERLAY,
            )

        self._patch_refresh(monkeypatch, _spy)
        runner.invoke(app, ["provider", "refresh-catalogue"])
        assert captured["source"] == "models_dev"
        assert captured["url"] is None
        assert captured["overlay_path"] == "./data/model_catalogue_overlay.json"

    def test_refresh_error_exits_1(
        self, runner: CliRunner, app: typer.Typer, monkeypatch: pytest.MonkeyPatch, patch_settings
    ) -> None:
        def _boom(**_: Any) -> CatalogueRefreshReport:
            raise CatalogueRefreshError("Catalogue refresh response was not valid JSON.")

        self._patch_refresh(monkeypatch, _boom)
        result = runner.invoke(app, ["provider", "refresh-catalogue"])
        assert result.exit_code == 1
        assert "Catalogue refresh failed" in _stripped(result.stdout)

    def test_os_error_does_not_leak_path(
        self, runner: CliRunner, app: typer.Typer, monkeypatch: pytest.MonkeyPatch, patch_settings
    ) -> None:
        def _disk(**_: Any) -> CatalogueRefreshReport:
            raise OSError(f"[Errno 13] Permission denied: '{self._OVERLAY}'")

        self._patch_refresh(monkeypatch, _disk)
        result = runner.invoke(app, ["provider", "refresh-catalogue"])
        assert result.exit_code == 1
        # The OS error's embedded path must never reach the rendered output.
        assert self._OVERLAY not in result.stdout


# ---------------------------------------------------------------------------
# Smoke import — placate ruff that imports are used
# ---------------------------------------------------------------------------


def test_provider_app_smoke() -> None:
    """The Typer instance is importable and carries every subcommand."""
    info_names = {cmd.name for cmd in provider_app.registered_commands}
    assert {"list", "validate", "set", "refresh-catalogue"} <= info_names


# Unused symbols carried for parametric extension — pin so ruff
# does not strip them.  ``ProviderEntry`` / ``ProviderAuthError`` are
# imported above purely so the security suite can grow without
# touching the import header.
_ = ProviderEntry
_ = ProviderAuthError
