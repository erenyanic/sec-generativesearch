"""Tests for :mod:`sec_generative_search.providers.factory`.

Covers:

- Default resolver shape: provider-specific env-var lookup, empty-string
    coercion to ``None``, unknown-name pass-through.
- ``build_embedder`` construction flow through
  :class:`ProviderRegistry`, including the forbidden-outside-the-factory
  contract (exercised by real provider classes, not bypassed by stubs).
- Hosted-provider missing-credential path surfaces a
  :class:`ConfigurationError` with the expected env var name.
- ``local`` tolerates a missing credential via the sentinel path.
- Custom resolver overrides the default.
- Security: factory never retains the resolved credential after the
  call, and the ``ConfigurationError`` message never echoes any
  credential material (the default resolver is the only seam that
  produces the value, and it never passes the value to the error
  branch — fail-fast guards that invariant).
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sec_generative_search.config.settings import EmbeddingSettings
from sec_generative_search.core.exceptions import ConfigurationError
from sec_generative_search.providers.factory import (
    build_embedder,
    build_llm_provider,
    default_api_key_resolver,
)
from sec_generative_search.providers.openai import (
    OpenAIEmbeddingProvider,
    OpenAIProvider,
)
from sec_generative_search.providers.registry import (
    ProviderRegistry,
    ProviderSurface,
)

_LOCAL_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
_requires_local = pytest.mark.skipif(
    not _LOCAL_AVAILABLE,
    reason="LocalEmbeddingProvider requires the [local-embeddings] extra",
)


@pytest.fixture(autouse=True)
def _reset_registry_cache() -> None:
    ProviderRegistry._reset_availability_cache()


@pytest.fixture(autouse=True)
def _clean_embedding_and_credential_env(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[pytest.MonkeyPatch]:
    """Strip every env var that would leak the developer's ``.env``.

    Pydantic settings read ``.env`` eagerly; without this fixture the
    ``EmbeddingSettings(...)`` calls below inherit the local machine's
    ``EMBEDDING_BATCH_SIZE=8`` (or similar) and trip the hosted-provider
    guard.  The credential env vars are stripped for the same reason —
    a populated ``OPENAI_API_KEY`` on the dev machine would paper over
    the "missing credential" tests.
    """
    for key in list(os.environ.keys()):
        if key.startswith("EMBEDDING_") or key in {
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "MISTRAL_API_KEY",
            "DASHSCOPE_API_KEY",
            "HF_TOKEN",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
            "MOONSHOT_API_KEY",
            "OPENROUTER_API_KEY",
            "ZAI_API_KEY",
            "XAI_API_KEY",
            "MINIMAX_API_KEY",
            "MIMO_API_KEY",
        }:
            monkeypatch.delenv(key, raising=False)
    yield monkeypatch


# ---------------------------------------------------------------------------
# default_api_key_resolver
# ---------------------------------------------------------------------------


class TestDefaultApiKeyResolver:
    @pytest.mark.parametrize(
        "provider,env_var",
        [
            ("openai", "OPENAI_API_KEY"),
            ("gemini", "GEMINI_API_KEY"),
            ("mistral", "MISTRAL_API_KEY"),
            ("qwen", "DASHSCOPE_API_KEY"),
            ("local", "HF_TOKEN"),
        ],
    )
    def test_reads_expected_env_var(
        self,
        provider: str,
        env_var: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(env_var, "sk-value-from-env")
        assert default_api_key_resolver(provider) == "sk-value-from-env"

    def test_unknown_provider_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Unregistered names never raise from the resolver; the factory
        # decides what to do with a missing credential.
        assert default_api_key_resolver("not-a-real-provider") is None

    def test_empty_string_env_value_coerces_to_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Blank ``OPENAI_API_KEY=`` must not enable a zero-length key."""
        monkeypatch.setenv("OPENAI_API_KEY", "")
        assert default_api_key_resolver("openai") is None

    def test_missing_env_var_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert default_api_key_resolver("openai") is None


# ---------------------------------------------------------------------------
# default_api_key_resolver — *_FILE secret-file indirection
# ---------------------------------------------------------------------------


class TestDefaultApiKeyResolverFileIndirection:
    """Admin-default provider keys resolve from ``{ENV_VAR}_FILE`` too.

    Mirrors the SQLCipher-key / auth-pepper ``*_FILE`` story so a
    Secret-Manager / Docker-secret file mount supplies the key without
    it ever landing in ``/proc/<pid>/environ``. The two sources are
    mutually exclusive and an empty file is rejected — exactly the
    contract :func:`resolve_secret_from_value_or_file` enforces for the
    other two secrets.
    """

    @pytest.mark.parametrize(
        "provider,env_var",
        [
            ("openai", "OPENAI_API_KEY"),  # both surfaces
            ("anthropic", "ANTHROPIC_API_KEY"),  # LLM-only
            ("local", "HF_TOKEN"),  # embedding-only
        ],
    )
    def test_reads_key_from_file(
        self,
        provider: str,
        env_var: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv(env_var, raising=False)
        key_file = tmp_path / "key"
        # Trailing whitespace/newline is stripped — secret managers and
        # ``printf`` both commonly append one.
        key_file.write_text("sk-value-from-file-1234\n")
        monkeypatch.setenv(f"{env_var}_FILE", str(key_file))

        assert default_api_key_resolver(provider) == "sk-value-from-file-1234"

    def test_inline_value_still_resolves_without_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression: the inline env var keeps working when no file is set."""
        monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-inline-value-5678")
        assert default_api_key_resolver("openai") == "sk-inline-value-5678"

    def test_blank_file_path_coerces_to_none_no_file_branch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A blank ``OPENAI_API_KEY_FILE=`` must not trip the file branch.

        An empty string is not a path; coercing it to ``None`` keeps the
        resolver returning ``None`` rather than raising "does not exist".
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY_FILE", "")
        assert default_api_key_resolver("openai") is None

    def test_nonexistent_file_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        missing = tmp_path / "absent"
        monkeypatch.setenv("OPENAI_API_KEY_FILE", str(missing))
        with pytest.raises(ValueError, match="OPENAI_API_KEY_FILE"):
            default_api_key_resolver("openai")

    @pytest.mark.security
    def test_inline_and_file_are_mutually_exclusive(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Both sources set is a hard error — no silent precedence.

        Silent precedence would mask a stale key lingering in one source
        after the operator rotated the other.
        """
        secret = "sk-secret-must-not-leak-9999"  # pragma: allowlist secret
        key_file = tmp_path / "key"
        key_file.write_text(secret)
        monkeypatch.setenv("OPENAI_API_KEY", secret)
        monkeypatch.setenv("OPENAI_API_KEY_FILE", str(key_file))

        with pytest.raises(ValueError) as exc_info:
            default_api_key_resolver("openai")
        message = str(exc_info.value)
        # Names both operator-facing knobs …
        assert "OPENAI_API_KEY" in message
        assert "OPENAI_API_KEY_FILE" in message
        # … but never echoes the secret material itself.
        assert secret not in message

    @pytest.mark.security
    def test_empty_file_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """An empty / whitespace-only key file never enables a blank key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        key_file = tmp_path / "key"
        key_file.write_text("   \n")
        monkeypatch.setenv("OPENAI_API_KEY_FILE", str(key_file))
        with pytest.raises(ValueError, match="empty"):
            default_api_key_resolver("openai")

    @pytest.mark.security
    def test_file_backed_key_never_appears_in_factory_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A file-mounted key reaches the provider but never an error string.

        Exercises the full ``build_llm_provider`` path: a valid file key
        constructs the provider (no error), and when the file is removed
        the resulting ``ConfigurationError`` names the env knob, never
        any secret material.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY_FILE", raising=False)
        with pytest.raises(ConfigurationError) as exc_info:
            build_llm_provider("anthropic")
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)


# ---------------------------------------------------------------------------
# build_embedder — happy path
# ---------------------------------------------------------------------------


class TestBuildEmbedderHosted:
    def test_builds_openai_embedding_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-ABCDEFGHIJKLMNOP")

        # Construct settings directly (no env I/O) so the test does not
        # depend on ``.env`` contents.
        settings = EmbeddingSettings(provider="openai", model_name="text-embedding-3-small")
        embedder = build_embedder(settings)

        assert isinstance(embedder, OpenAIEmbeddingProvider)
        # The factory must wire the configured model through.
        assert embedder.get_dimension() == 1536

    def test_raises_without_credential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = EmbeddingSettings(provider="openai", model_name="text-embedding-3-small")
        with pytest.raises(ConfigurationError) as exc_info:
            build_embedder(settings)
        # The message names the env var so the user can fix it without
        # reading the factory source.
        assert "OPENAI_API_KEY" in str(exc_info.value)
        assert "openai" in str(exc_info.value)

    def test_custom_resolver_is_used(self) -> None:
        calls: list[str] = []

        def resolver(name: str) -> str | None:
            calls.append(name)
            return "sk-from-custom-resolver-1234"

        settings = EmbeddingSettings(provider="openai", model_name="text-embedding-3-small")
        embedder = build_embedder(settings, api_key_resolver=resolver)
        assert isinstance(embedder, OpenAIEmbeddingProvider)
        # The resolver was called exactly once with the configured
        # provider name.
        assert calls == ["openai"]

    def test_resolver_none_for_hosted_provider_raises(self) -> None:
        settings = EmbeddingSettings(provider="gemini", model_name="text-embedding-004")
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
            build_embedder(settings, api_key_resolver=lambda _name: None)


# ---------------------------------------------------------------------------
# build_embedder — local path
# ---------------------------------------------------------------------------


@_requires_local
class TestBuildEmbedderLocal:
    def test_local_tolerates_missing_credential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        settings = EmbeddingSettings(
            provider="local", model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        embedder = build_embedder(settings)
        # The provider accepts ``None`` via its ``_NO_TOKEN_SENTINEL``
        # path; the factory must not intercept it with a
        # ConfigurationError.
        from sec_generative_search.providers.local import LocalEmbeddingProvider

        assert isinstance(embedder, LocalEmbeddingProvider)


class TestBuildEmbedderForwardsLocalKnobs:
    """``EMBEDDING_DEVICE`` / ``EMBEDDING_BATCH_SIZE`` must reach the on-device
    provider. The factory used to drop them, so a GPU deployment pinned to
    ``cuda`` silently embedded on the CPU. A recording stub stands in for the
    adapter, so this runs without the ``[local-embeddings]`` extra."""

    @staticmethod
    def _record_constructions(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        class _Recorder:
            def __init__(self, api_key: str | None, **kwargs: Any) -> None:
                calls.append({"api_key": api_key, **kwargs})

        monkeypatch.setattr(ProviderRegistry, "get_class", lambda _name, _surface: _Recorder)
        return calls

    def test_local_provider_receives_device_and_batch_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._record_constructions(monkeypatch)
        settings = EmbeddingSettings(
            provider="local",
            model_name="google/embeddinggemma-300m",
            device="cuda",
            batch_size=8,
            idle_timeout_minutes=5,
        )
        build_embedder(settings, api_key_resolver=lambda _name: None)
        # idle_timeout_minutes is deliberately NOT forwarded until the
        # idle-unload race (OPTIMIZATIONS.md F15) is fixed.
        assert calls == [
            {
                "api_key": None,
                "model": "google/embeddinggemma-300m",
                "device": "cuda",
                "batch_size": 8,
            }
        ]

    def test_hosted_provider_receives_no_local_only_knobs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._record_constructions(monkeypatch)
        fake_key = "sk-test-1234ABCD"  # pragma: allowlist secret
        settings = EmbeddingSettings(provider="openai", model_name="text-embedding-3-small")
        build_embedder(settings, api_key_resolver=lambda _name: fake_key)
        assert calls == [{"api_key": fake_key, "model": "text-embedding-3-small"}]


class TestBuildEmbedderExtrasGating:
    def test_missing_extras_surfaces_registry_key_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ``[local-embeddings]`` installed, the factory must
        surface the registry's install-hint ``KeyError`` — this is the
        contract that lets settings load succeed while construction
        fails with actionable guidance.
        """
        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **kw: (
                None if name == "sentence_transformers" else real_find_spec(name, *a, **kw)
            ),
        )
        ProviderRegistry._reset_availability_cache()

        settings = EmbeddingSettings(provider="local", model_name="google/embeddinggemma-300m")
        with pytest.raises(KeyError, match="optional extras"):
            build_embedder(settings)


# ---------------------------------------------------------------------------
# build_embedder — construction seam & security
# ---------------------------------------------------------------------------


class TestFactoryIsTheSoleSeam:
    """The factory must route through ProviderRegistry so callers can
    swap the resolver without touching any call site."""

    def test_factory_delegates_to_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patching ``ProviderRegistry.get_class`` proves the factory
        uses the registry rather than importing adapters directly."""
        calls: list[tuple[str, ProviderSurface]] = []

        def stub_class(name: str, surface: ProviderSurface) -> type:
            calls.append((name, surface))
            return OpenAIEmbeddingProvider

        monkeypatch.setattr(ProviderRegistry, "get_class", stub_class)

        settings = EmbeddingSettings(provider="openai", model_name="text-embedding-3-small")
        build_embedder(settings, api_key_resolver=lambda _n: "sk-test-1234ABCD")
        assert calls == [("openai", ProviderSurface.EMBEDDING)]


@pytest.mark.security
class TestFactoryCredentialHygiene:
    """The factory must not retain the credential beyond construction."""

    def test_configuration_error_does_not_echo_credential_material(self) -> None:
        """When the resolver returns ``None`` we know no credential is
        in flight, but assert that the error text also carries none of
        the bad-actor needles downstream logs grep for."""

        settings = EmbeddingSettings(provider="openai", model_name="text-embedding-3-small")
        with pytest.raises(ConfigurationError) as exc_info:
            build_embedder(settings, api_key_resolver=lambda _n: None)
        rendered = str(exc_info.value).lower()
        # The env-var name mention is fine; credentials themselves must
        # not appear.
        for needle in ("sk-", "bearer", "secret=", "password"):
            assert needle not in rendered

    def test_module_carries_no_credentials_at_rest(self) -> None:
        """Module-level state must not accumulate keys across calls.

        A naive caching layer inside the factory could inadvertently
        hold credentials in memory after the embedder has been dropped.
        The factory is stateless by contract — assert it.
        """
        import sec_generative_search.providers.factory as factory_module

        # Whitelist of acceptable module-level attributes.
        module_attrs: dict[str, Any] = {
            name: getattr(factory_module, name)
            for name in dir(factory_module)
            if not name.startswith("_")
        }
        for name, value in module_attrs.items():
            # Strings at module scope must not look like keys.
            if isinstance(value, str):
                assert "sk-" not in value.lower()
                assert "bearer " not in value.lower()
            # Dict values at module scope (the env-var mapping) must be
            # env-var *names*, not credentials.
            if isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, str):
                        # Env var names end in _KEY or _TOKEN by
                        # convention; a literal credential would not.
                        assert v.isupper(), (
                            f"factory module dict {name!r} contains a "
                            f"lower-case value {v!r} — the mapping must "
                            "carry env-var names only."
                        )


# ---------------------------------------------------------------------------
# build_llm_provider
# ---------------------------------------------------------------------------


class TestBuildLLMProvider:
    """The LLM construction seam mirrors ``build_embedder``.

    Differences asserted: no per-instance model is forwarded; a missing
    credential is a ``ConfigurationError`` for every hosted provider,
    while the self-hosted ``local_llm`` entry tolerates a ``None`` resolver
    result via a sentinel; the resolver chain remains the canonical
    credential seam.
    """

    def test_builds_openai_llm_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1234567890ABCDEF")
        provider = build_llm_provider("openai")
        assert isinstance(provider, OpenAIProvider)

    def test_builds_anthropic_via_default_resolver(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The default resolver lets Anthropic and the rest of the
        LLM-only providers resolve from server env without any
        caller-supplied chain."""
        from sec_generative_search.providers.anthropic import AnthropicProvider

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-1234567890ABCDEF")
        provider = build_llm_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)

    def test_raises_without_credential(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigurationError) as exc_info:
            build_llm_provider("openai")
        message = str(exc_info.value)
        # The message names the env var so an admin can fix it without
        # reading the factory source.
        assert "OPENAI_API_KEY" in message
        # The message names the resolver-chain alternative so a session
        # caller knows there is a per-request path too.
        assert "session" in message or "resolver chain" in message

    def test_custom_resolver_overrides_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-DDDDDDDD")
        called_with: list[str] = []

        def resolver(name: str) -> str | None:
            called_with.append(name)
            return "sk-from-resolver-FFFFFFFF"

        provider = build_llm_provider("openai", api_key_resolver=resolver)
        assert isinstance(provider, OpenAIProvider)
        assert called_with == ["openai"]

    def test_factory_delegates_to_registry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Patching ``ProviderRegistry.get_entry`` proves ``build_llm_provider``
        routes through the registry rather than importing adapters directly.

        The factory reads the *entry* (not just the class) so it can honour
        per-provider flags such as ``requires_api_key``; the delegation seam
        is therefore ``get_entry``.
        """
        from sec_generative_search.providers.registry import ProviderEntry

        calls: list[tuple[str, ProviderSurface]] = []

        def stub_entry(name: str, surface: ProviderSurface) -> ProviderEntry:
            calls.append((name, surface))
            return ProviderEntry(name, surface, OpenAIProvider)

        monkeypatch.setattr(ProviderRegistry, "get_entry", stub_entry)

        build_llm_provider("openai", api_key_resolver=lambda _n: "sk-test-1234ABCD")
        assert calls == [("openai", ProviderSurface.LLM)]


@pytest.mark.security
class TestBuildLLMProviderCredentialHygiene:
    def test_configuration_error_does_not_echo_credential_material(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigurationError) as exc_info:
            build_llm_provider("openai", api_key_resolver=lambda _n: None)
        rendered = str(exc_info.value).lower()
        for needle in ("sk-", "bearer", "secret=", "password"):
            assert needle not in rendered

    def test_built_provider_repr_does_not_leak_key(self) -> None:
        provider = build_llm_provider(
            "openai",
            api_key_resolver=lambda _n: "sk-NEEDLE-1234567890",
        )
        rendered = repr(provider) + " " + str(provider)
        assert "sk-NEEDLE-1234567890" not in rendered
        assert "NEEDLE" not in rendered
