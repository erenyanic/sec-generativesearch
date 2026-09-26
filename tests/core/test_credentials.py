"""Tests for :mod:`sec_generative_search.core.credentials`.

Coverage:

- ``InMemorySessionCredentialStore`` — set/get/delete/clear/list,
  TTL-based eviction (idle and absolute), per-key isolation,
  empty-key rejection, ``ttl_seconds<=0`` rejection.
- Resolver chain composition — first-hit-wins, ``None`` propagation,
  empty resolver list.
- ``session_resolver`` / ``encrypted_user_resolver`` — adapt the
  credential store to the ``ApiKeyResolver`` shape.
- ``validate_credential`` — happy path, ``ProviderAuthError`` collapses
  to ``False``, non-auth ``ProviderError`` propagates, type-checks
  ``surface``.
- Security: every credential touch surfaces only a masked tail in
  audit logs; raw key never appears in ``__repr__`` / exceptions /
  resolver outputs that should not return it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from sec_generative_search.core.credentials import (
    InMemorySessionCredentialStore,
    chain_resolvers,
    encrypted_user_resolver,
    session_resolver,
    validate_credential,
)
from sec_generative_search.core.exceptions import (
    ProviderAuthError,
    ProviderRateLimitError,
)
from sec_generative_search.core.logging import configure_logging
from sec_generative_search.providers.registry import ProviderSurface

# ---------------------------------------------------------------------------
# Audit-log capture — the package logger has propagate=False so caplog (which
# attaches to root) misses it without this opt-in.
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_caplog(
    caplog: pytest.LogCaptureFixture,
) -> Iterator[pytest.LogCaptureFixture]:
    """Yield a caplog with the package logger's propagate flag flipped on.

    ``configure_logging`` sets ``propagate = False`` on the package
    logger so production handlers (Rich console / RotatingFileHandler)
    are the single sink.  ``caplog`` attaches to root, so without
    flipping propagation here it never sees the audit records.  This
    mirrors the audit-log capture pattern used elsewhere in the test
    suite.

    ``configure_logging()`` is forced first so the flip is not silently
    undone: ``get_logger`` lazily configures logging on first use, and that
    call resets ``propagate = False``.  When the module-level
    ``_logging_configured`` flag is ``False`` at emit time (e.g. after
    ``tests/core/test_logging.py`` resets it in teardown), the lazy reconfigure
    fires *after* this fixture sets ``propagate = True`` and flips it back —
    the record then never reaches caplog.  Configuring up-front makes that lazy
    call an idempotent no-op.
    """
    configure_logging()
    pkg_logger = logging.getLogger("sec_generative_search")
    previous_propagate = pkg_logger.propagate
    pkg_logger.propagate = True
    caplog.set_level(logging.WARNING, logger="sec_generative_search.security.audit")
    try:
        yield caplog
    finally:
        pkg_logger.propagate = previous_propagate


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeClock:
    """Manually-advanceable monotonic clock for deterministic TTL tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# InMemorySessionCredentialStore
# ---------------------------------------------------------------------------


class TestInMemorySessionCredentialStore:
    def test_set_and_get_round_trip(self) -> None:
        store = InMemorySessionCredentialStore()
        store.set("sess-1", "openai", "sk-test-ABCDEFGHIJK")
        assert store.get("sess-1", "openai") == "sk-test-ABCDEFGHIJK"

    def test_get_unknown_returns_none(self) -> None:
        store = InMemorySessionCredentialStore()
        assert store.get("sess-1", "openai") is None

    def test_set_overwrites_previous(self) -> None:
        store = InMemorySessionCredentialStore()
        store.set("s", "openai", "sk-old-1234567890")
        store.set("s", "openai", "sk-new-ABCDEFGHIJ")
        assert store.get("s", "openai") == "sk-new-ABCDEFGHIJ"

    def test_per_session_isolation(self) -> None:
        store = InMemorySessionCredentialStore()
        store.set("sess-A", "openai", "sk-aaaaaaaaaaaa")
        store.set("sess-B", "openai", "sk-bbbbbbbbbbbb")
        assert store.get("sess-A", "openai") == "sk-aaaaaaaaaaaa"
        assert store.get("sess-B", "openai") == "sk-bbbbbbbbbbbb"

    def test_per_provider_isolation(self) -> None:
        store = InMemorySessionCredentialStore()
        store.set("s", "openai", "sk-openai-1234567890")
        store.set("s", "gemini", "gem-key-1234567890")
        assert store.list_providers("s") == {"openai", "gemini"}

    def test_delete_returns_true_when_removed(self) -> None:
        store = InMemorySessionCredentialStore()
        store.set("s", "openai", "sk-test-1234567890")
        assert store.delete("s", "openai") is True
        assert store.get("s", "openai") is None

    def test_delete_returns_false_when_absent(self) -> None:
        store = InMemorySessionCredentialStore()
        assert store.delete("s", "openai") is False

    def test_delete_removes_empty_session_entry(self) -> None:
        """The session is dropped once its last credential leaves.

        Keeping an empty entry around would leak memory under load
        (browser tabs come and go); the lazy eviction in ``get`` would
        eventually clean it but the explicit pop on ``delete`` is the
        guaranteed path.
        """
        store = InMemorySessionCredentialStore()
        store.set("s", "openai", "sk-1234567890ABCDEF")
        store.delete("s", "openai")
        assert store.list_providers("s") == set()

    def test_clear_removes_all_credentials(self) -> None:
        store = InMemorySessionCredentialStore()
        store.set("s", "openai", "sk-1111111111")
        store.set("s", "gemini", "gem-2222222")
        assert store.clear("s") == 2
        assert store.list_providers("s") == set()

    def test_clear_unknown_session_returns_zero(self) -> None:
        store = InMemorySessionCredentialStore()
        assert store.clear("sess-nope") == 0

    def test_set_rejects_empty_api_key(self) -> None:
        store = InMemorySessionCredentialStore()
        with pytest.raises(ValueError, match="non-empty"):
            store.set("s", "openai", "")

    def test_constructor_rejects_non_positive_ttl(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds"):
            InMemorySessionCredentialStore(ttl_seconds=0)
        with pytest.raises(ValueError, match="ttl_seconds"):
            InMemorySessionCredentialStore(ttl_seconds=-1)


class TestInMemorySessionCredentialStoreTTL:
    def test_idle_eviction_after_ttl(self) -> None:
        clock = _FakeClock()
        store = InMemorySessionCredentialStore(ttl_seconds=10, clock=clock)
        store.set("s", "openai", "sk-1234567890ABCDEF")
        clock.advance(11.0)
        assert store.get("s", "openai") is None

    def test_read_refreshes_ttl(self) -> None:
        """A successful ``get`` resets last-touched (sliding TTL).

        Matches the standard "user is active so keep them logged in"
        idiom.  The contract is documented in the in-memory store's
        docstring so callers do not assume absolute expiry.
        """
        clock = _FakeClock()
        store = InMemorySessionCredentialStore(ttl_seconds=10, clock=clock)
        store.set("s", "openai", "sk-aaaaaaaaaaaa")
        clock.advance(8.0)
        assert store.get("s", "openai") == "sk-aaaaaaaaaaaa"  # refreshes
        clock.advance(8.0)
        assert store.get("s", "openai") == "sk-aaaaaaaaaaaa"  # still alive

    def test_set_refreshes_ttl(self) -> None:
        clock = _FakeClock()
        store = InMemorySessionCredentialStore(ttl_seconds=10, clock=clock)
        store.set("s", "openai", "sk-aaaaaaaaaaaa")
        clock.advance(8.0)
        store.set("s", "gemini", "gem-bbbbbbbbb")  # refreshes
        clock.advance(8.0)
        assert store.list_providers("s") == {"openai", "gemini"}


class TestInMemorySessionCredentialStoreGlobalSweep:
    """Amortised global sweep of *abandoned* sessions.

    Per-key lazy eviction only collects a session that is accessed again,
    so ``store.get(abandoned)`` cannot distinguish "the global sweep
    dropped it" from "the read path lazily evicted it on this very
    access".  Every test here therefore triggers the sweep through a
    *different* key and inspects the internal ``_sessions`` map directly
    to prove the abandoned session was collected **without ever being
    touched again** — the property that bounds cleartext-credential
    residency to the TTL rather than to process lifetime.
    """

    @pytest.mark.security
    def test_abandoned_session_evicted_without_being_accessed(self) -> None:
        clock = _FakeClock()
        store = InMemorySessionCredentialStore(ttl_seconds=10, clock=clock)
        store.set("abandoned", "openai", "sk-abandoned-123456")
        clock.advance(11.0)  # past TTL — the session is now stale
        # Touch a *different* session; the abandoned one is never read.
        store.set("active", "openai", "sk-active-98765432")
        assert "abandoned" not in store._sessions  # collected by the sweep
        assert store.list_providers("active") == {"openai"}

    @pytest.mark.security
    def test_sweep_clears_credential_strings_before_delete(self) -> None:
        """Swept sessions have their key strings wiped, not just unlinked."""
        clock = _FakeClock()
        store = InMemorySessionCredentialStore(ttl_seconds=10, clock=clock)
        store.set("abandoned", "openai", "sk-abandoned-123456")
        entry = store._sessions["abandoned"]  # capture the entry pre-sweep
        clock.advance(11.0)
        store.get("someone-else", "openai")  # triggers the sweep
        assert "abandoned" not in store._sessions
        assert entry.credentials == {}  # strings dropped for prompt GC

    @pytest.mark.security
    def test_sweep_bounds_abandoned_session_count(self) -> None:
        """A pile of abandoned sessions collapses to zero on one later op."""
        clock = _FakeClock()
        store = InMemorySessionCredentialStore(ttl_seconds=10, clock=clock)
        for i in range(50):
            store.set(f"abandoned-{i}", "openai", f"sk-{i:016d}")
        assert len(store._sessions) == 50
        clock.advance(11.0)
        store.get("nobody", "openai")  # single unrelated op fires the sweep
        assert store._sessions == {}

    def test_sweep_spares_live_sessions(self) -> None:
        """A session touched inside the window survives the sweep that fires
        when the clock crosses the interval; an older one is collected."""
        clock = _FakeClock()
        store = InMemorySessionCredentialStore(ttl_seconds=10, clock=clock)
        store.set("stale", "openai", "sk-stale-123456789")
        clock.advance(6.0)  # still inside the TTL window
        store.set("live", "openai", "sk-live-1234567890")  # touched more recently
        clock.advance(5.0)  # now t=11: "stale" is 11s old (expired), "live" 5s (fresh)
        store.get("live", "openai")  # fires the sweep (11s since last sweep)
        assert "stale" not in store._sessions  # expired → collected
        assert store.get("live", "openai") == "sk-live-1234567890"  # spared

    @pytest.mark.security
    def test_sweep_is_caller_driven_not_timed(self) -> None:
        """No background timer: an expired entry is collected only when a
        caller next touches the store, never spontaneously.  Deterministic
        (fake clock) — a timer thread would have evicted it already."""
        clock = _FakeClock()
        store = InMemorySessionCredentialStore(ttl_seconds=10, clock=clock)
        store.set("abandoned", "openai", "sk-abandoned-123456")
        clock.advance(11.0)  # past TTL, but no store method is called
        assert "abandoned" in store._sessions  # nothing ran on its own
        store.get("trigger", "openai")  # the sweep fires only on this call
        assert "abandoned" not in store._sessions


class TestSessionUserBinding:
    """The user-tier ``session_id → user_id`` binding rides the session entry (F18).

    Before F18 it lived in a process-lifetime dict on ``app.state`` that no
    sweep reached — an abandoned login stayed bound forever.
    """

    def test_bind_and_resolve(self) -> None:
        store = InMemorySessionCredentialStore()
        assert store.user_for("s") is None
        store.bind_user("s", 7)
        assert store.user_for("s") == 7
        # A binding is not a credential.
        assert store.list_providers("s") == set()
        assert store.get("s", "openai") is None

    def test_rebind_replaces(self) -> None:
        store = InMemorySessionCredentialStore()
        store.bind_user("s", 7)
        store.bind_user("s", 8)
        assert store.user_for("s") == 8

    @pytest.mark.security
    def test_clear_drops_the_binding_with_the_credentials(self) -> None:
        """Every rotation / logout seam calls ``clear`` — so it unbinds too."""
        store = InMemorySessionCredentialStore()
        store.bind_user("s", 7)
        store.set("s", "openai", "sk-aaaaaaaaaaaa")
        assert store.clear("s") == 1  # counts credentials, not the binding
        assert store.user_for("s") is None
        assert "s" not in store._sessions

    def test_clear_of_a_binding_only_session_reports_zero(self) -> None:
        store = InMemorySessionCredentialStore()
        store.bind_user("s", 7)
        assert store.clear("s") == 0
        assert store.user_for("s") is None

    def test_deleting_the_last_credential_keeps_the_binding(self) -> None:
        store = InMemorySessionCredentialStore()
        store.bind_user("s", 7)
        store.set("s", "openai", "sk-aaaaaaaaaaaa")
        assert store.delete("s", "openai") is True
        assert store.user_for("s") == 7

    def test_unbind_keeps_credentials_and_reports(self) -> None:
        store = InMemorySessionCredentialStore()
        store.bind_user("s", 7)
        store.set("s", "openai", "sk-aaaaaaaaaaaa")
        assert store.unbind_user("s") is True
        assert store.unbind_user("s") is False
        assert store.user_for("s") is None
        assert store.get("s", "openai") == "sk-aaaaaaaaaaaa"

    def test_unbind_of_a_binding_only_session_drops_the_entry(self) -> None:
        store = InMemorySessionCredentialStore()
        store.bind_user("s", 7)
        assert store.unbind_user("s") is True
        assert "s" not in store._sessions

    @pytest.mark.security
    def test_binding_expires_at_ttl(self) -> None:
        clock = _FakeClock()
        store = InMemorySessionCredentialStore(ttl_seconds=10, clock=clock)
        store.bind_user("s", 7)
        clock.advance(11.0)
        assert store.user_for("s") is None

    @pytest.mark.security
    def test_expired_binding_is_refused_between_sweeps(self) -> None:
        """Per-key eviction, not only the amortised sweep, retires a binding."""
        clock = _FakeClock()
        store = InMemorySessionCredentialStore(ttl_seconds=10, clock=clock)
        clock.advance(5.0)
        store.bind_user("s", 7)
        clock.advance(5.0)
        store.get("other", "openai")  # sweep runs now; "s" is 5 s old and kept
        clock.advance(6.0)  # "s" is 11 s old; the next sweep is 4 s away
        assert store.user_for("s") is None

    def test_resolving_the_binding_slides_the_ttl(self) -> None:
        clock = _FakeClock()
        store = InMemorySessionCredentialStore(ttl_seconds=10, clock=clock)
        store.bind_user("s", 7)
        clock.advance(8.0)
        assert store.user_for("s") == 7  # refreshes
        clock.advance(8.0)
        assert store.user_for("s") == 7

    @pytest.mark.security
    def test_abandoned_binding_is_swept_without_being_accessed(self) -> None:
        """Residency is bounded (at most 2 * TTL) even if the session never returns."""
        clock = _FakeClock()
        store = InMemorySessionCredentialStore(ttl_seconds=10, clock=clock)
        store.bind_user("abandoned", 7)
        clock.advance(11.0)
        store.get("someone-else", "openai")  # an unrelated op fires the sweep
        assert "abandoned" not in store._sessions


# ---------------------------------------------------------------------------
# Resolver chain
# ---------------------------------------------------------------------------


class TestChainResolvers:
    def test_returns_first_hit(self) -> None:
        chain = chain_resolvers(
            lambda p: "from-A" if p == "openai" else None,
            lambda p: "from-B",  # would always answer if reached
        )
        assert chain("openai") == "from-A"  # A wins
        assert chain("gemini") == "from-B"  # falls through to B

    def test_empty_chain_returns_none(self) -> None:
        chain = chain_resolvers()
        assert chain("openai") is None

    def test_all_none_returns_none(self) -> None:
        chain = chain_resolvers(lambda _p: None, lambda _p: None)
        assert chain("openai") is None

    def test_does_not_skip_after_hit(self) -> None:
        """Once a resolver returns non-None the chain stops.  The next
        resolver MUST NOT be queried — otherwise an admin-env fallback
        would still observe a session-scoped lookup it has no business
        seeing."""
        b_called: list[str] = []

        def b(p: str) -> str | None:
            b_called.append(p)
            return None

        chain = chain_resolvers(lambda _p: "from-A", b)
        chain("openai")
        assert b_called == []


class TestSessionResolverAdapter:
    def test_resolves_from_store(self) -> None:
        store = InMemorySessionCredentialStore()
        store.set("sess-1", "openai", "sk-1234567890ABCDEF")
        resolver = session_resolver(store, "sess-1")
        assert resolver("openai") == "sk-1234567890ABCDEF"

    def test_returns_none_for_unknown_provider(self) -> None:
        store = InMemorySessionCredentialStore()
        resolver = session_resolver(store, "sess-1")
        assert resolver("openai") is None

    def test_logs_audit_event_on_hit(
        self,
        audit_caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = InMemorySessionCredentialStore()
        store.set("sess-1", "openai", "sk-test-VWXYZ12345")
        resolver = session_resolver(store, "sess-1")
        audit_caplog.clear()
        resolver("openai")
        messages = [r.getMessage() for r in audit_caplog.records]
        assert any("credential_resolved" in m for m in messages)
        assert any("resolver=session" in m for m in messages)


class TestEncryptedUserResolverAdapter:
    def test_resolves_from_store(self) -> None:
        store = InMemorySessionCredentialStore()  # protocol-compatible
        store.set("user-1", "openai", "sk-1234567890ABCDEF")
        resolver = encrypted_user_resolver(store, "user-1")
        assert resolver("openai") == "sk-1234567890ABCDEF"

    def test_logs_audit_event_on_hit(
        self,
        audit_caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = InMemorySessionCredentialStore()
        store.set("user-1", "openai", "sk-test-VWXYZ12345")
        resolver = encrypted_user_resolver(store, "user-1")
        audit_caplog.clear()
        resolver("openai")
        messages = [r.getMessage() for r in audit_caplog.records]
        assert any("resolver=encrypted_user" in m for m in messages)


# ---------------------------------------------------------------------------
# validate_credential
# ---------------------------------------------------------------------------


class _StubProvider:
    """Captures the key passed to validate_key and returns a configured verdict."""

    def __init__(self, *, raise_exc: Exception | None = None, ok: bool = True) -> None:
        self._exc = raise_exc
        self._ok = ok

    def validate_key(self) -> bool:
        if self._exc is not None:
            raise self._exc
        return self._ok


class TestValidateCredential:
    def test_returns_true_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sec_generative_search.providers.registry import ProviderRegistry

        monkeypatch.setattr(
            ProviderRegistry,
            "validate_key",
            classmethod(lambda cls, name, surface, key, *, model=None: True),
        )
        assert validate_credential("openai", ProviderSurface.LLM, "sk-test-1234567890") is True

    def test_returns_false_on_provider_auth_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # ProviderRegistry.validate_key already collapses ProviderAuthError
        # to False internally, but validate_credential must mirror that
        # behaviour transparently.
        from sec_generative_search.providers.registry import ProviderRegistry

        monkeypatch.setattr(
            ProviderRegistry,
            "validate_key",
            classmethod(lambda cls, name, surface, key, *, model=None: False),
        )
        assert validate_credential("openai", ProviderSurface.LLM, "sk-bad-1234567890") is False

    def test_propagates_non_auth_provider_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A rate-limit error is NOT a verdict on the key.  Reporting it
        as ``False`` would mislead the user into rotating a working key.
        """
        from sec_generative_search.providers.registry import ProviderRegistry

        def _raise(*_args: object, **_kw: object) -> bool:
            raise ProviderRateLimitError(
                "throttled",
                provider="openai",
            )

        monkeypatch.setattr(
            ProviderRegistry,
            "validate_key",
            classmethod(lambda cls, *a, **kw: _raise()),
        )
        with pytest.raises(ProviderRateLimitError):
            validate_credential("openai", ProviderSurface.LLM, "sk-test-1234567890")

    def test_rejects_non_provider_surface(self) -> None:
        with pytest.raises(TypeError, match="ProviderSurface"):
            validate_credential("openai", "llm", "sk-test-1234567890")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestCredentialSecurity:
    """The raw credential MUST NOT leak through any observable surface."""

    def test_audit_log_never_carries_raw_key_on_set(
        self,
        audit_caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = InMemorySessionCredentialStore()
        audit_caplog.clear()
        store.set("sess-1", "openai", "sk-veryunique-NEEDLE-12345")
        joined = "\n".join(r.getMessage() for r in audit_caplog.records)
        assert "sk-veryunique-NEEDLE-12345" not in joined
        # Masked tail is acceptable: last 4 chars only.
        assert "***2345" in joined

    def test_audit_log_never_carries_raw_key_on_resolve(
        self,
        audit_caplog: pytest.LogCaptureFixture,
    ) -> None:
        store = InMemorySessionCredentialStore()
        store.set("sess-1", "openai", "sk-veryunique-NEEDLE-12345")
        audit_caplog.clear()
        resolver = session_resolver(store, "sess-1")
        resolver("openai")
        joined = "\n".join(r.getMessage() for r in audit_caplog.records)
        assert "sk-veryunique-NEEDLE-12345" not in joined
        assert "NEEDLE" not in joined

    def test_audit_log_never_carries_raw_session_id(
        self,
        audit_caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A weak ``session_id`` is still sensitive enough to warrant masking."""
        store = InMemorySessionCredentialStore()
        audit_caplog.clear()
        store.set("very-unique-session-NEEDLE", "openai", "sk-test-1234567890")
        joined = "\n".join(r.getMessage() for r in audit_caplog.records)
        assert "very-unique-session-NEEDLE" not in joined

    def test_validate_credential_audit_does_not_carry_key(
        self,
        audit_caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sec_generative_search.providers.registry import ProviderRegistry

        monkeypatch.setattr(
            ProviderRegistry,
            "validate_key",
            classmethod(lambda cls, *a, **kw: True),
        )
        audit_caplog.clear()
        validate_credential(
            "openai",
            ProviderSurface.LLM,
            "sk-test-NEEDLE-abcdefghij",
        )
        joined = "\n".join(r.getMessage() for r in audit_caplog.records)
        assert "sk-test-NEEDLE-abcdefghij" not in joined
        assert "NEEDLE" not in joined

    def test_validate_credential_error_audit_records_type_only(
        self,
        audit_caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sec_generative_search.providers.registry import ProviderRegistry

        def _raise(*_a: object, **_kw: object) -> bool:
            raise ProviderRateLimitError("throttled", provider="openai")

        monkeypatch.setattr(
            ProviderRegistry,
            "validate_key",
            classmethod(lambda cls, *a, **kw: _raise()),
        )
        audit_caplog.clear()
        with pytest.raises(ProviderRateLimitError):
            validate_credential(
                "openai",
                ProviderSurface.LLM,
                "sk-test-NEEDLE-abcdefghij",
            )
        joined = "\n".join(r.getMessage() for r in audit_caplog.records)
        assert "sk-test-NEEDLE-abcdefghij" not in joined
        assert "credential_validate_error" in joined
        assert "ProviderRateLimitError" in joined

    def test_provider_auth_error_class_does_not_carry_key_in_message(
        self,
    ) -> None:
        """Smoke test: building a ProviderAuthError must never embed the
        key. The provider classes already guarantee this; assert it
        from the credential layer's perspective."""
        err = ProviderAuthError("invalid key", provider="openai")
        assert "sk-" not in str(err).lower()
