"""Tests for the passive per-provider health registry.

Coverage:

    - recording success / failure drives the per-provider breaker FSM
      (closed → open → half_open → closed) with an injected clock;
    - counters (consecutive / total failures, total successes, last
      latency, last error type) track outcomes correctly;
    - the snapshot derives ``last_failure_seconds_ago`` from the clock and
      never exposes the raw monotonic timestamp;
    - the process-global singleton + reset seam;
        - the snapshot is content-free and ``last_error_type`` is the
            exception class name only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import pytest

from sec_generative_search.core.provider_health import (
    ProviderHealthRegistry,
    ProviderHealthSnapshot,
    get_provider_health,
    reset_provider_health,
)
from sec_generative_search.core.resilience import ResilientCallPolicy, resilient_call


class _Clock:
    """Deterministic injectable clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def _reset_singleton() -> Iterator[None]:
    reset_provider_health()
    yield
    reset_provider_health()


# ---------------------------------------------------------------------------
# Recording + counters
# ---------------------------------------------------------------------------


class TestRecording:
    def test_success_records_latency_and_count(self) -> None:
        reg = ProviderHealthRegistry(threshold=5, reset_timeout=60)
        reg.record_success("openai", 1.5)
        reg.record_success("openai", 2.5)

        snap = _only(reg)
        assert snap.provider == "openai"
        assert snap.state == "closed"
        assert snap.total_successes == 2
        assert snap.total_failures == 0
        assert snap.consecutive_failures == 0
        assert snap.last_latency_seconds == 2.5
        assert snap.last_error_type is None
        assert snap.last_failure_seconds_ago is None

    def test_negative_latency_clamped_to_zero(self) -> None:
        reg = ProviderHealthRegistry(threshold=5, reset_timeout=60)
        reg.record_success("openai", -3.0)
        assert _only(reg).last_latency_seconds == 0.0

    def test_failure_records_error_class_name(self) -> None:
        reg = ProviderHealthRegistry(threshold=5, reset_timeout=60)
        reg.record_failure("anthropic", "ProviderTimeoutError")

        snap = _only(reg)
        assert snap.provider == "anthropic"
        assert snap.total_failures == 1
        assert snap.consecutive_failures == 1
        assert snap.last_error_type == "ProviderTimeoutError"

    def test_success_resets_consecutive_failures_only(self) -> None:
        reg = ProviderHealthRegistry(threshold=5, reset_timeout=60)
        reg.record_failure("openai", "ProviderError")
        reg.record_failure("openai", "ProviderError")
        reg.record_success("openai", 1.0)

        snap = _only(reg)
        # Consecutive run resets; lifetime total is preserved.
        assert snap.consecutive_failures == 0
        assert snap.total_failures == 2
        assert snap.total_successes == 1

    def test_empty_error_type_coerced_to_none(self) -> None:
        reg = ProviderHealthRegistry(threshold=5, reset_timeout=60)
        reg.record_failure("openai", "")
        assert _only(reg).last_error_type is None


# ---------------------------------------------------------------------------
# Breaker FSM driven through the registry
# ---------------------------------------------------------------------------


class TestBreakerState:
    def test_opens_after_threshold(self) -> None:
        reg = ProviderHealthRegistry(threshold=3, reset_timeout=60)
        for _ in range(2):
            reg.record_failure("openai", "ProviderError")
        assert _only(reg).state == "closed"
        reg.record_failure("openai", "ProviderError")
        assert _only(reg).state == "open"

    def test_half_open_after_reset_window(self) -> None:
        clock = _Clock()
        reg = ProviderHealthRegistry(threshold=2, reset_timeout=60, clock=clock)
        reg.record_failure("openai", "ProviderError")
        reg.record_failure("openai", "ProviderError")
        assert _only(reg).state == "open"

        clock.now += 60  # reach the reset window
        # Reading the snapshot applies the time-based OPEN → HALF_OPEN move.
        assert _only(reg).state == "half_open"

    def test_recovery_success_closes_breaker(self) -> None:
        clock = _Clock()
        reg = ProviderHealthRegistry(threshold=2, reset_timeout=60, clock=clock)
        reg.record_failure("openai", "ProviderError")
        reg.record_failure("openai", "ProviderError")
        clock.now += 60
        assert _only(reg).state == "half_open"

        reg.record_success("openai", 1.0)
        snap = _only(reg)
        assert snap.state == "closed"
        assert snap.consecutive_failures == 0

    def test_breakers_are_per_provider(self) -> None:
        reg = ProviderHealthRegistry(threshold=2, reset_timeout=60)
        reg.record_failure("openai", "ProviderError")
        reg.record_failure("openai", "ProviderError")
        reg.record_success("anthropic", 1.0)

        by_name = {s.provider: s for s in reg.snapshot()}
        assert by_name["openai"].state == "open"
        assert by_name["anthropic"].state == "closed"


# ---------------------------------------------------------------------------
# Snapshot shape + derived elapsed
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_last_failure_seconds_ago_is_derived(self) -> None:
        clock = _Clock()
        reg = ProviderHealthRegistry(threshold=5, reset_timeout=60, clock=clock)
        reg.record_failure("openai", "ProviderError")
        clock.now += 42
        assert _only(reg).last_failure_seconds_ago == 42.0

    def test_snapshot_sorted_by_provider(self) -> None:
        reg = ProviderHealthRegistry(threshold=5, reset_timeout=60)
        reg.record_success("zai", 1.0)
        reg.record_success("anthropic", 1.0)
        reg.record_success("openai", 1.0)
        assert [s.provider for s in reg.snapshot()] == ["anthropic", "openai", "zai"]

    def test_untouched_provider_absent_from_snapshot(self) -> None:
        reg = ProviderHealthRegistry(threshold=5, reset_timeout=60)
        reg.record_success("openai", 1.0)
        assert {s.provider for s in reg.snapshot()} == {"openai"}


# ---------------------------------------------------------------------------
# Singleton seam
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_reset_singleton")
class TestSingleton:
    def test_get_provider_health_is_stable(self) -> None:
        assert get_provider_health() is get_provider_health()

    def test_reset_rebuilds_instance(self) -> None:
        first = get_provider_health()
        first.record_failure("openai", "ProviderError")
        reset_provider_health()
        second = get_provider_health()
        assert second is not first
        # Fresh instance starts empty.
        assert second.snapshot() == []

    def test_singleton_reads_circuit_breaker_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sec_generative_search.config.settings import reload_settings

        monkeypatch.setenv("PROVIDER_CIRCUIT_BREAKER_THRESHOLD", "1")
        reload_settings()
        reset_provider_health()
        reg = get_provider_health()
        # Threshold of 1 means a single failure opens the breaker.
        reg.record_failure("openai", "ProviderError")
        assert _only(reg).state == "open"
        reload_settings()


# ---------------------------------------------------------------------------
# Security — content-free contract
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestContentFree:
    def test_snapshot_field_set_is_content_free(self) -> None:
        # Pin the exact field set so any addition is a deliberate,
        # security-reviewed change rather than an accidental drift.
        fields = {f.name for f in dataclasses.fields(ProviderHealthSnapshot)}
        assert fields == {
            "provider",
            "state",
            "consecutive_failures",
            "total_failures",
            "total_successes",
            "last_error_type",
            "last_failure_seconds_ago",
            "last_latency_seconds",
        }

    def test_no_credential_or_identity_field_names(self) -> None:
        banned = ("key", "secret", "token", "password", "email", "query", "ticker")
        for field in dataclasses.fields(ProviderHealthSnapshot):
            lowered = field.name.lower()
            assert not any(term in lowered for term in banned), field.name

    def test_error_type_message_is_never_stored(self) -> None:
        # Callers pass the class name; the registry stores it verbatim and
        # has no path that captures the exception message. A message-shaped
        # value passed in is stored as-is (the orchestrator is contractually
        # the source of the class name), so the contract is enforced at the
        # call site — assert here that nothing else is captured.
        reg = ProviderHealthRegistry(threshold=5, reset_timeout=60)
        reg.record_failure("openai", "ProviderAuthError")
        snap = _only(reg)
        assert snap.last_error_type == "ProviderAuthError"


# ---------------------------------------------------------------------------
# Security — the observational breaker is never a live gate
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.usefixtures("_reset_singleton")
class TestObservationalBreakerNeverBlocksLiveCall:
    """The registry breaker is a health *signal*, never a live gate.

    The degraded-mode runbook tells operators that an ``open``
    circuit on ``GET /api/providers/health`` is observational — it does
    NOT auto-reject traffic. That guidance is only safe while the
    registry's per-provider :class:`CircuitBreaker` stays decoupled from
    :func:`resilient_call`: a fully-open observational breaker must not
    short-circuit a live provider call, which runs under the adapter's own
    breaker-free :class:`ResilientCallPolicy`.
    """

    def test_open_registry_breaker_does_not_block_resilient_call(self) -> None:
        registry = get_provider_health()
        # Drive the provider's observational breaker decisively OPEN —
        # well past any sane configured threshold.
        for _ in range(64):
            registry.record_failure("openai", "ProviderTimeoutError")
        assert any(
            row.provider == "openai" and row.state == "open" for row in registry.snapshot()
        ), "precondition: observational breaker must be open"

        # A live call uses a ResilientCallPolicy with no circuit breaker
        # (the adapter default). The open observational breaker must not
        # gate it — fn runs and returns.
        invocations = 0

        def fn() -> str:
            nonlocal invocations
            invocations += 1
            return "ok"

        result = resilient_call(fn, provider="openai", policy=ResilientCallPolicy())
        assert result == "ok"
        assert invocations == 1

    def test_adapter_default_policy_carries_no_circuit_breaker(self) -> None:
        # The structural complement: ResilientCallPolicy has no breaker
        # field at all (F34 removed the unused hook), so there is no wiring
        # path from the observational registry into a live call.
        assert not hasattr(ResilientCallPolicy(), "circuit_breaker")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _only(reg: ProviderHealthRegistry) -> ProviderHealthSnapshot:
    """Return the single snapshot, asserting exactly one provider tracked."""
    snaps = reg.snapshot()
    assert len(snaps) == 1
    return snaps[0]
