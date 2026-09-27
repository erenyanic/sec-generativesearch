"""Tests for :mod:`sec_generative_search.core.resilience`.

Covers:

- :class:`RetryPolicy` validation and delay calculation.
- :class:`CircuitBreaker` state transitions: CLOSED → OPEN → HALF_OPEN
  → CLOSED (recovery) and HALF_OPEN → OPEN (failed probe).
- :class:`ExceptionMapping` / :func:`normalise_exception` mapping paths.
- :func:`with_timeout` success, timeout, and disabled-guard paths.
- :func:`resilient_call` composition: retry on transient errors, no
  retry on terminal errors, circuit-breaker integration, retry
  exhaustion.
- F16: full-jitter backoff, the untrusted ``Retry-After`` parse (floor on
  the backoff, never waited past ``max_delay``), the total retry deadline,
  and the interactive retry budget.
"""

from __future__ import annotations

import email.utils
import time
from collections.abc import Callable
from types import SimpleNamespace

import httpx
import pytest

from sec_generative_search.core.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.resilience import (
    INTERACTIVE_RETRY_POLICY,
    CircuitBreaker,
    CircuitState,
    ExceptionMapping,
    ResilientCallPolicy,
    RetryPolicy,
    normalise_exception,
    resilient_call,
    retry_after_from_exception,
    with_timeout,
)

# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_defaults(self) -> None:
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.backoff_base == 2.0
        assert policy.initial_delay == 1.0
        assert policy.max_delay == 30.0
        assert policy.jitter is True
        assert policy.deadline_seconds == 90.0

    def test_delay_for_attempt_exponential(self) -> None:
        policy = RetryPolicy(
            max_retries=5,
            backoff_base=2.0,
            initial_delay=1.0,
            max_delay=100.0,
            jitter=False,
        )
        assert policy.delay_for_attempt(1) == pytest.approx(1.0)
        assert policy.delay_for_attempt(2) == pytest.approx(2.0)
        assert policy.delay_for_attempt(3) == pytest.approx(4.0)
        assert policy.delay_for_attempt(4) == pytest.approx(8.0)

    def test_delay_honours_max(self) -> None:
        policy = RetryPolicy(
            max_retries=10,
            backoff_base=2.0,
            initial_delay=1.0,
            max_delay=5.0,
            jitter=False,
        )
        # 2^4 = 16 would exceed the 5s cap.
        assert policy.delay_for_attempt(5) == pytest.approx(5.0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_retries", -1),
            ("backoff_base", 0.5),
            ("initial_delay", -0.1),
            ("deadline_seconds", -1.0),
            ("deadline_seconds", float("nan")),
            ("deadline_seconds", float("inf")),
        ],
    )
    def test_invalid_constructor_args_rejected(self, field: str, value: float) -> None:
        kwargs: dict[str, float] = {field: value}
        with pytest.raises(ValueError):
            RetryPolicy(**kwargs)  # type: ignore[arg-type]

    def test_max_delay_below_initial_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            RetryPolicy(initial_delay=10.0, max_delay=1.0)

    def test_delay_for_attempt_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            RetryPolicy().delay_for_attempt(0)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class _FakeClock:
    """Monotonic-like clock whose advance is explicitly driven in tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestCircuitBreaker:
    def test_starts_closed(self) -> None:
        breaker = CircuitBreaker(threshold=3, reset_timeout=10)
        assert breaker.state is CircuitState.CLOSED

    def test_opens_after_threshold_failures(self) -> None:
        breaker = CircuitBreaker(threshold=3, reset_timeout=10)
        for _ in range(3):
            breaker.on_failure()
        assert breaker.state is CircuitState.OPEN

    def test_before_call_raises_when_open(self) -> None:
        breaker = CircuitBreaker(threshold=1, reset_timeout=10)
        breaker.on_failure()
        with pytest.raises(ProviderError):
            breaker.before_call()

    def test_transitions_to_half_open_after_timeout(self) -> None:
        clock = _FakeClock()
        breaker = CircuitBreaker(threshold=1, reset_timeout=5, clock=clock)
        breaker.on_failure()
        assert breaker.state is CircuitState.OPEN
        clock.advance(6)
        # Querying state triggers the time-based transition.
        assert breaker.state is CircuitState.HALF_OPEN

    def test_half_open_closes_on_success(self) -> None:
        clock = _FakeClock()
        breaker = CircuitBreaker(threshold=1, reset_timeout=5, clock=clock)
        breaker.on_failure()
        clock.advance(6)
        breaker.before_call()  # no raise — probe is allowed
        breaker.on_success()
        assert breaker.state is CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self) -> None:
        clock = _FakeClock()
        breaker = CircuitBreaker(threshold=1, reset_timeout=5, clock=clock)
        breaker.on_failure()
        clock.advance(6)
        breaker.before_call()  # moves to half-open
        breaker.on_failure()
        assert breaker.state is CircuitState.OPEN

    def test_success_resets_failure_count(self) -> None:
        breaker = CircuitBreaker(threshold=3, reset_timeout=10)
        breaker.on_failure()
        breaker.on_failure()
        breaker.on_success()
        # Next two failures should NOT open — count was reset.
        breaker.on_failure()
        breaker.on_failure()
        assert breaker.state is CircuitState.CLOSED

    @pytest.mark.parametrize(
        ("threshold", "reset_timeout"),
        [(0, 10), (-1, 10), (1, -0.1)],
    )
    def test_invalid_constructor_args_rejected(
        self,
        threshold: int,
        reset_timeout: float,
    ) -> None:
        with pytest.raises(ValueError):
            CircuitBreaker(threshold=threshold, reset_timeout=reset_timeout)


# ---------------------------------------------------------------------------
# Exception normalisation
# ---------------------------------------------------------------------------


class _FakeAuthError(Exception):
    """Stand-in for an SDK-specific authentication exception."""


class _FakeRateLimitError(Exception):
    """Stand-in for an SDK-specific rate-limit exception."""


class _FakeContentFilterError(Exception):
    """Stand-in for an SDK-specific safety-filter exception."""


class _FakeConnectionError(Exception):
    """Stand-in for an SDK-specific endpoint-unreachable exception."""


class _FakeTimeoutSubclassingConnectionError(_FakeConnectionError):
    """A timeout type that *subclasses* the connection type.

    Mirrors the real SDKs (``openai.APITimeoutError`` derives from
    ``openai.APIConnectionError``), so it pins that ``normalise_exception``
    consults ``timeout`` before ``connection``.
    """


_MAPPING = ExceptionMapping(
    auth=(_FakeAuthError,),
    rate_limit=(_FakeRateLimitError,),
    timeout=(_FakeTimeoutSubclassingConnectionError,),
    connection=(_FakeConnectionError,),
    content_filter=(_FakeContentFilterError,),
)


class TestNormaliseException:
    def test_passes_provider_error_through(self) -> None:
        original = ProviderRateLimitError("already normalised", provider="x")
        assert normalise_exception(original, provider="x", mapping=_MAPPING) is original

    def test_auth_mapping(self) -> None:
        normalised = normalise_exception(_FakeAuthError("401"), provider="openai", mapping=_MAPPING)
        assert isinstance(normalised, ProviderAuthError)
        assert normalised.provider == "openai"
        assert normalised.hint is not None

    def test_rate_limit_mapping(self) -> None:
        normalised = normalise_exception(
            _FakeRateLimitError("429"), provider="openai", mapping=_MAPPING
        )
        assert isinstance(normalised, ProviderRateLimitError)

    def test_timeout_mapping_uses_builtin_default(self) -> None:
        # The default ExceptionMapping already recognises TimeoutError.
        default_mapping = ExceptionMapping()
        normalised = normalise_exception(
            TimeoutError("slow"), provider="openai", mapping=default_mapping
        )
        assert isinstance(normalised, ProviderTimeoutError)

    def test_content_filter_mapping(self) -> None:
        normalised = normalise_exception(
            _FakeContentFilterError("blocked"), provider="openai", mapping=_MAPPING
        )
        assert isinstance(normalised, ProviderContentFilterError)

    def test_connection_mapping(self) -> None:
        # An endpoint-unreachable error normalises to ProviderConnectionError
        # with an actionable, key-preserving hint.
        normalised = normalise_exception(
            _FakeConnectionError("connection refused"),
            provider="local_llm",
            mapping=_MAPPING,
        )
        assert isinstance(normalised, ProviderConnectionError)
        assert normalised.provider == "local_llm"
        assert normalised.hint is not None
        # The hint must steer away from rotating a (possibly fine) key.
        assert "do not rotate the key" in normalised.hint.lower()

    def test_timeout_subclass_of_connection_resolves_to_timeout(self) -> None:
        # Ordering guard: a timeout type that *subclasses* the connection
        # type must resolve to ProviderTimeoutError, never
        # ProviderConnectionError — a slow response is not an unreachable
        # endpoint.  This mirrors the real openai/anthropic SDK hierarchy.
        normalised = normalise_exception(
            _FakeTimeoutSubclassingConnectionError("slow"),
            provider="local_llm",
            mapping=_MAPPING,
        )
        assert isinstance(normalised, ProviderTimeoutError)
        assert not isinstance(normalised, ProviderConnectionError)

    def test_unknown_exception_falls_back_to_provider_error(self) -> None:
        normalised = normalise_exception(RuntimeError("boom"), provider="openai", mapping=_MAPPING)
        assert isinstance(normalised, ProviderError)
        # Not a known subclass.
        assert not isinstance(
            normalised,
            (
                ProviderAuthError,
                ProviderRateLimitError,
                ProviderTimeoutError,
                ProviderConnectionError,
                ProviderContentFilterError,
            ),
        )

    def test_details_are_exception_string(self) -> None:
        normalised = normalise_exception(
            _FakeAuthError("bad token"), provider="openai", mapping=_MAPPING
        )
        assert normalised.details == "bad token"


# ---------------------------------------------------------------------------
# with_timeout
# ---------------------------------------------------------------------------


class TestWithTimeout:
    def test_runs_in_calling_thread_when_disabled(self) -> None:
        assert with_timeout(lambda: 42, seconds=0) == 42

    def test_returns_value_when_fast_enough(self) -> None:
        assert with_timeout(lambda: "ok", seconds=5) == "ok"

    def test_raises_timeout_error_when_slow(self) -> None:
        def slow() -> None:
            time.sleep(0.5)

        with pytest.raises(TimeoutError):
            with_timeout(slow, seconds=0.05)


# ---------------------------------------------------------------------------
# resilient_call
# ---------------------------------------------------------------------------


def _zero_sleep(_: float) -> None:
    """Sleep stub — lets tests exhaust retries without waiting."""


def _make_policy(
    *,
    max_retries: int = 2,
    mapping: ExceptionMapping | None = None,
    breaker: CircuitBreaker | None = None,
    timeout: float = 0.0,
    jitter: bool = False,
) -> ResilientCallPolicy:
    return ResilientCallPolicy(
        retry_policy=RetryPolicy(
            max_retries=max_retries,
            backoff_base=2.0,
            initial_delay=0.01,
            max_delay=0.1,
            jitter=jitter,
        ),
        exception_mapping=mapping or _MAPPING,
        circuit_breaker=breaker,
        timeout=timeout,
    )


class TestResilientCall:
    def test_returns_on_first_success(self) -> None:
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        result = resilient_call(fn, provider="test", policy=_make_policy(), sleep=_zero_sleep)
        assert result == "ok"
        assert calls == 1

    def test_retries_on_rate_limit(self) -> None:
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise _FakeRateLimitError("429")
            return "ok"

        result = resilient_call(
            fn, provider="test", policy=_make_policy(max_retries=3), sleep=_zero_sleep
        )
        assert result == "ok"
        assert calls == 3

    def test_auth_error_is_not_retried(self) -> None:
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            raise _FakeAuthError("401")

        with pytest.raises(ProviderAuthError):
            resilient_call(
                fn,
                provider="test",
                policy=_make_policy(max_retries=5),
                sleep=_zero_sleep,
            )
        assert calls == 1

    def test_content_filter_error_is_not_retried(self) -> None:
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            raise _FakeContentFilterError("blocked")

        with pytest.raises(ProviderContentFilterError):
            resilient_call(
                fn,
                provider="test",
                policy=_make_policy(max_retries=5),
                sleep=_zero_sleep,
            )
        assert calls == 1

    def test_connection_error_is_retried_then_raised(self) -> None:
        # An endpoint-unreachable error is transient, not terminal — it is
        # retried within the budget and only surfaces after exhaustion.
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            raise _FakeConnectionError("connection refused")

        with pytest.raises(ProviderConnectionError):
            resilient_call(
                fn,
                provider="local_llm",
                policy=_make_policy(max_retries=2),
                sleep=_zero_sleep,
            )
        # Initial + 2 retries = 3 total attempts (not terminal like auth).
        assert calls == 3

    def test_connection_error_recovers_on_retry(self) -> None:
        # A transient blip that clears on the next attempt returns normally.
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            if calls < 2:
                raise _FakeConnectionError("connection refused")
            return "ok"

        result = resilient_call(
            fn, provider="local_llm", policy=_make_policy(max_retries=3), sleep=_zero_sleep
        )
        assert result == "ok"
        assert calls == 2

    def test_raises_after_exhausting_retries(self) -> None:
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            raise _FakeRateLimitError("429")

        with pytest.raises(ProviderRateLimitError):
            resilient_call(
                fn,
                provider="test",
                policy=_make_policy(max_retries=2),
                sleep=_zero_sleep,
            )
        # Initial + 2 retries = 3 total attempts.
        assert calls == 3

    def test_circuit_breaker_opens_and_blocks_calls(self) -> None:
        breaker = CircuitBreaker(threshold=2, reset_timeout=60)

        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            raise _FakeRateLimitError("429")

        # First invocation: retry policy will keep hitting the breaker
        # until threshold is reached, at which point the breaker opens
        # and raises a ProviderError (not further retried).
        with pytest.raises(ProviderError):
            resilient_call(
                fn,
                provider="test",
                policy=_make_policy(max_retries=5, breaker=breaker),
                sleep=_zero_sleep,
            )
        # Breaker is open now.
        assert breaker.state is CircuitState.OPEN

        # A subsequent call must fail immediately without invoking fn.
        calls_at_open = calls

        def fn2() -> str:
            nonlocal calls
            calls += 1
            return "won't run"

        with pytest.raises(ProviderError):
            resilient_call(
                fn2,
                provider="test",
                policy=_make_policy(max_retries=2, breaker=breaker),
                sleep=_zero_sleep,
            )
        assert calls == calls_at_open  # fn2 was never executed

    def test_timeout_is_retried(self) -> None:
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            if calls < 2:
                raise TimeoutError("slow")
            return "ok"

        result = resilient_call(
            fn, provider="test", policy=_make_policy(max_retries=3), sleep=_zero_sleep
        )
        assert result == "ok"
        assert calls == 2

    def test_sleep_is_invoked_between_retries(self) -> None:
        sleeps: list[float] = []

        def recording_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        attempts = 0

        def fn() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise _FakeRateLimitError("429")
            return "ok"

        resilient_call(
            fn,
            provider="test",
            policy=_make_policy(max_retries=3),
            sleep=recording_sleep,
        )
        # Two retries → two sleeps; with jitter off, delays follow the
        # exponential schedule.
        assert len(sleeps) == 2
        assert sleeps[0] < sleeps[1]

    def test_provider_error_subclass_raised_directly_is_respected(self) -> None:
        """A concrete provider may raise a normalised error to bypass the
        mapping — the wrapper must not double-normalise it."""

        def fn() -> str:
            raise ProviderAuthError("bad key", provider="test")

        with pytest.raises(ProviderAuthError):
            resilient_call(
                fn,
                provider="test",
                policy=_make_policy(max_retries=5),
                sleep=_zero_sleep,
            )


# ---------------------------------------------------------------------------
# F16 — jitter, Retry-After, deadline, interactive budget
# ---------------------------------------------------------------------------


class _RateLimitWithHeadersError(_FakeRateLimitError):
    """A mapped rate-limit error carrying SDK-shaped ``response.headers``."""

    def __init__(self, headers: dict[str, str]) -> None:
        super().__init__("429")
        self.response = SimpleNamespace(headers=httpx.Headers(headers))


class _FakeMonotonic:
    """Manually advanced monotonic clock for deadline tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _deadline_policy(
    *,
    max_retries: int = 3,
    deadline_seconds: float = 90.0,
    max_delay: float = 30.0,
) -> ResilientCallPolicy:
    return ResilientCallPolicy(
        retry_policy=RetryPolicy(
            max_retries=max_retries,
            backoff_base=2.0,
            initial_delay=1.0,
            max_delay=max_delay,
            jitter=False,
            deadline_seconds=deadline_seconds,
        ),
        exception_mapping=_MAPPING,
    )


class TestRetryJitter:
    def test_full_jitter_scales_the_capped_backoff(self) -> None:
        policy = RetryPolicy(initial_delay=1.0, backoff_base=2.0, max_delay=30.0)
        assert policy.delay_for_attempt(3, rng=lambda: 0.25) == pytest.approx(1.0)
        assert policy.delay_for_attempt(3, rng=lambda: 0.0) == 0.0
        # The cap applies before the draw: ceiling 30, not 2**9.
        assert policy.delay_for_attempt(10, rng=lambda: 0.5) == pytest.approx(15.0)

    def test_default_draw_stays_within_the_ceiling(self) -> None:
        policy = RetryPolicy(initial_delay=1.0, backoff_base=2.0, max_delay=30.0)
        draws = [policy.delay_for_attempt(3) for _ in range(200)]
        assert all(0.0 <= d <= 4.0 for d in draws)

    def test_callers_failing_together_do_not_retry_together(self) -> None:
        """§5 plan: 100 callers hitting one 429 storm get spread delays."""
        sleeps: list[float] = []

        def fn() -> str:
            raise _FakeRateLimitError("429")

        for _ in range(100):
            with pytest.raises(ProviderRateLimitError):
                resilient_call(
                    fn,
                    provider="test",
                    policy=_make_policy(max_retries=1, jitter=True),
                    sleep=sleeps.append,
                )
        assert len(sleeps) == 100
        # Synchronised schedules would give one distinct value.
        assert len(set(sleeps)) > 90

    def test_retry_after_is_a_floor_under_jitter(self) -> None:
        policy = RetryPolicy(initial_delay=1.0, max_delay=30.0)
        assert policy.delay_for_attempt(1, retry_after=5.0, rng=lambda: 0.1) == 5.0
        # A jittered backoff above the floor is kept.
        assert policy.delay_for_attempt(5, retry_after=1.0, rng=lambda: 1.0) == 16.0


@pytest.mark.security
class TestRetryAfterParsing:
    """``Retry-After`` is untrusted upstream input (F16).

    A value drives how long a worker thread sleeps, so only a finite,
    non-negative number of seconds is ever honoured.
    """

    def test_milliseconds_header(self) -> None:
        exc = _RateLimitWithHeadersError({"retry-after-ms": "1500"})
        assert retry_after_from_exception(exc) == pytest.approx(1.5)

    def test_milliseconds_win_over_seconds(self) -> None:
        exc = _RateLimitWithHeadersError({"retry-after-ms": "250", "retry-after": "9"})
        assert retry_after_from_exception(exc) == pytest.approx(0.25)

    def test_invalid_milliseconds_fall_back_to_seconds(self) -> None:
        exc = _RateLimitWithHeadersError({"retry-after-ms": "soon", "retry-after": "3"})
        assert retry_after_from_exception(exc) == 3.0

    @pytest.mark.parametrize(("raw", "expected"), [("7", 7.0), ("2.5", 2.5), ("0", 0.0)])
    def test_delta_seconds(self, raw: str, expected: float) -> None:
        exc = _RateLimitWithHeadersError({"Retry-After": raw})
        assert retry_after_from_exception(exc) == expected

    def test_future_http_date(self) -> None:
        now = 1_800_000_000.0
        exc = _RateLimitWithHeadersError(
            {"retry-after": email.utils.formatdate(now + 20, usegmt=True)}
        )
        assert retry_after_from_exception(exc, now=lambda: now) == pytest.approx(20.0)

    def test_naive_http_date_is_read_as_gmt(self) -> None:
        now = 1_800_000_000.0
        raw = email.utils.formatdate(now + 12, usegmt=True).replace("GMT", "-0000")
        exc = _RateLimitWithHeadersError({"retry-after": raw})
        assert retry_after_from_exception(exc, now=lambda: now) == pytest.approx(12.0)

    def test_past_http_date_is_zero(self) -> None:
        now = 1_800_000_000.0
        exc = _RateLimitWithHeadersError(
            {"retry-after": email.utils.formatdate(now - 600, usegmt=True)}
        )
        assert retry_after_from_exception(exc, now=lambda: now) == 0.0

    @pytest.mark.parametrize(
        "raw",
        ["nan", "inf", "-inf", "1e309", "-5", "", "soon", "9" * 65, "Fri, 99 Foo 2026"],
    )
    def test_hostile_or_malformed_values_are_ignored(self, raw: str) -> None:
        exc = _RateLimitWithHeadersError({"retry-after": raw, "retry-after-ms": raw})
        assert retry_after_from_exception(exc) is None

    def test_missing_response_or_headers(self) -> None:
        assert retry_after_from_exception(_FakeRateLimitError("429")) is None
        exc = _FakeRateLimitError("429")
        exc.response = SimpleNamespace()  # type: ignore[attr-defined]
        assert retry_after_from_exception(exc) is None

    def test_headers_that_raise_are_ignored(self) -> None:
        class _Exploding:
            def get(self, _key: str) -> str:
                raise RuntimeError("boom")

        exc = _FakeRateLimitError("429")
        exc.response = SimpleNamespace(headers=_Exploding())  # type: ignore[attr-defined]
        assert retry_after_from_exception(exc) is None

    def test_normalised_rate_limit_carries_it(self) -> None:
        exc = _RateLimitWithHeadersError({"retry-after": "7"})
        normalised = normalise_exception(exc, provider="x", mapping=_MAPPING)
        assert isinstance(normalised, ProviderRateLimitError)
        assert normalised.retry_after == 7.0

    def test_rate_limit_error_defaults_to_none(self) -> None:
        assert ProviderRateLimitError("rl", provider="x").retry_after is None
        normalised = normalise_exception(_FakeRateLimitError("429"), provider="x", mapping=_MAPPING)
        assert normalised.retry_after is None  # type: ignore[attr-defined]


@pytest.mark.security
class TestResilientCallRetryAfter:
    def test_retry_after_is_honoured_as_the_floor(self) -> None:
        sleeps: list[float] = []
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _RateLimitWithHeadersError({"retry-after": "2"})
            return "ok"

        result = resilient_call(fn, provider="test", policy=_deadline_policy(), sleep=sleeps.append)
        assert result == "ok"
        assert sleeps == [2.0]

    def test_retry_after_beyond_max_delay_ends_the_loop(self) -> None:
        sleeps: list[float] = []
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            raise _RateLimitWithHeadersError({"retry-after": "120"})

        with pytest.raises(ProviderRateLimitError) as info:
            resilient_call(fn, provider="test", policy=_deadline_policy(), sleep=sleeps.append)
        # Never retried early against the server's own deadline, never slept on it.
        assert calls == 1
        assert sleeps == []
        assert info.value.retry_after == 120.0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -3.0, True])
    def test_unusable_adapter_supplied_value_is_ignored(self, bad: float) -> None:
        """An adapter may raise the error itself — the value is re-checked."""
        sleeps: list[float] = []
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ProviderRateLimitError("rl", provider="test", retry_after=bad)
            return "ok"

        resilient_call(fn, provider="test", policy=_deadline_policy(), sleep=sleeps.append)
        # Plain backoff (jitter off: 1.0), not NaN / inf / a negative sleep.
        assert sleeps == [1.0]


@pytest.mark.security
class TestResilientCallDeadline:
    """The deadline bounds how long one call can hold a threadpool slot."""

    def test_no_retry_starts_once_the_deadline_would_pass(self) -> None:
        clock = _FakeMonotonic()
        sleeps: list[float] = []
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            clock.now += 60.0  # each attempt burns a full SDK timeout
            raise _FakeRateLimitError("429")

        with pytest.raises(ProviderRateLimitError):
            resilient_call(
                fn,
                provider="test",
                policy=_deadline_policy(max_retries=3),
                sleep=sleeps.append,
                clock=clock,
            )
        # 60 s + 1 s backoff < 90 → one retry; at 121 s nothing more starts.
        assert calls == 2
        assert sleeps == [1.0]
        # Honest bound: the deadline plus at most one in-flight attempt.
        assert clock.now <= 90.0 + 60.0

    def test_a_backoff_ending_past_the_deadline_is_not_slept(self) -> None:
        clock = _FakeMonotonic()
        sleeps: list[float] = []

        def fn() -> str:
            clock.now = 89.5
            raise _FakeRateLimitError("429")

        with pytest.raises(ProviderRateLimitError):
            resilient_call(
                fn,
                provider="test",
                policy=_deadline_policy(),
                sleep=sleeps.append,
                clock=clock,
            )
        assert sleeps == []

    def test_zero_deadline_disables_the_budget(self) -> None:
        clock = _FakeMonotonic()
        calls = 0

        def fn() -> str:
            nonlocal calls
            calls += 1
            clock.now += 60.0
            raise _FakeRateLimitError("429")

        with pytest.raises(ProviderRateLimitError):
            resilient_call(
                fn,
                provider="test",
                policy=_deadline_policy(max_retries=3, deadline_seconds=0.0),
                sleep=_zero_sleep,
                clock=clock,
            )
        assert calls == 4

    def test_interactive_policy_is_one_retry_under_the_deadline(self) -> None:
        assert INTERACTIVE_RETRY_POLICY.max_retries == 1
        assert INTERACTIVE_RETRY_POLICY.jitter is True
        assert INTERACTIVE_RETRY_POLICY.deadline_seconds == 90.0
        assert RetryPolicy().max_retries == 3


# ---------------------------------------------------------------------------
# Composite typing sanity
# ---------------------------------------------------------------------------


def test_sleep_default_is_time_sleep() -> None:
    """Regression: callers must not accidentally pass ``None`` for ``sleep``."""
    sleep_default: Callable[[float], None] = time.sleep  # noqa: F841
