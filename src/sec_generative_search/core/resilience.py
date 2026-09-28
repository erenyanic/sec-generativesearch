"""Resilience primitives for provider calls and future retrieval work.

This module is deliberately dependency-free: it imports only from the
standard library and :mod:`core.exceptions` / :mod:`core.logging`.  The
provider adapters compose these primitives, and the retrieval service
reuses the same building blocks, so no piece of this file may reach
into any SDK.

Contents:

- :class:`RetryPolicy` — exponential-backoff retry configuration with
  full jitter and a total retry deadline.
- :data:`INTERACTIVE_RETRY_POLICY` — the shorter budget for
  request-scoped LLM calls.
- :class:`CircuitBreaker` — thread-safe three-state circuit breaker
  (``CLOSED`` → ``OPEN`` → ``HALF_OPEN`` → ``CLOSED``/``OPEN``), used
  **observationally** by :mod:`~sec_generative_search.core.provider_health`
  only — it records outcomes and reports a state; nothing consults it
  before a call.
- :class:`ExceptionMapping` — declarative mapping from SDK-specific
  exception types to :class:`ProviderError` subclasses.
- :func:`normalise_exception` — consult a mapping and return the
  corresponding :class:`ProviderError` subclass.
- :func:`retry_after_from_exception` — bounded parse of a provider's
  ``Retry-After`` from an SDK exception's response headers.
- :func:`resilient_call` — the top-level composer: retry with jittered
  exponential backoff (never sooner than a provider's ``Retry-After``,
  never past the policy deadline) and exception normalisation, driven by
  the policies above.

Design notes:

- No ``ProviderError`` carries the raw API key or any user-supplied
  prompt text.  The mapping only inspects exception *types* — the
  original exception message is copied into ``details`` after the caller
  has ensured it is safe to log.  Never pass a raw prompt string through
  this layer.
- ``resilient_call`` treats :class:`ProviderAuthError` and
  :class:`ProviderContentFilterError` as terminal: both indicate an
  input that will not change on retry, so retrying wastes quota and
  extends the blast radius of a bad key.  All other
  :class:`ProviderError` subclasses are retried within the policy.
- Retries hold the caller's thread — on the API that is one of the
  worker's ~40 threadpool slots — so the budget is bounded three ways:
  full jitter keeps callers that failed together from retrying together,
  a ``Retry-After`` above ``max_delay`` ends the loop instead of
  retrying early, and ``deadline_seconds`` stops starting new attempts
  once the budget is spent.
- The per-attempt timeout is the SDK's own ``timeout=`` argument
  (``PROVIDER_TIMEOUT`` via ``providers/network_policy.py``).  There is
  no thread-based wall-clock wrapper: the former ``with_timeout`` was
  never enabled by any adapter, and its pool shutdown waited for the
  call anyway, so it could not have returned early (F34).
- No circuit breaker is wired in — a failing provider is reported by the
  passive health registry, never short-circuited (F34 removed the unused
  hook, so there is no path to wire one).
"""

from __future__ import annotations

import email.utils
import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC
from enum import Enum

from sec_generative_search.core.exceptions import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "INTERACTIVE_RETRY_POLICY",
    "CircuitBreaker",
    "CircuitState",
    "ExceptionMapping",
    "RetryPolicy",
    "normalise_exception",
    "resilient_call",
    "retry_after_from_exception",
]

# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


# Jitter source.  ``SystemRandom`` draws from the OS, so it is immune to a
# test's ``random.seed`` and — the property that matters for a retry storm —
# to the PRNG state a forked worker inherits from its parent: identical
# state would re-synchronise the very schedules jitter exists to spread.
_JITTER_RNG = random.SystemRandom()


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential-backoff retry configuration with full jitter and a deadline.

    Construction is cheap; instances are frozen so they can be shared
    across threads without synchronisation.  This module stays
    settings-free: ``providers/network_policy.py`` builds the policy each
    adapter gets from ``PROVIDER_MAX_RETRIES`` /
    ``PROVIDER_RETRY_BACKOFF_BASE`` (the defaults here mirror theirs).

    Attributes:
        max_retries: Number of *retry* attempts after the initial call.
            ``max_retries=3`` means up to four total attempts.
        backoff_base: Exponential base.  The backoff ceiling before
            retry *n* (1-indexed, retries only) is
            ``min(initial_delay * backoff_base ** (n - 1), max_delay)``.
        initial_delay: Backoff ceiling (seconds) before the first retry.
        max_delay: Upper bound on the computed backoff — prevents the
            schedule from drifting into hour-long waits.  Also the longest
            provider ``Retry-After`` :func:`resilient_call` will wait for;
            a longer one ends the retry loop instead of retrying early.
        jitter: Full jitter — each delay is drawn uniformly from
            ``[0, ceiling]`` so callers that failed together (one upstream
            429 storm) do not retry together.  ``False`` restores the
            fixed schedule.
        deadline_seconds: Total retry budget, measured from the start of
            the first attempt.  A retry whose backoff would end at or past
            it is never started; the last error is raised instead.  It
            cannot interrupt an attempt already in flight (the SDK's own
            ``timeout`` bounds that), so the worst case is roughly the
            deadline plus one SDK timeout.  ``0`` disables the budget.
    """

    max_retries: int = 3
    backoff_base: float = 2.0
    initial_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True
    deadline_seconds: float = 90.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.backoff_base < 1.0:
            raise ValueError("backoff_base must be >= 1.0")
        if self.initial_delay < 0:
            raise ValueError("initial_delay must be >= 0")
        if self.max_delay < self.initial_delay:
            raise ValueError("max_delay must be >= initial_delay")
        if not math.isfinite(self.deadline_seconds) or self.deadline_seconds < 0:
            raise ValueError("deadline_seconds must be finite and >= 0")

    def delay_for_attempt(
        self,
        attempt: int,
        *,
        retry_after: float | None = None,
        rng: Callable[[], float] | None = None,
    ) -> float:
        """Return the delay (seconds) before retry *attempt*.

        ``attempt`` is 1-indexed, counting only retries — attempt 1 is
        the delay before the *first* retry (i.e. after the initial call
        failed).

        With ``jitter`` the capped exponential backoff is scaled by a draw
        from *rng* (``[0, 1)``; defaults to an OS-backed source).  A
        provider's *retry_after* is a floor: the delay is never shorter
        than the server asked for.  Whether a *retry_after* above
        ``max_delay`` is worth waiting for is the caller's decision —
        :func:`resilient_call` does not wait for one.
        """
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        delay = min(self.initial_delay * (self.backoff_base ** (attempt - 1)), self.max_delay)
        if self.jitter:
            delay *= (rng or _JITTER_RNG.random)()
        if retry_after is not None:
            delay = max(delay, retry_after)
        return delay


# Retry budget cap for request-scoped (interactive) calls: one retry, not
# three.  The caller holds one of the API worker's threadpool slots while it
# waits, and a third or fourth attempt against an upstream that already
# failed twice rarely lands inside a user's patience while pinning the slot
# for another SDK timeout.  ``providers/network_policy.py`` applies it to
# every LLM build and key-validation probe (``PROVIDER_MAX_RETRIES`` can
# lower it, never raise it); embedders — built once per process and shared
# by ingest and search — take ``PROVIDER_MAX_RETRIES`` as set.
INTERACTIVE_RETRY_POLICY = RetryPolicy(max_retries=1)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class CircuitState(Enum):
    """States of the :class:`CircuitBreaker` finite-state machine."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe three-state circuit breaker.

    The breaker opens after ``threshold`` consecutive failures and reports
    ``OPEN`` for ``reset_timeout`` seconds; after the cool-down, reading
    :attr:`state` moves it to ``HALF_OPEN``, and the next recorded
    outcome closes it (success) or re-opens it (failure).  It is purely
    observational — nothing asks it for permission before a call.

    The clock is injectable so unit tests can drive the FSM
    deterministically without :func:`time.sleep`.
    """

    def __init__(
        self,
        *,
        threshold: int,
        reset_timeout: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        if reset_timeout < 0:
            raise ValueError("reset_timeout must be >= 0")
        self._threshold = threshold
        self._reset_timeout = reset_timeout
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        """Current state, after applying any time-based transitions."""
        with self._lock:
            self._maybe_transition_to_half_open_locked()
            return self._state

    def _maybe_transition_to_half_open_locked(self) -> None:
        """OPEN → HALF_OPEN once ``reset_timeout`` has elapsed.

        Must be called under ``self._lock``.
        """
        if (
            self._state is CircuitState.OPEN
            and self._opened_at is not None
            and self._clock() - self._opened_at >= self._reset_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            logger.info("Circuit breaker entering HALF_OPEN for probe")

    def on_success(self) -> None:
        """Record a successful call — closes the breaker."""
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                logger.info("Circuit breaker CLOSED after successful probe")
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._opened_at = None

    def on_failure(self) -> None:
        """Record a failed call — may open or re-open the breaker."""
        with self._lock:
            if self._state is CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                logger.warning("Circuit breaker re-OPENED after failed probe")
                return
            self._failures += 1
            if self._failures >= self._threshold and self._state is CircuitState.CLOSED:
                self._state = CircuitState.OPEN
                self._opened_at = self._clock()
                logger.warning(
                    "Circuit breaker OPENED after %d consecutive failures",
                    self._failures,
                )


# ---------------------------------------------------------------------------
# Exception normalisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExceptionMapping:
    """Declarative mapping from SDK-specific exception types to :class:`ProviderError`.

    Each concrete provider constructs one at module load
    using the SDK's public exception hierarchy.  The resilience layer
    consults the mapping to translate every failure into the small set
    of types that the RAG orchestrator and UI can reason about.

    ``timeout`` defaults to ``(TimeoutError,)`` so the thread-timeout
    raised by :func:`with_timeout` is always normalised, even when a
    caller forgets to extend the tuple.  Callers may override this
    default to disable or extend the mapping.

    ``connection`` captures endpoint-unreachable failures (refused
    connection, no route to host, DNS failure).  Because SDK timeout
    types are often *subclasses* of the SDK's connection-error type
    (e.g. ``openai.APITimeoutError`` derives from
    ``openai.APIConnectionError``), :func:`normalise_exception` consults
    ``timeout`` **before** ``connection`` so a slow response is never
    mislabelled as an unreachable endpoint.

    Empty tuples are fine — ``isinstance(exc, ())`` returns ``False``
    and the default branch of :func:`normalise_exception` returns a
    generic :class:`ProviderError`.
    """

    auth: tuple[type[BaseException], ...] = ()
    rate_limit: tuple[type[BaseException], ...] = ()
    timeout: tuple[type[BaseException], ...] = (TimeoutError,)
    connection: tuple[type[BaseException], ...] = ()
    content_filter: tuple[type[BaseException], ...] = ()


def normalise_exception(
    exc: BaseException,
    *,
    provider: str,
    mapping: ExceptionMapping,
) -> ProviderError:
    """Translate an arbitrary exception into a :class:`ProviderError` subclass.

    If *exc* is already a :class:`ProviderError`, it is returned
    unchanged — concrete providers may raise these directly to bypass
    the mapping.  ``details`` on the returned error holds ``str(exc)``;
    callers are responsible for ensuring the exception message is safe
    to log (no prompt text, no key material).
    """
    if isinstance(exc, ProviderError):
        return exc
    detail = str(exc) or type(exc).__name__
    if mapping.auth and isinstance(exc, mapping.auth):
        return ProviderAuthError(
            f"Authentication failed against {provider}",
            provider=provider,
            hint="Verify the API key is correct and not expired.",
            details=detail,
        )
    if mapping.rate_limit and isinstance(exc, mapping.rate_limit):
        return ProviderRateLimitError(
            f"Rate limit exceeded for {provider}",
            provider=provider,
            hint="Retry after the Retry-After interval or lower request volume.",
            details=detail,
            retry_after=retry_after_from_exception(exc),
        )
    if mapping.timeout and isinstance(exc, mapping.timeout):
        return ProviderTimeoutError(
            f"{provider} call timed out",
            provider=provider,
            hint="Increase PROVIDER_TIMEOUT or retry with fewer tokens.",
            details=detail,
        )
    # Consulted after ``timeout`` on purpose: SDK timeout types usually
    # subclass the connection-error type, so a slow response must resolve
    # to ProviderTimeoutError, not ProviderConnectionError.
    if mapping.connection and isinstance(exc, mapping.connection):
        return ProviderConnectionError(
            f"Could not reach {provider} endpoint",
            provider=provider,
            hint=(
                "Verify the endpoint is running and reachable "
                "(for local_llm, that the local model server is up); "
                "retry once it recovers. Do not rotate the key."
            ),
            details=detail,
        )
    if mapping.content_filter and isinstance(exc, mapping.content_filter):
        return ProviderContentFilterError(
            f"{provider} safety filter blocked the request",
            provider=provider,
            hint="Reformulate the prompt or route to a different provider.",
            details=detail,
        )
    return ProviderError(
        f"{provider} call failed: {type(exc).__name__}",
        provider=provider,
        details=detail,
    )


# ---------------------------------------------------------------------------
# Retry-After
# ---------------------------------------------------------------------------

# Longest ``Retry-After`` header value we hand to a parser.  Real values
# are a few digits or a 29-character IMF-fixdate; anything longer is
# malformed and ignored.
_MAX_RETRY_AFTER_CHARS = 64


def retry_after_from_exception(
    exc: BaseException,
    *,
    now: Callable[[], float] = time.time,
) -> float | None:
    """Return the retry delay (seconds) a provider asked for, or ``None``.

    Duck-typed over ``exc.response.headers`` — the shape the OpenAI,
    Anthropic and google-genai SDK errors share — so this module stays
    SDK-free.  ``retry-after-ms`` (OpenAI's precise variant) wins, then
    ``retry-after`` as delta-seconds, then as an HTTP-date.

    The header is **untrusted upstream input**: an OpenRouter upstream, a
    non-loopback ``local_llm`` server or an intermediary can send
    anything.  A value is honoured only when it parses to a finite,
    non-negative number of seconds (``float`` happily accepts ``nan``,
    ``inf`` and ``1e309``); a past HTTP-date yields ``0.0``; everything
    else is ``None``.  No upper bound is applied here —
    :func:`resilient_call` refuses to wait past ``RetryPolicy.max_delay``.
    Never raises.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        raw_ms = headers.get("retry-after-ms")
        raw = headers.get("retry-after")
    except Exception:
        return None
    millis = _parse_non_negative_seconds(raw_ms)
    if millis is not None:
        return millis / 1000.0
    seconds = _parse_non_negative_seconds(raw)
    if seconds is not None:
        return seconds
    return _parse_http_date_delay(raw, now=now)


def _parse_non_negative_seconds(raw: object) -> float | None:
    """Parse a delta-seconds header value; ``None`` unless finite and ``>= 0``."""
    if not isinstance(raw, str) or len(raw) > _MAX_RETRY_AFTER_CHARS:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _parse_http_date_delay(raw: object, *, now: Callable[[], float]) -> float | None:
    """Seconds until an HTTP-date ``Retry-After``; ``0.0`` once it has passed."""
    if not isinstance(raw, str) or len(raw) > _MAX_RETRY_AFTER_CHARS:
        return None
    try:
        when = email.utils.parsedate_to_datetime(raw)
        if when.tzinfo is None:
            # An HTTP-date is always GMT; ``-0000`` parses as naive.
            when = when.replace(tzinfo=UTC)
        delay = when.timestamp() - now()
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if not math.isfinite(delay):
        return None
    return max(0.0, delay)


def _usable_retry_after(value: object) -> float | None:
    """Re-check a ``retry_after`` attribute before it drives a sleep.

    Adapters may construct :class:`ProviderRateLimitError` themselves, so
    the value reaching :func:`resilient_call` is not guaranteed to have
    come through :func:`retry_after_from_exception`.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return float(value)


# ---------------------------------------------------------------------------
# Composite wrapper
# ---------------------------------------------------------------------------


# Terminal error types — never retried.  Auth and content-filter failures
# are deterministic given the input; retrying wastes quota and extends
# the blast radius of a bad key or a blocked prompt.
_TERMINAL_PROVIDER_ERRORS: tuple[type[ProviderError], ...] = (
    ProviderAuthError,
    ProviderContentFilterError,
)


@dataclass
class ResilientCallPolicy:
    """Bundle of resilience policies applied by :func:`resilient_call`.

    Pass-through container so that provider subclasses can configure
    the retry budget and exception mapping in one place and forward them
    to :func:`resilient_call` without a long argument list.
    """

    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    exception_mapping: ExceptionMapping = field(default_factory=ExceptionMapping)


def resilient_call[T](
    fn: Callable[[], T],
    *,
    provider: str,
    policy: ResilientCallPolicy,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    rng: Callable[[], float] | None = None,
) -> T:
    """Execute *fn* under the retry and exception-mapping policies.

    Flow per attempt:

    1. Run *fn* (its per-attempt timeout is the SDK's own).
    2. On success: return the result.
    3. On failure: normalise the exception.
       *Terminal* errors (auth, content-filter) raise immediately.
       *Retryable* errors raise when the retry budget is exhausted, when
       the provider's ``Retry-After`` exceeds ``max_delay``, or when the
       backoff would end past ``deadline_seconds``; otherwise they sleep
       (jittered, never shorter than ``Retry-After``) and loop.

    ``sleep``, ``clock`` (monotonic, for the deadline) and ``rng`` (the
    jitter draw) are injectable so tests can drive the loop without real
    time passing.
    """
    retry = policy.retry_policy
    last_exc: ProviderError | None = None
    max_attempts = retry.max_retries + 1
    started = clock()
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as raw:
            normalised = normalise_exception(
                raw,
                provider=provider,
                mapping=policy.exception_mapping,
            )
            if isinstance(normalised, _TERMINAL_PROVIDER_ERRORS):
                raise normalised from raw
            last_exc = normalised
            if attempt >= max_attempts:
                raise last_exc from raw
            delay = _next_retry_delay(
                normalised,
                retry=retry,
                attempt=attempt,
                elapsed=clock() - started,
                rng=rng,
                provider=provider,
            )
            if delay is None:
                raise last_exc from raw
            sleep(delay)

    # The loop always returns on success or raises on exhaustion; this
    # line is unreachable.  Present to reassure type checkers.
    assert last_exc is not None  # pragma: no cover
    raise last_exc  # pragma: no cover


def _next_retry_delay(
    error: ProviderError,
    *,
    retry: RetryPolicy,
    attempt: int,
    elapsed: float,
    rng: Callable[[], float] | None,
    provider: str,
) -> float | None:
    """Backoff before retry *attempt*, or ``None`` when it must not happen.

    Log lines carry the curated provider slug and numbers only.
    """
    retry_after = _usable_retry_after(getattr(error, "retry_after", None))
    if retry_after is not None and retry_after > retry.max_delay:
        # Retrying before the server's own deadline is known to fail and
        # extends the penalty; waiting it out would hold the caller's
        # thread past the backoff cap.  Surface the error now.
        logger.info(
            "Not retrying %s: Retry-After %.1fs exceeds the %.1fs backoff cap",
            provider,
            retry_after,
            retry.max_delay,
        )
        return None
    delay = retry.delay_for_attempt(attempt, retry_after=retry_after, rng=rng)
    if retry.deadline_seconds > 0 and elapsed + delay >= retry.deadline_seconds:
        logger.info(
            "Not retrying %s: the %.0fs retry deadline would pass",
            provider,
            retry.deadline_seconds,
        )
        return None
    return delay
