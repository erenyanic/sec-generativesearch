"""User-supplied provider credential management.

The module sits alongside :mod:`core.security` in the import graph: low
in the layering, dependency-free outside of the standard library plus
``core.security``, ``core.logging``, ``core.exceptions``, and the
provider registry.  No reach into ``database`` or ``providers``
internals other than the registry's class-level look-up.

Three concerns live here:

1. **The ``CredentialStore`` protocol** — the shape every credential
   backend conforms to (in-memory session map, SQLCipher-encrypted
   table, future remote KMS, ...).  Stores are keyed by an opaque
   ``key_id`` (``session_id`` for the in-memory store, ``user_id``
   for the encrypted store) — the protocol is intentionally agnostic
   to which one, because the resolver chain treats them uniformly.

2. **The resolver chain** — three composable callables of the
   :data:`~sec_generative_search.providers.factory.ApiKeyResolver`
   shape that ``providers.factory.build_embedder`` /
   ``build_llm_provider`` already consume.  No new abstraction tax;
   the chain is just function composition over the existing seam.

3. **``validate_credential``** — a thin audit-logged wrapper around
    :meth:`ProviderRegistry.validate_key`.  The HTTP route handler is a
    thin adapter around this seam.

Security contracts the rest of the package relies on:

- The raw credential value MUST NOT appear in any log record, exception
  message, ``__repr__``, or persisted row.  Every audit-log call here
  passes the value through :func:`mask_secret`; the encrypted store
  (see :mod:`database.credentials`) writes ciphertext under SQLCipher
  and never echoes the plaintext back from any read API except the
  explicit ``get`` path.
- ``InMemorySessionCredentialStore`` keys are opaque strings the caller
    mints.  The request layer owns generation (``secrets.token_urlsafe(32)``,
    HTTP-only ``Secure`` cookie, rotated on auth events).  This module
    enforces no policy on the string itself; weak ``session_id``s collapse
    the entire credential isolation model and are a cross-layer concern.
- Resolver chain stops on first hit.  A more-specific resolver
  (request-bound session) shadows a less-specific one (admin env);
  there is no "merge" semantics.

The ``TaskManager`` zero-key contract: any background task that needs a
provider key carries it as a field populated from the same per-request
resolver chain that the foreground handler used, and the field is
explicitly cleared in the ``finally`` block that runs on completion /
cancellation / error.  The serialised ``task_history`` row written by
:meth:`MetadataRegistry.save_task_history` MUST never carry a key — that
contract is enforced by every write site, not by this module.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from sec_generative_search.core.exceptions import ProviderError
from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.core.security import mask_secret

__all__ = [
    "ApiKeyResolver",
    "CredentialStore",
    "InMemorySessionCredentialStore",
    "chain_resolvers",
    "encrypted_user_resolver",
    "session_resolver",
    "validate_credential",
]


logger = get_logger(__name__)


# Re-declared locally rather than imported from ``providers.factory`` to
# keep the layering clean — ``core`` does not depend on ``providers``.
# The two declarations are validated to be identical by a security test.
ApiKeyResolver = Callable[[str], str | None]


# ---------------------------------------------------------------------------
# CredentialStore protocol
# ---------------------------------------------------------------------------


class CredentialStore(Protocol):
    """The shape every credential backend must satisfy.

    Implementations are free to differ on lifecycle (in-memory + TTL,
    persistent, remote KMS) and on what ``key_id`` represents (an
    opaque ``session_id`` for the in-memory store, a stable ``user_id``
    for the encrypted store).  The resolver chain treats every store
    identically: it asks for a value and either gets one or moves on.

    Implementations MUST NOT log, return, or otherwise echo the raw
    credential value outside the explicit :meth:`get` path.  Listing
    methods MUST return only the provider names, never the values.
    """

    def get(self, key_id: str, provider: str) -> str | None:
        """Return the stored credential, or ``None`` if absent / evicted."""

    def set(self, key_id: str, provider: str, api_key: str) -> None:
        """Store ``api_key`` for ``(key_id, provider)``, replacing any prior value."""

    def delete(self, key_id: str, provider: str) -> bool:
        """Remove the stored credential.  Returns ``True`` iff one was removed."""

    def list_providers(self, key_id: str) -> set[str]:
        """Return the provider names for which ``key_id`` has a stored credential."""

    def clear(self, key_id: str) -> int:
        """Remove every credential for ``key_id``.  Returns the count removed."""


# ---------------------------------------------------------------------------
# In-memory session-scoped store
# ---------------------------------------------------------------------------


# Default TTL for an idle session entry.  Matches the rough lifetime of
# a typical browser session — long enough to support a multi-tab workflow,
# short enough that an abandoned tab does not leak credentials forever.
# Operators can override per-instance via the constructor.
_DEFAULT_SESSION_TTL_SECONDS = 60 * 60  # one hour


@dataclass
class _SessionEntry:
    """One session's credentials.  Mutable; held only inside the store's lock.

    ``user_id`` is the user-tier login bound to the session (``None`` for an
    operator / anonymous session).  It rides the entry so every seam that
    drops the session — :meth:`InMemorySessionCredentialStore.clear`, TTL
    eviction, the global sweep — drops the binding in the same step.
    """

    credentials: dict[str, str] = field(default_factory=dict)
    last_touched: float = 0.0
    user_id: int | None = None


class InMemorySessionCredentialStore:
    """Process-local credential store, keyed by an opaque ``session_id``.

    Holds user-supplied provider keys for the lifetime of a logical
    session.  Eviction is **lazy** — there is no background thread, by
    deliberate analogy to caller-driven lifecycle management: a leaked
    timer thread holding live credentials is strictly worse than lazy
    eviction.  Two lazy tiers run under the lock: per-key eviction on
    the accessed session (:meth:`_evict_if_expired`) plus an amortised
    **global** sweep (:meth:`_maybe_sweep_locked`, at most one pass per
    TTL window) so an *abandoned* session's cleartext keys are collected
    on the next operation past its TTL rather than lingering until the
    process restarts.

    Concurrency: a single :class:`threading.Lock` serialises every
    mutating and observing operation.  Keys are short, sessions are
    bounded, contention is negligible — keep it simple.

    Lifecycle expectations from callers:

        - ``session_id`` MUST be cryptographically random and minted
            server-side.
    - On user logout the route handler MUST call :meth:`clear` so the
      session's credentials are dropped immediately rather than
      lingering until TTL expiry.
        - The store bounds itself to *live* sessions only: abandoned
            sessions are swept at TTL by :meth:`_maybe_sweep_locked`, but
            there is no cap on genuinely-active concurrent sessions.  Higher
            layers own rate-limiting session minting if a hard cap is needed.

    Session → user binding: :meth:`bind_user` / :meth:`user_for` /
    :meth:`unbind_user` attach the user-tier login to the session's entry
    (not part of :class:`CredentialStore`).  The binding shares the entry's
    sliding TTL and sweep, and :meth:`clear` drops it with the credentials,
    so a rotated, logged-out or abandoned ``session_id`` can never keep
    authorising user-tier writes.

    Audit logging: :meth:`set` and :meth:`delete` emit structured
    audit-log entries with the session-id tail and the masked
    credential tail.  :meth:`get` does not log on every call (that
    would flood logs at request rate); the resolver-level audit is
    handled in :func:`session_resolver` so each resolution event is
    logged exactly once.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = _DEFAULT_SESSION_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            # A non-positive TTL would evict every entry on the next get.
            # Reject to make the misconfiguration visible at construction
            # time rather than at first read.
            raise ValueError(
                f"ttl_seconds must be > 0; got {ttl_seconds}. "
                "An always-expired store is never useful."
            )
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionEntry] = {}
        # Monotonic timestamp of the last full sweep.  Seeded at
        # construction so the first sweep cannot fire until one TTL window
        # has elapsed.  See :meth:`_maybe_sweep_locked`.
        self._last_sweep = self._clock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_sweep_locked(self) -> None:
        """Amortised full sweep of expired sessions.  Caller MUST hold the lock.

        Per-key lazy eviction (:meth:`_evict_if_expired`) only collects a
        session that is *accessed again*.  An abandoned session — the
        browser tab that never comes back — would otherwise keep its
        cleartext provider keys resident in RAM until the process
        restarts, silently widening the live-host-compromise window past
        the sliding TTL the Data Classification table advertises, and
        leaving the session count unbounded (a slow memory-exhaustion
        vector under Scenario C).

        This closes that gap without a background thread (the deliberate
        constraint in the class docstring stands): every public operation
        already takes ``self._lock``; piggybacked on it, at most **one**
        ``O(sessions)`` pass runs per TTL window, triggered by whichever
        operation first crosses the interval.  Time-based rather than
        op-count-based so residency is bounded by wall-clock, not by
        traffic volume — a mostly-idle server still collects abandoned
        sessions on the next touch after the window elapses.  An
        abandoned session is therefore evicted within ``2 * ttl_seconds``
        of its last use (expiry at ``ttl``, swept on the next op up to one
        further ``ttl`` later).
        """
        now = self._clock()
        if (now - self._last_sweep) < self._ttl_seconds:
            return
        self._last_sweep = now
        expired = [
            key_id
            for key_id, entry in self._sessions.items()
            if (now - entry.last_touched) > self._ttl_seconds
        ]
        for key_id in expired:
            # Clear the credential strings before del so they become
            # collectable immediately (mirrors :meth:`_evict_if_expired`).
            self._sessions[key_id].credentials.clear()
            del self._sessions[key_id]

    def _evict_if_expired(self, key_id: str) -> None:
        """Drop the session entry if its TTL has elapsed.

        Caller MUST hold ``self._lock``.  Idle eviction is intentionally
        coupled to the access path — every public method runs through
        here on first touch so an abandoned session is collected at the
        next read attempt.
        """
        entry = self._sessions.get(key_id)
        if entry is None:
            return
        if (self._clock() - entry.last_touched) > self._ttl_seconds:
            # Replace the value with an empty dict before del to make
            # the credential strings garbage-collectable as fast as
            # CPython's reference-counter allows.  Belt + braces.
            entry.credentials.clear()
            del self._sessions[key_id]

    def _touch(self, entry: _SessionEntry) -> None:
        entry.last_touched = self._clock()

    # ------------------------------------------------------------------
    # CredentialStore protocol
    # ------------------------------------------------------------------

    def get(self, key_id: str, provider: str) -> str | None:
        with self._lock:
            self._maybe_sweep_locked()
            self._evict_if_expired(key_id)
            entry = self._sessions.get(key_id)
            if entry is None:
                return None
            value = entry.credentials.get(provider)
            if value is not None:
                self._touch(entry)
            return value

    def set(self, key_id: str, provider: str, api_key: str) -> None:
        if not api_key:
            # Refuse the empty-string case explicitly.  Storing ``""``
            # would silently shadow a working admin-env fallback in the
            # resolver chain (chain stops on first non-``None`` hit).
            raise ValueError(
                "api_key must be a non-empty string. Use delete() to remove a stored credential."
            )
        with self._lock:
            self._maybe_sweep_locked()
            self._evict_if_expired(key_id)
            entry = self._sessions.get(key_id)
            if entry is None:
                entry = _SessionEntry()
                self._sessions[key_id] = entry
            entry.credentials[provider] = api_key
            self._touch(entry)
        audit_log(
            "credential_set",
            detail=(
                f"store=in_memory_session "
                f"key_id_tail={mask_secret(key_id)} "
                f"provider={provider} "
                f"key_tail={mask_secret(api_key)}"
            ),
        )

    def delete(self, key_id: str, provider: str) -> bool:
        with self._lock:
            self._maybe_sweep_locked()
            self._evict_if_expired(key_id)
            entry = self._sessions.get(key_id)
            if entry is None:
                removed = False
            else:
                removed = entry.credentials.pop(provider, None) is not None
                # A bound entry outlives its last credential — deleting a
                # provider key must not sign the user out.
                if not entry.credentials and entry.user_id is None:
                    del self._sessions[key_id]
        if removed:
            audit_log(
                "credential_delete",
                detail=(
                    f"store=in_memory_session key_id_tail={mask_secret(key_id)} provider={provider}"
                ),
            )
        return removed

    def list_providers(self, key_id: str) -> set[str]:
        with self._lock:
            self._maybe_sweep_locked()
            self._evict_if_expired(key_id)
            entry = self._sessions.get(key_id)
            if entry is None:
                return set()
            return set(entry.credentials.keys())

    def clear(self, key_id: str) -> int:
        """Drop the whole session — credentials **and** any user binding.

        Returns the number of credentials removed (the binding is not
        counted).
        """
        with self._lock:
            self._maybe_sweep_locked()
            entry = self._sessions.pop(key_id, None)
            if entry is None:
                return 0
            count = len(entry.credentials)
            entry.credentials.clear()
        if count:
            audit_log(
                "credential_clear",
                detail=(
                    f"store=in_memory_session key_id_tail={mask_secret(key_id)} removed={count}"
                ),
            )
        return count

    # ------------------------------------------------------------------
    # Session → user binding (user tier; outside the CredentialStore protocol)
    # ------------------------------------------------------------------

    def bind_user(self, key_id: str, user_id: int) -> None:
        """Bind a user-tier login to the session, replacing any prior binding."""
        with self._lock:
            self._maybe_sweep_locked()
            self._evict_if_expired(key_id)
            entry = self._sessions.get(key_id)
            if entry is None:
                entry = _SessionEntry()
                self._sessions[key_id] = entry
            entry.user_id = user_id
            self._touch(entry)

    def user_for(self, key_id: str) -> int | None:
        """Return the bound user id, or ``None`` if unbound / expired.

        A hit refreshes the sliding TTL, as a credential read does.
        """
        with self._lock:
            self._maybe_sweep_locked()
            self._evict_if_expired(key_id)
            entry = self._sessions.get(key_id)
            if entry is None or entry.user_id is None:
                return None
            self._touch(entry)
            return entry.user_id

    def unbind_user(self, key_id: str) -> bool:
        """Drop the session's user binding.  Returns ``True`` iff one existed.

        Credentials stay; an entry left with neither is removed.
        """
        with self._lock:
            self._maybe_sweep_locked()
            self._evict_if_expired(key_id)
            entry = self._sessions.get(key_id)
            if entry is None or entry.user_id is None:
                return False
            entry.user_id = None
            if not entry.credentials:
                del self._sessions[key_id]
            return True


# ---------------------------------------------------------------------------
# Resolver chain — composes callables of the ApiKeyResolver shape
# ---------------------------------------------------------------------------


def chain_resolvers(*resolvers: ApiKeyResolver) -> ApiKeyResolver:
    """Combine resolvers into a single first-hit-wins resolver.

    The returned callable iterates the resolvers in order and returns
    the first non-``None`` value.  No merge semantics — a more-specific
    resolver always shadows a less-specific one.  An empty resolver
    list returns ``None`` for every provider, which is the correct
    behaviour for "no credentials configured anywhere".

    The contract matches
    :data:`~sec_generative_search.providers.factory.ApiKeyResolver`
    exactly so the chain plugs straight into ``build_embedder`` /
    ``build_llm_provider`` without any adapter layer.
    """
    resolver_chain = tuple(resolvers)

    def chained(provider: str) -> str | None:
        for resolver in resolver_chain:
            value = resolver(provider)
            if value is not None:
                return value
        return None

    return chained


def session_resolver(
    store: CredentialStore,
    session_id: str,
) -> ApiKeyResolver:
    """Adapt a ``CredentialStore`` to the ``ApiKeyResolver`` shape.

    Curries the ``key_id`` so the resulting resolver matches the
    factory's expected ``Callable[[str], str | None]`` shape.  Logs an
    audit-log entry on each non-``None`` hit so a credential's
    actually-resolved-from-session lineage is greppable in operator
    logs (compare to the noisy alternative of logging on every
    :meth:`CredentialStore.get`, which would flood under high RPS).
    """

    def resolver(provider: str) -> str | None:
        value = store.get(session_id, provider)
        if value is not None:
            audit_log(
                "credential_resolved",
                detail=(
                    f"resolver=session "
                    f"session_id_tail={mask_secret(session_id)} "
                    f"provider={provider} "
                    f"key_tail={mask_secret(value)}"
                ),
            )
        return value

    return resolver


def encrypted_user_resolver(
    store: CredentialStore,
    user_id: str,
) -> ApiKeyResolver:
    """Adapt the encrypted persistent store to the ``ApiKeyResolver`` shape.

    Functionally identical to :func:`session_resolver`; the split
    exists so audit-log lines can name *which* tier the credential came
    from, which matters when reasoning about whether a stored key
    should be rotated.
    """

    def resolver(provider: str) -> str | None:
        value = store.get(user_id, provider)
        if value is not None:
            audit_log(
                "credential_resolved",
                detail=(
                    f"resolver=encrypted_user "
                    f"user_id_tail={mask_secret(user_id)} "
                    f"provider={provider} "
                    f"key_tail={mask_secret(value)}"
                ),
            )
        return value

    return resolver


# ---------------------------------------------------------------------------
# Provider credential validation
# ---------------------------------------------------------------------------


def validate_credential(
    provider: str,
    surface: object,
    api_key: str,
    *,
    model: str | None = None,
) -> bool:
    """Validate ``api_key`` against ``(provider, surface)``.

    Thin audit-logged wrapper around
    :meth:`ProviderRegistry.validate_key`.  The split exists so route
    handlers can call a single canonical validation seam that already
    takes care of audit-log discipline, instead of re-creating the
    masking + logging at every call site.

    The signature accepts ``surface`` as ``object`` to keep this module
    free of an import-time dependency on ``providers.registry`` (which
    pulls in every adapter class via the registry's ``_ENTRIES``).  The
    runtime contract is that ``surface`` is a
    :class:`~sec_generative_search.providers.registry.ProviderSurface`
    member — callers pass it as keyword for clarity.

    Returns ``True`` when the provider accepts the key.  Returns
    ``False`` when the provider explicitly rejects it
    (:class:`ProviderAuthError`).  Every other ``ProviderError``
    propagates intentionally — see ``ProviderRegistry.validate_key``'s
    docstring.

    The raw key NEVER appears in the audit log; only its masked tail.
    """
    # Deferred import — see docstring for the layering rationale.
    from sec_generative_search.providers.registry import (
        ProviderRegistry,
        ProviderSurface,
    )

    if not isinstance(surface, ProviderSurface):
        raise TypeError(f"surface must be a ProviderSurface member; got {type(surface).__name__}.")

    try:
        ok = ProviderRegistry.validate_key(provider, surface, api_key, model=model)
    except ProviderError as exc:
        # Non-auth provider errors (rate limit, timeout, content filter,
        # network) should NOT be reported as a verdict on the key.
        # Audit the attempt and re-raise.
        audit_log(
            "credential_validate_error",
            detail=(
                f"provider={provider} "
                f"surface={surface.value} "
                f"key_tail={mask_secret(api_key)} "
                f"error={type(exc).__name__}"
            ),
        )
        raise

    audit_log(
        "credential_validate",
        detail=(
            f"provider={provider} "
            f"surface={surface.value} "
            f"key_tail={mask_secret(api_key)} "
            f"result={'ok' if ok else 'rejected'}"
        ),
    )
    return ok
