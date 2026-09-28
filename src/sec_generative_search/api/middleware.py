"""ASGI middleware stack for the FastAPI surface.

Stack ordering matters and is documented at the call site
(:func:`sec_generative_search.api.app.create_app`).  Read top-to-bottom
of the ``add_middleware`` block as outermost-to-innermost — the
*last-added* middleware is the *first* to see an inbound request.

Components in this module:

    - :class:`CorrelationIdMiddleware` — pure ASGI; binds a per-request
      correlation ID and echoes it as ``X-Request-ID``.
    - :class:`SecurityHeadersMiddleware` — pure ASGI; appends fixed
      response headers (CSP, X-Frame-Options, ...).
    - :class:`ContentSizeLimitMiddleware` — pure ASGI; rejects oversize
      bodies before they reach a handler, defends against chunked
      transfer encoding mid-stream.
    - :class:`RateLimitMiddleware` — pure ASGI; per-IP sliding-window
      limiter with a separate per-``session_id`` window enforced on top
      of the per-IP window for the provider-validation route (both must
      allow; exhausting either returns 429).  It also carries the
      one-shot **insecure-transport warning** (folded in from a former
      standalone ``BaseHTTPMiddleware`` layer — a whole middleware for a
      single log line was over-structure): it logs once when
      authenticated traffic arrives over plain HTTP.

CORS is supplied by Starlette's :class:`CORSMiddleware` directly; we do
not wrap it.

Security notes:

    - Every bespoke middleware is **pure ASGI** — no
      ``BaseHTTPMiddleware`` remains, so none pays the per-request
      task-group + memory-stream hop, and the correlation-ID
      ``ContextVar`` stays visible to the same task that runs the route.
    - All bespoke middlewares are dependency-free outside the Python
      standard library, Starlette, and ``core.logging``.  Do not add
      provider, database, or chromadb imports here.
    - The per-route rate-limit / body-cap policy is resolved **once per
      request** and cached on the ASGI ``scope`` (:func:`_resolve_policy_cached`)
      so the rate limiter and the content-size limiter share the single
      :func:`~sec_generative_search.api.policies.resolve_policy` scan.
    - The error responses sent from middleware MUST carry the same
      :data:`~sec_generative_search.api.errors.ErrorEnvelope` shape as
      handler-raised errors so downstream consumers do not need to
      branch on response source.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from typing import Any

from starlette.requests import cookie_parser
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sec_generative_search.api.errors import envelope
from sec_generative_search.api.policies import RoutePolicy, resolve_policy
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.correlation import (
    new_correlation_id,
    reset_correlation_id,
    set_correlation_id,
    validate_request_id,
)
from sec_generative_search.core.logging import get_logger

# Imported lazily inside the dispatch path so the middleware module
# stays cheap at import time. The dependency is one-way.
_SESSION_COOKIE_LAZY: str | None = None


def _session_cookie_name() -> str:
    """Return the session-cookie name without forcing an early import.

    The cookie name lives in ``api.dependencies``, which itself does not
    import middleware. Read it once and cache so the per-request rate
    limit path stays O(1).
    """
    global _SESSION_COOKIE_LAZY
    if _SESSION_COOKIE_LAZY is None:
        from sec_generative_search.api.dependencies import SESSION_COOKIE_NAME

        _SESSION_COOKIE_LAZY = SESSION_COOKIE_NAME
    return _SESSION_COOKIE_LAZY


__all__ = [
    "DEFAULT_MAX_CONTENT_LENGTH",
    "REQUEST_ID_HEADER",
    "ContentSizeLimitMiddleware",
    "CorrelationIdMiddleware",
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware",
]


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-request policy cache
# ---------------------------------------------------------------------------


# Scope key under which the resolved :class:`RoutePolicy` is memoised so
# the rate limiter and the content-size limiter share a single
# :func:`~sec_generative_search.api.policies.resolve_policy` scan per
# request instead of one each.  Namespaced to avoid colliding with any
# ASGI-server or framework key on the scope.
_POLICY_SCOPE_KEY = "sec_generative_search.route_policy"


def _resolve_policy_cached(scope: Scope) -> RoutePolicy:
    """Resolve — and memoise on ``scope`` — the per-route policy.

    Whichever bespoke middleware runs first (the rate limiter, outermost
    of the two) populates the cache; the inner content-size limiter reads
    it back.  Order-independent: if the cache is empty (e.g. a request
    that reached the content-size limiter for a non-``/api/`` path the
    rate limiter skipped without resolving) it resolves and caches here.
    """
    cached = scope.get(_POLICY_SCOPE_KEY)
    if cached is not None:
        return cached
    policy = resolve_policy(scope.get("path", ""), scope.get("method", ""))
    scope[_POLICY_SCOPE_KEY] = policy
    return policy


# ---------------------------------------------------------------------------
# Correlation ID
# ---------------------------------------------------------------------------


# Both the inbound (operator-supplied) and outbound (echoed) header name.
REQUEST_ID_HEADER = b"x-request-id"


class CorrelationIdMiddleware:
    """Bind a per-request correlation ID and echo it as ``X-Request-ID``.

    Pure ASGI — avoids the ``BaseHTTPMiddleware`` threadpool hop so the
    :class:`~contextvars.ContextVar` it sets stays visible to the same
    task that runs the route, retrieval, and generation. The ID is:

    - adopted from an inbound ``X-Request-ID`` **only** when it passes
      :func:`~sec_generative_search.core.correlation.validate_request_id`
      (bounded alphanumeric/``-``/``_``; CR/LF and control characters
      rejected — a log/header-injection guard); otherwise
    - freshly minted.

    The ID is echoed back on the ``http.response.start`` headers so a
    caller can correlate its request with server logs, and it is reset
    in ``finally`` so the ContextVar never leaks across requests sharing
    a worker.

    Placed outermost among the bespoke middlewares (just inside CORS) so
    even rate-limit rejections and oversize-body 413s emit log records —
    and a response header — carrying the ID.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        raw = headers.get(REQUEST_ID_HEADER)
        inbound: str | None = None
        if raw is not None:
            # Header bytes are latin-1 on the wire; a decode failure or a
            # shape-check miss both fall through to a freshly minted ID.
            try:
                inbound = validate_request_id(raw.decode("latin-1"))
            except (UnicodeDecodeError, ValueError):
                inbound = None

        correlation_id = inbound or new_correlation_id()
        cid_bytes = correlation_id.encode("ascii")
        token = set_correlation_id(correlation_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                # Drop any X-Request-ID the app set so the response
                # carries exactly the one we bound, never a duplicate.
                response_headers = [
                    (n, v) for n, v in message.get("headers", []) if n != REQUEST_ID_HEADER
                ]
                response_headers.append((REQUEST_ID_HEADER, cid_bytes))
                message["headers"] = response_headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            reset_correlation_id(token)


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------


_CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "base-uri 'self'",
        "frame-ancestors 'none'",
        "img-src 'self' data: blob: https:",
        "font-src 'self' data: https:",
        "style-src 'self' 'unsafe-inline'",
        "script-src 'self'",
        "connect-src 'self' ws: wss:",
    ]
)

_PERMISSIONS_POLICY = (
    "camera=(), microphone=(), geolocation=(), payment=(), usb=(), interest-cohort=()"
)

_SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"x-xss-protection", b"1; mode=block"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (b"content-security-policy", _CONTENT_SECURITY_POLICY.encode()),
    (b"permissions-policy", _PERMISSIONS_POLICY.encode()),
)

# Header names we force onto every response — computed once at import
# time rather than rebuilt per response inside the send hook.
_FORCED_SECURITY_HEADER_NAMES: frozenset[bytes] = frozenset(name for name, _ in _SECURITY_HEADERS)


class SecurityHeadersMiddleware:
    """Append a fixed set of security headers to every HTTP response.

    Pure ASGI — intercepts the ``http.response.start`` message and
    extends the headers list directly.  Avoids Starlette's
    ``BaseHTTPMiddleware`` threadpool dispatch, which adds latency for
    a no-op pass-through.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                # Filter out any header we are about to set so the final
                # response carries exactly one of each (callers that try
                # to override are silently corrected, not duplicated).
                headers = [(n, v) for n, v in headers if n not in _FORCED_SECURITY_HEADER_NAMES]
                headers.extend(_SECURITY_HEADERS)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


# ---------------------------------------------------------------------------
# Content size limit
# ---------------------------------------------------------------------------


# 1 MiB.  Matches the nginx ``client_max_body_size`` we expect at the
# reverse proxy; the in-process check is defence in depth for direct
# uvicorn access (local dev without a proxy).
DEFAULT_MAX_CONTENT_LENGTH = 1 * 1024 * 1024


class ContentSizeLimitMiddleware:
    """Reject oversize request bodies, per-route.

    The effective bound is the **smaller** of:

    1. The constructor's ``max_bytes`` (global ceiling — kept for the
       scenarios in which an operator wants a tighter cap than any
       route's table entry, e.g. an ingress proxy is already enforcing
       a smaller limit and we want consistency below it).
    2. The route policy's ``max_body_bytes`` resolved from
       :func:`~sec_generative_search.api.policies.resolve_policy`.

    Two layers of defence inside the chosen bound:

    1. *Content-Length check* — short-circuits before reading the body
       when the declared size already exceeds the limit.
    2. *Stream-counting wrapper* — intercepts ``http.request`` messages
       and tallies the bytes received so far.  Rejects mid-stream when
       the tally exceeds the limit, which catches chunked transfer
       encoding (no ``Content-Length`` header).

    The mid-stream 413 is only substituted while the app has **not yet**
    sent ``http.response.start``.  Every current route reads its whole
    body before responding, so that is the reachable path; but a
    hypothetical handler that starts responding before draining the body
    would have its status line already on the wire when the tally trips
    — the middleware then lets that in-flight response finish untouched
    rather than emitting a **second** ``http.response.start`` (an ASGI
    protocol violation the server surfaces as a broken connection).  The
    body cap is still enforced in that case: the stream-counting wrapper
    stops feeding the app once the limit is exceeded.
    """

    def __init__(self, app: ASGIApp, *, max_bytes: int = DEFAULT_MAX_CONTENT_LENGTH) -> None:
        self.app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Resolve the per-route cap (once per request, shared with the
        # rate limiter via the scope cache) and clamp by the global
        # ceiling so a tighter constructor-supplied bound always wins.
        policy_max = _resolve_policy_cached(scope).max_body_bytes
        effective_max = min(policy_max, self._max_bytes)

        headers = dict(scope.get("headers", []))
        content_length_raw = headers.get(b"content-length")

        if content_length_raw is not None:
            try:
                length = int(content_length_raw)
            except (ValueError, UnicodeDecodeError):
                await self._send_payload_error(
                    send,
                    status=400,
                    error="invalid_content_length",
                    message="Invalid Content-Length header.",
                    hint="Send Content-Length as an unsigned decimal integer.",
                )
                return

            if length > effective_max:
                await self._send_payload_error(
                    send,
                    status=413,
                    error="payload_too_large",
                    message=(
                        f"Request body too large ({length:,} bytes). "
                        f"Maximum allowed: {effective_max:,} bytes."
                    ),
                    hint="Reduce the request payload size.",
                )
                return

        bytes_received = 0
        rejected = False
        response_started = False

        async def receive_with_limit() -> Message:
            nonlocal bytes_received, rejected
            message = await receive()
            if message["type"] == "http.request":
                bytes_received += len(message.get("body", b""))
                if bytes_received > effective_max:
                    rejected = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def send_with_guard(message: Message) -> None:
            nonlocal response_started
            # If the tally has tripped but the app has NOT yet committed
            # to a response, swallow its start/body so we can substitute
            # our own 413 after it returns. Once ``http.response.start``
            # has passed through, the status line + headers are already
            # on the wire and a 413 can no longer be substituted — so we
            # stop guarding and let the app's in-flight response finish
            # untouched (sending our own start now would be a second
            # ``http.response.start``, an ASGI protocol violation). The
            # body cap is still enforced regardless: ``receive_with_limit``
            # stops feeding the app once the limit is exceeded.
            if (
                rejected
                and not response_started
                and message["type"]
                in (
                    "http.response.start",
                    "http.response.body",
                )
            ):
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        await self.app(scope, receive_with_limit, send_with_guard)

        # Only issue the 413 when the app never began responding. If the
        # response already started, we cannot override it — the cap was
        # still enforced on the received body, and emitting a start here
        # would be a duplicate.
        if rejected and not response_started:
            await self._send_payload_error(
                send,
                status=413,
                error="payload_too_large",
                message=(
                    f"Request body too large (>{effective_max:,} bytes). "
                    f"Maximum allowed: {effective_max:,} bytes."
                ),
                hint="Reduce the request payload size.",
            )

    @staticmethod
    async def _send_payload_error(
        send: Send,
        *,
        status: int,
        error: str,
        message: str,
        hint: str,
    ) -> None:
        body = json.dumps(envelope(error=error, message=message, hint=hint)).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


# ---------------------------------------------------------------------------
# Rate limiting (per-IP sliding window)
# ---------------------------------------------------------------------------


# How often to prune stale buckets (seconds).
_CLEANUP_INTERVAL = 300.0


class _SlidingWindow:
    """Per-key sliding-window counter with periodic cleanup."""

    __slots__ = ("_last_cleanup", "_limit", "_lock", "_requests", "_window")

    def __init__(self, requests_per_minute: int) -> None:
        self._limit = requests_per_minute
        self._window = 60.0
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    @property
    def limit(self) -> int:
        return self._limit

    def is_allowed(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            if now - self._last_cleanup > _CLEANUP_INTERVAL:
                self._prune(now)
                self._last_cleanup = now

            cutoff = now - self._window
            timestamps = self._requests[key]
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= self._limit:
                retry_after = int(self._window - (now - timestamps[0])) + 1
                return False, max(retry_after, 1)

            timestamps.append(now)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        stale = [k for k, ts in self._requests.items() if not ts or ts[-1] <= cutoff]
        for k in stale:
            del self._requests[k]


def _rate_category_for_scope(scope: Scope) -> str | None:
    """Map a request to its rate-limit category through the policy cache.

    - ``None`` — the route is exempt from the rate limiter (e.g.
      ``/api/health``) **or** the path falls outside ``/api/`` (Swagger /
      Redoc / static — unauthenticated read surfaces gated by ``API_KEY``
      presence in the FastAPI factory).
    - any other string — the rate-limit bucket name.

    Resolves through :func:`_resolve_policy_cached`, so the single
    per-request table scan is shared with the content-size limiter.
    Non-``/api/`` paths short-circuit *without* touching the cache — the
    content-size limiter resolves its own policy for those.
    """
    path = scope.get("path", "")
    if not path.startswith("/api/") and path != "/api":
        return None
    return _resolve_policy_cached(scope).rate_category


def _client_ip_from_scope(scope: Scope) -> str:
    """Return the peer host from the ASGI ``scope`` (``"unknown"`` if absent).

    ``scope["client"]`` is a ``(host, port)`` tuple or ``None`` — the
    pure-ASGI equivalent of ``request.client.host``.
    """
    client = scope.get("client")
    if client:
        return client[0]
    return "unknown"


def _session_id_from_scope(scope: Scope) -> str | None:
    """Extract the raw ``session_id`` cookie value from the ASGI ``scope``.

    Reads the first ``cookie`` header and parses it with Starlette's own
    :func:`cookie_parser` — byte-for-byte the semantics
    ``request.cookies`` used before this middleware went pure-ASGI. The
    value is *not* shape-validated here (the rate limiter keys on
    whatever the client sent, exactly as before); a missing/empty cookie
    yields ``None`` so the caller falls back to the per-IP window alone.
    """
    for name, value in scope.get("headers", []):
        if name == b"cookie":
            sid = cookie_parser(value.decode("latin-1")).get(_session_cookie_name())
            return sid or None
    return None


class RateLimitMiddleware:
    """Pure-ASGI per-IP sliding-window limiter with optional per-session keying.

    Buckets are keyed by the peer host (``scope["client"]``).  A ``0``
    limit disables that category.

    The validate category additionally enforces a per-``session_id``
    sliding window on top of the per-IP bucket: both must allow the
    request. A shared NAT cannot consume the per-IP bucket on behalf of
    one tenant, and a single session cannot brute-force key validation
    across many origins. Requests without a ``session_id`` cookie skip
    the per-session check and rely on the per-IP bucket alone.

    This layer also carries the **one-shot insecure-transport warning**
    (folded in from a former standalone ``BaseHTTPMiddleware``): it logs
    once, before the rate-limit gate, when authentication or per-session
    EDGAR credentials are configured and traffic arrives over plain HTTP
    (``X-Forwarded-Proto: http``). The check is observational and runs on
    every request — including exempt and 429-rejected ones — so a
    misconfigured deployment is flagged regardless of the outcome.

    Pure ASGI: no ``BaseHTTPMiddleware`` task-group / memory-stream hop.
    A 429 is emitted with a :class:`~starlette.responses.JSONResponse`
    whose ``http.response.start`` still flows back out through the
    outer :class:`SecurityHeadersMiddleware` and
    :class:`CorrelationIdMiddleware`, so the rejection carries the full
    security-header set and the bound ``X-Request-ID``.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        search_rpm: int = 60,
        ingest_rpm: int = 10,
        delete_rpm: int = 30,
        general_rpm: int = 120,
        rag_rpm: int = 60,
        validate_rpm: int = 10,
        validate_per_session_rpm: int = 5,
        session_rpm: int = 20,
        login_rpm: int = 5,
    ) -> None:
        self.app = app
        self._buckets: dict[str, _SlidingWindow] = {}
        for category, rpm in (
            ("search", search_rpm),
            ("ingest", ingest_rpm),
            ("delete", delete_rpm),
            ("general", general_rpm),
            ("rag", rag_rpm),
            ("validate", validate_rpm),
            ("session", session_rpm),
            ("login", login_rpm),
        ):
            if rpm > 0:
                self._buckets[category] = _SlidingWindow(rpm)
        # Separate bucket so the per-IP and per-session windows do not
        # share a counter.
        self._validate_session_bucket: _SlidingWindow | None = (
            _SlidingWindow(validate_per_session_rpm) if validate_per_session_rpm > 0 else None
        )
        # One-shot insecure-transport warning state (folded in).
        self._insecure_warned = False
        self._insecure_lock = threading.Lock()

    def reset(self) -> None:
        for bucket in self._buckets.values():
            bucket.reset()
        if self._validate_session_bucket is not None:
            self._validate_session_bucket.reset()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Observational one-shot warning, before the gate so it fires
        # even on exempt / rejected requests.
        self._maybe_warn_insecure_transport(scope)

        category = _rate_category_for_scope(scope)
        if category is None or category not in self._buckets:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")
        bucket = self._buckets[category]
        client_ip = _client_ip_from_scope(scope)

        allowed, retry_after = bucket.is_allowed(client_ip)
        if not allowed:
            await self._reject(category, method, path, client_ip, bucket.limit, retry_after)(
                scope, receive, send
            )
            return

        # Per-session check piggy-backs on the per-IP gate above for the
        # validate route only. We do not log on success; rejection uses
        # the same structured 429 envelope as every other rate-limit response.
        if category == "validate" and self._validate_session_bucket is not None:
            sid = _session_id_from_scope(scope)
            if sid:
                allowed_s, retry_s = self._validate_session_bucket.is_allowed(sid)
                if not allowed_s:
                    await self._reject(
                        "validate",
                        method,
                        path,
                        f"session={sid[:6]}...",
                        self._validate_session_bucket.limit,
                        retry_s,
                    )(scope, receive, send)
                    return

        await self.app(scope, receive, send)

    def _maybe_warn_insecure_transport(self, scope: Scope) -> None:
        # Cheap fast-path before we read settings or take the lock.
        if self._insecure_warned:
            return
        api = get_settings().api
        if not (api.key or api.admin_key or api.edgar_session_required):
            return

        forwarded_proto: str | None = None
        for name, value in scope.get("headers", []):
            if name == b"x-forwarded-proto":
                forwarded_proto = value.decode("latin-1")
                break
        if forwarded_proto is None:
            return
        if forwarded_proto.split(",", 1)[0].strip().lower() != "http":
            return

        with self._insecure_lock:
            if self._insecure_warned:
                return
            self._insecure_warned = True
            logger.warning(
                "Insecure transport detected (X-Forwarded-Proto=http) while "
                "authentication or per-session EDGAR credentials are enabled. "
                "Scenarios B/C require TLS — terminate HTTPS at the reverse "
                "proxy or launch uvicorn with --ssl-certfile/--ssl-keyfile."
            )

    def _reject(
        self,
        category: str,
        method: str,
        path: str,
        key_label: str,
        limit: int,
        retry_after: int,
    ) -> JSONResponse:
        logger.warning(
            "Rate limit exceeded: %s from %s on %s %s (limit=%d/min)",
            category,
            key_label,
            method,
            path,
            limit,
        )
        return JSONResponse(
            status_code=429,
            content=envelope(
                error="rate_limited",
                message=(f"Rate limit exceeded for {category}. Retry in {retry_after}s."),
                details={"category": category, "limit_per_minute": limit},
                hint=f"Maximum {limit} {category} requests per minute.",
            ),
            headers={"Retry-After": str(retry_after)},
        )


# ---------------------------------------------------------------------------
# Internal: typing helper for tests
# ---------------------------------------------------------------------------


def _security_headers_for_test() -> dict[str, str]:
    """Expose the static security-headers map for unit assertions."""
    return {name.decode(): value.decode() for name, value in _SECURITY_HEADERS}


_security_headers_for_test.__test__ = False  # type: ignore[attr-defined]


# Re-exported for tests that want the raw bytes pairs.
SECURITY_HEADERS_RAW: tuple[tuple[bytes, bytes], ...] = _SECURITY_HEADERS


# Re-exported for tests that want to inspect the typing of dispatched messages.
_MessageT = dict[str, Any]
