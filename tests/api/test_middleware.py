"""Security tests for the middleware stack."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sec_generative_search.api.middleware import (
    _SECURITY_HEADERS,
    DEFAULT_MAX_CONTENT_LENGTH,
    ContentSizeLimitMiddleware,
)


@pytest.mark.security
class TestSecurityHeaders:
    def test_all_security_headers_present_on_health(self, api_client: TestClient) -> None:
        response = api_client.get("/api/health")
        assert response.status_code == 200
        for name, value in _SECURITY_HEADERS:
            assert response.headers.get(name.decode()) == value.decode()

    def test_security_headers_on_error_response(self, api_client: TestClient) -> None:
        # 404 — must still carry the headers.
        response = api_client.get("/api/does-not-exist")
        assert response.status_code == 404
        assert response.headers.get("x-frame-options") == "DENY"
        assert response.headers.get("x-content-type-options") == "nosniff"

    def test_csp_is_restrictive(self, api_client: TestClient) -> None:
        response = api_client.get("/api/health")
        csp = response.headers.get("content-security-policy", "")
        # The policy MUST include frame-ancestors 'none' (clickjacking
        # defence) and a default-src self anchor.
        assert "frame-ancestors 'none'" in csp
        assert "default-src 'self'" in csp


@pytest.mark.security
class TestContentSizeLimit:
    def test_oversize_content_length_rejected(self, api_client: TestClient) -> None:
        oversize = DEFAULT_MAX_CONTENT_LENGTH + 1
        response = api_client.post(
            "/api/session",
            content=b"x" * oversize,
            headers={"content-length": str(oversize)},
        )
        assert response.status_code == 413
        body = response.json()
        assert body["error"] == "payload_too_large"

    def test_valid_size_passes(self, api_client: TestClient) -> None:
        response = api_client.post("/api/session")
        assert response.status_code == 201

    def test_per_route_cap_tighter_than_global_default(self, api_client: TestClient) -> None:
        # Per-route policy on /api/session is 1 KiB — well below the
        # 1 MiB global cap. A 2 KiB declared body MUST 413 even
        # though it would have passed the legacy global ceiling.
        oversize = 2 * 1024
        response = api_client.post(
            "/api/session",
            content=b"x" * oversize,
            headers={"content-length": str(oversize)},
        )
        assert response.status_code == 413
        assert response.json()["error"] == "payload_too_large"

    def test_per_route_cap_envelope_message(self, api_client: TestClient) -> None:
        # 413 response messages must reference the per-route cap, not
        # the global default — operators tracing a 413 need to know
        # which cap fired.
        oversize = 2 * 1024
        response = api_client.post(
            "/api/session",
            content=b"x" * oversize,
            headers={"content-length": str(oversize)},
        )
        body = response.json()
        # The 1 KiB per-route bound on /api/session.
        assert "1,024 bytes" in body["message"]

    def test_health_route_rejects_huge_declared_body(self, api_client: TestClient) -> None:
        # /api/health is unauthenticated. Without the per-route cap a
        # caller could declare a 1 MiB Content-Length and force a
        # body read on every probe. The 1 KiB cap rejects pre-read.
        oversize = 100 * 1024
        response = api_client.post(
            "/api/health",
            content=b"x" * oversize,
            headers={"content-length": str(oversize)},
        )
        assert response.status_code == 413

    def test_rag_query_accepts_realistic_plan_envelope(self, api_client: TestClient) -> None:
        # Sanity check: a realistic ~16 KiB plan body MUST land at the
        # handler (returns 422 / 401 from missing fields, not 413).
        # Confirms the /api/rag/query 64 KiB cap holds for plausible
        # real-world payloads.
        body = b"{" + b"x" * (16 * 1024) + b"}"
        response = api_client.post(
            "/api/rag/query",
            content=body,
            headers={"content-type": "application/json"},
        )
        assert response.status_code != 413


@pytest.mark.security
class TestContentSizeLimitStreamRejection:
    """The stream-counting rejection path (chunked / no ``Content-Length``).

    The ``Content-Length`` short-circuit above is exercised by the real
    routes through :class:`TestClient`; this drives the middleware
    directly at the ASGI layer because **no current route responds
        before it has finished consuming the request body**, so the
        protocol edge these tests pin is otherwise unreachable through the
        app.  The invariant under test: the middleware MUST NEVER emit a
        second ``http.response.start`` — sending one after the downstream
        app already sent its own is an ASGI protocol violation the server
        surfaces as a broken connection.
    """

    @staticmethod
    def _run(
        app: ASGIApp,
        chunks: list[tuple[bytes, bool]],
        *,
        max_bytes: int = 64,
        path: str = "/api/rag/query",
        method: str = "POST",
    ) -> list[Message]:
        """Run ``ContentSizeLimitMiddleware`` over ``app``, feeding ``chunks``.

        Returns every ASGI message the middleware forwarded *downstream*
        (i.e. towards the server / client).  ``max_bytes`` is the global
        ceiling; with the 1 KiB smallest route cap it always wins, so 64
        bytes is the effective body limit.
        """
        scope: Scope = {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [],  # no Content-Length → the stream-counting path
            "client": ("test-client", 0),
        }
        queue = list(chunks)

        async def receive() -> Message:
            if queue:
                body, more_body = queue.pop(0)
                return {"type": "http.request", "body": body, "more_body": more_body}
            return {"type": "http.request", "body": b"", "more_body": False}

        sent: list[Message] = []

        async def send(message: Message) -> None:
            sent.append(message)

        middleware = ContentSizeLimitMiddleware(app, max_bytes=max_bytes)
        asyncio.run(middleware(scope, receive, send))
        return sent

    @staticmethod
    def _starts(sent: list[Message]) -> list[Message]:
        return [m for m in sent if m["type"] == "http.response.start"]

    # -- Downstream ASGI apps with distinct receive/send orderings -------

    @staticmethod
    async def _respond_before_body(scope: Scope, receive: Receive, send: Send) -> None:
        """Commit to a 200 *before* consuming the body (streaming shape)."""
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        more = True
        while more:
            message = await receive()
            more = bool(message.get("more_body", False))
        await send({"type": "http.response.body", "body": b"partial", "more_body": False})

    @staticmethod
    async def _read_before_respond(scope: Scope, receive: Receive, send: Send) -> None:
        """Drain the whole body first, then respond (every real route's shape)."""
        more = True
        while more:
            message = await receive()
            more = bool(message.get("more_body", False))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    # -- Tests -----------------------------------------------------------

    def test_no_second_start_when_response_already_began(self) -> None:
        # 40 + 40 = 80 bytes > the 64-byte effective cap → the tally trips
        # on the second chunk, AFTER the app has already sent its 200
        # start. The middleware must NOT then send its own 413 start:
        # that would be a second http.response.start (protocol violation).
        oversize = [(b"x" * 40, True), (b"x" * 40, True)]
        sent = self._run(self._respond_before_body, oversize)

        starts = self._starts(sent)
        assert len(starts) == 1, (
            f"expected exactly one http.response.start, got {len(starts)} "
            "(a second one is an ASGI protocol violation)"
        )
        # The single start is the app's own 200 — the middleware left the
        # in-flight response alone rather than corrupting it with a 413.
        assert starts[0]["status"] == 200

    def test_read_before_respond_still_413s(self) -> None:
        # The reachable path: the app reads the whole body before
        # responding, so the tally trips before any http.response.start.
        # The middleware suppresses the app's 200 and issues its own 413.
        oversize = [(b"x" * 40, True), (b"x" * 40, True)]
        sent = self._run(self._read_before_respond, oversize)

        starts = self._starts(sent)
        assert len(starts) == 1
        assert starts[0]["status"] == 413
        body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
        assert b"payload_too_large" in body

    def test_under_limit_stream_passes_through(self) -> None:
        # A body under the cap is forwarded untouched — the app's 200 and
        # its body reach the client, no 413.
        under = [(b"x" * 20, True), (b"x" * 20, False)]
        sent = self._run(self._read_before_respond, under)

        starts = self._starts(sent)
        assert len(starts) == 1
        assert starts[0]["status"] == 200


@pytest.mark.security
class TestErrorEnvelope:
    def test_404_returns_envelope_shape(self, api_client: TestClient) -> None:
        response = api_client.get("/api/does-not-exist")
        body = response.json()
        # Strict envelope: error / message / details / hint — no leaked
        # FastAPI ``detail`` shape.
        assert set(body.keys()) >= {"error", "message"}
        assert "detail" not in body

    def test_validation_error_returns_envelope(self, api_client: TestClient) -> None:
        # POST /api/session does not require a body, so trigger a
        # validation path indirectly: GET on a POST-only route returns
        # 405 with the envelope.
        response = api_client.get("/api/session")
        assert response.status_code == 405
        body = response.json()
        assert "error" in body and "message" in body


@pytest.mark.security
class TestOpenAPIGating:
    def test_docs_available_without_api_key(self, api_client: TestClient) -> None:
        response = api_client.get("/docs")
        # With no API key configured, the docs endpoints stay exposed.
        assert response.status_code == 200

    def test_openapi_schema_available_without_api_key(self, api_client: TestClient) -> None:
        response = api_client.get("/openapi.json")
        assert response.status_code == 200
        assert "openapi" in response.json()

    def test_docs_disabled_when_api_key_set(self, api_client_factory) -> None:
        client = api_client_factory(API_KEY="rotated-test-key")  # pragma: allowlist secret
        # Docs / Redoc / openapi.json all 404 when auth is enabled.
        for path in ("/docs", "/redoc", "/openapi.json"):
            response = client.get(path)
            assert response.status_code == 404, f"{path} leaked when API_KEY set"


@pytest.mark.security
class TestRateLimitOnSessionRoute:
    def test_session_route_is_rate_limited(self, api_client_factory) -> None:
        # Make the bucket small so we can hit it deterministically.
        client = api_client_factory(API_RATE_LIMIT_GENERAL="120")
        # Mint many times — the dedicated session bucket has a lower cap
        # than ``general`` (default 20).  Exceed it deliberately.
        last = None
        for _ in range(40):
            last = client.post("/api/session")
            if last.status_code == 429:
                break
        assert last is not None
        assert last.status_code == 429
        body = last.json()
        assert body["error"] == "rate_limited"
        assert "Retry-After" in last.headers


@pytest.mark.security
class TestMiddlewareOrdering:
    def test_security_headers_added_to_rate_limited_response(self, api_client_factory) -> None:
        # Rate-limit the session bucket and assert the 429 still carries
        # the full security header set — proves SecurityHeadersMiddleware
        # sits OUTSIDE RateLimitMiddleware.
        client = api_client_factory()
        last = None
        for _ in range(40):
            last = client.post("/api/session")
            if last.status_code == 429:
                break
        assert last is not None and last.status_code == 429
        assert last.headers.get("x-frame-options") == "DENY"
        assert last.headers.get("content-security-policy") is not None

    def test_correlation_id_present_on_rate_limited_response(self, api_client_factory) -> None:
        # The pure-ASGI RateLimitMiddleware emits its 429 via a
        # JSONResponse whose http.response.start still flows back out
        # through the outermost CorrelationIdMiddleware. The rejection
        # MUST carry the bound X-Request-ID.
        client = api_client_factory()
        last = None
        for _ in range(40):
            last = client.post("/api/session")
            if last.status_code == 429:
                break
        assert last is not None and last.status_code == 429
        assert last.headers.get("x-request-id") is not None


@pytest.mark.security
class TestPureAsgiMiddlewareStack:
    """Every bespoke middleware is pure ASGI, and the standalone
    insecure-transport layer has been folded into RateLimitMiddleware."""

    def test_no_middleware_subclasses_basehttpmiddleware(self) -> None:
        from starlette.middleware.base import BaseHTTPMiddleware

        from sec_generative_search.api import middleware as mw

        for cls in (
            mw.CorrelationIdMiddleware,
            mw.SecurityHeadersMiddleware,
            mw.ContentSizeLimitMiddleware,
            mw.RateLimitMiddleware,
        ):
            assert not issubclass(cls, BaseHTTPMiddleware), cls.__name__
            # A pure-ASGI middleware exposes the wrapped app and the
            # (scope, receive, send) callable — never a `dispatch`.
            assert not hasattr(cls, "dispatch"), cls.__name__

    def test_standalone_insecure_transport_middleware_removed(self) -> None:
        # The class was folded into RateLimitMiddleware; leaving a stale
        # export around would tempt a future re-wire of a dead layer.
        from sec_generative_search.api import middleware as mw

        assert not hasattr(mw, "InsecureTransportWarningMiddleware")
        assert "InsecureTransportWarningMiddleware" not in mw.__all__


class TestSinglePolicyResolution:
    """The per-route policy table is scanned once per request and
    shared between the rate limiter and the content-size limiter via the
    scope cache."""

    def test_resolve_policy_called_once_per_request(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sec_generative_search.api import middleware as mw
        from sec_generative_search.api.policies import resolve_policy as real_resolve

        calls: list[tuple[str, str]] = []

        def counting(path: str, method: str):
            calls.append((path, method))
            return real_resolve(path, method)

        monkeypatch.setattr(mw, "resolve_policy", counting)

        response = api_client.get("/api/health")
        assert response.status_code == 200
        # The scope cache collapses the scan count to 1.
        assert len(calls) == 1, calls


@pytest.mark.security
class TestInsecureTransportWarning:
    """The one-shot insecure-transport warning, folded into the rate
    limiter. It fires exactly once when authenticated traffic arrives
    over plain HTTP, and stays silent otherwise."""

    @staticmethod
    def _spy_warnings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        from sec_generative_search.api import middleware as mw

        recorded: list[str] = []

        def spy(msg, *args, **kwargs) -> None:
            recorded.append(str(msg))

        monkeypatch.setattr(mw.logger, "warning", spy)
        return recorded

    def test_warns_once_over_plain_http_when_auth_enabled(
        self, api_client_factory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorded = self._spy_warnings(monkeypatch)
        client = api_client_factory(API_KEY="rotated-test-key")  # pragma: allowlist secret
        for _ in range(3):
            client.get("/api/health", headers={"X-Forwarded-Proto": "http"})
        insecure = [m for m in recorded if "Insecure transport detected" in m]
        assert len(insecure) == 1  # one-shot across the three probes

    def test_silent_over_https(self, api_client_factory, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded = self._spy_warnings(monkeypatch)
        client = api_client_factory(API_KEY="rotated-test-key")  # pragma: allowlist secret
        client.get("/api/health", headers={"X-Forwarded-Proto": "https"})
        assert not [m for m in recorded if "Insecure transport detected" in m]

    def test_silent_when_no_auth_configured(
        self, api_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Local dev: no API key / admin key / EDGAR-session requirement,
        # so plain HTTP is expected and MUST NOT warn.
        recorded = self._spy_warnings(monkeypatch)
        api_client.get("/api/health", headers={"X-Forwarded-Proto": "http"})
        assert not [m for m in recorded if "Insecure transport detected" in m]
