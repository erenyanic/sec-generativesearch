"""Auth-boundary and admin-operation route sweep.

The per-surface auth tests already exercise the *helpers*:

    - :mod:`tests.api.test_auth_tiers` walks the ``API_KEY`` x
      ``API_ADMIN_KEY`` matrix against ``admin_route_dependencies()`` on a
      synthetic sentinel route.
    - :mod:`tests.api.test_dependencies` walks the resolver chain.

What no existing test does is assert the invariant **across the real,
registered route surface of the live app**. This module closes that gap
by catching a future route wired the wrong way:

    1. A new business route accidentally shipped without
       ``verify_api_key`` (silent unauthenticated surface).
    2. A new destructive route wired with ``verify_admin_key`` *alone*
       — the documented footgun.  Because ``verify_admin_key`` precedes
       no API-key check, an ``X-Admin-Key``-only request would surface
       ``403`` instead of ``401``, and (worse) an unauthenticated request
       would pass the read tier it never had.
    3. A known destructive route silently down-tiered out of the admin
       gate.

The sweep enumerates the live app's registered routes and classifies each
one's auth tier by **observable HTTP behaviour**, not by introspecting
FastAPI's internal dependency structures — the latter is not stable across
FastAPI versions (where router-level ``dependencies=[...]`` surface moved
between releases) and silently misclassified every route under CI's newer
pin. Behaviour is the ground truth and what actually protects the route.
A small probe (:class:`TestAdminKeyOnlyProbe`) cross-checks the same
invariant on a curated set of routes.
"""

from __future__ import annotations

import re

import pytest
from starlette.routing import Route

# ---------------------------------------------------------------------------
# Route-tier inventory
# ---------------------------------------------------------------------------

# The complete set of ``/api`` routes that are deliberately reachable
# **without** an API key.  Each entry is justified — a new line here is a
# security decision that must be made consciously, which is the whole
# point of pinning the set exactly.
#
#   - /api/health                 liveness probe; tiny body; rate-exempt.
#   - /api/session*               session-cookie lifecycle.  The cookie is
#                                 the bootstrap an anonymous browser needs
#                                 *before* it can carry any other tier, so
#                                 it cannot itself be API-key gated.
#   - /api/auth/*                 user-tier auth. These carry their own
#                                 ``auth_proof`` / enrolment-token /
#                                 brute-force controls and are not part
#                                 of the infrastructure ``API_KEY`` tier.
_EXPECTED_OPEN_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/api/health"),
        ("POST", "/api/session"),
        ("POST", "/api/session/logout"),
        ("POST", "/api/session/edgar"),
        ("DELETE", "/api/session/edgar"),
        ("GET", "/api/auth/login-params"),
        ("POST", "/api/auth/login"),
        ("POST", "/api/auth/enrol"),
        ("POST", "/api/auth/password"),
        ("DELETE", "/api/auth/session"),
        ("POST", "/api/auth/vault"),
    }
)

# Destructive / privileged routes that must stay behind the admin tier.
# A regression that down-tiers any of these is caught by the inventory.
_EXPECTED_ADMIN_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/api/admin/users"),
        ("DELETE", "/api/admin/users/{user_id}"),
        ("POST", "/api/admin/users/{user_id}/unlock"),
        ("GET", "/api/providers/health"),
        ("POST", "/api/providers/catalogue/refresh"),
        ("DELETE", "/api/filings/{accession}"),
        ("POST", "/api/filings/delete-by-ids"),
        ("POST", "/api/filings/bulk-delete"),
        ("DELETE", "/api/filings/"),
        ("GET", "/api/metrics/"),
    }
)

# Routes gated by a dedicated shared-secret token instead of the API /
# admin keys — a deliberate, reviewed exception to rule A. Adding one is a
# security decision on the same footing as widening the open allow-list.
#   - POST /api/admin/demo-reset: the scheduled demo reset (F27). Cloud
#     Scheduler holds ``API_DEMO_RESET_TOKEN``, a secret that can do
#     nothing else; the API / admin keys must NOT open it.
_EXPECTED_TOKEN_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/api/admin/demo-reset"),
    }
)


# Generous per-category rate-limit ceilings so the multi-route sweep (≈2
# requests per route) never trips a limiter and turns into a flaky 429.
_RELAXED_RATE_LIMITS = {
    "API_RATE_LIMIT_SEARCH": "100000",
    "API_RATE_LIMIT_INGEST": "100000",
    "API_RATE_LIMIT_DELETE": "100000",
    "API_RATE_LIMIT_GENERAL": "100000",
    "API_RATE_LIMIT_RAG": "100000",
    "API_RATE_LIMIT_VALIDATE": "100000",
    "API_RATE_LIMIT_VALIDATE_PER_SESSION": "100000",
    "API_RATE_LIMIT_SESSION": "100000",
    "API_RATE_LIMIT_LOGIN": "100000",
    "API_RATE_LIMIT_LOGIN_PER_USERNAME": "100000",
}

_SWEEP_API_KEY = "sweep-shared-api-key"  # pragma: allowlist secret
_SWEEP_ADMIN_KEY = "sweep-shared-admin-key"  # pragma: allowlist secret


def _api_routes(app) -> list[Route]:
    return [
        r for r in app.routes if isinstance(r, Route) and r.path.startswith("/api") and r.methods
    ]


def _concrete_path(path: str) -> str:
    """Fill path params with a benign placeholder.

    The auth gate is a dependency that runs *before* path-parameter
    validation, so the placeholder value is irrelevant to classification —
    a gated route rejects with 401/403 long before the segment is parsed.
    """
    return re.sub(r"\{[^}]+\}", "x", path)


# ---------------------------------------------------------------------------
# 1. Auth-tier inventory — behavioural
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestRouteAuthTierInventory:
    """Every ``/api`` route sits in its intended auth tier.

    Classification is by HTTP behaviour against the live app with both
    ``API_KEY`` and ``API_ADMIN_KEY`` configured, so the sweep tracks the
    app as it grows and is immune to FastAPI-internal restructuring.

    Every assertion relies ONLY on the gate-level signals that fire inside
    the auth dependencies *before* any route handler runs — a no-header
    request to a gated route returns ``401`` and an api-key-only request to
    an admin route returns ``403``, neither of which touches handler code
    or ``app.state``. That keeps the sweep robust (no 500s from the test's
    minimal app stubs) and lets it use the standard test client directly.

    Route ENUMERATION reads the raw ``api_app`` FastAPI instance, never
    ``TestClient.app`` — the latter does not reliably surface every router
    under CI's newer Starlette/httpx, dropping the open routers from its
    ``.routes`` view. HTTP REQUESTS go through ``sweep_client`` (both keys
    configured). The two fixtures build the same route table, so mixing
    them is sound.
    """

    @pytest.fixture
    def sweep_client(self, api_client_factory):
        return api_client_factory(
            API_KEY=_SWEEP_API_KEY,
            API_ADMIN_KEY=_SWEEP_ADMIN_KEY,
            **_RELAXED_RATE_LIMITS,
        )

    def _route_keys(self, app) -> set[tuple[str, str]]:
        keys: set[tuple[str, str]] = set()
        for r in _api_routes(app):
            for method in r.methods or set():
                if method not in {"HEAD", "OPTIONS"}:
                    keys.add((method, r.path))
        return keys

    def test_no_unlisted_route_is_unauthenticated(self, api_app, sweep_client) -> None:
        """No ``/api`` route outside the reviewed open allow-list is anonymous.

        For every route NOT on the allow-list, a no-header request MUST be
        rejected by the API-key gate (``401``). A new business route that
        forgot ``verify_api_key`` would instead admit the anonymous request
        (any non-401 status) and is surfaced here. This is the load-bearing
        direction: a route silently becoming *open* is the regression we
        must catch. (Auth gates reject before the handler, so no request in
        this loop reaches route logic.)
        """
        offenders: dict[tuple[str, str], int] = {}
        for method, path in self._route_keys(api_app):
            if (method, path) in _EXPECTED_OPEN_ROUTES:
                continue
            status = sweep_client.request(method, _concrete_path(path)).status_code
            # A gated route ALWAYS returns 401 to a no-header request (the
            # gate runs before the handler). 404/405 mean the concretised
            # path did not resolve to this route on this client/Starlette
            # version — a routing artifact, not an unauthenticated surface —
            # so they are ignored. Anything else (200/422/…) means the route
            # admitted the anonymous caller: a real missing gate.
            if status not in {401, 404, 405}:
                offenders[(method, path)] = status
        assert not offenders, (
            "/api routes outside the reviewed unauthenticated allow-list that "
            f"do not require an API key (no-header status not in 401/404/405): {offenders}"
        )

    def test_open_allowlist_routes_are_registered(self, sweep_client) -> None:
        """Each open allow-list entry maps to a live route (no typos / removals).

        Checked over HTTP rather than by enumerating ``.routes`` — a missing
        or mistyped entry resolves to ``404`` (unknown path) or ``405``
        (wrong method); a real route returns anything else (200/201/401/422
        …). This keeps the hygiene check independent of route-enumeration
        quirks across client/Starlette versions.
        """
        missing: dict[tuple[str, str], int] = {}
        for method, path in _EXPECTED_OPEN_ROUTES:
            status = sweep_client.request(method, _concrete_path(path)).status_code
            if status in {404, 405}:
                missing[(method, path)] = status
        assert not missing, (
            f"open allow-list entries that resolve to no live route (404/405): {missing}"
        )

    def test_known_destructive_routes_are_admin_tier(self, sweep_client) -> None:
        """Each reviewed destructive route sits behind the admin gate.

        Gate-level signature: no-header → ``401`` (API tier first) AND
        api-key-only → ``403`` (admin gate fires once the read tier clears).
        Both fire inside the auth dependencies, before the handler.
        """
        misgated: dict[tuple[str, str], tuple[int, int]] = {}
        for method, path in _EXPECTED_ADMIN_ROUTES:
            cp = _concrete_path(path)
            no_header = sweep_client.request(method, cp).status_code
            api_only = sweep_client.request(
                method, cp, headers={"X-API-Key": _SWEEP_API_KEY}
            ).status_code
            if not (no_header == 401 and api_only == 403):
                misgated[(method, path)] = (no_header, api_only)
        assert not misgated, (
            "destructive routes not behind the admin gate "
            f"(want no-header=401, api-key-only=403): {misgated}"
        )

    def test_token_routes_refuse_the_api_and_admin_keys(self, sweep_client) -> None:
        """A token-gated route is 401 with no header AND with both keys — the
        admin credentials are not a way in (least privilege both ways)."""
        for method, path in _EXPECTED_TOKEN_ROUTES:
            cp = _concrete_path(path)
            no_header = sweep_client.request(method, cp).status_code
            both_keys = sweep_client.request(
                method,
                cp,
                headers={"X-API-Key": _SWEEP_API_KEY, "X-Admin-Key": _SWEEP_ADMIN_KEY},
            ).status_code
            assert (no_header, both_keys) == (401, 401), (
                f"{method} {path}: token-gated route must refuse both no-header and "
                f"admin-key requests with 401, got {(no_header, both_keys)}"
            )

    def test_every_admin_path_is_classified(self, api_app) -> None:
        """Every ``/api/admin/*`` route is reviewed as admin-tier or token-gated."""
        unclassified = {
            (method, path)
            for method, path in self._route_keys(api_app)
            if path.startswith("/api/admin/")
            and (method, path) not in _EXPECTED_ADMIN_ROUTES | _EXPECTED_TOKEN_ROUTES
        }
        assert not unclassified, f"unreviewed /api/admin routes: {sorted(unclassified)}"

    def test_every_admin_route_runs_api_key_before_admin_key(self, sweep_client) -> None:
        """The footgun guard, behaviourally.

        For an admin route, a no-header request must surface ``401`` (the
        API tier runs first), never ``403``. The status flips to ``403``
        iff the admin gate is wired ahead of — or instead of — the
        API-key gate, which is exactly the "never wire ``verify_admin_key``
        alone" footgun.
        """
        for method, path in _EXPECTED_ADMIN_ROUTES:
            status = sweep_client.request(method, _concrete_path(path)).status_code
            assert status == 401, (
                f"{method} {path}: no-header request must surface 401 (API tier "
                f"first), got {status} — admin gate wired ahead of the API gate?"
            )


# ---------------------------------------------------------------------------
# 2. Live behaviour probe
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestAdminKeyOnlyProbe:
    """With both keys configured, an admin-key-only request is ``401``.

    The structural sweep proves the *wiring*; this proves the *behaviour*
    so the two cannot silently diverge. Routes chosen carry no required
    body, so the auth dependency is the only gate and the status is an
    unambiguous 401/403 (never a 422 from body validation).
    """

    _API_KEY = "shared-team-key-PROBE"  # pragma: allowlist secret
    _ADMIN_KEY = "secret-admin-key-PROBE"  # pragma: allowlist secret

    # (method, path) — body-free admin routes spanning metrics,
    # provider-health, filings, and users.
    _PROBE_ROUTES = (
        ("GET", "/api/metrics/"),
        ("GET", "/api/providers/health"),
        ("DELETE", "/api/filings/0000320193-23-000077"),
        ("POST", "/api/admin/users/some-user-id/unlock"),
    )

    @pytest.fixture
    def client(self, api_client_factory):
        return api_client_factory(API_KEY=self._API_KEY, API_ADMIN_KEY=self._ADMIN_KEY)

    @pytest.mark.parametrize(("method", "path"), _PROBE_ROUTES)
    def test_no_header_returns_401_not_403(self, client, method: str, path: str) -> None:
        # Neither header supplied.  The API tier runs first and rejects
        # with 401 — the caller must never learn the admin gate even
        # exists until they clear the read tier.  This is the case whose
        # status code *flips* (401 -> 403) if the admin gate is ever
        # ordered ahead of the API-key gate, so it is the behavioural
        # mirror of the structural ordering lock above.
        response = client.request(method, path)
        assert response.status_code == 401, (
            f"{method} {path}: no-header request must surface 401 (API tier first), "
            f"got {response.status_code}"
        )
        assert response.json()["error"] == "unauthorised"

    @pytest.mark.parametrize(("method", "path"), _PROBE_ROUTES)
    def test_admin_key_only_returns_401_not_403(self, client, method: str, path: str) -> None:
        # X-Admin-Key supplied, X-API-Key omitted.
        response = client.request(method, path, headers={"X-Admin-Key": self._ADMIN_KEY})
        assert response.status_code == 401, (
            f"{method} {path}: admin-key-only must surface 401 (API tier first), "
            f"got {response.status_code}"
        )
        assert response.json()["error"] == "unauthorised"

    @pytest.mark.parametrize(("method", "path"), _PROBE_ROUTES)
    def test_api_key_only_returns_403_admin_required(self, client, method: str, path: str) -> None:
        # X-API-Key supplied, X-Admin-Key omitted.
        response = client.request(method, path, headers={"X-API-Key": self._API_KEY})
        assert response.status_code == 403, (
            f"{method} {path}: api-key-only must surface 403 admin_required, "
            f"got {response.status_code}"
        )
        assert response.json()["error"] == "admin_required"


# ---------------------------------------------------------------------------
# 3. Secret handling
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestConfiguredSecretNeverInResponseBody:
    """No configured infrastructure secret is ever echoed to the wire.

    A denial or validation envelope must never carry the server's
    configured ``API_KEY`` / ``API_ADMIN_KEY``, nor the wrong credential
    the caller supplied.
    """

    _API_KEY = "configured-api-key-SENTINEL-A"  # pragma: allowlist secret
    _ADMIN_KEY = "configured-admin-key-SENTINEL-B"  # pragma: allowlist secret
    _WRONG_KEY = "supplied-wrong-key-SENTINEL-C"  # pragma: allowlist secret
    _PROVIDER_KEY = "sk-supplied-provider-SENTINEL-D"  # pragma: allowlist secret
    _RESET_TOKEN = "configured-demo-reset-token-SENTINEL-E"  # pragma: allowlist secret

    @pytest.fixture
    def client(self, api_client_factory):
        return api_client_factory(
            API_KEY=self._API_KEY,
            API_ADMIN_KEY=self._ADMIN_KEY,
            API_DEMO_MODE="true",
            API_DEMO_RESET_TOKEN=self._RESET_TOKEN,
        )

    def test_no_configured_or_supplied_secret_appears_in_any_body(self, client) -> None:
        bodies: list[str] = []

        # 401: missing key on a read-tier route.
        bodies.append(client.get("/api/status/").text)
        # 401: wrong key on a read-tier route.
        bodies.append(client.get("/api/status/", headers={"X-API-Key": self._WRONG_KEY}).text)
        # 403: api-key valid, admin key missing on a destructive route.
        bodies.append(
            client.delete(
                "/api/filings/0000320193-23-000077",
                headers={"X-API-Key": self._API_KEY},
            ).text
        )
        # 403: api-key valid, admin key wrong on a destructive route.
        bodies.append(
            client.delete(
                "/api/filings/0000320193-23-000077",
                headers={"X-API-Key": self._API_KEY, "X-Admin-Key": self._WRONG_KEY},
            ).text
        )
        # Validate route: a supplied provider key must never be echoed back.
        bodies.append(
            client.post(
                "/api/providers/validate",
                headers={"X-API-Key": self._API_KEY},
                json={"provider": "openai", "api_key": self._PROVIDER_KEY},
            ).text
        )

        # 401: the demo-reset route with a wrong / admin credential.
        bodies.append(
            client.post(
                "/api/admin/demo-reset", headers={"X-Demo-Reset-Token": self._WRONG_KEY}
            ).text
        )
        bodies.append(
            client.post(
                "/api/admin/demo-reset",
                headers={"X-API-Key": self._API_KEY, "X-Admin-Key": self._ADMIN_KEY},
            ).text
        )

        joined = "\n".join(bodies)
        for secret in (
            self._API_KEY,
            self._ADMIN_KEY,
            self._WRONG_KEY,
            self._PROVIDER_KEY,
            self._RESET_TOKEN,
        ):
            assert secret not in joined, f"secret leaked into a response body: {secret!r}"
