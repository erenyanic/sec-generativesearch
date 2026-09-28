"""Per-route rate-limit + request-size policy table.

Single source of truth consulted by both
:class:`~sec_generative_search.api.middleware.RateLimitMiddleware`
(via ``_rate_category_for_scope``) and
:class:`~sec_generative_search.api.middleware.ContentSizeLimitMiddleware`
(per-route body cap).

Design notes
------------

- The table is **code, not configuration**.  A misconfigured body
  cap on a destructive route is a worse footgun than the absence of
  an env knob, and operator-tunable caps would diverge between
  deployments under no-one's audit.  The schema-layer bounds in
  ``api/schemas.py`` defend content semantics; this defends memory
  before any handler runs.
- Per-route caps are **upper bounds**, not the operator-facing rate
  limits — those still live in ``ApiSettings.rate_limit_*``.  The
  table maps each route to a category name, and the rate limiter
  reads the category's bucket from settings.
- Lookup is **first-hit-wins** over an ordered tuple.  More-specific
  prefixes (e.g. ``/api/filings/delete-by-ids``) MUST appear before
  their parent (``/api/filings``).
- Path matching uses ``==`` *or* ``startswith(prefix + "/")``.  A
  bare ``startswith`` would let ``/api/searcher`` collide with the
  ``/api/search`` entry, leaking a stricter cap onto an unrelated
  route.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

__all__ = [
    "DEFAULT_POLICY",
    "ROUTE_POLICIES",
    "RoutePolicy",
    "all_rate_categories",
    "resolve_policy",
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoutePolicy:
    """Effective per-route policy.

    ``rate_category=None`` means the route is exempt from the rate
    limiter (currently only ``/api/health``).  ``max_body_bytes`` is
    inclusive of headers' declared ``Content-Length`` and of the
    running tally of streamed body bytes.
    """

    rate_category: str | None
    max_body_bytes: int


# ---------------------------------------------------------------------------
# Bounds catalogue
# ---------------------------------------------------------------------------


_KIB = 1024
_MIB = 1024 * 1024


# Fallback for any path not matched by the table.  The ``general``
# rate bucket and the historical 1 MiB cap are deliberately preserved
# so unmatched future routes inherit the same envelope today's code
# enforces.
DEFAULT_POLICY = RoutePolicy(rate_category="general", max_body_bytes=1 * _MIB)


# Each entry: ``(path_prefix, method or None, policy)``.
#
# Method ``None`` matches any HTTP verb on that prefix.  Order is
# load-bearing: ``/api/filings/delete-by-ids`` MUST come before
# ``/api/filings`` so the longer prefix wins, and the (POST, /api/rag/*)
# entries must cover plan / query / stream individually because each
# carries a different worst-case payload envelope.
ROUTE_POLICIES: tuple[tuple[str, str | None, RoutePolicy], ...] = (
    # Health is unauthenticated; the rate limiter is skipped so that a
    # liveness probe storm cannot trigger 429 on the very signal that
    # tells the orchestrator the API is alive.  The body cap still
    # applies and rejects spurious large declared payloads.
    ("/api/health", None, RoutePolicy(rate_category=None, max_body_bytes=1 * _KIB)),
    # Session lifecycle.  ``/api/session/edgar`` carries a name+email
    # body (rejected at the schema layer for control characters); the
    # mint and logout routes carry no body at all.  Both share the
    # ``session`` rate bucket so a probe cannot rotate cookies in a
    # tight loop.
    (
        "/api/session/edgar",
        None,
        RoutePolicy(rate_category="session", max_body_bytes=4 * _KIB),
    ),
    (
        "/api/session",
        None,
        RoutePolicy(rate_category="session", max_body_bytes=1 * _KIB),
    ),
    # User-tier authentication. ``login`` + ``login-params`` are the
    # brute-force surface — kept under a tight per-IP bucket (default
    # 5 rpm) AND a per-username sliding window enforced inside the
    # route handler (default 3 rpm). Both must allow the request.
    # ``password`` and ``session`` (DELETE / sign-out) are authenticated
    # follow-ups and ride the ``session`` bucket — they are not the
    # online-guess surface.
    (
        "/api/auth/login-params",
        "GET",
        RoutePolicy(rate_category="login", max_body_bytes=1 * _KIB),
    ),
    (
        "/api/auth/login",
        "POST",
        RoutePolicy(rate_category="login", max_body_bytes=1 * _KIB),
    ),
    (
        "/api/auth/password",
        "POST",
        RoutePolicy(rate_category="session", max_body_bytes=4 * _KIB),
    ),
    (
        "/api/auth/session",
        "DELETE",
        RoutePolicy(rate_category="session", max_body_bytes=1 * _KIB),
    ),
    # Admin user management. ``delete`` bucket reuses the
    # destructive-tier rate so a leaking admin key cannot churn user
    # rows in a tight loop.
    (
        "/api/admin/users",
        None,
        RoutePolicy(rate_category="delete", max_body_bytes=4 * _KIB),
    ),
    # Provider-key validation: bound generously above the worst-case
    # envelope (provider slug + bearer key + optional model name).
    (
        "/api/providers/validate",
        "POST",
        RoutePolicy(rate_category="validate", max_body_bytes=16 * _KIB),
    ),
    # Catalogue refresh trigger: admin-gated POST with no body. It drives an
    # outbound HTTPS fetch, so it rides the destructive ``delete`` bucket — a
    # leaking admin key must not be able to hammer the upstream metadata source
    # in a tight loop. MUST precede the ``/api/providers`` GET entry so the
    # longer prefix wins. 1 KiB cap defends against declared-Content-Length
    # probes against a body-free route.
    (
        "/api/providers/catalogue",
        "POST",
        RoutePolicy(rate_category="delete", max_body_bytes=1 * _KIB),
    ),
    # Provider catalogue: read-tier GET with no body. Tight cap defends
    # against declared-Content-Length probes against an authenticated
    # route. MUST come after ``/api/providers/validate`` (which pins the
    # POST verb) so the longer path keeps priority on its own method.
    (
        "/api/providers",
        "GET",
        RoutePolicy(rate_category="general", max_body_bytes=1 * _KIB),
    ),
    # Retrieval-only search.  Query is ≤ 1024 chars at the schema
    # layer; list filters are ≤ 50 entries.  32 KiB clears that with
    # margin for client whitespace / quoting.
    (
        "/api/search",
        "POST",
        RoutePolicy(rate_category="search", max_body_bytes=32 * _KIB),
    ),
    # RAG endpoints: ``/plan`` carries only the raw query envelope;
    # ``/query`` and ``/stream`` carry the full ``QueryPlanSchema``.
    (
        "/api/rag/plan",
        "POST",
        RoutePolicy(rate_category="rag", max_body_bytes=16 * _KIB),
    ),
    (
        "/api/rag/query",
        "POST",
        RoutePolicy(rate_category="rag", max_body_bytes=64 * _KIB),
    ),
    (
        "/api/rag/stream",
        "POST",
        RoutePolicy(rate_category="rag", max_body_bytes=64 * _KIB),
    ),
    # Ingestion surface; rate bucket preserved.
    (
        "/api/ingest",
        "POST",
        RoutePolicy(rate_category="ingest", max_body_bytes=64 * _KIB),
    ),
    # Filings: destructive POSTs (longer prefix MUST come before
    # ``/api/filings``).  Bulk-delete carries a filter envelope only;
    # delete-by-ids accepts up to 500 SEC accession numbers
    # (~26 chars each + JSON overhead).
    (
        "/api/filings/delete-by-ids",
        "POST",
        RoutePolicy(rate_category="delete", max_body_bytes=64 * _KIB),
    ),
    (
        "/api/filings/bulk-delete",
        "POST",
        RoutePolicy(rate_category="delete", max_body_bytes=16 * _KIB),
    ),
    # Filings: DELETE verb on any sub-path (single delete + clear-all).
    (
        "/api/filings",
        "DELETE",
        RoutePolicy(rate_category="delete", max_body_bytes=1 * _KIB),
    ),
    # Filings: read tier (list + detail).
    (
        "/api/filings",
        "GET",
        RoutePolicy(rate_category="general", max_body_bytes=1 * _KIB),
    ),
    # Status snapshot.
    (
        "/api/status",
        None,
        RoutePolicy(rate_category="general", max_body_bytes=1 * _KIB),
    ),
    # Embedder resource introspection.  Read-tier GET with no body —
    # the 1 KiB cap defends against declared-Content-Length probes
    # against an authenticated route.
    (
        "/api/resources",
        None,
        RoutePolicy(rate_category="general", max_body_bytes=1 * _KIB),
    ),
    # Metrics exposition (admin-gated GET, no body). 1 KiB cap defends
    # against declared-Content-Length probes; the route auth is the real
    # control. Shares the ``general`` bucket — a scraper polls on a fixed
    # cadence well under that ceiling.
    (
        "/api/metrics",
        None,
        RoutePolicy(rate_category="general", max_body_bytes=1 * _KIB),
    ),
)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def resolve_policy(path: str, method: str) -> RoutePolicy:
    """Return the effective policy for a request.

    First-hit-wins over :data:`ROUTE_POLICIES`.  Unmatched paths
    inherit :data:`DEFAULT_POLICY` (``general`` rate bucket, 1 MiB
    body cap) — that includes Swagger / OpenAPI artefacts and any
    future route added before its policy entry lands.
    """
    for prefix, allowed_method, policy in ROUTE_POLICIES:
        if not _path_matches(path, prefix):
            continue
        if allowed_method is not None and method != allowed_method:
            continue
        return policy
    return DEFAULT_POLICY


def _path_matches(path: str, prefix: str) -> bool:
    """Return True if ``path`` is exactly ``prefix`` or strictly below it.

    Trailing-slash handling: ``/api/filings/`` matches the
    ``/api/filings`` entry because Starlette routes both forms to the
    same handler.  The ``+ "/"`` boundary check prevents
    ``/api/searcher`` from colliding with the ``/api/search`` entry.
    """
    if path == prefix or path == prefix + "/":
        return True
    return path.startswith(prefix + "/")


def all_rate_categories() -> Iterable[str]:
    """Audit helper: yield every non-``None`` rate category in use.

    Used by tests to confirm every category named in the table has a
    matching ``ApiSettings.rate_limit_*`` knob.
    """
    seen: set[str] = set()
    for _, _, policy in ROUTE_POLICIES:
        if policy.rate_category is not None and policy.rate_category not in seen:
            seen.add(policy.rate_category)
            yield policy.rate_category
    if DEFAULT_POLICY.rate_category is not None and DEFAULT_POLICY.rate_category not in seen:
        seen.add(DEFAULT_POLICY.rate_category)
        yield DEFAULT_POLICY.rate_category
