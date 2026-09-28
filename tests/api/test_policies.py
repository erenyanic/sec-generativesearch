"""Tests for the per-route rate-limit + request-size policy table.

Covers:

- :func:`resolve_policy` returns the right rate category and body
  cap for every shipped route.
- The rate limiter's scope-level classifier (``_rate_category_for_scope``)
  keeps the category mapping every route test relies on.
- Every rate category named in the table has a backing
  ``ApiSettings.rate_limit_*`` knob — drift between the table and
  settings would silently disable a bucket.
"""

from __future__ import annotations

import pytest

from sec_generative_search.api.policies import (
    DEFAULT_POLICY,
    ROUTE_POLICIES,
    RoutePolicy,
    all_rate_categories,
    resolve_policy,
)
from sec_generative_search.config.settings import ApiSettings


class TestResolvePolicyHappyPaths:
    @pytest.mark.parametrize(
        ("path", "method", "expected_category"),
        [
            ("/api/health", "GET", None),
            ("/api/session", "POST", "session"),
            ("/api/session/logout", "POST", "session"),
            ("/api/session/edgar", "POST", "session"),
            ("/api/session/edgar", "DELETE", "session"),
            ("/api/providers/validate", "POST", "validate"),
            ("/api/search", "POST", "search"),
            ("/api/rag/plan", "POST", "rag"),
            ("/api/rag/query", "POST", "rag"),
            ("/api/rag/stream", "POST", "rag"),
            ("/api/ingest/add", "POST", "ingest"),
            ("/api/ingest/batch", "POST", "ingest"),
            ("/api/filings/delete-by-ids", "POST", "delete"),
            ("/api/filings/bulk-delete", "POST", "delete"),
            ("/api/filings/0000320193-23-000077", "DELETE", "delete"),
            ("/api/filings/", "GET", "general"),
            ("/api/filings/0000320193-23-000077", "GET", "general"),
            ("/api/filings/", "DELETE", "delete"),
            ("/api/status/", "GET", "general"),
        ],
    )
    def test_rate_category(self, path: str, method: str, expected_category: str | None) -> None:
        assert resolve_policy(path, method).rate_category == expected_category

    @pytest.mark.parametrize(
        ("path", "method", "min_bytes"),
        [
            # Each cap MUST be at least the worst-case schema bound for
            # that route's body. These assertions encode the design
            # invariants in policies.py — bumping a cap is fine, but
            # never below the schema-layer bound.
            ("/api/providers/validate", "POST", 1024),
            ("/api/search", "POST", 1024),
            ("/api/rag/plan", "POST", 1024),
            ("/api/rag/query", "POST", 1024),
            ("/api/rag/stream", "POST", 1024),
            # 500 accessions of 26 chars JSON-quoted ~= 13 KiB envelope
            ("/api/filings/delete-by-ids", "POST", 13 * 1024),
        ],
    )
    def test_max_body_bytes_above_schema_bound(
        self, path: str, method: str, min_bytes: int
    ) -> None:
        assert resolve_policy(path, method).max_body_bytes >= min_bytes


class TestPathBoundary:
    def test_prefix_does_not_leak_to_unrelated_route(self) -> None:
        # ``/api/searcher`` MUST NOT inherit the ``/api/search`` entry.
        # Falls through to the default policy.
        assert resolve_policy("/api/searcher", "POST") == DEFAULT_POLICY

    def test_trailing_slash_matches_bare_prefix(self) -> None:
        # Starlette routes ``/api/filings`` and ``/api/filings/`` to
        # the same handler — the table must not differentiate.
        bare = resolve_policy("/api/filings", "GET")
        trailing = resolve_policy("/api/filings/", "GET")
        assert bare == trailing

    def test_unknown_api_path_falls_back_to_default(self) -> None:
        # A future ``/api/foo`` route inherits the DEFAULT_POLICY so
        # an un-tabled surface is still rate-limited at "general"
        # and capped at 1 MiB rather than reaching a handler with
        # an unbounded body.
        assert resolve_policy("/api/foo", "POST") == DEFAULT_POLICY

    def test_non_api_path_falls_back_to_default(self) -> None:
        # /docs, /redoc, /openapi.json all inherit the default. The
        # rate-limit middleware separately exempts non-/api/ paths
        # in ``_rate_category_for_scope``.
        assert resolve_policy("/docs", "GET") == DEFAULT_POLICY
        assert resolve_policy("/openapi.json", "GET") == DEFAULT_POLICY


class TestMethodMatching:
    def test_method_specific_entry_only_matches_listed_verb(self) -> None:
        # /api/search is POST-only in the table.
        assert resolve_policy("/api/search", "POST").rate_category == "search"
        # GET on /api/search would fall through to the next matching
        # entry (or DEFAULT_POLICY) — there is no GET entry, so it
        # ends at the default.
        assert resolve_policy("/api/search", "GET") == DEFAULT_POLICY

    def test_method_none_matches_any_verb(self) -> None:
        # /api/health has method=None — every verb matches.
        assert resolve_policy("/api/health", "GET").rate_category is None
        assert resolve_policy("/api/health", "POST").rate_category is None


@pytest.mark.security
class TestPolicyTableInvariants:
    def test_destructive_routes_use_delete_bucket(self) -> None:
        # Every route that mutates filings must land on the ``delete``
        # bucket so the per-IP cap cannot be side-stepped by switching
        # method or path.
        destructive_routes = [
            ("/api/filings/0000320193-23-000077", "DELETE"),
            ("/api/filings/delete-by-ids", "POST"),
            ("/api/filings/bulk-delete", "POST"),
            ("/api/filings/", "DELETE"),
        ]
        for path, method in destructive_routes:
            policy = resolve_policy(path, method)
            assert policy.rate_category == "delete", (path, method)

    def test_health_is_never_rate_limited(self) -> None:
        # Liveness probe storms must never trip 429 — that would
        # cascade into the orchestrator marking the API dead.
        for method in ("GET", "POST", "OPTIONS", "HEAD"):
            assert resolve_policy("/api/health", method).rate_category is None

    def test_health_has_non_zero_body_cap(self) -> None:
        # Defensive: prevents declared-Content-Length probes against
        # the unauthenticated route.
        assert resolve_policy("/api/health", "GET").max_body_bytes > 0

    def test_default_policy_caps_at_1mib(self) -> None:
        # Historical bound — preserved so unmatched routes inherit
        # the same envelope today's code enforces.
        assert DEFAULT_POLICY.max_body_bytes == 1 * 1024 * 1024
        assert DEFAULT_POLICY.rate_category == "general"

    def test_every_category_has_settings_knob(self) -> None:
        # Drift between the table and ApiSettings would silently
        # disable a bucket: the middleware constructor reads
        # ``rate_limit_<category>``; a missing field would 0 the
        # bucket and let traffic through unmetered.
        api = ApiSettings()
        for category in all_rate_categories():
            attr = f"rate_limit_{category}"
            assert hasattr(api, attr), f"ApiSettings missing {attr} for table category {category!r}"

    def test_policy_table_entries_are_immutable(self) -> None:
        # Frozen dataclass + tuple table — defends against accidental
        # mutation from a route handler reading the policy and
        # mutating it for "just this one request".
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            DEFAULT_POLICY.max_body_bytes = 99  # type: ignore[misc]
        # Tuple immutability:
        with pytest.raises(TypeError):
            ROUTE_POLICIES[0] = ("/x", None, RoutePolicy("x", 1))  # type: ignore[index]


class TestClassifyPathShim:
    """The legacy classifier shim must keep returning the same values
    as before the refactor — there are seven existing tests across
    test_search / test_rag_* / test_filings that import it directly."""

    @pytest.mark.parametrize(
        ("path", "method", "expected"),
        [
            # Mirror every assertion in the existing test files.
            ("/api/search", "POST", "search"),
            ("/api/rag/query", "POST", "rag"),
            ("/api/rag/plan", "POST", "rag"),
            ("/api/rag/stream", "POST", "rag"),
            ("/api/filings/delete-by-ids", "POST", "delete"),
            ("/api/filings/bulk-delete", "POST", "delete"),
            ("/api/filings/", "GET", "general"),
            ("/api/filings/0000320193-23-000077", "DELETE", "delete"),
            ("/api/health", "GET", None),
            # Out-of-/api/ path must return None so docs / static are
            # unmetered.
            ("/docs", "GET", None),
        ],
    )
    def test_legacy_contract(self, path: str, method: str, expected: str | None) -> None:
        from sec_generative_search.api.middleware import _rate_category_for_scope

        assert (
            _rate_category_for_scope({"type": "http", "path": path, "method": method}) == expected
        )
