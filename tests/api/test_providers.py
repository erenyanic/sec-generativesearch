"""Security tests for the provider key-validation route.

Covers body validation, registry lookup failures, auth verdicts,
transient provider errors, response-body redaction, and per-IP versus
per-session rate limiting.

Strategy: the registry lookup is monkeypatched to return a stub
provider with a controllable ``validate_key`` so the tests never touch
a real upstream SDK.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sec_generative_search.core.exceptions import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.providers.registry import ProviderRegistry, ProviderSurface


class _StubProvider:
    """Minimal stand-in matching ``BaseLLMProvider.validate_key()`` shape."""

    def __init__(self, *args, behaviour: str = "ok", **kwargs) -> None:
        self._behaviour = behaviour

    def validate_key(self) -> bool:
        if self._behaviour == "ok":
            return True
        if self._behaviour == "auth_error":
            raise ProviderAuthError("rejected")
        if self._behaviour == "rate_limit":
            raise ProviderRateLimitError("upstream rate limited")
        if self._behaviour == "timeout":
            raise ProviderTimeoutError("upstream timed out")
        if self._behaviour == "generic":
            raise ProviderError("transport blew up")
        raise AssertionError(f"unknown stub behaviour: {self._behaviour}")

    def close(self) -> None:
        # ``ProviderRegistry.validate_key`` closes the provider it constructs
        # in a ``finally``; the stub mirrors the no-network contract.
        self._closed = True


@pytest.fixture
def stub_provider(monkeypatch: pytest.MonkeyPatch):
    """Return a factory that points the registry at ``_StubProvider``.

    Yields a callable that pins the stub's behaviour for a single test.
    """
    state = {"behaviour": "ok"}

    class _Factory(_StubProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(behaviour=state["behaviour"], **kwargs)

    def _set(behaviour: str) -> None:
        state["behaviour"] = behaviour

    # Direct ``get_entry`` so :func:`validate_credential` finds an
    # entry that "exists" for the surface under test.
    from sec_generative_search.providers.registry import ProviderEntry

    fake_entry = ProviderEntry(
        name="openai",
        surface=ProviderSurface.LLM,
        provider=_Factory,
    )

    def _get_entry(name, surface):
        if name == "openai" and surface == ProviderSurface.LLM:
            return fake_entry
        # Defer to the original for everything else so "unknown_provider"
        # tests still get the real registry behaviour.
        raise KeyError(f"No provider registered for name='{name}', surface='{surface.value}'.")

    monkeypatch.setattr(
        ProviderRegistry,
        "get_entry",
        classmethod(lambda cls, n, s: _get_entry(n, s)),
    )
    return _set


@pytest.mark.security
class TestValidateRouteSchemaGuards:
    def test_missing_body_rejected(self, api_client: TestClient) -> None:
        response = api_client.post("/api/providers/validate")
        assert response.status_code == 422
        assert response.json()["error"] == "validation_failed"

    def test_empty_api_key_rejected(self, api_client: TestClient) -> None:
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": ""},
        )
        assert response.status_code == 422

    def test_uppercase_provider_rejected(self, api_client: TestClient) -> None:
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "OPENAI", "api_key": "sk-x" * 8},  # pragma: allowlist secret
        )
        # Slug pattern is lower-case only.
        assert response.status_code == 422

    def test_oversize_api_key_rejected(self, api_client: TestClient) -> None:
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": "x" * 5000},
        )
        assert response.status_code == 422


@pytest.mark.security
class TestValidateRouteVerdicts:
    def test_valid_key(self, api_client: TestClient, stub_provider) -> None:
        stub_provider("ok")
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": "sk-good-key-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        body = response.json()
        assert body == {"valid": True, "provider": "openai", "surface": "llm"}

    def test_auth_error_collapses_to_invalid(self, api_client: TestClient, stub_provider) -> None:
        stub_provider("auth_error")
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": "sk-bad-key-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 200
        assert response.json()["valid"] is False

    def test_rate_limit_does_not_become_verdict(
        self, api_client: TestClient, stub_provider
    ) -> None:
        stub_provider("rate_limit")
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": "sk-key-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 503
        body = response.json()
        assert body["error"] == "provider_unavailable"
        # Hint MUST steer the caller away from rotating a working key.
        assert "do not rotate" in body["hint"].lower()

    def test_generic_provider_error_is_502(self, api_client: TestClient, stub_provider) -> None:
        stub_provider("generic")
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": "sk-key-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 502
        body = response.json()
        assert body["error"] == "provider_error"


@pytest.mark.security
class TestValidateRouteNoKeyLeak:
    def test_response_body_does_not_echo_key(self, api_client: TestClient, stub_provider) -> None:
        stub_provider("ok")
        secret = "sk-NEVER-ECHO-ME-1234-abcd"  # pragma: allowlist secret
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": secret},  # pragma: allowlist secret
        )
        # Whether valid or not, the body MUST NOT carry the raw key.
        assert secret not in response.text


@pytest.mark.security
class TestValidateRouteUnknownProvider:
    def test_unknown_provider_400(self, api_client: TestClient) -> None:
        # No fixture: real registry lookup is exercised.
        response = api_client.post(
            "/api/providers/validate",
            json={"provider": "definitely-not-real", "api_key": "x" * 16},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "unknown_provider"


@pytest.mark.security
class TestValidateRouteAuthGate:
    def test_api_key_required_when_configured(self, api_client_factory, stub_provider) -> None:
        stub_provider("ok")
        client = api_client_factory(API_KEY="rotated-key")  # pragma: allowlist secret
        # Missing X-API-Key header → 401.
        response = client.post(
            "/api/providers/validate",
            json={"provider": "openai", "api_key": "sk-test-1234"},  # pragma: allowlist secret
        )
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"


@pytest.mark.security
class TestValidateRouteRateLimit:
    def test_per_session_window_is_separate_from_per_ip(
        self, api_client_factory, stub_provider
    ) -> None:
        stub_provider("ok")
        # Tight limits so we hit the boundary fast.
        client = api_client_factory(
            API_RATE_LIMIT_VALIDATE="100",
            API_RATE_LIMIT_VALIDATE_PER_SESSION="3",
        )
        # Mint a session so the per-session bucket has a key.
        client.post("/api/session")

        # Burn through the per-session budget on the validate route.
        statuses: list[int] = []
        for _ in range(8):
            r = client.post(
                "/api/providers/validate",
                json={"provider": "openai", "api_key": "sk-test-1234"},  # pragma: allowlist secret
            )
            statuses.append(r.status_code)
        # Must include at least one 429 — the session bucket of 3 was
        # exhausted before the IP bucket of 100.
        assert 429 in statuses
        # Must include at least some 200s before the limiter kicked in.
        assert 200 in statuses


# ---------------------------------------------------------------------------
# GET /api/providers — read-tier catalogue
# ---------------------------------------------------------------------------


class TestListProvidersShape:
    """The list route is the registry's wire projection.

    Asserts the curated tuple makes it onto the wire untransformed and
    the allow-list lift drops every internal field on
    :class:`ProviderEntry`.
    """

    def test_returns_200_with_entries(self, api_client: TestClient) -> None:
        response = api_client.get("/api/providers/")
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, dict)
        assert set(body.keys()) == {"providers", "total"}
        assert body["total"] == len(body["providers"])
        # Registry ships at least one LLM provider (openai).
        assert body["total"] >= 1

    def test_entry_fields_are_exactly_three(self, api_client: TestClient) -> None:
        response = api_client.get("/api/providers/")
        assert response.status_code == 200
        providers = response.json()["providers"]
        assert providers, "registry must surface at least one provider"
        for entry in providers:
            # The explicit allow-list lift in ProviderInfoSchema means
            # the wire is exactly these three fields — no more, no less.
            # If a future ProviderEntry attribute leaks through, this
            # assertion fails and forces a security review.
            assert set(entry.keys()) == {
                "name",
                "surface",
                "supports_upstream_routing",
            }

    def test_known_providers_present(self, api_client: TestClient) -> None:
        response = api_client.get("/api/providers/")
        body = response.json()
        names = {(p["name"], p["surface"]) for p in body["providers"]}
        # Spot-check a few entries that ship unconditionally — every
        # listed (name, surface) pair is part of the registry's curated
        # tuple and does not require an optional extra.
        assert ("openai", "llm") in names
        assert ("anthropic", "llm") in names
        assert ("openrouter", "llm") in names
        assert ("openai", "embedding") in names

    def test_openrouter_advertises_upstream_routing(self, api_client: TestClient) -> None:
        response = api_client.get("/api/providers/")
        providers = response.json()["providers"]
        by_key = {(p["name"], p["surface"]): p for p in providers}
        # OpenRouter is the only provider that supports upstream-routing
        # hints in the current registry. The UI uses this flag to decide
        # whether to render the upstream-provider picker.
        assert by_key[("openrouter", "llm")]["supports_upstream_routing"] is True
        # Every other provider must report False — rendering the picker
        # elsewhere would be a misleading UX.
        for entry in providers:
            if (entry["name"], entry["surface"]) == ("openrouter", "llm"):
                continue
            assert entry["supports_upstream_routing"] is False, entry

    def test_local_llm_present_and_non_routing(self, api_client: TestClient) -> None:
        # The self-hosted ``local_llm`` provider is a curated LLM-surface
        # entry, so it rides this list onto the SPA model-picker dropdown
        # like every other vendor. It drops upstream-routing hints
        # (``supports_upstream_routing`` False), so the picker hides the
        # routing UI for it — same as every non-OpenRouter vendor.
        response = api_client.get("/api/providers/")
        by_key = {(p["name"], p["surface"]): p for p in response.json()["providers"]}
        assert ("local_llm", "llm") in by_key
        assert by_key[("local_llm", "llm")]["supports_upstream_routing"] is False


class TestListProvidersOrdering:
    def test_curated_order_preserved(self, api_client: TestClient) -> None:
        # The web UI is allowed to render the tuple verbatim; the
        # backend's curated order is the presentation contract.
        from sec_generative_search.providers.registry import ProviderRegistry

        expected = [(entry.name, entry.surface.value) for entry in ProviderRegistry.all_entries()]
        actual = [
            (entry["name"], entry["surface"])
            for entry in api_client.get("/api/providers/").json()["providers"]
        ]
        assert actual == expected


@pytest.mark.security
class TestListProvidersNoSecrets:
    """The list route MUST NOT leak credential-shaped strings.

    The allow-list schema already prevents an :class:`ApiKey`-shaped
    field from landing on the wire, but a future field rename could
    accidentally surface ``api_key`` or a masked tail. Belt-and-braces.
    """

    _FORBIDDEN_SUBSTRINGS = (
        "api_key",
        "api-key",
        "secret",
        "bearer",
        "authorization",
        "token",
        # Masked-tail shape from ``mask_secret`` (`***xxxx`) — also
        # banned even though the schema cannot carry it.
        "***",
    )

    def test_response_body_carries_no_credential_marker(self, api_client: TestClient) -> None:
        response = api_client.get("/api/providers/")
        assert response.status_code == 200
        text = response.text.lower()
        for needle in self._FORBIDDEN_SUBSTRINGS:
            assert needle not in text, f"forbidden substring {needle!r} leaked into list response"

    def test_response_carries_no_model_catalogue(self, api_client: TestClient) -> None:
        # The per-model catalogue is internal backend data. The ``GET
        # /api/providers/`` list route deliberately does NOT surface it —
        # listing models there would couple the UI to backend slug
        # renames. (The dedicated ``GET /api/providers/{name}/models``
        # route is the right shape for that.)
        response = api_client.get("/api/providers/")
        providers = response.json()["providers"]
        for entry in providers:
            assert "models" not in entry
            assert "model_catalogue" not in entry
            assert "default_model" not in entry


@pytest.mark.security
class TestListProvidersAuthGate:
    def test_api_key_required_when_configured(self, api_client_factory) -> None:
        client = api_client_factory(API_KEY="rotated-key")  # pragma: allowlist secret
        # Missing X-API-Key header → 401, even though the route is a
        # read-only catalogue (the catalogue's existence is a deployment
        # fingerprint and is gated when auth is on).
        response = client.get("/api/providers/")
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"

    def test_admin_key_not_required(self, api_client_factory) -> None:
        # Read-tier: an X-API-Key alone (no X-Admin-Key) MUST succeed.
        client = api_client_factory(
            API_KEY="read-key",  # pragma: allowlist secret
            API_ADMIN_KEY="admin-key",  # pragma: allowlist secret
        )
        response = client.get(
            "/api/providers/",
            headers={"X-API-Key": "read-key"},  # pragma: allowlist secret
        )
        assert response.status_code == 200


@pytest.mark.security
class TestListProvidersDoesNotInstantiate:
    """The list route is O(1) and credential-free.

    A regression here (e.g. switching to ``validate_key`` to "probe
    availability") would call into provider SDKs and expose the route to
    upstream network failures + key requirements. Guard by patching the
    constructors of every shipped adapter to raise — the route must
    still complete.
    """

    def test_handler_never_constructs_a_provider(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sec_generative_search.providers import registry as registry_module

        def _boom(*_args, **_kwargs):
            raise AssertionError("provider was instantiated by the list route")

        # ``_construct`` is the single seam ProviderRegistry uses to
        # instantiate a provider (only called from ``validate_key``).
        # The list route must never reach it.
        monkeypatch.setattr(
            registry_module.ProviderRegistry,
            "_construct",
            classmethod(_boom),
        )
        response = api_client.get("/api/providers/")
        assert response.status_code == 200


class TestListProvidersBodyCap:
    def test_post_to_get_only_route_is_rejected(self, api_client: TestClient) -> None:
        # FastAPI returns 405 Method Not Allowed for verbs the router
        # does not bind. Defensive: a future POST handler at the same
        # path would need its own ROUTE_POLICIES entry.
        response = api_client.post("/api/providers/", json={})
        assert response.status_code in (405, 422)


# ---------------------------------------------------------------------------
# GET /api/providers/{provider}/models — pricing-tier catalogue
# ---------------------------------------------------------------------------


class TestProviderModelsShape:
    """The models route lifts the LLM model catalogue of one provider.

    Asserts the catalogue makes it onto the wire as ``(model,
    pricing_tier)`` rows in declaration order, that the tier is the
    registry's single source of truth, and that the allow-list lift drops
    every other field on the capability matrix.
    """

    def test_returns_200_with_models(self, api_client: TestClient) -> None:
        response = api_client.get("/api/providers/openai/models")
        assert response.status_code == 200
        body = response.json()
        assert set(body.keys()) == {
            "provider",
            "surface",
            "supports_arbitrary_models",
            "models",
            "total",
        }
        assert body["provider"] == "openai"
        assert body["surface"] == "llm"
        assert body["supports_arbitrary_models"] is False
        assert body["total"] == len(body["models"])
        assert body["total"] >= 1

    def test_each_row_is_exactly_four_fields(self, api_client: TestClient) -> None:
        response = api_client.get("/api/providers/openai/models")
        models = response.json()["models"]
        assert models, "openai must surface at least one catalogued model"
        for row in models:
            # The explicit allow-list lift means a future
            # ProviderCapability field (context window, tool_use, ...)
            # cannot leak onto this surface. The row is intentionally four
            # fields wide: model, tier, and exact per-MTok input/output cost.
            # If a *different* field leaks through, this fails and forces a
            # review.
            assert set(row.keys()) == {
                "model",
                "pricing_tier",
                "input_cost_per_mtok",
                "output_cost_per_mtok",
            }

    def test_rows_match_registry_catalogue(self, api_client: TestClient) -> None:
        # The wire is the registry's single source of truth, in
        # declaration order, with the same tier the metrics facade reads and
        # the exact per-MTok cost the tier was derived from.
        expected = [
            (
                slug,
                cap.pricing_tier.value,
                cap.input_cost_per_mtok,
                cap.output_cost_per_mtok,
            )
            for slug in ProviderRegistry.list_models("openai", ProviderSurface.LLM)
            for cap in [ProviderRegistry.get_capability("openai", ProviderSurface.LLM, slug)]
        ]
        actual = [
            (
                row["model"],
                row["pricing_tier"],
                row["input_cost_per_mtok"],
                row["output_cost_per_mtok"],
            )
            for row in api_client.get("/api/providers/openai/models").json()["models"]
        ]
        assert actual == expected

    def test_baseline_cost_is_known_and_non_negative(self, api_client: TestClient) -> None:
        # Every vendored OpenAI row carries exact cost (the baseline is
        # fully priced) and the derived tier is never UNKNOWN. Cost on the
        # wire keeps that invariant visible end to end.
        from sec_generative_search.core.types import PricingTier

        for row in api_client.get("/api/providers/openai/models").json()["models"]:
            assert row["input_cost_per_mtok"] is not None
            assert row["output_cost_per_mtok"] is not None
            assert row["input_cost_per_mtok"] >= 0.0
            assert row["output_cost_per_mtok"] >= 0.0
            assert row["pricing_tier"] != PricingTier.UNKNOWN.value

    def test_tier_values_are_valid_pricing_tiers(self, api_client: TestClient) -> None:
        from sec_generative_search.core.types import PricingTier

        valid = {tier.value for tier in PricingTier}
        for row in api_client.get("/api/providers/openai/models").json()["models"]:
            assert row["pricing_tier"] in valid
            # The vendored baseline must never surface UNKNOWN — that would
            # defeat the single-source-of-truth contract.
            assert row["pricing_tier"] != PricingTier.UNKNOWN.value


class TestProviderModelsArbitraryProvider:
    def test_openrouter_returns_empty_catalogue(self, api_client: TestClient) -> None:
        # OpenRouter's catalogue is intentionally empty — the UI renders
        # a free-text slug input and treats any slug as UNKNOWN pricing.
        response = api_client.get("/api/providers/openrouter/models")
        assert response.status_code == 200
        body = response.json()
        assert body["models"] == []
        assert body["total"] == 0
        assert body["supports_arbitrary_models"] is True

    def test_local_llm_returns_empty_catalogue(self, api_client: TestClient) -> None:
        # The self-hosted ``local_llm`` provider surfaces OpenRouter-style
        # — an empty vendored catalogue with ``supports_arbitrary_models``
        # set — so the SPA renders a free-text slug input for whatever the
        # local server has pulled. This route is the model-picker's data
        # source; no new route was added for it (the handler is generic over
        # :class:`ProviderRegistry`).
        response = api_client.get("/api/providers/local_llm/models")
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "local_llm"
        assert body["surface"] == "llm"
        assert body["models"] == []
        assert body["total"] == 0
        assert body["supports_arbitrary_models"] is True


@pytest.mark.security
class TestProviderModelsOverlayUnknown:
    """Overlay-only unknown pricing, end-to-end at the read surface.

    A closed-catalogue provider gains an overlay / auto-discovered model
    that arrived without pricing. The read route must *surface* it — with
    ``pricing_tier == "unknown"`` and ``None`` costs — rather than rejecting
    it or coercing a misleading price. The vendored baseline rows stay fully
    priced alongside it (that invariant is unchanged).
    """

    def test_overlay_only_model_surfaces_with_unknown_tier_and_null_cost(
        self, api_client: TestClient
    ) -> None:
        from sec_generative_search.core.types import PricingTier, ProviderCapability
        from sec_generative_search.providers.catalogue import (
            model_catalogue,
            reset_catalogue,
            set_catalogue,
        )

        unpriced = ProviderCapability(chat=True, streaming=True)  # both costs None
        assert unpriced.pricing_tier is PricingTier.UNKNOWN
        set_catalogue(model_catalogue().with_provider("openai", {"gpt-overlay-x": unpriced}))
        try:
            rows = api_client.get("/api/providers/openai/models").json()["models"]
        finally:
            reset_catalogue()

        by_slug = {row["model"]: row for row in rows}
        overlay_row = by_slug["gpt-overlay-x"]
        assert overlay_row["pricing_tier"] == PricingTier.UNKNOWN.value
        assert overlay_row["input_cost_per_mtok"] is None
        assert overlay_row["output_cost_per_mtok"] is None
        # The baseline rows are untouched — still priced, never UNKNOWN.
        assert by_slug["gpt-4o"]["pricing_tier"] != PricingTier.UNKNOWN.value
        assert by_slug["gpt-4o"]["input_cost_per_mtok"] is not None


@pytest.mark.security
class TestProviderModelsUnknownProvider:
    def test_unknown_provider_404(self, api_client: TestClient) -> None:
        response = api_client.get("/api/providers/definitely-not-real/models")
        assert response.status_code == 404
        assert response.json()["error"] == "unknown_provider"

    def test_embedding_only_provider_is_404_on_llm_surface(self, api_client: TestClient) -> None:
        # ``local`` ships only an embedding adapter. The models route is
        # LLM-surface only — an embedding-only provider is "not
        # registered" here, exactly like an unknown name.
        response = api_client.get("/api/providers/local/models")
        assert response.status_code == 404
        assert response.json()["error"] == "unknown_provider"


@pytest.mark.security
class TestProviderModelsSlugGuard:
    def test_uppercase_slug_rejected(self, api_client: TestClient) -> None:
        # The path validator mirrors the lower-case provider-slug shape.
        response = api_client.get("/api/providers/OpenAI/models")
        assert response.status_code == 422

    def test_control_character_slug_rejected(self, api_client: TestClient) -> None:
        # A slug carrying a path/control character must never reach the
        # registry lookup — the anchored pattern rejects it.
        response = api_client.get("/api/providers/open%0aai/models")
        assert response.status_code in (404, 422)


@pytest.mark.security
class TestProviderModelsNoSecrets:
    def test_response_body_carries_no_credential_marker(self, api_client: TestClient) -> None:
        response = api_client.get("/api/providers/openai/models")
        assert response.status_code == 200
        text = response.text.lower()
        for needle in ("api_key", "api-key", "secret", "bearer", "authorization", "***"):
            assert needle not in text, f"forbidden substring {needle!r} leaked into models response"


@pytest.mark.security
class TestProviderModelsAuthGate:
    def test_api_key_required_when_configured(self, api_client_factory) -> None:
        client = api_client_factory(API_KEY="rotated-key")  # pragma: allowlist secret
        response = client.get("/api/providers/openai/models")
        assert response.status_code == 401
        assert response.json()["error"] == "unauthorised"

    def test_admin_key_not_required(self, api_client_factory) -> None:
        client = api_client_factory(
            API_KEY="read-key",  # pragma: allowlist secret
            API_ADMIN_KEY="admin-key",  # pragma: allowlist secret
        )
        response = client.get(
            "/api/providers/openai/models",
            headers={"X-API-Key": "read-key"},  # pragma: allowlist secret
        )
        assert response.status_code == 200


@pytest.mark.security
class TestProviderModelsDoesNotInstantiate:
    """The models route is O(1) and credential-free — same contract as
    the list route. A regression that instantiates the provider would
    expose the route to upstream network failures and key requirements.
    """

    def test_handler_never_constructs_a_provider(
        self, api_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from sec_generative_search.providers import registry as registry_module

        def _boom(*_args, **_kwargs):
            raise AssertionError("provider was instantiated by the models route")

        monkeypatch.setattr(
            registry_module.ProviderRegistry,
            "_construct",
            classmethod(_boom),
        )
        response = api_client.get("/api/providers/openai/models")
        assert response.status_code == 200
