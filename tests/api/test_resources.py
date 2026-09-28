"""Tests for the embedder resource introspection route.

Single endpoint under test: ``GET /api/resources/gpu``.

Strategy
--------

The production lifespan is skipped — :class:`BaseEmbeddingProvider`
stubs are attached to ``app.state.embedder`` so the tests never load
``sentence-transformers`` or call out to a hosted vendor.

Coverage focuses on:

    - Hosted-embedder collapse to a no-op shape
      (``is_local=False`` / ``is_loaded=True`` / no device / no idle).
    - On-device load-state surfacing for
      :class:`LocalEmbeddingProvider`, including the unloaded shape
      and the elapsed-idle scalar.
    - Privacy contract: response NEVER carries a file path, a token /
      key value, or the internal monotonic ``_last_used`` timestamp.
    - Read-tier auth gate (401 without ``X-API-Key`` when configured)
      and admin-key independence (admin alone MUST NOT pass).
    - The route MUST NOT trigger a model load — reading
      ``is_loaded`` is a pure property and reaching the route on an
      unloaded provider keeps it unloaded.
    - Rate-limit classification: the path falls into the ``general``
      bucket via ``resolve_policy``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sec_generative_search.api.app import create_app
from sec_generative_search.config.settings import reload_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.edgar_identity import InMemorySessionEdgarIdentityStore
from sec_generative_search.providers.base import BaseEmbeddingProvider
from sec_generative_search.providers.local import LocalEmbeddingProvider

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _HostedEmbedderStub(BaseEmbeddingProvider):
    """Minimal stand-in for a hosted embedding provider.

    Mirrors :class:`OpenAIEmbeddingProvider` / :class:`GeminiEmbeddingProvider`
    in shape — stateless, no ``is_loaded`` / ``device`` / ``_last_used``
    — without pulling in the real SDK clients.
    """

    provider_name = "openai"

    def __init__(self) -> None:
        super().__init__("admin-env-key")  # pragma: allowlist secret

    def validate_key(self) -> bool:  # pragma: no cover - not exercised
        return True

    def get_capabilities(self) -> Any:  # pragma: no cover - not exercised
        from sec_generative_search.core.types import ProviderCapability

        return ProviderCapability(embeddings=True)

    def embed_texts(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        return np.zeros((len(texts), 8), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:  # pragma: no cover
        return np.zeros((8,), dtype=np.float32)

    def get_dimension(self) -> int:  # pragma: no cover
        return 8


def _fake_st_loader(**_kwargs: Any) -> Any:
    """Lightweight ``sentence-transformers`` replacement.

    Returns an object that satisfies the ``encode`` contract for the
    encode paths we never exercise here; the resource route only reads
    ``is_loaded`` / ``_resolved_device`` / ``_last_used``, so the
    encode surface is incidental. Accepts arbitrary kwargs because
    :class:`LocalEmbeddingProvider` passes ``model_name`` / ``device``
    / ``token`` by name.
    """

    class _Model:
        def encode(self, texts: list[str], **_kwargs: Any) -> np.ndarray:  # pragma: no cover
            return np.zeros((len(texts), 384), dtype=np.float32)

        def to(self, *_args: Any, **_kwargs: Any) -> _Model:  # pragma: no cover
            return self

    return _Model()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_resources_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip ``API_*`` / ``EMBEDDING_*`` env vars to a deterministic baseline."""
    for key in list(os.environ.keys()):
        if key.startswith(("API_", "EMBEDDING_")):
            monkeypatch.delenv(key, raising=False)
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def resources_app_factory(monkeypatch: pytest.MonkeyPatch):
    """Build a fresh app with an embedder stub attached.

    The factory accepts either a constructed ``BaseEmbeddingProvider``
    instance (typical) or callers can override ``app.state.embedder``
    after the fact when they need to retain a handle on a real
    :class:`LocalEmbeddingProvider`.
    """

    def factory(
        *,
        embedder: BaseEmbeddingProvider | None = None,
        env: dict[str, str] | None = None,
    ) -> Any:
        if env:
            for key, value in env.items():
                monkeypatch.setenv(key, value)
        reload_settings()

        app = create_app()
        if embedder is not None:
            app.state.embedder = embedder
        else:
            app.state.embedder = _HostedEmbedderStub()
        app.state.session_store = InMemorySessionCredentialStore(ttl_seconds=300)
        app.state.edgar_identity_store = InMemorySessionEdgarIdentityStore(ttl_seconds=300)
        app.state.encrypted_credential_store = None
        return app

    return factory


# ---------------------------------------------------------------------------
# Hosted embedder shape
# ---------------------------------------------------------------------------


class TestHostedEmbedderShape:
    def test_hosted_collapses_to_no_op_shape(self, resources_app_factory) -> None:
        app = resources_app_factory(embedder=_HostedEmbedderStub())
        client = TestClient(app, base_url="https://testserver")

        response = client.get("/api/resources/gpu")
        assert response.status_code == 200
        body = response.json()

        # Hosted embedders are stateless; the schema collapses the load
        # state to a single "ready" snapshot.  ``provider`` / ``model``
        # are taken from settings, never from the stub instance.
        assert body["is_local"] is False
        assert body["is_loaded"] is True
        assert body["device"] is None
        assert body["idle_seconds"] is None
        assert body["idle_timeout_minutes"] == 0

    def test_response_field_set_is_pinned(self, resources_app_factory) -> None:
        app = resources_app_factory(embedder=_HostedEmbedderStub())
        client = TestClient(app, base_url="https://testserver")
        body = client.get("/api/resources/gpu").json()

        # A future field addition on :class:`GpuStatusResponse` should
        # land deliberately; this assertion pins the wire shape.
        assert set(body.keys()) == {
            "provider",
            "model",
            "is_local",
            "is_loaded",
            "device",
            "idle_seconds",
            "idle_timeout_minutes",
        }


# ---------------------------------------------------------------------------
# On-device embedder load-state surfacing
# ---------------------------------------------------------------------------


class TestLocalEmbedderShape:
    def test_unloaded_local_reports_no_device(self, resources_app_factory) -> None:
        # Construction never loads the model; only :meth:`_ensure_model`
        # does.  The route MUST observe the unloaded state cleanly.
        local = LocalEmbeddingProvider(
            hf_token=None,
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            idle_timeout_minutes=5,
            loader=_fake_st_loader,
        )
        app = resources_app_factory(embedder=local)
        client = TestClient(app, base_url="https://testserver")

        body = client.get("/api/resources/gpu").json()
        assert body["is_local"] is True
        assert body["is_loaded"] is False
        # No device / idle data while the model is not resident.
        assert body["device"] is None
        assert body["idle_seconds"] is None
        # Operator-configured timeout survives the unload state.
        assert body["idle_timeout_minutes"] == 5

    def test_loaded_local_reports_device_and_idle(self, resources_app_factory) -> None:
        ticks = iter([100.0, 100.0, 107.5])  # construction, _mark_used, route read

        def _clock() -> float:
            return next(ticks)

        local = LocalEmbeddingProvider(
            hf_token=None,
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            idle_timeout_minutes=10,
            loader=_fake_st_loader,
            clock=_clock,
        )
        # Force a load + ``_mark_used`` so the route observes the
        # loaded shape; we cannot rely on the real encode path because
        # the loader stub does not implement ``.encode`` outside the
        # ``pragma: no cover`` branch.
        local._ensure_model()
        local._mark_used()

        app = resources_app_factory(embedder=local)
        client = TestClient(app, base_url="https://testserver")

        body = client.get("/api/resources/gpu").json()
        assert body["is_local"] is True
        assert body["is_loaded"] is True
        assert body["device"] == "cpu"
        # idle_seconds = clock(read) - clock(mark_used) = 107.5 - 100.0
        assert body["idle_seconds"] == pytest.approx(7.5, rel=1e-3)
        assert body["idle_timeout_minutes"] == 10

    def test_route_does_not_trigger_model_load(self, resources_app_factory) -> None:
        """The route MUST be a pure read — hitting it on an unloaded
        provider MUST NOT cause :meth:`_ensure_model` to fire.
        """
        load_calls = 0

        def _counting_loader(*args: Any, **kwargs: Any) -> Any:
            nonlocal load_calls
            load_calls += 1
            return _fake_st_loader(*args, **kwargs)

        local = LocalEmbeddingProvider(
            hf_token=None,
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            loader=_counting_loader,
        )
        app = resources_app_factory(embedder=local)
        client = TestClient(app, base_url="https://testserver")

        assert load_calls == 0
        for _ in range(5):
            response = client.get("/api/resources/gpu")
            assert response.status_code == 200
        assert load_calls == 0
        assert local.is_loaded is False


# ---------------------------------------------------------------------------
# Auth tier — read-tier only
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestResourcesAuthGate:
    def test_api_key_required_when_configured(self, resources_app_factory) -> None:
        app = resources_app_factory(
            embedder=_HostedEmbedderStub(),
            env={"API_KEY": "shared-team-key"},  # pragma: allowlist secret
        )
        client = TestClient(app, base_url="https://testserver")

        unauthed = client.get("/api/resources/gpu")
        assert unauthed.status_code == 401
        assert unauthed.json()["error"] == "unauthorised"

        ok = client.get(
            "/api/resources/gpu",
            headers={"X-API-Key": "shared-team-key"},  # pragma: allowlist secret
        )
        assert ok.status_code == 200

    def test_admin_key_alone_is_not_sufficient(self, resources_app_factory) -> None:
        # Mirrors the search-route guarantee: the route is read-tier,
        # so the read-tier gate alone allows the request; an admin-key
        # alone (without ``X-API-Key``) MUST still be rejected as
        # ``401 unauthorised`` — not ``403 admin_required`` — because
        # the read-tier dependency is what runs first.
        app = resources_app_factory(
            embedder=_HostedEmbedderStub(),
            env={
                "API_KEY": "shared-team-key",  # pragma: allowlist secret
                "API_ADMIN_KEY": "secret-admin-key",  # pragma: allowlist secret
            },
        )
        client = TestClient(app, base_url="https://testserver")

        admin_only = client.get(
            "/api/resources/gpu",
            headers={"X-Admin-Key": "secret-admin-key"},  # pragma: allowlist secret
        )
        assert admin_only.status_code == 401
        assert admin_only.json()["error"] == "unauthorised"


# ---------------------------------------------------------------------------
# Privacy contract — never leak paths / keys / monotonic timestamps
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestResourcesPrivacy:
    def test_response_carries_no_path_or_key(self, resources_app_factory) -> None:
        sentinel_token = "hf-test-secret-token-zzz"  # pragma: allowlist secret
        local = LocalEmbeddingProvider(
            hf_token=sentinel_token,
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            idle_timeout_minutes=3,
            loader=_fake_st_loader,
        )
        local._ensure_model()
        local._mark_used()

        app = resources_app_factory(embedder=local)
        client = TestClient(app, base_url="https://testserver")
        raw = client.get("/api/resources/gpu").text

        # No HF token / API key value reaches the wire.
        assert sentinel_token not in raw
        # No filesystem paths: the response carries slugs only, never
        # an absolute or home-directory path fragment.
        assert "/home/" not in raw
        assert "/var/" not in raw

    def test_idle_seconds_is_elapsed_not_monotonic(self, resources_app_factory) -> None:
        """The schema surfaces an elapsed scalar, not the raw monotonic
        ``_last_used`` timestamp (which would leak process start-time
        information).  The elapsed value is always small relative to
        ``time.monotonic()``.
        """
        local = LocalEmbeddingProvider(
            hf_token=None,
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            idle_timeout_minutes=10,
            loader=_fake_st_loader,
        )
        local._ensure_model()
        local._mark_used()
        # Sleep a tiny bit so ``elapsed`` is observable but bounded.
        time.sleep(0.02)

        app = resources_app_factory(embedder=local)
        client = TestClient(app, base_url="https://testserver")
        body = client.get("/api/resources/gpu").json()

        # Elapsed must be small and non-negative; the raw monotonic
        # clock would be many seconds since process start, comfortably
        # > 1.
        assert body["idle_seconds"] is not None
        assert 0.0 <= body["idle_seconds"] < 1.0


# ---------------------------------------------------------------------------
# Rate-limit classification
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestResourcesRateLimitClassification:
    def test_path_classifies_as_general(self) -> None:
        from sec_generative_search.api.middleware import _rate_category_for_scope

        # The route shares the ``general`` rate bucket with ``/api/status``
        # — same operator-facing read tier, same envelope.
        assert (
            _rate_category_for_scope(
                {"type": "http", "path": "/api/resources/gpu", "method": "GET"}
            )
            == "general"
        )

    def test_policy_caps_body_at_1kib(self) -> None:
        from sec_generative_search.api.policies import resolve_policy

        policy = resolve_policy("/api/resources/gpu", "GET")
        assert policy.rate_category == "general"
        # 1 KiB defends against declared-Content-Length probes on a
        # read-tier route that takes no body.
        assert policy.max_body_bytes == 1024
