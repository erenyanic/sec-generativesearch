"""FastAPI application factory and lifespan.

What this file owns:

        - Build the FastAPI ``app`` instance via :func:`create_app`.
        - Initialise singletons in :func:`lifespan`: ``MetadataRegistry``,
            ``ChromaDBClient``, the embedder, ``FilingStore``,
            ``RetrievalService``, the in-memory session store, and the
            optional ``EncryptedCredentialStore``.
        - Wire the middleware stack and the structured-error handlers.
        - Mount the API routes exposed by this application module.
        - Toggle OpenAPI docs off when ``API_KEY`` is configured.

Notes for the lifespan:

    - The embedder is constructed eagerly at startup using the
    *administrative* default-env-var resolver. A startup failure here
    is the right behaviour: a deployment that cannot embed will not
    serve correct retrievals; refuse early.
    - ``EncryptedCredentialStore`` is optional and gated by
      ``DB_PERSIST_PROVIDER_CREDENTIALS``.  When disabled, the resolver
      chain falls through to the session store and the admin-env
      fallback, which is the documented Scenario-A behaviour.
    - ``app.state`` is a Starlette-supplied attribute namespace; we
      attach typed names that ``api.dependencies`` reads back.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sec_generative_search import __version__
from sec_generative_search.api.errors import install_error_handlers
from sec_generative_search.api.middleware import (
    ContentSizeLimitMiddleware,
    CorrelationIdMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)
from sec_generative_search.api.routes.auth import router as auth_router
from sec_generative_search.api.routes.catalogue import router as catalogue_router
from sec_generative_search.api.routes.filings import router as filings_router
from sec_generative_search.api.routes.health import router as health_router
from sec_generative_search.api.routes.ingest import router as ingest_router
from sec_generative_search.api.routes.metrics import router as metrics_router
from sec_generative_search.api.routes.provider_health import router as provider_health_router
from sec_generative_search.api.routes.providers import router as providers_router
from sec_generative_search.api.routes.rag import router as rag_router
from sec_generative_search.api.routes.resources import router as resources_router
from sec_generative_search.api.routes.search import router as search_router
from sec_generative_search.api.routes.session import router as session_router
from sec_generative_search.api.routes.status import router as status_router
from sec_generative_search.api.routes.users import router as users_router
from sec_generative_search.api.tasks import TaskManager, run_retention_eviction_safe
from sec_generative_search.api.websocket import router as websocket_router
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.credentials import InMemorySessionCredentialStore
from sec_generative_search.core.edgar_identity import InMemorySessionEdgarIdentityStore
from sec_generative_search.core.exceptions import ConfigurationError
from sec_generative_search.core.logging import get_logger
from sec_generative_search.core.types import EmbedderStamp
from sec_generative_search.database import ChromaDBClient, FilingStore, MetadataRegistry
from sec_generative_search.database.credentials import EncryptedCredentialStore
from sec_generative_search.database.users import UserStore
from sec_generative_search.pipeline.fetch import FilingFetcher
from sec_generative_search.pipeline.orchestrator import PipelineOrchestrator
from sec_generative_search.providers.factory import build_embedder
from sec_generative_search.providers.registry import ProviderRegistry
from sec_generative_search.search import RetrievalService

__all__ = ["app", "create_app", "lifespan"]


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


async def _warm_embedder(embedder: object, model_name: str) -> None:
    """Load the embedder's model now, failing the boot if it cannot load.

    Duck-typed on ``warm_up`` (never ``isinstance``), mirroring the
    ``maybe_unload`` seam.  Settings already reject the flag for hosted
    embedders, so a hosted provider is never called at boot.  The load
    runs in a worker thread to keep the loop free.

    An operator who asked for readiness-before-traffic gets a refusal,
    not a silent fallback to the first-request download.  This module's
    log line and the raised message carry the model slug and exception
    *type* only — never the exception text.  The cause stays chained so
    the boot traceback keeps the loader's diagnosis (no request data
    exists at boot).
    """
    warm_up = getattr(embedder, "warm_up", None)
    if not callable(warm_up):
        return
    logger.info("Embedder warm-up: loading model=%s before accepting traffic", model_name)
    try:
        await asyncio.to_thread(warm_up)
    except Exception as exc:
        logger.error(
            "Embedder warm-up failed (model=%s): %s",
            model_name,
            type(exc).__name__,
        )
        raise ConfigurationError(
            "EMBEDDING_WARM_ON_BOOT=true but the embedding model could not be loaded.",
            details=(
                "Check HF_TOKEN (the default model is gated), outbound access to "
                "huggingface.co or a pre-populated HF_HOME cache, the "
                "[local-embeddings] extra, and — with EMBEDDING_DEVICE=cuda — that "
                "a usable GPU is visible to the process (NVIDIA driver + container "
                "toolkit, a CUDA torch build)."
            ),
        ) from exc


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Boot every singleton, expose them on ``app.state``, and tear down.

    Order is load-bearing:

    1. Embedder — fails early on missing admin env var; optionally
       warmed (``EMBEDDING_WARM_ON_BOOT``), failing the boot on error.
    2. ``EmbedderStamp`` from the registry — the storage seal.
    3. ``ChromaDBClient`` — opens (or creates) the sealed collection.
    4. ``MetadataRegistry`` — opens SQLite/SQLCipher and applies
       migrations including v2 (``provider_credentials``).
    5. ``FilingStore`` — coordinator over the two stores above.
    6. ``RetrievalService`` — pre-built single-query primitive.
    7. ``InMemorySessionCredentialStore`` — in-memory session store.
    8. ``EncryptedCredentialStore`` — optional, gated by settings.
    """
    settings = get_settings()
    logger.info("SEC-GenerativeSearch API starting up (v%s)", __version__)

    # 1. Embedder — administrative selection, default-env resolver only.
    embedder = build_embedder(settings.embedding)

    # 1b. Optional warm-up (``EMBEDDING_WARM_ON_BOOT``, local-only).  Runs
    # before uvicorn binds, so a cold weight download finishes before the
    # startup probe can pass instead of stalling the first user request.
    if settings.embedding.warm_on_boot:
        await _warm_embedder(embedder, settings.embedding.model_name)

    # 2. Stamp via the registry — single source of truth for dimension.
    dimension = ProviderRegistry.get_dimension(
        settings.embedding.provider,
        settings.embedding.model_name,
    )
    stamp = EmbedderStamp(
        provider=settings.embedding.provider,
        model=settings.embedding.model_name,
        dimension=dimension,
    )

    # 3-5. Storage chain.
    chroma = ChromaDBClient(stamp)
    registry = MetadataRegistry()
    filing_store = FilingStore(chroma, registry)

    # 6. Retrieval — pre-built; provider-neutral by design.
    retrieval_service = RetrievalService(embedder=embedder, chroma_client=chroma)

    # 7. In-memory session credential store.  TTL mirrors the cookie's
    # ``Max-Age`` so a credential cannot outlive the cookie that points
    # at it (lazy eviction; no background thread).
    session_store = InMemorySessionCredentialStore(
        ttl_seconds=settings.api.session_ttl_seconds,
    )

    # 7b. In-memory per-session EDGAR identity store. Same TTL as the
    # credential store so the (name, email) tuple cannot outlive the
    # cookie that points at it.
    edgar_identity_store = InMemorySessionEdgarIdentityStore(
        ttl_seconds=settings.api.session_ttl_seconds,
    )

    # 8. Optional encrypted store.  Settings validation already rejected
    # ``persist=true`` without SQLCipher at load time; we still defend
    # at this seam by letting the store's own constructor refuse loudly.
    encrypted_store: EncryptedCredentialStore | None = None
    if settings.database.persist_provider_credentials:
        try:
            encrypted_store = EncryptedCredentialStore(registry)
        except Exception:  # pragma: no cover - defensive log
            logger.exception(
                "Encrypted credential store failed to initialise — "
                "continuing without persistent credential tier."
            )
            encrypted_store = None

    # UserStore is gated on SQLCipher (the table holds ``auth_hash`` +
    # the ciphertext vault; plaintext is unacceptable) and on the
    # pepper-required-when-non-empty contract. Construction refuses
    # loudly when either invariant is violated; we surface the refusal
    # at startup so an operator who rotates the pepper without
    # realising loses the API process, not their users' next login.
    user_store: UserStore | None = None
    if registry.encrypted:
        try:
            user_store = UserStore(registry)
        except Exception:  # pragma: no cover — defensive log
            logger.exception(
                "UserStore failed to initialise — user-tier auth surface will not be available."
            )
            user_store = None

    # Per-username login rate limiter. Sibling to the ``validate``
    # per-session window in :class:`RateLimitMiddleware` — built here
    # rather than in middleware because the username arrives in the
    # JSON body, not the cookie header, and consuming the body in
    # middleware would break Starlette's request flow.
    login_username_window = None
    if settings.api.rate_limit_login_per_username > 0:
        from sec_generative_search.api.middleware import _SlidingWindow

        login_username_window = _SlidingWindow(settings.api.rate_limit_login_per_username)

    # 9c. ``session_id → user_id`` mapping for the user-tier routes.
    # Process-local dict; entries evict on logout and on rotation.  The
    # parallel ``session_store`` (provider-key cache) is keyed by
    # ``session_id`` opaquely — this dict adds the typed link the auth
    # follow-up routes need without widening that store's protocol.
    session_user_index: dict[str, int] = {}

    # Background ingestion TaskManager. The fetcher / orchestrator chain
    # is built fresh per app — both are stateless apart from the
    # process-global ``edgar.set_identity`` mutation, which the manager
    # re-applies under its own lock before every EDGAR call cluster. The
    # embedder is the same singleton attached above, so a successful boot
    # here also proves the GPU/local-extras gating already passed.
    fetcher = FilingFetcher()
    pipeline_orchestrator = PipelineOrchestrator(embedder=embedder)
    task_manager = TaskManager(
        filing_store=filing_store,
        registry=registry,
        fetcher=fetcher,
        orchestrator=pipeline_orchestrator,
        # Thread the embedder reference through so the worker's
        # post-task cleanup and the lazy-eviction sweep can fire the
        # idle-unload hook. Hosted embedders no-op via duck-typing in
        # the manager, which keeps the optional local-embeddings
        # import path out of the app wiring.
        embedder=embedder,
    )
    # The running asyncio loop is what ``_push`` uses to bridge the sync
    # worker thread to the per-task message queue. Capture it once here;
    # ``set_event_loop`` is the only public seam for the bind.
    task_manager.set_event_loop(asyncio.get_running_loop())

    # Expose every singleton on ``app.state``.
    app.state.settings = settings
    app.state.embedder = embedder
    app.state.chroma = chroma
    app.state.registry = registry
    app.state.filing_store = filing_store
    app.state.retrieval_service = retrieval_service
    app.state.session_store = session_store
    app.state.edgar_identity_store = edgar_identity_store
    app.state.encrypted_credential_store = encrypted_store
    app.state.user_store = user_store
    app.state.login_username_window = login_username_window
    app.state.session_user_index = session_user_index
    app.state.task_manager = task_manager

    logger.info(
        "API ready: embedder=%s/%s, dim=%d, encrypted_store=%s, user_store=%s",
        stamp.provider,
        stamp.model,
        stamp.dimension,
        "yes" if encrypted_store is not None else "no",
        "yes" if user_store is not None else "no",
    )

    # One-shot startup retention sweep. Best-effort: a failure inside
    # the helper is logged and swallowed so a transient backend hiccup
    # never blocks the API from coming up. The helper short-circuits
    # when ``DB_RETENTION_MAX_AGE_DAYS=0`` (Scenario A default).
    run_retention_eviction_safe(
        filing_store,
        settings.database.retention_max_age_days,
        context_label="startup",
    )

    try:
        yield
    finally:
        logger.info("SEC-GenerativeSearch API shutting down.")
        # Stop accepting new tasks and cancel any in-flight duration
        # timers before the registry connection closes.  Existing
        # daemon worker threads will reach their own teardown when the
        # interpreter exits — that is the documented behaviour.
        task_manager.shutdown()
        # ``MetadataRegistry`` owns the SQLite/SQLCipher connection that
        # the encrypted store also uses — close it here, never inside
        # the encrypted store, so we don't double-close from two sites.
        registry.close()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and configure the ASGI application.

    The factory is the single seam every entry point goes through —
    uvicorn, the test client, and downstream embedders that wrap the
    app.  Settings are resolved here, not in module-level code, so a
    test fixture that patches the environment before calling
    :func:`create_app` gets fresh wiring.
    """
    settings = get_settings()

    # Disable OpenAPI / Swagger / Redoc when API_KEY is set.
    # In Scenario-A development the docs remain available; in
    # Scenarios B/C they are off so an unauthenticated probe cannot
    # discover the surface.
    is_protected = bool(settings.api.key)

    app = FastAPI(
        title="SEC-GenerativeSearch API",
        description=(
            "REST API for retrieval-augmented generation over SEC filings. "
            "Health, status, session, provider validation, and retrieval routes."
        ),
        version=__version__,
        docs_url=None if is_protected else "/docs",
        redoc_url=None if is_protected else "/redoc",
        openapi_url=None if is_protected else "/openapi.json",
        lifespan=lifespan,
    )

    install_error_handlers(app)

    # ------------------------------------------------------------------
    # Middleware stack — read top-to-bottom as outermost-to-innermost.
    # ASGI middleware is applied LIFO: the *last* ``add_middleware``
    # call is the *first* to see an inbound request. Every layer is
    # pure ASGI (no ``BaseHTTPMiddleware`` task-group / memory-stream
    # hop remains).
    #
    # Order rationale (outer → inner):
    #
    #   1. CORSMiddleware           — first to see preflight; emits
    #                                 headers without invoking the app.
    #   2. CorrelationIdMiddleware  — binds the per-request correlation ID
    #                                 before any inner middleware logs, so
    #                                 even rate-limit / 413 rejections carry
    #                                 it; echoes X-Request-ID on the way out.
    #   3. SecurityHeadersMiddleware — touches every response, including
    #                                  rate-limit and 413 responses.
    #   4. RateLimitMiddleware       — rejects abusive callers before we
    #                                  pay the cost of body parsing; also
    #                                  carries the one-shot insecure-transport
    #                                  warning (folded in from a former
    #                                  standalone layer).
    #   5. ContentSizeLimitMiddleware — cheap reject on the body stream;
    #                                  shares the rate limiter's cached
    #                                  per-route policy resolution.
    # ------------------------------------------------------------------
    app.add_middleware(ContentSizeLimitMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        search_rpm=settings.api.rate_limit_search,
        ingest_rpm=settings.api.rate_limit_ingest,
        delete_rpm=settings.api.rate_limit_delete,
        general_rpm=settings.api.rate_limit_general,
        rag_rpm=settings.api.rate_limit_rag,
        validate_rpm=settings.api.rate_limit_validate,
        validate_per_session_rpm=settings.api.rate_limit_validate_per_session,
        session_rpm=settings.api.rate_limit_session,
        login_rpm=settings.api.rate_limit_login,
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "X-API-Key",
            "X-Admin-Key",
            "X-Edgar-Name",
            "X-Edgar-Email",
        ],
    )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(health_router, prefix="/api", tags=["meta"])
    app.include_router(status_router, prefix="/api/status", tags=["status"])
    app.include_router(session_router, prefix="/api", tags=["session"])
    app.include_router(auth_router, prefix="/api", tags=["auth"])
    app.include_router(users_router, prefix="/api/admin", tags=["admin-users"])
    app.include_router(providers_router, prefix="/api/providers", tags=["providers"])
    app.include_router(provider_health_router, prefix="/api/providers", tags=["providers"])
    app.include_router(catalogue_router, prefix="/api/providers", tags=["providers"])
    app.include_router(filings_router, prefix="/api/filings", tags=["filings"])
    app.include_router(search_router, prefix="/api/search", tags=["search"])
    app.include_router(rag_router, prefix="/api/rag", tags=["rag"])
    app.include_router(ingest_router, prefix="/api/ingest", tags=["ingest"])
    app.include_router(resources_router, prefix="/api/resources", tags=["resources"])
    app.include_router(metrics_router, prefix="/api/metrics", tags=["metrics"])
    # WebSocket router is mounted at the root — the path itself
    # (``/ws/ingest/{task_id}``) carries the namespace.  Browser
    # ``WebSocket`` constructors cannot supply custom headers, so the
    # route owns its own origin + API-key + ownership handshake; the
    # HTTP middleware stack is bypassed for upgrades.
    app.include_router(websocket_router, tags=["ingest"])

    return app


# ---------------------------------------------------------------------------
# Module-level ASGI app (used by uvicorn and the FastAPI test client).
# Cheap at import time: only the FastAPI instance + middleware + routes
# are wired here; storage / embedder / credential-store construction
# happens inside ``lifespan`` when uvicorn enters the lifespan context.
# ---------------------------------------------------------------------------


app = create_app()
