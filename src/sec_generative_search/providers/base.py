"""Abstract base classes for LLM, embedding, and reranker providers.

Every concrete provider subclasses exactly one of the ABCs defined here and supplies:

- its SDK-specific :class:`ExceptionMapping` for error normalisation,
- a capability probe populating :class:`ProviderCapability`, and
- the actual network calls wrapped by :func:`resilient_call`.

Intentionally defines *only* the interface surface — no SDK
dependency is pulled in yet.  This keeps the abstractions stable before
any concrete adapter commits to a vendor quirk.

Security constraints on API keys:

- API keys are stored on ``self._api_key`` (leading underscore so
  dataclass-style serialisers skip them).
- ``__repr__`` and ``__str__`` render the key via
  :func:`mask_secret` — they are marked :data:`typing.final` so
  subclasses cannot override the redaction.
- Neither the base classes nor any helper here log the key.
- No method signature carries a credential — ``api_key`` is supplied
  once, at construction time, and never leaves the instance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, final

from sec_generative_search.core.logging import get_logger
from sec_generative_search.core.security import mask_secret
from sec_generative_search.core.types import (
    Chunk,
    ProviderCapability,
    TokenUsage,
)

if TYPE_CHECKING:
    import numpy as np

    from sec_generative_search.providers.routing_hints import OpenRouterRoutingHints


__all__ = [
    "BaseEmbeddingProvider",
    "BaseLLMProvider",
    "BaseRerankerProvider",
    "GenerationRequest",
    "GenerationResponse",
    "RerankResult",
]


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM request / response types
# ---------------------------------------------------------------------------


@dataclass
class GenerationRequest:
    """Input to :meth:`BaseLLMProvider.generate` / :meth:`generate_stream`.

    Keeps the set of fields small and vendor-neutral.  Provider-specific
    extensions (tool definitions, cache-control markers, JSON schemas)
    land via subclass-specific request types in later.
    Attributes:
        prompt: The user-facing question / instruction.  Tier 3 data —
            callers must not log this without redaction.
        model: Provider-specific model slug (e.g. ``"gpt-4o"``).
        system: Optional system prompt.  May carry SEC-analysis framing
            for templates.
        temperature: Sampling temperature.  Defaults to 0.1 for factual
            filings analysis.
        max_output_tokens: Upper bound on the response length.
        routing_hints: Optional upstream-routing hints consumed only by
            :class:`~sec_generative_search.providers.openrouter.OpenRouterProvider`.
            Every other provider ignores the field — the OpenAI-compatible
            base's ``_extra_request_kwargs`` hook returns an empty dict by
            default, so unrelated vendors never see the hint reach their
            SDK call.  The hint object is pass-through and carries no
            credential material; it is a *routing* channel, not an
            authentication one.
        response_format: ``"text"`` (default) for plain free-form output,
            or ``"json"`` to ask the provider to constrain the output to
            a JSON object.  Providers that do not advertise
            :attr:`ProviderCapability.structured_output` may ignore
            ``"json"`` (best-effort) — the orchestrator's hybrid citation
            extractor handles a parse failure by falling back to inline
            markers.  The flag is consumed by each provider's
            ``_extra_request_kwargs`` hook (or its SDK equivalent for
            non-OpenAI-compatible providers) so the same surface works
            across every adapter.
        response_schema: Optional JSON Schema describing the expected
            output shape.  Used only when ``response_format == "json"``;
            providers that support schema-constrained output (e.g.
            OpenAI, Gemini) consume it directly.  Providers that only
            support plain JSON-mode ignore the schema and rely on the
            prompt to enforce shape.
    """

    prompt: str
    model: str
    system: str | None = None
    temperature: float = 0.1
    max_output_tokens: int = 2048
    routing_hints: OpenRouterRoutingHints | None = None
    response_format: Literal["text", "json"] = "text"
    response_schema: dict | None = None


@dataclass
class GenerationResponse:
    """Output of :meth:`BaseLLMProvider.generate`.

    Carries the *model's text* plus accounting — not citations or
    retrieved chunks.  The RAG orchestrator composes this with
    retrieval state to produce the higher-level
    :class:`~sec_generative_search.core.types.GenerationResult`.

    Attributes:
        text: The model's generated text.
        model: Model slug that produced the output (echoed back for
            logging and provenance).
        token_usage: Input/output token accounting.  For streaming
            responses, the final yielded chunk carries the final total;
            earlier chunks may leave this zero.
        finish_reason: Provider-reported reason the generation stopped
            (``"stop"``, ``"length"``, ``"content_filter"``, ...).
    """

    text: str
    model: str
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = "stop"


# ---------------------------------------------------------------------------
# Reranker result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankResult:
    """Single reranker output — the index into the original document
    list and the reranker's score for it.

    Frozen so result lists can be safely shared across threads without
    defensive copying.
    """

    index: int
    score: float


# ---------------------------------------------------------------------------
# Shared base — key handling, repr, capabilities
# ---------------------------------------------------------------------------


class _ProviderBase(ABC):
    """Shared construction, key-handling, and :func:`repr` logic.

    Not intended for direct subclassing by concrete providers — pick one
    of :class:`BaseLLMProvider`, :class:`BaseEmbeddingProvider`, or
    :class:`BaseRerankerProvider`.
    """

    # Concrete subclasses override this.  Empty on the ABCs triggers a
    # clear error at instantiation time; it is checked in ``__init__``
    # rather than via ``__init_subclass__`` so test doubles can inherit
    # from the ABCs directly and set ``provider_name`` in the subclass
    # body.
    provider_name: ClassVar[str] = ""

    def __init__(self, api_key: str) -> None:
        cls = type(self)
        if not cls.provider_name:
            raise TypeError(f"{cls.__name__} must set the `provider_name` class attribute")
        if not isinstance(api_key, str):
            raise TypeError(f"api_key must be a str, got {type(api_key).__name__}")
        if not api_key:
            raise ValueError("api_key must not be empty")
        # Leading underscore — never serialised by dataclass helpers,
        # never read by public accessors.  The only sanctioned way to
        # expose the key is the SDK call the concrete provider makes.
        self._api_key = api_key

    @final
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(provider={self.provider_name!r}, "
            f"api_key={mask_secret(self._api_key)})"
        )

    @final
    def __str__(self) -> str:
        return self.__repr__()

    def close(self) -> None:
        """Release the provider's network client, if it owns one.

        Every hosted adapter constructs an SDK client (``OpenAI`` /
        ``Anthropic`` / ``genai.Client``) that owns an
        :mod:`httpx` connection pool; that pool — and the
        credential-bearing transport it holds — is otherwise released
        only at garbage collection, which is non-deterministic under a
        reference cycle.  Closing it at the end of a request gives
        deterministic file-descriptor and transport teardown and shrinks
        the window in which the cleartext key sits in process memory.

        The client attribute is always ``self._client`` where present;
        providers with no network client (the on-device
        :class:`~sec_generative_search.providers.local.LocalEmbeddingProvider`)
        inherit a safe no-op.  Idempotent and quiet: closing an
        already-closed or client-less provider never raises, so the call
        is safe inside a ``finally``.
        """
        client = getattr(self, "_client", None)
        client_close = getattr(client, "close", None)
        if not callable(client_close):
            return
        try:
            client_close()
        except Exception:
            # close() is called from route / producer-thread ``finally``
            # blocks: a failed transport teardown is not actionable and must
            # not shadow the request's own exception path.
            logger.debug("provider client close failed", exc_info=True)

    def __enter__(self) -> _ProviderBase:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @abstractmethod
    def validate_key(self) -> bool:
        """Verify the stored key against the provider.

        Concrete implementations should perform the cheapest call that
        requires authentication (e.g. list models, 1-token completion).
        Returns ``True`` on success.  Raises
        :class:`~sec_generative_search.core.exceptions.ProviderAuthError`
        on an invalid key; other failures raise the appropriate
        :class:`ProviderError` subclass via the resilience wrapper.
        """

    @abstractmethod
    def get_capabilities(self) -> ProviderCapability:
        """Return the capability matrix for this provider/model pair.

        Concrete implementations are expected to cache this
        populates the matrix once at registration.
        """


# ---------------------------------------------------------------------------
# LLM ABC
# ---------------------------------------------------------------------------


class BaseLLMProvider(_ProviderBase):
    """Abstract chat-completion provider.

    Subclasses must:

    - Implement :meth:`generate` and :meth:`generate_stream` so that
      both populate a :class:`TokenUsage` (``generate_stream`` on the
      final yielded chunk).
    - Use :func:`resilient_call` with a provider-specific
      :class:`ExceptionMapping` for every network call so the
      orchestrator sees only normalised :class:`ProviderError`
      subclasses.
    """

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Non-streaming chat completion."""

    @abstractmethod
    def generate_stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationResponse]:
        """Streaming chat completion.

        Yields partial :class:`GenerationResponse` values.  The final
        yielded value MUST populate ``token_usage`` with the total for
        the stream; earlier values MAY leave it empty.
        """


# ---------------------------------------------------------------------------
# Embedding ABC
# ---------------------------------------------------------------------------


class BaseEmbeddingProvider(_ProviderBase):
    """Abstract text-embedding provider.

    Every concrete embedding provider automatically satisfies the
    :class:`~sec_generative_search.pipeline.orchestrator.ChunkEmbedder`
    protocol via the default :meth:`embed_chunks` implementation, so
    :class:`PipelineOrchestrator` can accept any embedding provider
    without a bespoke adapter.
    """

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed *texts* as a ``(len(texts), dimension)`` float32 array."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query as a ``(dimension,)`` float32 array.

        Separate from :meth:`embed_texts` because some providers use a
        distinct instruction prefix for queries versus documents
        (e.g. Cohere, Voyage AI).  Providers without that distinction
        may simply call :meth:`embed_texts` with a one-element list.
        """

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension (columns in
        :meth:`embed_texts` output)."""

    def embed_chunks(
        self,
        chunks: list[Chunk],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Bridge to the orchestrator's ``ChunkEmbedder`` protocol.

        Default implementation pulls ``chunk.content`` from every chunk
        and delegates to :meth:`embed_texts`.  Providers that can emit
        progress updates may override to drive a progress bar; the
        default silently ignores ``show_progress`` so every provider
        works with :class:`PipelineOrchestrator` out of the box.
        """
        del show_progress  # default implementation has no progress surface
        return self.embed_texts([c.content for c in chunks])


# ---------------------------------------------------------------------------
# Reranker ABC
# ---------------------------------------------------------------------------


class BaseRerankerProvider(_ProviderBase):
    """Abstract reranker provider.

    Reranking is a post-retrieval re-ordering of a short candidate
    list.  Subclasses call the vendor's rerank endpoint and return
    :class:`RerankResult` values sorted by score descending.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank *documents* relative to *query*.

        Returns a list of :class:`RerankResult`.  When ``top_k`` is
        supplied the list is truncated to ``top_k`` entries.  The
        original input order is preserved only in the
        :attr:`RerankResult.index` field — the returned list is always
        sorted by score.
        """
