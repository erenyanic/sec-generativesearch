"""OpenAI-compatible provider adapters.

Every vendor that speaks the OpenAI Chat Completions / Embeddings wire
protocol — OpenAI itself, Mistral, Kimi, DeepSeek, Qwen, OpenRouter —
shares the same client construction, error normalisation, and request
shape.  This module captures that shared surface so each concrete
vendor differs only by ``provider_name`` and an optional ``base_url``;
model capabilities come from the shared vendored
:mod:`~sec_generative_search.providers.catalogue`.

Design notes:

- The OpenAI Python SDK exposes one ``OpenAI`` client class for both
  chat and embeddings.  We do **not** combine them into a single
  provider class because :class:`BaseLLMProvider` and
  :class:`BaseEmbeddingProvider` are deliberately separate ABCs
  (some vendors offer only one surface).  Instead, both providers
  share a tiny :class:`_OpenAIClientMixin` that owns the client
  instance, the per-call resilience policy, and the SDK-specific
  exception mapping.

- Every network call goes through :func:`resilient_call` with the
  vendor-supplied :class:`ResilientCallPolicy`.  The orchestrator and
  RAG pipeline therefore only ever see normalised
  :class:`ProviderError` subclasses.

- The OpenAI SDK does not raise an exception for content-filter blocks;
  it sets ``finish_reason="content_filter"`` on the response choice.
  We surface that as :class:`ProviderContentFilterError` directly from
  the response handler — terminal by definition.

- API keys are stored on ``self._api_key`` by the base class.  They
  are passed to the SDK once at client construction and never logged,
  re-exported, or attached to any error message.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from sec_generative_search.core.exceptions import (
    ProviderContentFilterError,
    ProviderError,
)
from sec_generative_search.core.logging import get_logger
from sec_generative_search.core.resilience import (
    ExceptionMapping,
    ResilientCallPolicy,
    RetryPolicy,
    normalise_exception,
    resilient_call,
)
from sec_generative_search.core.types import (
    ProviderCapability,
    TokenUsage,
)
from sec_generative_search.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    GenerationRequest,
    GenerationResponse,
)
from sec_generative_search.providers.catalogue import model_catalogue

if TYPE_CHECKING:
    import numpy as np


__all__ = [
    "OPENAI_EXCEPTION_MAPPING",
    "OpenAICompatibleEmbeddingProvider",
    "OpenAICompatibleLLMProvider",
]


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exception mapping shared by every OpenAI-compatible vendor
# ---------------------------------------------------------------------------

# The OpenAI SDK's hierarchy is uniform across compatible vendors —
# they re-emit the same exception classes through this client.  We map:
#
#   AuthenticationError, PermissionDeniedError -> ProviderAuthError (terminal)
#   RateLimitError                             -> ProviderRateLimitError
#   APITimeoutError, builtin TimeoutError      -> ProviderTimeoutError
#   APIConnectionError                         -> ProviderConnectionError
#
# ``APITimeoutError`` subclasses ``APIConnectionError``, so it appears in
# ``timeout`` and is matched there first (``normalise_exception``
# consults ``timeout`` before ``connection``) — a slow response stays a
# timeout, while a genuinely unreachable endpoint (refused / no route /
# DNS) maps to ProviderConnectionError.  This is the dominant failure of
# a self-hosted ``local_llm`` endpoint that is not running.
#
# Content-filter blocks are surfaced from the response body, not as an
# exception, so they do not appear in this mapping.
OPENAI_EXCEPTION_MAPPING = ExceptionMapping(
    auth=(AuthenticationError, PermissionDeniedError),
    rate_limit=(RateLimitError,),
    timeout=(APITimeoutError, TimeoutError),
    connection=(APIConnectionError,),
)


# ---------------------------------------------------------------------------
# Shared client mixin
# ---------------------------------------------------------------------------


class _OpenAIClientMixin:
    """Owns the OpenAI client and the resilient-call plumbing.

    Subclasses combine this with one of the provider ABCs.  The mixin
    relies on attributes set by :class:`_ProviderBase.__init__`
    (``self._api_key``, ``self.provider_name``) — pick a concrete
    provider ABC as the second base so that ``__init__`` runs first.
    """

    # Concrete subclasses override.  ``None`` -> SDK default endpoint.
    default_base_url: ClassVar[str | None] = None

    # SDK-level timeout in seconds.  Subclasses or callers can override
    # by passing ``timeout=`` to :meth:`_init_client`.
    default_timeout: ClassVar[float] = 60.0

    # Hooks present on the concrete provider via _ProviderBase.
    _api_key: str
    provider_name: ClassVar[str]

    def _init_client(
        self,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        """Construct the ``OpenAI`` client and the resilient-call policy.

        Concrete providers call this from their ``__init__`` after the
        base ABC has stored the API key.  We deliberately disable the
        SDK's own retry logic (``max_retries=0``) — retries are owned
        by :func:`resilient_call` so the circuit breaker and
        exponential backoff stay coordinated.
        """
        url = base_url if base_url is not None else self.default_base_url
        wall_clock = timeout if timeout is not None else self.default_timeout
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=url,
            timeout=wall_clock,
            max_retries=0,
        )
        self._policy = ResilientCallPolicy(
            retry_policy=retry_policy or RetryPolicy(),
            exception_mapping=OPENAI_EXCEPTION_MAPPING,
        )

    def _call[T](self, fn: Callable[[], T]) -> T:
        """Run *fn* under the shared resilient-call policy."""
        return resilient_call(fn, provider=self.provider_name, policy=self._policy)


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------


class OpenAICompatibleLLMProvider(_OpenAIClientMixin, BaseLLMProvider):
    """Chat-completion provider over the OpenAI wire protocol.

    Subclasses override ``provider_name`` and ``default_model`` and,
    optionally, ``default_base_url``.  Model capabilities are no longer
    declared per class: the vendored :mod:`~sec_generative_search.providers.catalogue`
    is the single source, keyed by ``(provider_name, slug)``.
    """

    # Default model used when a caller does not supply one.  Concrete
    # subclasses override; left empty here so the ABC fail-fast logic
    # forces every vendor to declare a default explicitly.
    default_model: ClassVar[str] = ""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(api_key)
        self._init_client(base_url=base_url, timeout=timeout, retry_policy=retry_policy)

    # ------------------------------------------------------------------
    # Capability and validation
    # ------------------------------------------------------------------

    def validate_key(self) -> bool:
        """Probe the key with the cheapest authenticated call.

        ``models.list`` returns a small page and counts as one call
        against the lowest-cost rate-limit bucket.  Failure raises a
        normalised :class:`ProviderError` subclass via the resilience
        wrapper; success returns ``True``.
        """
        self._call(lambda: self._client.models.list())
        return True

    def get_capabilities(self, model: str | None = None) -> ProviderCapability:
        """Return the static capability matrix for *model*.

        Falls back to the provider's ``default_model`` when *model* is
        ``None``.  Reads the vendored catalogue keyed by
        ``(provider_name, slug)``; unknown models receive a permissive
        ``ProviderCapability(chat=True, streaming=True)`` rather than a
        raise — the SDK will reject the slug at call time with a clear
        error if the vendor does not actually serve it.
        """
        slug = model or self.default_model
        cap = model_catalogue().get_llm_capability(self.provider_name, slug)
        if cap is not None:
            return cap
        return ProviderCapability(chat=True, streaming=True)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Non-streaming chat completion."""
        messages = self._build_messages(request)
        extra = self._extra_request_kwargs(request)

        def call() -> Any:
            return self._client.chat.completions.create(
                model=request.model or self.default_model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_output_tokens,
                stream=False,
                **extra,
            )

        completion = self._call(call)
        choice = completion.choices[0]
        finish_reason = choice.finish_reason or "stop"
        # Content-filter terminations are terminal — surface them as a
        # ProviderError before the orchestrator sees an empty string.
        if finish_reason == "content_filter":
            raise ProviderContentFilterError(
                f"{self.provider_name} safety filter blocked the response",
                provider=self.provider_name,
                hint="Reformulate the prompt or route to a different provider.",
            )

        usage = completion.usage
        token_usage = TokenUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        return GenerationResponse(
            text=choice.message.content or "",
            model=completion.model,
            token_usage=token_usage,
            finish_reason=finish_reason,
        )

    def generate_stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationResponse]:
        """Streaming chat completion.

        Yields one :class:`GenerationResponse` per SDK chunk.  Earlier
        chunks carry the partial text and an empty :class:`TokenUsage`;
        the final chunk carries the empty text plus the aggregate token
        accounting reported by the ``stream_options`` payload.
        """
        messages = self._build_messages(request)
        extra = self._extra_request_kwargs(request)

        def call() -> Any:
            return self._client.chat.completions.create(
                model=request.model or self.default_model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_output_tokens,
                stream=True,
                stream_options={"include_usage": True},
                **extra,
            )

        stream = self._call(call)
        model_slug = request.model or self.default_model

        # Iteration runs outside ``_call`` because it spans many SDK
        # network round-trips; if a chunk raises, surface the exception
        # without retry.  This matches the API contract — partial
        # output has already been observed by the caller.  The loop is
        # still wrapped so a mid-stream SDK failure (a dropped
        # connection, an in-flight rate-limit) is normalised to the same
        # ``ProviderError`` subclass the initial ``_call`` would have
        # produced — otherwise the raw SDK type escapes the orchestrator's
        # ``except ProviderError`` and is misclassified as an internal
        # error.  Classification only; the no-retry-after-partial-output
        # contract stays intact.
        try:
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    # Final usage-only chunk has empty ``choices``.
                    usage = getattr(chunk, "usage", None)
                    if usage is not None:
                        yield GenerationResponse(
                            text="",
                            model=getattr(chunk, "model", model_slug),
                            token_usage=TokenUsage(
                                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                            ),
                            finish_reason="stop",
                        )
                    continue
                choice = choices[0]
                finish_reason = choice.finish_reason or "stop"
                if finish_reason == "content_filter":
                    raise ProviderContentFilterError(
                        f"{self.provider_name} safety filter blocked the response",
                        provider=self.provider_name,
                        hint="Reformulate the prompt or route to a different provider.",
                    )
                delta_text = getattr(choice.delta, "content", None) or ""
                if not delta_text and finish_reason == "stop":
                    # Empty deltas are common; emit only when there is real
                    # text to convey or a non-default finish reason.
                    continue
                yield GenerationResponse(
                    text=delta_text,
                    model=getattr(chunk, "model", model_slug),
                    token_usage=TokenUsage(),
                    finish_reason=finish_reason,
                )
        except ProviderError:
            # Already normalised (e.g. the content-filter block above, or
            # a ``ProviderError`` raised deeper in) — re-raise as-is so a
            # terminal type stays terminal.
            raise
        except Exception as exc:  # SDK-raised mid-stream
            raise normalise_exception(
                exc,
                provider=self.provider_name,
                mapping=OPENAI_EXCEPTION_MAPPING,
            ) from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(request: GenerationRequest) -> list[dict[str, str]]:
        """Render a :class:`GenerationRequest` into the OpenAI message list."""
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        return messages

    def _extra_request_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        """Hook for vendor-specific SDK kwargs.

        Splatted into :meth:`generate` / :meth:`generate_stream`'s call
        to ``chat.completions.create``.  The default implementation
        forwards :attr:`GenerationRequest.response_format` ``"json"``
        as the OpenAI Chat Completions ``response_format`` argument.
        When a JSON schema is supplied, it is forwarded as
        ``json_schema``-shaped ``response_format``; without a schema,
        ``json_object`` is used (plain JSON-mode).

        Overridden by :class:`~sec_generative_search.providers.openrouter.OpenRouterProvider`
        to forward routing hints via ``extra_body``; subclasses must
        ``super()`` into this default so JSON-mode keeps working.
        """
        kwargs: dict[str, Any] = {}
        if request.response_format == "json":
            if request.response_schema is not None:
                # OpenAI's "json_schema" response_format expects a
                # ``{"name": ..., "schema": ...}`` envelope.  We fill in
                # a stable ``name`` because the SDK rejects an empty
                # value; callers who want a custom name can override the
                # ``$id`` field on the schema instead.
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": request.response_schema.get("$id", "rag_response"),
                        "schema": request.response_schema,
                        "strict": True,
                    },
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}
        return kwargs


# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------


class OpenAICompatibleEmbeddingProvider(_OpenAIClientMixin, BaseEmbeddingProvider):
    """Embedding provider over the OpenAI wire protocol.

    Subclasses declare a ``MODEL_DIMENSIONS`` mapping plus a
    ``default_model``; everything else is shared.  Numpy is imported
    lazily so embedding providers do not pull numpy into modules that
    only need the LLM surface.
    """

    MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {}
    default_model: ClassVar[str] = ""

    def __init__(
        self,
        api_key: str,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(api_key)
        self._init_client(base_url=base_url, timeout=timeout, retry_policy=retry_policy)
        self._model = model or self.default_model
        if not self._model:
            raise ValueError(
                f"{type(self).__name__} requires a model — pass one explicitly "
                "or set `default_model` on the subclass."
            )
        if self._model not in self.MODEL_DIMENSIONS:
            raise ValueError(
                f"Unknown embedding model '{self._model}' for "
                f"{self.provider_name}. Add it to MODEL_DIMENSIONS."
            )

    # ------------------------------------------------------------------
    # Capability and validation
    # ------------------------------------------------------------------

    def validate_key(self) -> bool:
        self._call(lambda: self._client.models.list())
        return True

    def get_capabilities(self) -> ProviderCapability:
        return ProviderCapability(embeddings=True)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed *texts* in a single SDK call.

        Empty input yields an empty ``(0, dimension)`` array — never an
        SDK round-trip.  The OpenAI embeddings endpoint rejects empty
        input arrays, so this branch is both an optimisation and a
        defensive guard.
        """
        import numpy as np

        if not texts:
            return np.zeros((0, self.get_dimension()), dtype=np.float32)

        def call() -> Any:
            return self._client.embeddings.create(model=self._model, input=texts)

        response = self._call(call)
        # Embeddings come back in input order; build a (n, d) array.
        return np.asarray(
            [item.embedding for item in response.data],
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query as a 1-D float32 array."""
        matrix = self.embed_texts([text])
        return matrix[0]

    def get_dimension(self) -> int:
        return self.MODEL_DIMENSIONS[self._model]
