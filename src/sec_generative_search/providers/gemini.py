"""Google Gemini provider adapters.

Concrete adapters that target Google's Gemini API via the first-party
``google-genai`` SDK.  Unlike the OpenAI SDK, the genai client raises a
single :class:`~google.genai.errors.APIError` hierarchy and encodes
error category in an HTTP status ``.code`` attribute — so error
classification happens inside the per-call wrapper (by inspecting
``exc.code``) rather than via a type-only :class:`ExceptionMapping`.
:class:`resilient_call` still sees :class:`ProviderError` subclasses
after the translation and applies retry / terminal semantics exactly
as it does for every other provider.

Chat safety blocks do *not* surface as exceptions.  They arrive on the
response body as ``candidates[0].finish_reason`` ``SAFETY``
(or one of the ``PROHIBITED_CONTENT`` / ``BLOCKLIST`` variants) or as
``prompt_feedback.block_reason`` on fully-blocked prompts.  Both paths
raise :class:`ProviderContentFilterError` directly — terminal, matching
the OpenAI content-filter contract.

Token counting is offline via :mod:`tiktoken`'s ``cl100k_base`` for the
same reason as the Anthropic adapter: the context-window packer
needs a cheap estimate *before* calling the model.  ``cl100k_base`` is
not Gemini's native tokeniser, so the count is an approximation biased
slightly high — the right direction for a budget guard.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar

from google import genai
from google.genai import errors, types

from sec_generative_search.core.exceptions import (
    ProviderAuthError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sec_generative_search.core.logging import get_logger
from sec_generative_search.core.resilience import (
    ExceptionMapping,
    ResilientCallPolicy,
    RetryPolicy,
    normalise_exception,
    resilient_call,
    retry_after_from_exception,
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
    "GEMINI_EXCEPTION_MAPPING",
    "GeminiEmbeddingProvider",
    "GeminiProvider",
]


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exception mapping
# ---------------------------------------------------------------------------

# The genai SDK collapses every failure into :class:`errors.APIError`
# with an HTTP ``.code`` attribute.  We translate those to our terminal
# / retryable types *inside* the call wrapper (see ``_translate``),
# then let :func:`resilient_call` treat the :class:`ProviderError`
# subclasses as usual.  Only ``TimeoutError`` needs a type-based
# mapping, since :func:`with_timeout` raises that stdlib exception
# rather than an SDK one.
GEMINI_EXCEPTION_MAPPING = ExceptionMapping(
    timeout=(TimeoutError,),
)


# HTTP status codes the Gemini API uses for authentication / rate-limit
# signals.  ``401`` is the canonical auth failure and ``403`` covers
# API-key permission denials (e.g. key not enabled for the model).
_AUTH_STATUS_CODES = frozenset({401, 403})
_RATE_LIMIT_STATUS_CODES = frozenset({429})
_TIMEOUT_STATUS_CODES = frozenset({408, 504})

# Google APIs signal "retry after N" in the JSON error body rather than a
# header: a ``google.rpc.RetryInfo`` entry in ``error.details`` whose
# ``retryDelay`` is a protobuf Duration string (``"37s"``, ``"0.5s"``).
# The body is untrusted upstream input — only a bounded number of detail
# entries is scanned and only a strictly shaped duration is accepted.
_RETRY_INFO_TYPE = "type.googleapis.com/google.rpc.RetryInfo"
_RETRY_DELAY_RE = re.compile(r"\d{1,9}(?:\.\d{1,9})?s")
_MAX_ERROR_DETAILS_SCANNED = 16

# Finish reasons that indicate Gemini's safety system suppressed the
# response.  ``SAFETY``, ``PROHIBITED_CONTENT``, ``BLOCKLIST``, and
# ``SPII`` all mean "blocked content" from the user's perspective; the
# orchestrator does not need to distinguish between them.
_CONTENT_FILTER_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "PROHIBITED_CONTENT",
        "BLOCKLIST",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
    }
)


def _translate_api_error(exc: errors.APIError, *, provider: str) -> ProviderError:
    """Convert a Gemini ``APIError`` into a :class:`ProviderError` subclass.

    Classification is purely by HTTP status code — the SDK does not
    carry structured error categories.  Anything outside the known
    buckets falls through to a generic :class:`ProviderError` so
    retry policy still kicks in.
    """
    code = getattr(exc, "code", None) or 0
    detail = str(exc) or type(exc).__name__
    if code in _AUTH_STATUS_CODES:
        return ProviderAuthError(
            f"Authentication failed against {provider}",
            provider=provider,
            hint="Verify the API key is correct and enabled for the requested model.",
            details=detail,
        )
    if code in _RATE_LIMIT_STATUS_CODES:
        return ProviderRateLimitError(
            f"Rate limit exceeded for {provider}",
            provider=provider,
            hint="Retry after backoff or request a higher quota from Google AI Studio.",
            details=detail,
            retry_after=_gemini_retry_after(exc),
        )
    if code in _TIMEOUT_STATUS_CODES:
        return ProviderTimeoutError(
            f"{provider} call timed out (HTTP {code})",
            provider=provider,
            hint="Increase PROVIDER_TIMEOUT or retry with fewer tokens.",
            details=detail,
        )
    return ProviderError(
        f"{provider} call failed with HTTP {code}",
        provider=provider,
        details=detail,
    )


def _gemini_retry_after(exc: errors.APIError) -> float | None:
    """Seconds a Gemini 429 asked the caller to wait, or ``None``.

    A ``Retry-After`` header (the generic path) wins; otherwise the
    ``google.rpc.RetryInfo`` detail in the error body.  Never raises.
    """
    from_header = retry_after_from_exception(exc)
    if from_header is not None:
        return from_header
    body = getattr(exc, "details", None)
    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    details = error.get("details") if isinstance(error, dict) else None
    if not isinstance(details, list):
        return None
    for entry in details[:_MAX_ERROR_DETAILS_SCANNED]:
        if isinstance(entry, dict) and entry.get("@type") == _RETRY_INFO_TYPE:
            delay = entry.get("retryDelay")
            if isinstance(delay, str) and _RETRY_DELAY_RE.fullmatch(delay):
                return float(delay[:-1])
            return None
    return None


# ---------------------------------------------------------------------------
# Shared client/policy construction
# ---------------------------------------------------------------------------


class _GeminiClientMixin:
    """Owns the ``genai.Client`` and the resilient-call plumbing.

    Kept separate from the ABCs so both the LLM and embedding providers
    can share the construction path without depending on each other's
    abstract surface.
    """

    default_timeout: ClassVar[float] = 60.0

    # Hooks populated by ``_ProviderBase.__init__``.
    _api_key: str
    provider_name: ClassVar[str]

    def _init_client(
        self,
        *,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        wall_clock = timeout if timeout is not None else self.default_timeout
        # The SDK reads the timeout from ``HttpOptions``.  Values are in
        # milliseconds; convert from seconds so callers use the same
        # unit across every provider in the project.
        http_options = types.HttpOptions(timeout=int(wall_clock * 1000))
        self._client = genai.Client(api_key=self._api_key, http_options=http_options)
        self._policy = ResilientCallPolicy(
            retry_policy=retry_policy or RetryPolicy(),
            exception_mapping=GEMINI_EXCEPTION_MAPPING,
            timeout=0.0,
        )

    def _call[T](self, fn: Callable[[], T]) -> T:
        """Run *fn* under the shared resilient-call policy.

        Translates the genai ``APIError`` hierarchy into our
        :class:`ProviderError` subclasses *before* :func:`resilient_call`
        sees the exception; that way terminal (auth) errors are
        honoured without extra retries.
        """

        def wrapped() -> T:
            try:
                return fn()
            except errors.APIError as exc:
                raise _translate_api_error(exc, provider=self.provider_name) from exc

        return resilient_call(wrapped, provider=self.provider_name, policy=self._policy)


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------


class GeminiProvider(_GeminiClientMixin, BaseLLMProvider):
    """Chat-completion provider for Google's Gemini models."""

    provider_name: ClassVar[str] = "gemini"
    default_model: ClassVar[str] = "gemini-3-flash-preview"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(api_key)
        self._init_client(timeout=timeout, retry_policy=retry_policy)
        self._encoder: Any | None = None

    # ------------------------------------------------------------------
    # Capability and validation
    # ------------------------------------------------------------------

    def validate_key(self) -> bool:
        """Probe the key by listing available models."""

        # ``models.list`` returns a pager that lazily fetches the first
        # page on iteration; consuming ``next`` is enough to exercise
        # the auth path without buffering the whole catalogue.
        def call() -> Any:
            pager = self._client.models.list()
            iterator = iter(pager)
            # Pull the first entry to force an HTTP round-trip.  An
            # empty model list is theoretically possible, hence the
            # ``next(..., None)`` fallback instead of raising.
            next(iterator, None)
            return True

        self._call(call)
        return True

    def get_capabilities(self, model: str | None = None) -> ProviderCapability:
        slug = model or self.default_model
        cap = model_catalogue().get_llm_capability(self.provider_name, slug)
        if cap is not None:
            return cap
        return ProviderCapability(chat=True, streaming=True)

    # ------------------------------------------------------------------
    # Generation — non-streaming
    # ------------------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        model_slug = request.model or self.default_model
        config = self._build_config(request)

        def call() -> Any:
            return self._client.models.generate_content(
                model=model_slug,
                contents=request.prompt,
                config=config,
            )

        response = self._call(call)
        self._check_prompt_feedback(response)
        finish_reason = self._finish_reason(response)
        if finish_reason in _CONTENT_FILTER_FINISH_REASONS:
            raise ProviderContentFilterError(
                f"{self.provider_name} safety filter blocked the response ({finish_reason})",
                provider=self.provider_name,
                hint="Reformulate the prompt or route to a different provider.",
            )

        text = self._extract_text(response)
        usage = getattr(response, "usage_metadata", None)
        token_usage = TokenUsage(
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )
        return GenerationResponse(
            text=text,
            model=getattr(response, "model_version", model_slug) or model_slug,
            token_usage=token_usage,
            finish_reason=self._normalise_finish_reason(finish_reason),
        )

    # ------------------------------------------------------------------
    # Generation — streaming
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationResponse]:
        """Streaming ``generate_content_stream`` call.

        Each SDK chunk is a :class:`GenerateContentResponse` — we yield
        the text delta as its own frame and carry forward the final
        usage metadata into a trailing usage-only frame, matching the
        contract every other provider in the package follows.
        """
        model_slug = request.model or self.default_model
        config = self._build_config(request)

        def call() -> Any:
            return self._client.models.generate_content_stream(
                model=model_slug,
                contents=request.prompt,
                config=config,
            )

        stream = self._call(call)

        input_tokens = 0
        output_tokens = 0
        final_finish_reason = "STOP"

        # Iteration runs outside ``_call`` (it spans many SDK network
        # round-trips), so the loop mirrors ``_call``'s translation: a
        # mid-stream ``errors.APIError`` is routed through
        # ``_translate_api_error`` and any other SDK failure through
        # ``normalise_exception`` — otherwise the raw SDK type escapes the
        # orchestrator's ``except ProviderError`` and is misclassified as
        # an internal error.  Classification only, no retry after partial
        # output.
        try:
            for chunk in stream:
                self._check_prompt_feedback(chunk)

                finish_reason = self._finish_reason(chunk)
                if finish_reason:
                    if finish_reason in _CONTENT_FILTER_FINISH_REASONS:
                        raise ProviderContentFilterError(
                            f"{self.provider_name} safety filter blocked mid-stream "
                            f"({finish_reason})",
                            provider=self.provider_name,
                            hint="Reformulate the prompt or route to a different provider.",
                        )
                    final_finish_reason = finish_reason

                text = self._extract_text(chunk)
                if text:
                    yield GenerationResponse(
                        text=text,
                        model=model_slug,
                        token_usage=TokenUsage(),
                        finish_reason="stop",
                    )

                usage = getattr(chunk, "usage_metadata", None)
                if usage is not None:
                    input_tokens = (
                        getattr(usage, "prompt_token_count", input_tokens) or input_tokens
                    )
                    output_tokens = (
                        getattr(usage, "candidates_token_count", output_tokens) or output_tokens
                    )
        except ProviderError:
            # Already normalised (the content-filter block above, or a
            # ``ProviderError`` raised deeper in) — re-raise as-is.
            raise
        except errors.APIError as exc:
            raise _translate_api_error(exc, provider=self.provider_name) from exc
        except Exception as exc:  # non-APIError SDK failure mid-stream
            raise normalise_exception(
                exc,
                provider=self.provider_name,
                mapping=GEMINI_EXCEPTION_MAPPING,
            ) from exc

        yield GenerationResponse(
            text="",
            model=model_slug,
            token_usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            finish_reason=self._normalise_finish_reason(final_finish_reason),
        )

    # ------------------------------------------------------------------
    # Token counting — offline approximation
    # ------------------------------------------------------------------

    def count_tokens(self, text: str, model: str | None = None) -> int:
        """Offline token approximation via ``cl100k_base``.

        The SDK ships ``models.count_tokens`` but it is a network call.
        The context-window packer budgets prompts *before* a
        generation call, so we favour a cheap, over-estimating local
        counter rather than round-tripping every budget check.
        """
        del model
        if self._encoder is None:
            import tiktoken

            self._encoder = tiktoken.get_encoding("cl100k_base")
        return len(self._encoder.encode(text))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_config(request: GenerationRequest) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            system_instruction=request.system or None,
        )

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the text from a ``GenerateContentResponse``-shaped object.

        Prefers the SDK's own ``.text`` property (which concatenates
        every text part) and falls back to manual part concatenation
        so tests can mock minimal response shapes without having to
        replicate the full property implementation.
        """
        text = getattr(response, "text", None)
        if isinstance(text, str) and text:
            return text
        parts_text: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", None) or []:
                piece = getattr(part, "text", "")
                if piece:
                    parts_text.append(piece)
        return "".join(parts_text)

    @staticmethod
    def _finish_reason(response: Any) -> str:
        """Return the first candidate's ``finish_reason`` as a plain string."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""
        reason = getattr(candidates[0], "finish_reason", None)
        if reason is None:
            return ""
        # ``FinishReason`` is an ``Enum``; fall back to ``str(reason)``
        # so bare strings from test doubles also work.
        return getattr(reason, "name", None) or str(reason)

    def _check_prompt_feedback(self, response: Any) -> None:
        """Raise :class:`ProviderContentFilterError` on a blocked prompt.

        Some Gemini refusals never return a candidate — the API signals
        the block on ``prompt_feedback.block_reason`` instead.  Surface
        that as a terminal error, same as a safety ``finish_reason``.
        """
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is None:
            return
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason is None:
            return
        name = getattr(block_reason, "name", None) or str(block_reason)
        if name and name.upper() not in {"BLOCK_REASON_UNSPECIFIED", ""}:
            raise ProviderContentFilterError(
                f"{self.provider_name} safety filter blocked the prompt ({name})",
                provider=self.provider_name,
                hint="Reformulate the prompt or route to a different provider.",
            )

    @staticmethod
    def _normalise_finish_reason(finish_reason: str) -> str:
        """Map Gemini finish reasons onto the shared vocabulary."""
        if not finish_reason:
            return "stop"
        normalised = finish_reason.upper()
        if normalised in {"STOP", "FINISH_REASON_UNSPECIFIED"}:
            return "stop"
        if normalised == "MAX_TOKENS":
            return "length"
        if normalised in _CONTENT_FILTER_FINISH_REASONS:
            return "content_filter"
        return normalised.lower()


# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------


class GeminiEmbeddingProvider(_GeminiClientMixin, BaseEmbeddingProvider):
    """Embedding provider for Google's hosted embedding models."""

    provider_name: ClassVar[str] = "gemini"
    default_model: ClassVar[str] = "gemini-embedding-2"

    # Native dimensions. The GA ``gemini-embedding-2`` is the default — a
    # defaulted run must not depend on a preview endpoint. It emits 3072
    # dimensions natively while also supporting smaller request-time
    # projections; the registry uses the native default. The superseded
    # ``gemini-embedding-2-preview`` slug is retained so a collection
    # already stamped against it still opens (selection is explicit).
    MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "gemini-embedding-2": 3072,
        "gemini-embedding-2-preview": 3072,
        "gemini-embedding-001": 3072,
    }

    def __init__(
        self,
        api_key: str,
        *,
        model: str | None = None,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(api_key)
        self._init_client(timeout=timeout, retry_policy=retry_policy)
        self._model = model or self.default_model
        if self._model not in self.MODEL_DIMENSIONS:
            raise ValueError(
                f"Unknown embedding model '{self._model}' for "
                f"{self.provider_name}. Add it to MODEL_DIMENSIONS."
            )

    # ------------------------------------------------------------------
    # Capability and validation
    # ------------------------------------------------------------------

    def validate_key(self) -> bool:
        def call() -> Any:
            pager = self._client.models.list()
            iterator = iter(pager)
            next(iterator, None)
            return True

        self._call(call)
        return True

    def get_capabilities(self) -> ProviderCapability:
        return ProviderCapability(embeddings=True)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed *texts* as a ``(len(texts), dimension)`` float32 array.

        Empty input short-circuits without an SDK call — the embed
        endpoint rejects empty arrays, and skipping the round-trip is
        also a meaningful optimisation when callers filter out empty
        chunks upstream.
        """
        import numpy as np

        if not texts:
            return np.zeros((0, self.get_dimension()), dtype=np.float32)

        def call() -> Any:
            return self._client.models.embed_content(
                model=self._model,
                contents=list(texts),
            )

        response = self._call(call)
        embeddings = getattr(response, "embeddings", None) or []
        vectors = [getattr(emb, "values", []) or [] for emb in embeddings]
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        matrix = self.embed_texts([text])
        return matrix[0]

    def get_dimension(self) -> int:
        return self.MODEL_DIMENSIONS[self._model]
