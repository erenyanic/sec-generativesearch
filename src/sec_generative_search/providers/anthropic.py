"""Anthropic provider adapter.

Concrete :class:`BaseLLMProvider` that targets Anthropic's Messages API
via the first-party ``anthropic`` SDK.  The SDK's retry loop is disabled
(``max_retries=0``) so retry, timeout, and circuit-breaker behaviour
stay owned by :func:`resilient_call` — the same contract every other
provider in this package follows.

Key notes:

- Anthropic does not ship a hosted embedding model; the embedding path
  is out of scope.  Callers needing embeddings pair this with an
  embedding-only provider (OpenAI, Gemini) — the ``ProviderRegistry``
  in handles the dual-provider wiring.
- The Messages API returns a ``stop_reason`` of ``"refusal"`` when the
  safety system blocks the response.  We treat that as terminal and
  surface it as :class:`ProviderContentFilterError` — same shape as the
  OpenAI content-filter path.  Doing the check on the response body
  (rather than via the exception mapping) matches the SDK's own
  behaviour: refusals come back as valid HTTP 200 responses.
- The adapter carries no tokeniser: prompt budgeting uses the one
  shared offline ``cl100k_base`` counter (``search/retrieval.py``) for
  every provider — never the SDK's network ``messages.count_tokens``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar

from anthropic import (
    Anthropic,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
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
    BaseLLMProvider,
    GenerationRequest,
    GenerationResponse,
)
from sec_generative_search.providers.catalogue import model_catalogue

if TYPE_CHECKING:
    pass


__all__ = [
    "ANTHROPIC_EXCEPTION_MAPPING",
    "AnthropicProvider",
]


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exception mapping
# ---------------------------------------------------------------------------

# The Anthropic SDK raises distinct typed exceptions, so a plain
# ``ExceptionMapping`` captures every non-terminal class cleanly.  Note
# that content-filter refusals are *not* represented here — they arrive
# as valid responses with ``stop_reason="refusal"`` and are raised from
# the response handler (terminal, bypassing the retry loop).
#
# As with the OpenAI SDK, ``APITimeoutError`` subclasses
# ``APIConnectionError``; ``normalise_exception`` consults ``timeout``
# before ``connection`` so a slow response stays a timeout and only a
# genuinely unreachable endpoint maps to ProviderConnectionError.
ANTHROPIC_EXCEPTION_MAPPING = ExceptionMapping(
    auth=(AuthenticationError, PermissionDeniedError),
    rate_limit=(RateLimitError,),
    timeout=(APITimeoutError, TimeoutError),
    connection=(APIConnectionError,),
)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class AnthropicProvider(BaseLLMProvider):
    """Chat-completion provider for Anthropic's hosted Claude models."""

    provider_name: ClassVar[str] = "anthropic"
    default_model: ClassVar[str] = "claude-haiku-4-5"
    default_timeout: ClassVar[float] = 60.0

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        super().__init__(api_key)
        wall_clock = timeout if timeout is not None else self.default_timeout
        # Disable the SDK's own retry loop — ``resilient_call`` owns it.
        self._client = Anthropic(
            api_key=self._api_key,
            base_url=base_url,
            timeout=wall_clock,
            max_retries=0,
        )
        self._policy = ResilientCallPolicy(
            retry_policy=retry_policy or RetryPolicy(),
            exception_mapping=ANTHROPIC_EXCEPTION_MAPPING,
        )

    def _call[T](self, fn: Callable[[], T]) -> T:
        return resilient_call(fn, provider=self.provider_name, policy=self._policy)

    # ------------------------------------------------------------------
    # Capability and validation
    # ------------------------------------------------------------------

    def validate_key(self) -> bool:
        """Probe the key with the cheapest authenticated call.

        ``client.models.list`` returns a small page behind the same
        authentication bucket as generation, so it is the right
        validation surface.  Any failure propagates as a normalised
        :class:`ProviderError` subclass via the resilience wrapper.
        """
        self._call(lambda: self._client.models.list())
        return True

    def get_capabilities(self, model: str | None = None) -> ProviderCapability:
        """Return the static capability matrix for *model*.

        Reads the vendored catalogue keyed by ``(provider_name, slug)``;
        unknown slugs receive a permissive
        ``ProviderCapability(chat=True, streaming=True)`` — same
        semantics as the OpenAI-compatible bases, so the SDK rejects
        unsupported slugs at call time with a clear error rather than
        here.
        """
        slug = model or self.default_model
        cap = model_catalogue().get_llm_capability(self.provider_name, slug)
        if cap is not None:
            return cap
        return ProviderCapability(chat=True, streaming=True)

    # ------------------------------------------------------------------
    # Generation — non-streaming
    # ------------------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Non-streaming Messages API call."""
        kwargs = self._build_kwargs(request)

        def call() -> Any:
            return self._client.messages.create(stream=False, **kwargs)

        message = self._call(call)
        stop_reason = message.stop_reason or "end_turn"
        # Terminal safety refusal — surface before the caller ever sees
        # the empty content list.
        if stop_reason == "refusal":
            raise ProviderContentFilterError(
                f"{self.provider_name} safety filter refused to respond",
                provider=self.provider_name,
                hint="Reformulate the prompt or route to a different provider.",
            )

        text = self._concat_text_blocks(message.content)
        usage = message.usage
        token_usage = TokenUsage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
        return GenerationResponse(
            text=text,
            model=getattr(message, "model", request.model or self.default_model),
            token_usage=token_usage,
            finish_reason=self._normalise_stop_reason(stop_reason),
        )

    # ------------------------------------------------------------------
    # Generation — streaming
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        request: GenerationRequest,
    ) -> Iterator[GenerationResponse]:
        """Streaming Messages API call.

        Emits one :class:`GenerationResponse` per text-delta event plus
        a final usage-only frame so callers can aggregate token counts
        consistently with the OpenAI-compatible path.
        """
        kwargs = self._build_kwargs(request)
        model_slug = request.model or self.default_model

        def call() -> Any:
            return self._client.messages.create(stream=True, **kwargs)

        stream = self._call(call)

        # Anthropic separates input and output token reporting: the
        # opening ``message_start`` event carries ``input_tokens`` but
        # ``output_tokens=0``; each subsequent ``message_delta`` event
        # replaces ``output_tokens`` with the running total.  Hold the
        # latest value and emit a final usage-only frame.
        input_tokens = 0
        output_tokens = 0
        final_stop_reason = "end_turn"

        # Iteration runs outside ``_call`` (it spans many SDK network
        # round-trips), so wrap the loop to normalise a mid-stream SDK
        # failure to the same ``ProviderError`` subclass the opening call
        # would have produced — a raw ``anthropic.APIConnectionError`` /
        # ``RateLimitError`` raised mid-iteration must not escape the
        # orchestrator's ``except ProviderError``.  Classification only,
        # no retry after partial output.
        try:
            for event in stream:
                event_type = getattr(event, "type", "")

                if event_type == "message_start":
                    message = getattr(event, "message", None)
                    if message is not None:
                        usage = getattr(message, "usage", None)
                        if usage is not None:
                            input_tokens = getattr(usage, "input_tokens", 0) or 0
                            output_tokens = getattr(usage, "output_tokens", 0) or 0
                    continue

                if event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    delta_text = getattr(delta, "text", "") if delta is not None else ""
                    if delta_text:
                        yield GenerationResponse(
                            text=delta_text,
                            model=model_slug,
                            token_usage=TokenUsage(),
                            finish_reason="stop",
                        )
                    continue

                if event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    stop_reason = getattr(delta, "stop_reason", None) if delta is not None else None
                    if stop_reason == "refusal":
                        raise ProviderContentFilterError(
                            f"{self.provider_name} safety filter refused mid-stream",
                            provider=self.provider_name,
                            hint="Reformulate the prompt or route to a different provider.",
                        )
                    if stop_reason:
                        final_stop_reason = stop_reason
                    usage = getattr(event, "usage", None)
                    if usage is not None:
                        output_tokens = (
                            getattr(usage, "output_tokens", output_tokens) or output_tokens
                        )
                    continue

                # ``message_stop`` (and any unknown event types) ends the
                # conversation — nothing to yield here.
                if event_type == "message_stop":
                    break
        except ProviderError:
            # Already normalised (the refusal block above, or a deeper
            # ``ProviderError``) — re-raise so a terminal type stays terminal.
            raise
        except Exception as exc:  # SDK-raised mid-stream
            raise normalise_exception(
                exc,
                provider=self.provider_name,
                mapping=ANTHROPIC_EXCEPTION_MAPPING,
            ) from exc

        yield GenerationResponse(
            text="",
            model=model_slug,
            token_usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
            finish_reason=self._normalise_stop_reason(final_stop_reason),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        """Render a :class:`GenerationRequest` into Messages API kwargs.

        The Anthropic API separates the ``system`` prompt from the
        ``messages`` array; we honour that split so system framing is
        not accidentally interpreted as a user turn.
        """
        kwargs: dict[str, Any] = {
            "model": request.model or self.default_model,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            kwargs["system"] = request.system
        return kwargs

    @staticmethod
    def _concat_text_blocks(blocks: list[Any]) -> str:
        """Concatenate the text of every ``type=="text"`` block."""
        parts: list[str] = []
        for block in blocks or []:
            if getattr(block, "type", "") == "text":
                text = getattr(block, "text", "") or ""
                if text:
                    parts.append(text)
        return "".join(parts)

    @staticmethod
    def _normalise_stop_reason(stop_reason: str) -> str:
        """Map Anthropic stop reasons onto the shared finish-reason vocabulary.

        ``GenerationResponse.finish_reason`` is provider-neutral; a
        ``"length"`` signal from the OpenAI path and Anthropic's
        ``"max_tokens"`` should read the same downstream.
        """
        return {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_use",
            "pause_turn": "pause",
            "refusal": "content_filter",
        }.get(stop_reason, stop_reason or "stop")
