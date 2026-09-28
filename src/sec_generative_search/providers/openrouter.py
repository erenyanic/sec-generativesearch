"""OpenRouter meta-provider adapter.

OpenRouter is a proxy that multiplexes across many upstream providers
using the OpenAI wire protocol at ``https://openrouter.ai/api/v1``.  The
catalogue is open-ended — OpenRouter can gain or lose models daily — so
this adapter is deliberately absent from the vendored model catalogue and
relies on the base class's permissive-default branch: unknown slugs yield
``ProviderCapability(chat=True, streaming=True)`` and the SDK rejects
unserviceable slugs at call time with a clear error.

Lazy validation against OpenRouter's model list is the ``models.list`` round-trip that
:meth:`validate_key` already performs via the inherited base behaviour.
A richer "does this slug exist right now?" probe belongs in the registry,
not in this adapter.

OpenRouter slugs use the ``vendor/model`` form (e.g. ``openai/gpt-4o``,
``anthropic/claude-sonnet-4-6``).  :data:`default_model` picks a cheap,
widely-available slug so omitting ``model`` on a request still reaches a
live endpoint.

Upstream-provider routing:

- OpenRouter accepts an optional ``provider`` block in the request body
  that pins, allowlists, or blocklists the upstream providers it routes
  to.  Callers pass :class:`OpenRouterRoutingHints` on
  :class:`~sec_generative_search.providers.base.GenerationRequest` and
  :class:`OpenRouterProvider` forwards the block via the SDK's
  ``extra_body`` kwarg.  Every non-OpenRouter adapter silently ignores
  the hint — the OpenAI-compatible base's :meth:`_extra_request_kwargs`
  hook returns an empty dict by default.
- The hint object is pass-through: no validation, no auth material.  The
  authoritative error surface is OpenRouter's API at call time.  The
  OpenRouter API key remains the sole credential in flight — security
  tests in ``tests/providers/test_openrouter.py`` enforce this.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sec_generative_search.providers.openai_compat import (
    OpenAICompatibleLLMProvider,
)

# The hints dataclass lives in an SDK-free module so the CLI / API request
# layers can build it without importing the ``openai`` SDK (F28); re-exported
# here so this remains its documented home.
from sec_generative_search.providers.routing_hints import OpenRouterRoutingHints

if TYPE_CHECKING:
    from sec_generative_search.providers.base import GenerationRequest

__all__ = [
    "OpenRouterProvider",
    "OpenRouterRoutingHints",
]


class OpenRouterProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting ``openrouter.ai``.

    Any model slug accepted by OpenRouter works here — the capability
    probe returns a permissive default for every slug not in the vendored
    catalogue (OpenRouter has no catalogued models by design).  Callers
    that need an accurate capability matrix for a specific slug should
    consult the provider registry.

    When the :class:`~sec_generative_search.providers.base.GenerationRequest`
    carries :class:`OpenRouterRoutingHints`, the hints are forwarded into
    OpenRouter's ``provider`` request block via the SDK's ``extra_body``
    kwarg.  No other provider honours the hint — see
    :meth:`OpenAICompatibleLLMProvider._extra_request_kwargs`.
    """

    provider_name = "openrouter"
    default_base_url = "https://openrouter.ai/api/v1"
    default_model = "qwen/qwen3.6-plus"

    # Intentionally absent from the vendored catalogue — see module
    # docstring.  The base class :meth:`get_capabilities` returns a
    # permissive default for any slug not catalogued, which is the correct
    # semantics for a meta-provider.

    def _extra_request_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        """Forward routing hints into OpenRouter's ``provider`` block.

        Returns an empty dict when no hints are attached, keeping the
        common path identical to every other OpenAI-compatible vendor.
        The hint object is rendered via
        :meth:`OpenRouterRoutingHints.to_provider_block` — a pass-through
        copy with empty / ``None`` fields dropped.  No credential is
        added here; the OpenRouter API key already flows through the SDK
        client's ``Authorization`` header and is the sole credential in
        flight.
        """
        # Inherit the base default's ``response_format`` handling so
        # JSON-mode requests still reach OpenRouter even when routing
        # hints are also present.
        kwargs = super()._extra_request_kwargs(request)
        hints = request.routing_hints
        if hints is None:
            return kwargs
        block = hints.to_provider_block()
        if not block:
            return kwargs
        kwargs["extra_body"] = {"provider": block}
        return kwargs
