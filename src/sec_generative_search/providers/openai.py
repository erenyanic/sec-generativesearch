"""OpenAI provider adapters.

Concrete subclasses of the OpenAI-compatible bases that target
``api.openai.com`` directly.  No ``base_url`` override; relies on the
SDK's default endpoint.

The ``MODEL_DIMENSIONS`` mapping is the cheap embedding-dimension probe;
LLM capabilities and exact per-MTok cost come from the shared vendored
:mod:`~sec_generative_search.providers.catalogue`, so no extra network
round-trip is needed to know which models support streaming, structured
output, prompt caching, or what they cost.

The coarse :class:`~sec_generative_search.core.types.PricingTier` is
derived from cost at load; the exact spend on a real call is reported via
:class:`TokenUsage`.
"""

from __future__ import annotations

from typing import ClassVar

from sec_generative_search.providers.openai_compat import (
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleLLMProvider,
)

__all__ = [
    "OpenAIEmbeddingProvider",
    "OpenAIProvider",
]


class OpenAIProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider for OpenAI's hosted models."""

    provider_name = "openai"
    default_model = "gpt-6-luna"


class OpenAIEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """Embedding provider for OpenAI's hosted ``text-embedding-3-*`` models."""

    provider_name = "openai"
    default_model = "text-embedding-3-small"

    # Native dimensions (OpenAI also supports the ``dimensions`` request
    # parameter for shrinking; v1 sticks to the native sizes for
    # ChromaDB collection stability).
    MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
