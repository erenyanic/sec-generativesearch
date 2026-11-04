"""Mistral provider adapters.

Mistral's La Plateforme exposes an OpenAI-compatible Chat Completions
and Embeddings surface at ``https://api.mistral.ai/v1``.  Both the chat
and embedding surfaces subclass the shared OpenAI-compatible bases and
differ only in ``provider_name``, ``default_base_url``, ``default_model``,
and the static catalogue that drives the O(1) capability probe.

Pricing tiers are coarse buckets, not real cost figures (same pattern as
:mod:`providers.openai`); the tier exists so the UI can hint
"cheap vs. premium" without consulting a per-token cost table.
"""

from __future__ import annotations

from typing import ClassVar

from sec_generative_search.providers.openai_compat import (
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleLLMProvider,
)

__all__ = [
    "MistralEmbeddingProvider",
    "MistralProvider",
]


class MistralProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting ``api.mistral.ai``."""

    provider_name = "mistral"
    default_base_url = "https://api.mistral.ai/v1"
    default_model = "mistral-small-2603"


class MistralEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """Embedding provider for Mistral's hosted embedding models."""

    provider_name = "mistral"
    default_base_url = "https://api.mistral.ai/v1"
    default_model = "codestral-embed"

    MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "mistral-embed": 1024,
        "codestral-embed": 1536,
    }
