"""Kimi (Moonshot AI) provider adapter.

Moonshot AI ships an OpenAI-compatible Chat Completions surface at
``https://api.moonshot.ai/v1`` (international) — Mainland callers use the
``.cn`` mirror, which shares the wire protocol.  Only the chat surface
is exposed; Moonshot does not currently offer a first-party embedding
API, so pair with :class:`OpenAIEmbeddingProvider` /
:class:`MistralEmbeddingProvider` / :class:`GeminiEmbeddingProvider` when
using this adapter end-to-end.
"""

from __future__ import annotations

from sec_generative_search.providers.openai_compat import (
    OpenAICompatibleLLMProvider,
)

__all__ = ["KimiProvider"]


class KimiProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting Moonshot AI's Kimi endpoints."""

    provider_name = "kimi"
    default_base_url = "https://api.moonshot.ai/v1"
    default_model = "kimi-k3.1-pro"
