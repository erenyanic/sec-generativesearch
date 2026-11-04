"""Qwen (Alibaba DashScope) provider adapters.

Alibaba's DashScope exposes an OpenAI-compatible surface at
``https://dashscope-intl.aliyuncs.com/compatible-mode/v1`` for
international accounts.  Both chat and embedding surfaces are available
and are modelled separately here so each can be used independently.
"""

from __future__ import annotations

from typing import ClassVar

from sec_generative_search.providers.openai_compat import (
    OpenAICompatibleEmbeddingProvider,
    OpenAICompatibleLLMProvider,
)

__all__ = [
    "QwenEmbeddingProvider",
    "QwenProvider",
]


_QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class QwenProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting DashScope's Qwen models."""

    provider_name = "qwen"
    default_base_url = _QWEN_BASE_URL
    default_model = "qwen3.8-flash"


class QwenEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """Embedding provider for DashScope's ``text-embedding-v*`` models."""

    provider_name = "qwen"
    default_base_url = _QWEN_BASE_URL
    default_model = "qwen3-rerank"

    MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "qwen3.7-text-embedding": 1024,
        "qwen3-rerank": 1024,
        "text-embedding-v4": 1024,
        "text-embedding-v3": 1024,
    }
