"""Z.ai / Zhipu AI GLM provider adapter.

Z.ai ships an OpenAI-compatible Chat Completions surface at
``https://api.z.ai/api/paas/v4`` (the general-purpose PaaS endpoint; a
separate ``/api/coding/paas/v4`` endpoint exists specifically for the
Coding plan and is intentionally not used here).  Mainland callers use
the ``open.bigmodel.cn`` mirror, which shares the wire protocol.

Only the chat surface is modelled.  Zhipu does ship first-party
embedding models (``embedding-3`` / ``embedding-2``) but the project's
curated embedding surface is intentionally kept narrow — the existing
OpenAI / Gemini / Mistral / Qwen adapters plus the optional
:class:`LocalEmbeddingProvider` already cover the cloud and on-device
dimensions the storage layer requires, and adding another vendor on the
embedding surface would multiply the ChromaDB collection-dimension
matrix without a matching use case.  Pair ``ZaiProvider`` with one of
the existing embedding adapters when using this vendor end-to-end.
"""

from __future__ import annotations

from sec_generative_search.providers.openai_compat import (
    OpenAICompatibleLLMProvider,
)

__all__ = ["ZaiProvider"]


class ZaiProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting Z.ai's GLM family."""

    provider_name = "zai"
    default_base_url = "https://api.z.ai/api/paas/v4"
    default_model = "glm-5.3-flash"
