"""DeepSeek provider adapter.

DeepSeek's API is OpenAI-compatible at ``https://api.deepseek.com/v1``.
Only the chat surface is exposed — DeepSeek does not ship a first-party
embedding API, so pair with a separate embedding provider when using
this adapter end-to-end.
"""

from __future__ import annotations

from sec_generative_search.providers.openai_compat import (
    OpenAICompatibleLLMProvider,
)

__all__ = ["DeepSeekProvider"]


class DeepSeekProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting DeepSeek's hosted models."""

    provider_name = "deepseek"
    default_base_url = "https://api.deepseek.com"
    default_model = "deepseek-flash"
