"""Provider abstraction layer for SEC-GenerativeSearch.

Exports the abstract provider contracts, the concrete hosted-vendor
adapters, the optional on-device embedding provider, and the curated
registry used for capability lookup, model listings, and key
validation.

Every name is imported **on first access** (PEP 562, F28): the adapters
import their vendor SDKs (``openai`` / ``anthropic`` / ``google.genai``,
~1.75 s together), and an eager package ``__init__`` made *any*
``sec_generative_search.providers.*`` import — the registry, the
settings validator, ``sec-rag --help`` — pay for all three.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_EXPORTS: dict[str, str] = {
    "AnthropicProvider": "anthropic",
    "ApiKeyResolver": "factory",  # pragma: allowlist secret — a type name, not a key
    "BaseEmbeddingProvider": "base",
    "BaseLLMProvider": "base",
    "BaseRerankerProvider": "base",
    "DeepSeekProvider": "deepseek",
    "GeminiEmbeddingProvider": "gemini",
    "GeminiProvider": "gemini",
    "GenerationRequest": "base",
    "GenerationResponse": "base",
    "GrokProvider": "grok",
    "KimiProvider": "kimi",
    "LocalEmbeddingProvider": "local",
    "LocalLLMProvider": "local_llm",
    "MimoProvider": "mimo",
    "MiniMaxProvider": "minimax",
    "MistralEmbeddingProvider": "mistral",
    "MistralProvider": "mistral",
    "OpenAIEmbeddingProvider": "openai",
    "OpenAIProvider": "openai",
    "OpenRouterProvider": "openrouter",
    "ProviderEntry": "registry",
    "ProviderRegistry": "registry",
    "ProviderSurface": "registry",
    "QwenEmbeddingProvider": "qwen",
    "QwenProvider": "qwen",
    "RerankResult": "base",
    "ZaiProvider": "zai",
    "build_embedder": "factory",
    "default_api_key_resolver": "factory",  # pragma: allowlist secret — a function name
}

__all__ = [
    "AnthropicProvider",
    "ApiKeyResolver",
    "BaseEmbeddingProvider",
    "BaseLLMProvider",
    "BaseRerankerProvider",
    "DeepSeekProvider",
    "GeminiEmbeddingProvider",
    "GeminiProvider",
    "GenerationRequest",
    "GenerationResponse",
    "GrokProvider",
    "KimiProvider",
    "LocalEmbeddingProvider",
    "LocalLLMProvider",
    "MimoProvider",
    "MiniMaxProvider",
    "MistralEmbeddingProvider",
    "MistralProvider",
    "OpenAIEmbeddingProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "ProviderEntry",
    "ProviderRegistry",
    "ProviderSurface",
    "QwenEmbeddingProvider",
    "QwenProvider",
    "RerankResult",
    "ZaiProvider",
    "build_embedder",
    "default_api_key_resolver",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:
    from sec_generative_search.providers.anthropic import (
        AnthropicProvider,
    )
    from sec_generative_search.providers.base import (
        BaseEmbeddingProvider,
        BaseLLMProvider,
        BaseRerankerProvider,
        GenerationRequest,
        GenerationResponse,
        RerankResult,
    )
    from sec_generative_search.providers.deepseek import (
        DeepSeekProvider,
    )
    from sec_generative_search.providers.factory import (
        ApiKeyResolver,
        build_embedder,
        default_api_key_resolver,
    )
    from sec_generative_search.providers.gemini import (
        GeminiEmbeddingProvider,
        GeminiProvider,
    )
    from sec_generative_search.providers.grok import (
        GrokProvider,
    )
    from sec_generative_search.providers.kimi import (
        KimiProvider,
    )
    from sec_generative_search.providers.local import (
        LocalEmbeddingProvider,
    )
    from sec_generative_search.providers.local_llm import (
        LocalLLMProvider,
    )
    from sec_generative_search.providers.mimo import (
        MimoProvider,
    )
    from sec_generative_search.providers.minimax import (
        MiniMaxProvider,
    )
    from sec_generative_search.providers.mistral import (
        MistralEmbeddingProvider,
        MistralProvider,
    )
    from sec_generative_search.providers.openai import (
        OpenAIEmbeddingProvider,
        OpenAIProvider,
    )
    from sec_generative_search.providers.openrouter import (
        OpenRouterProvider,
    )
    from sec_generative_search.providers.qwen import (
        QwenEmbeddingProvider,
        QwenProvider,
    )
    from sec_generative_search.providers.registry import (
        ProviderEntry,
        ProviderRegistry,
        ProviderSurface,
    )
    from sec_generative_search.providers.zai import (
        ZaiProvider,
    )
