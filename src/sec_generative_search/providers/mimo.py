"""Xiaomi MiMo provider adapter.

Xiaomi ships an OpenAI-compatible Chat Completions surface at
``https://api.xiaomimimo.com/v1``.  The service sits behind the
``platform.xiaomimimo.com`` token-plan portal and uses standard OpenAI
``/chat/completions`` paths — the SDK works unchanged via a
``base_url`` override.

The catalogue covers the text-only MiMo-V2 slugs.  ``MiMo-V2-Omni``
(multimodal) and ``MiMo-V2-TTS`` (speech synthesis) are intentionally
excluded: the project's current surface is grounded SEC text analysis
and neither modality contributes to that pipeline today. If voice or
vision support is added to the orchestrator, those slugs can
be added with the appropriate capability flags.
"""

from __future__ import annotations

from sec_generative_search.providers.openai_compat import (
    OpenAICompatibleLLMProvider,
)

__all__ = ["MimoProvider"]


class MimoProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting Xiaomi's MiMo API."""

    provider_name = "mimo"
    default_base_url = "https://api.xiaomimimo.com/v1"
    default_model = "mimo-v2.6-pro"
