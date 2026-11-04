"""MiniMax provider adapter.

MiniMax exposes an OpenAI-compatible Chat Completions surface at
``https://api.minimax.io/v1`` (international).  The mainland mirror
lives at ``api.minimaxi.chat`` and shares the wire protocol.

Only the chat surface is modelled here.  MiniMax does ship a first-party
embedding model (``embo-01``, 1536-dim) but it is served under a
MiniMax-proprietary request shape rather than the OpenAI embeddings
schema, so it does not slot into
:class:`OpenAICompatibleEmbeddingProvider` without a custom request
handler.  Pair with an existing embedding adapter when using this
vendor end-to-end.

The ``*-highspeed`` variants expose the same model family with a
latency-optimised schedule and lower throughput ceiling — the catalogue
surfaces both because they are separately priced and the orchestrator
may want to pick one per request mode (interactive vs. batch).
"""

from __future__ import annotations

from sec_generative_search.providers.openai_compat import (
    OpenAICompatibleLLMProvider,
)

__all__ = ["MiniMaxProvider"]


class MiniMaxProvider(OpenAICompatibleLLMProvider):
    """Chat-completion provider targeting MiniMax's hosted models."""

    provider_name = "minimax"
    default_base_url = "https://api.minimax.io/v1"
    default_model = "MiniMax-M3.1-Flash"
