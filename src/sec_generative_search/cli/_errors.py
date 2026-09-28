"""The CLI's one provider-failure classification ladder.

``cli/rag.py`` wrote the ladder out four times (query understanding,
generation, the chat REPL's plan step and its streaming classifier) and
``cli/provider.py`` twice (OPTIMIZATIONS F32) — a security-relevant table
(it decides whether the operator is told to rotate a key) kept in lockstep
by hand.  It now lives here once.

It deliberately **mirrors, never imports**, the API's
``api.routes.rag._PROVIDER_ERROR_LADDER`` (AGENT.md §CLI): the operator
sees the same label, wording and hint as an API client for the same
failure, while the CLI and API layers stay decoupled.  Ordering is
load-bearing — subclasses before the :class:`ProviderError` base, and
:class:`ProviderConnectionError` (a ``ProviderError`` subclass) before the
base branch.  Only the exception *type* ever reaches ``details`` — never
its text, which can carry provider URLs or request fragments.
"""

from __future__ import annotations

from dataclasses import dataclass

from sec_generative_search.core.exceptions import (
    GenerationError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

__all__ = ["LOCAL_ENDPOINT_HINT", "CliError", "classify_provider_error"]

# Hint for an unreachable endpoint — chiefly a self-hosted ``local_llm``
# server that is down — distinct from the rate-limit / timeout wording.
LOCAL_ENDPOINT_HINT = (
    "Verify the endpoint is running and reachable (for local_llm, that the "
    "local model server is up, e.g. `ollama serve`, and LOCAL_LLM_BASE_URL is "
    "correct); retry once it recovers."
)


@dataclass(frozen=True)
class CliError:
    """An operator-facing error: the arguments ``print_error`` renders."""

    label: str
    message: str
    hint: str
    error_code: str
    details: str | None = None


def classify_provider_error(
    exc: BaseException,
    *,
    provider_name: str,
    phase: str,
    connection_is_distinct: bool = True,
) -> CliError | None:
    """Map a provider / generation failure to its operator envelope.

    ``phase`` is the noun in the generic provider message ("query
    understanding", "generation", "validation").  ``connection_is_distinct``
    is ``False`` for key validation, which — like ``POST
    /api/providers/validate`` — reports an unreachable endpoint as a
    generic provider error.  Returns ``None`` for anything else (retrieval,
    storage, unexpected errors), which callers render themselves.
    """
    if isinstance(exc, ProviderAuthError):
        return CliError(
            "Provider unauthorised",
            "The upstream provider rejected the supplied API key.",
            (
                f"Verify or rotate the provider key for {provider_name!r}; "
                "do not retry until corrected."
            ),
            "provider_unauthorized",
        )
    if isinstance(exc, ProviderRateLimitError | ProviderTimeoutError):
        return CliError(
            "Provider unavailable",
            "The upstream provider is rate-limited or timed out.",
            "Retry after a short backoff; do not rotate the key.",
            "provider_unavailable",
            details=type(exc).__name__,
        )
    if connection_is_distinct and isinstance(exc, ProviderConnectionError):
        return CliError(
            "Provider unavailable",
            "The upstream provider endpoint could not be reached.",
            LOCAL_ENDPOINT_HINT,
            "provider_unavailable",
        )
    if isinstance(exc, ProviderError):
        return CliError(
            "Provider error",
            f"The upstream provider returned an error during {phase}.",
            "Inspect the audit log; do not rotate the key on a non-auth error.",
            "provider_error",
            details=type(exc).__name__,
        )
    if isinstance(exc, GenerationError):
        return CliError(
            "Generation failed",
            "The orchestrator could not assemble a valid answer.",
            "Retry the request; if the failure persists, switch model or provider.",
            "generation_failed",
        )
    return None
