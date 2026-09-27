"""SDK timeout and retry budget for hosted provider clients.

The single place ``ProviderSettings`` (``PROVIDER_TIMEOUT`` /
``PROVIDER_MAX_RETRIES`` / ``PROVIDER_RETRY_BACKOFF_BASE``) becomes the
keyword arguments every hosted adapter is constructed with.  Used by the
factory (``build_llm_provider`` / ``build_embedder``) and by
``ProviderRegistry._construct`` (key-validation probes), so no construction
seam can drift to its own numbers.

Two budgets:

- **interactive** — request-scoped LLM calls and validation probes, where a
  caller is waiting on an API worker thread: at most
  :data:`~sec_generative_search.core.resilience.INTERACTIVE_RETRY_POLICY`'s
  one retry.  ``PROVIDER_MAX_RETRIES`` can lower it (``0`` = no retry) but
  never raise it — that would re-open the threadpool-occupancy problem F16
  closed.
- **background** — the embedder (built once per process, shared by ingest
  and search): ``PROVIDER_MAX_RETRIES`` as set.

The on-device embedder makes no network call and never receives these.
Settings are read per construction, so a fresh value applies to the next
request once the settings singleton is reloaded.
"""

from __future__ import annotations

from typing import Any

from sec_generative_search.config.settings import ProviderSettings, get_settings
from sec_generative_search.core.resilience import INTERACTIVE_RETRY_POLICY, RetryPolicy

__all__ = ["hosted_client_kwargs"]


def hosted_client_kwargs(
    *,
    interactive: bool,
    settings: ProviderSettings | None = None,
) -> dict[str, Any]:
    """Return ``{"timeout", "retry_policy"}`` for a hosted adapter constructor."""
    provider = settings if settings is not None else get_settings().provider
    max_retries = provider.max_retries
    if interactive:
        max_retries = min(max_retries, INTERACTIVE_RETRY_POLICY.max_retries)
    return {
        "timeout": float(provider.timeout),
        "retry_policy": RetryPolicy(
            max_retries=max_retries,
            backoff_base=provider.retry_backoff_base,
        ),
    }
