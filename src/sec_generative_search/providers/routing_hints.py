"""Upstream-routing hints for OpenRouter — SDK-free.

:class:`OpenRouterRoutingHints` is plain request data, but it used to live
in :mod:`~sec_generative_search.providers.openrouter`, which imports the
``openai`` SDK (~0.65 s); the CLI ``rag`` commands and the API RAG routes
build one per request, so importing them paid for the SDK even for
``sec-rag --help`` (F28).  It is re-exported from ``openrouter`` — that
remains its documented home; the identity is the same class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["OpenRouterRoutingHints"]


@dataclass(frozen=True)
class OpenRouterRoutingHints:
    """Provider-neutral upstream-routing hints for OpenRouter.

    Forwarded to OpenRouter's ``provider`` request block when attached
    to a :class:`~sec_generative_search.providers.base.GenerationRequest`
    that hits :class:`OpenRouterProvider`.  Every other provider ignores
    the hint — see :meth:`OpenAICompatibleLLMProvider._extra_request_kwargs`.

    Frozen and tuple-valued so the hint object is hashable, immutable,
    and safe to share across threads without defensive copying.
    Values are **pass-through**: this adapter does not validate them.
    An unknown upstream slug or malformed ``data_collection`` value
    surfaces at call time with OpenRouter's own error message, which is
    the authoritative source of truth for what OpenRouter currently
    accepts.

    The field set mirrors OpenRouter's documented ``provider`` block keys:

    - ``order``: preferred upstream order, first match wins.
    - ``allow_fallbacks``: when ``False``, refuse to fall back to
      upstreams outside ``order`` / ``only``.
    - ``only``: allowlist of upstream slugs.
    - ``ignore``: blocklist of upstream slugs.
    - ``require_parameters``: when ``True``, only route to upstreams
      that honour every request parameter (e.g. reasoning, tools).
    - ``data_collection``: ``"allow"`` or ``"deny"`` — refuse upstreams
      that log prompt/response content.

    This object carries no credential-shaped fields by design.  A
    parametrised security test enforces the field set against the same
    credential-hint list used for :class:`GenerationRequest` and the
    provider registry rows.
    """

    order: tuple[str, ...] = ()
    allow_fallbacks: bool | None = None
    only: tuple[str, ...] = ()
    ignore: tuple[str, ...] = ()
    require_parameters: bool | None = None
    data_collection: str | None = None

    def to_provider_block(self) -> dict[str, Any]:
        """Render the hints as the ``provider`` block OpenRouter expects.

        Omitted / empty fields are dropped so OpenRouter's own defaults
        apply — a hint object with no fields set yields an empty dict and
        is equivalent to not attaching hints at all.  Tuples are coerced
        to lists because OpenRouter's JSON schema expects arrays; the
        conversion is safe because :class:`OpenRouterRoutingHints` is
        frozen — we never hand callers a mutable reference to the
        original storage.
        """
        block: dict[str, Any] = {}
        if self.order:
            block["order"] = list(self.order)
        if self.allow_fallbacks is not None:
            block["allow_fallbacks"] = self.allow_fallbacks
        if self.only:
            block["only"] = list(self.only)
        if self.ignore:
            block["ignore"] = list(self.ignore)
        if self.require_parameters is not None:
            block["require_parameters"] = self.require_parameters
        if self.data_collection is not None:
            block["data_collection"] = self.data_collection
        return block
