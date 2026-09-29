"""Header redaction helpers used by middleware-level access logging.

Redaction is a defence-in-depth control. The load-bearing rule is that
no log record carries the raw value, even at ``DEBUG`` level, so the
originating record never had the secret in the first place.

This module is intentionally small and dependency-free outside of
``core.security``: it sits below middleware in the import graph and is
easy to unit-test without a FastAPI runtime.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from sec_generative_search.core.security import mask_secret

__all__ = [
    "REDACTED_HEADER_NAMES",
    "REDACTED_HEADER_PREFIXES",
    "SUPPRESSED_HEADER_NAMES",
    "redact_header_value",
    "redact_headers",
]


# Headers whose value is fully suppressed — no tail shown.
SUPPRESSED_HEADER_NAMES: frozenset[str] = frozenset(
    {
        "x-edgar-name",
        "x-edgar-email",
    }
)


# Headers whose value is replaced by the masked tail (same rule as
# ``mask_secret``).
REDACTED_HEADER_NAMES: frozenset[str] = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-admin-key",
        "x-demo-reset-token",
    }
)


# Header *prefixes* whose value is masked.
REDACTED_HEADER_PREFIXES: tuple[str, ...] = ("x-provider-key",)


def _normalise(name: str) -> str:
    """Lowercase a header name (HTTP names are case-insensitive)."""
    return name.lower()


def redact_header_value(name: str, value: str) -> str:
    """Return the redacted form of ``value`` for header ``name``.

    Returns the original value when ``name`` is not a redaction target,
    so the helper is safe to call eagerly inside a logging filter.
    """
    lowered = _normalise(name)

    if lowered in SUPPRESSED_HEADER_NAMES:
        return "***"

    if lowered in REDACTED_HEADER_NAMES:
        return mask_secret(value)

    for prefix in REDACTED_HEADER_PREFIXES:
        if lowered.startswith(prefix):
            return mask_secret(value)

    return value


def redact_headers(
    headers: Mapping[str, str] | Iterable[tuple[str, str]],
) -> dict[str, str]:
    """Apply :func:`redact_header_value` across an entire header mapping.

    Accepts both ``dict``-shaped headers and the ``list[tuple[str, str]]``
    shape Starlette uses internally.  Returns a fresh ``dict`` keyed by
    the original-case header names — callers that need lower-cased keys
    should normalise on the way out.
    """
    items = headers.items() if isinstance(headers, Mapping) else headers
    return {name: redact_header_value(name, value) for name, value in items}
