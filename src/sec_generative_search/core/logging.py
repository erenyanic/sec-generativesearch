"""
Logging configuration for SEC-GenerativeSearch.

This module provides a consistent logging setup across all package modules.
It uses Rich for beautiful console output when running interactively.

Configuration:
    LOG_LEVEL environment variable controls the logging level.
    Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)

    LOG_REDACT_QUERIES gates research-identifier redaction.  When
    enabled it hashes query text at the call sites that wrap a value in
    :func:`redact_for_log`, and :class:`AccessionRedactionFilter`
    additionally scrubs every SEC accession number out of the rendered
    message of *every* record.

    LOG_FILE_PATH environment variable enables optional file logging via
    RotatingFileHandler.  LOG_FILE_MAX_BYTES (default 10 MB) and
    LOG_FILE_BACKUP_COUNT (default 3) control rotation.

Usage:
    from sec_generative_search.core.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Processing filing", extra={"ticker": "AAPL"})
"""

import hashlib
import json
import logging
import logging.handlers
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from sec_generative_search.core.correlation import get_correlation_id

# Package-level logger name
LOGGER_NAME = "sec_generative_search"

# Default format for non-Rich handlers (e.g., file output). The
# ``correlation_id`` field is injected onto every record by
# :class:`CorrelationIdFilter`, so it is always present at format time.
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(correlation_id)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track whether logging has been configured
_logging_configured = False

# SEC accession number: ``NNNNNNNNNN-NN-NNNNNN``, with or without the
# dashes.  This is the one research identifier that is unambiguous enough
# to scrub with a global regex — hence the split control described in
# :class:`AccessionRedactionFilter`.  ``database.metadata`` reuses this
# pattern so the two surfaces cannot drift apart.
ACCESSION_RE = re.compile(r"\b\d{10}-?\d{2}-?\d{6}\b")


def _get_log_level() -> int:
    """
    Get log level from environment variable.

    Returns:
        Logging level constant (e.g., logging.INFO)
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def _get_log_format() -> str:
    """Return the configured log format mode: ``"console"`` or ``"json"``.

    Read directly from ``LOG_FORMAT`` (same os.environ pattern as
    :func:`_get_log_level`) to avoid a circular import with
    pydantic-settings. ``console`` (Rich for interactive terminals, plain
    text otherwise) is the Scenario-A default; ``json`` emits one JSON
    object per line for Scenario B/C log aggregators. Unknown values fall
    back to ``console``.
    """
    value = os.environ.get("LOG_FORMAT", "console").strip().lower()
    return value if value in ("console", "json") else "console"


class CorrelationIdFilter(logging.Filter):
    """Inject the active correlation ID onto every log record.

    Attached to each handler (not the logger) so that both
    directly-logged and propagated child-logger records carry the
    attribute before any formatter references it. Absent a request scope
    the value is ``-``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id() or "-"
        return True


class AccessionRedactionFilter(logging.Filter):
    """Scrub SEC accession numbers out of every rendered log record.

    Gated on ``LOG_REDACT_QUERIES``, read **per record** so an operator
    (or a test) toggling the flag takes effect without re-configuring
    logging.

    Why a filter and not call-site wrapping: accessions are logged from
    roughly thirty sites across the ingest pipeline, the storage layer,
    and three route audit lines, and new sites are easy to add.  One
    handler-level control closes all of them, including sites nobody
    enumerated.  Tickers deliberately do **not** get the same treatment —
    a regex able to spot an arbitrary symbol would also match ``INFO``,
    ``POST``, ``API``, ``GPU`` and ``JSON`` — so those are wrapped in
    :func:`redact_for_log` where they are logged.

    Two mechanics are load-bearing:

    * Most sites log ``logger.info("... %s", accession)``, so the value
      lives in ``record.args``, never in ``record.msg``.  The filter
      therefore rewrites the **rendered** message and clears ``args``.
    * It attaches to each *handler* (not the logger), matching
      :class:`CorrelationIdFilter` — the package logger sets
      ``propagate = False``, so a logger-level filter would miss
      records emitted by child loggers.

    The substitution reuses :func:`redact_for_log`'s
    ``<redacted:XXXXXXXX>`` form, so a scrubbed accession correlates
    with one redacted at a call site, and repeated filtering (one pass
    per handler) is idempotent.

    ``RichHandler(rich_tracebacks=True)`` renders tracebacks straight
    from ``exc_info`` and so bypasses the ``exc_text`` scrub below.
    That is console-only, interactive, Scenario-A output — outside the
    B/C log-aggregator threat model this control exists for.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # Never raise: a filter that throws drops the record, which on a
        # request path would silently discard audit evidence.
        try:
            if not _redaction_enabled():
                return True
            message = record.getMessage()
            if ACCESSION_RE.search(message):
                record.msg = ACCESSION_RE.sub(_redact_accession_match, message)
                record.args = ()
            if record.exc_info or record.exc_text:
                exc_text = record.exc_text or logging.Formatter().formatException(
                    record.exc_info  # type: ignore[arg-type]
                )
                record.exc_text = ACCESSION_RE.sub(_redact_accession_match, exc_text)
        except Exception:  # pragma: no cover - defensive, must never break logging
            return True
        return True


class JsonFormatter(logging.Formatter):
    """Dependency-free JSON-lines formatter for log aggregators.

    Emits a fixed, **content-free** field set: timestamp, level, logger
    name, correlation ID, and the rendered message. Arbitrary ``extra``
    record attributes are deliberately NOT serialised — a stray ``extra``
    could otherwise smuggle a ticker, query, or secret into the log
    stream. Exception text is included only when the record carries
    ``exc_info``.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, DEFAULT_DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": getattr(record, "correlation_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info or record.exc_text:
            # ``exc_text`` may already have been rendered *and scrubbed*
            # by :class:`AccessionRedactionFilter`; re-formatting from
            # ``exc_info`` here would undo that.
            payload["exc"] = record.exc_text or self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _make_formatter(log_format: str) -> logging.Formatter:
    """Return the formatter for the non-Rich handlers given the format mode."""
    if log_format == "json":
        return JsonFormatter()
    return logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)


def _add_file_handler(
    logger: logging.Logger,
    file_path: str,
    log_level: int,
    log_format: str,
) -> None:
    """Attach a ``RotatingFileHandler`` to *logger*.

    Creates parent directories if they do not exist.  Rotation is
    controlled by ``LOG_FILE_MAX_BYTES`` (default 10 MB) and
    ``LOG_FILE_BACKUP_COUNT`` (default 3). The handler honours the
    ``LOG_FORMAT`` mode so file output matches the console stream.
    """
    max_bytes = int(os.environ.get("LOG_FILE_MAX_BYTES", 10_485_760))
    backup_count = int(os.environ.get("LOG_FILE_BACKUP_COUNT", 3))

    # Ensure the parent directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(_make_formatter(log_format))
    file_handler.addFilter(CorrelationIdFilter())
    file_handler.addFilter(AccessionRedactionFilter())
    logger.addHandler(file_handler)


def configure_logging(
    level: int | None = None,
    use_rich: bool = True,
) -> None:
    """
    Configure the package-level logger.

    This function sets up the root logger for the sec_generative_search package.
    It should be called once at application startup (e.g., in CLI main).

    Args:
        level: Logging level. If None, reads from LOG_LEVEL env var.
        use_rich: Whether to use RichHandler for console output.
                  Set to False when output is being piped or redirected.
    """
    global _logging_configured

    if _logging_configured:
        return

    log_level = level if level is not None else _get_log_level()
    log_format = _get_log_format()

    # Get the package-level logger
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(log_level)

    # Remove any existing handlers
    logger.handlers.clear()

    # Rich is reserved for human-facing console output. In ``json`` mode
    # we always use a plain stream handler so the stream stays
    # machine-parseable for a B/C log aggregator even on an interactive
    # terminal.
    is_interactive = sys.stdout.isatty() and use_rich and log_format != "json"

    if is_interactive:
        # Rich handler for beautiful console output
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        # Standard handler for non-interactive environments
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_make_formatter(log_format))

    handler.setLevel(log_level)
    # The correlation-ID filter runs on every handler so propagated
    # child-logger records carry the attribute before the formatter
    # references it.
    handler.addFilter(CorrelationIdFilter())
    # Accession scrubbing sits beside it, for the same reason: a
    # handler-level filter also covers propagated child-logger records.
    handler.addFilter(AccessionRedactionFilter())
    logger.addHandler(handler)

    # Optional file logging via RotatingFileHandler.
    # Reads from os.environ directly (same pattern as _get_log_level)
    # to avoid circular imports with pydantic-settings.
    log_file_path = os.environ.get("LOG_FILE_PATH")
    if log_file_path:
        _add_file_handler(logger, log_file_path, log_level, log_format)

    # Prevent propagation to root logger
    logger.propagate = False

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.

    This function returns a child logger of the package-level logger.
    If logging hasn't been configured yet, it will be configured with
    default settings.

    Args:
        name: Module name, typically __name__ from the calling module.
              If the name doesn't start with the package name, it will
              be prefixed automatically.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing filing for %s", ticker)
    """
    # Ensure logging is configured
    if not _logging_configured:
        configure_logging()

    # Ensure the logger is under our package namespace
    if not name.startswith(LOGGER_NAME):
        name = f"{LOGGER_NAME}.{name}"

    return logging.getLogger(name)


def audit_log(
    action: str,
    *,
    client_ip: str = "unknown",
    detail: str = "",
    endpoint: str = "",
) -> None:
    """Log a security-relevant action with structured context.

    All destructive operations (delete, clear, cancel, GPU unload) should
    call this so that security events are identifiable in log output
    without needing to parse generic log messages.

    The ``SECURITY_AUDIT:`` prefix makes entries easy to grep/filter.
    """
    logger = get_logger("security.audit")
    logger.warning(
        "SECURITY_AUDIT: action=%s client=%s endpoint=%s %s",
        action,
        client_ip,
        endpoint,
        detail,
    )


def _redaction_enabled() -> bool:
    """Whether ``LOG_REDACT_QUERIES`` is set to a truthy value.

    Read from ``os.environ`` on every call (never cached) so the flag can
    be toggled at runtime and so the check works from any module without
    depending on the Pydantic settings hierarchy (avoids circular
    imports).
    """
    return os.environ.get("LOG_REDACT_QUERIES", "").lower() in ("1", "true", "yes")


def _digest(value: str) -> str:
    """Return the ``<redacted:XXXXXXXX>`` placeholder for *value*."""
    return f"<redacted:{hashlib.sha256(value.encode()).hexdigest()[:8]}>"


def _redact_accession_match(match: re.Match[str]) -> str:
    """``re.sub`` replacement returning the placeholder for a matched accession."""
    return _digest(match.group(0))


def redact_for_log(value: str) -> str:
    """Return *value* unchanged or a SHA-256 digest prefix when redaction is enabled.

    Controlled by the ``LOG_REDACT_QUERIES`` environment variable.  When set
    to a truthy value (``1``, ``true``, ``yes`` — case-insensitive), the
    original text is replaced with ``<redacted:XXXXXXXX>`` where ``XXXXXXXX``
    is the first 8 hex characters of its SHA-256 hash.  This preserves log
    correlation (same input → same hash) while hiding the actual content.

    Wrap **query text and ticker symbols** here at the call site.  SEC
    accession numbers are handled centrally by
    :class:`AccessionRedactionFilter`, so they need no call-site wrapping
    (wrapping one anyway is harmless — the placeholder no longer matches
    the accession pattern).
    """
    if _redaction_enabled():
        return _digest(value)
    return value


def redact_all_for_log(values: Iterable[str]) -> str:
    """Render a collection of identifiers for a log line, each redacted.

    Used for ticker lists so a log line keeps its cardinality (and its
    per-symbol correlation hashes) without carrying the symbols
    themselves.  Returns ``"none"`` for an empty collection.
    """
    rendered = ", ".join(redact_for_log(value) for value in values)
    return rendered or "none"


def suppress_third_party_loggers() -> None:
    """
    Suppress verbose logging from third-party libraries.

    Some libraries (sentence-transformers, chromadb, httpx) are quite
    verbose at INFO level. This function sets them to WARNING.
    """
    noisy_loggers = [
        "sentence_transformers",
        "chromadb",
        "httpx",
        "httpcore",
        "urllib3",
        "transformers",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
