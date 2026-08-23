"""Tests for logging configuration."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from pathlib import Path

import pytest

from sec_generative_search.core import logging as sgs_logging
from sec_generative_search.core.correlation import bind_correlation_id
from sec_generative_search.core.logging import (
    LOGGER_NAME,
    AccessionRedactionFilter,
    CorrelationIdFilter,
    JsonFormatter,
    audit_log,
    configure_logging,
    get_logger,
    redact_for_log,
    suppress_third_party_loggers,
)


@pytest.fixture(autouse=True)
def _reset_logging_config() -> None:
    """Reset the module-level ``_logging_configured`` flag between tests."""
    # Pre-test: clear handlers and flag
    sgs_logging._logging_configured = False
    logging.getLogger(LOGGER_NAME).handlers.clear()
    yield
    # Post-test: same
    sgs_logging._logging_configured = False
    logging.getLogger(LOGGER_NAME).handlers.clear()


class TestLoggerName:
    def test_package_logger_uses_new_namespace(self) -> None:
        assert LOGGER_NAME == "sec_generative_search"

    def test_get_logger_prefixes_module_name(self) -> None:
        logger = get_logger("pipeline.fetch")
        assert logger.name == "sec_generative_search.pipeline.fetch"

    def test_get_logger_respects_full_prefix(self) -> None:
        logger = get_logger("sec_generative_search.core.logging")
        assert logger.name == "sec_generative_search.core.logging"


class TestConfiguration:
    def test_configures_once_and_is_idempotent(self) -> None:
        configure_logging(level=logging.DEBUG, use_rich=False)
        handlers_after_first = list(logging.getLogger(LOGGER_NAME).handlers)
        configure_logging(level=logging.INFO, use_rich=False)
        handlers_after_second = list(logging.getLogger(LOGGER_NAME).handlers)
        # Idempotent: the second call must not add new handlers.
        assert len(handlers_after_first) == len(handlers_after_second)

    def test_file_handler_added_when_env_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        log_file = tmp_path / "nested" / "run.log"
        monkeypatch.setenv("LOG_FILE_PATH", str(log_file))
        configure_logging(level=logging.DEBUG, use_rich=False)

        handlers = logging.getLogger(LOGGER_NAME).handlers
        # Must have at least a stream handler + a rotating file handler.
        assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in handlers)
        # Parent directory was created automatically.
        assert log_file.parent.is_dir()


class TestRedactForLog:
    """Security: redact_for_log must not leak the original value when enabled."""

    def test_no_redaction_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOG_REDACT_QUERIES", raising=False)
        assert redact_for_log("hello") == "hello"

    @pytest.mark.parametrize("flag", ["1", "true", "yes", "TRUE", "Yes"])
    def test_redaction_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        flag: str,
    ) -> None:
        monkeypatch.setenv("LOG_REDACT_QUERIES", flag)
        out = redact_for_log("revenue forecast for AAPL")
        assert out.startswith("<redacted:")
        assert out.endswith(">")
        # Hash prefix must be deterministic.
        expected = hashlib.sha256(b"revenue forecast for AAPL").hexdigest()[:8]
        assert expected in out
        # Original must not appear anywhere.
        assert "revenue forecast" not in out
        assert "AAPL" not in out

    def test_redaction_is_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")
        # Same input → same redaction (preserves log correlation).
        assert redact_for_log("query X") == redact_for_log("query X")

    @pytest.mark.parametrize("flag", ["0", "false", "no", "", "maybe"])
    def test_redaction_disabled_for_falsy_flags(
        self,
        monkeypatch: pytest.MonkeyPatch,
        flag: str,
    ) -> None:
        monkeypatch.setenv("LOG_REDACT_QUERIES", flag)
        assert redact_for_log("hello") == "hello"


class TestAuditLog:
    def test_audit_entry_uses_warning_level_and_prefix(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # ``configure_logging`` sets ``propagate = False`` on the package
        # logger, so caplog (which attaches to root) can't see records.
        # Enable propagation for the duration of this test so the
        # WARNING reaches caplog's handler.
        configure_logging(level=logging.DEBUG, use_rich=False)
        pkg_logger = logging.getLogger(LOGGER_NAME)
        pkg_logger.propagate = True
        audit_logger_name = f"{LOGGER_NAME}.security.audit"

        try:
            with caplog.at_level(logging.WARNING, logger=audit_logger_name):
                audit_log(
                    action="delete_filing",
                    client_ip="127.0.0.1",
                    endpoint="/api/filings/AAPL",
                    detail="accession=0000320193-23-000077",
                )
        finally:
            pkg_logger.propagate = False

        records = [r for r in caplog.records if r.name == audit_logger_name]
        assert records, "audit_log should emit on security.audit logger"
        msg = records[0].getMessage()
        assert "SECURITY_AUDIT:" in msg
        assert "action=delete_filing" in msg
        assert "client=127.0.0.1" in msg


def _make_record(msg: str = "hello %s", *args: object) -> logging.LogRecord:
    return logging.LogRecord(
        name="sec_generative_search.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


class TestCorrelationIdFilter:
    def test_injects_dash_without_scope(self) -> None:
        record = _make_record()
        assert CorrelationIdFilter().filter(record) is True
        assert record.correlation_id == "-"

    def test_injects_bound_id(self) -> None:
        record = _make_record()
        with bind_correlation_id("cid-12345678"):
            CorrelationIdFilter().filter(record)
        assert record.correlation_id == "cid-12345678"


class TestJsonFormatter:
    def test_emits_fixed_field_set(self) -> None:
        record = _make_record("processed %s filings", 3)
        CorrelationIdFilter().filter(record)
        line = JsonFormatter().format(record)
        payload = json.loads(line)
        assert payload["level"] == "INFO"
        assert payload["logger"] == "sec_generative_search.test"
        assert payload["message"] == "processed 3 filings"
        assert payload["correlation_id"] == "-"
        assert set(payload) == {"ts", "level", "logger", "correlation_id", "message"}

    def test_carries_bound_correlation_id(self) -> None:
        record = _make_record("x")
        with bind_correlation_id("req-abcdef12"):
            CorrelationIdFilter().filter(record)
        payload = json.loads(JsonFormatter().format(record))
        assert payload["correlation_id"] == "req-abcdef12"

    @pytest.mark.security
    def test_does_not_serialise_arbitrary_extra(self) -> None:
        # A stray ``extra`` must never reach the JSON stream — it could
        # smuggle a ticker / query / secret into the aggregator.
        record = _make_record("x")
        record.ticker = "AAPL"  # type: ignore[attr-defined]
        record.api_key = "sk-secret"  # type: ignore[attr-defined]
        line = JsonFormatter().format(record)
        assert "AAPL" not in line
        assert "sk-secret" not in line

    def test_includes_exception_text_when_present(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="sec_generative_search.test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        payload = json.loads(JsonFormatter().format(record))
        assert "ValueError" in payload["exc"]


class TestLogFormatSelection:
    def test_json_format_uses_json_formatter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_FORMAT", "json")
        configure_logging(level=logging.INFO, use_rich=False)
        handlers = logging.getLogger(LOGGER_NAME).handlers
        assert any(isinstance(h.formatter, JsonFormatter) for h in handlers)

    def test_console_format_does_not_use_json_formatter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LOG_FORMAT", "console")
        configure_logging(level=logging.INFO, use_rich=False)
        handlers = logging.getLogger(LOGGER_NAME).handlers
        assert not any(isinstance(h.formatter, JsonFormatter) for h in handlers)

    def test_unknown_format_falls_back_to_console(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOG_FORMAT", "yaml-nonsense")
        configure_logging(level=logging.INFO, use_rich=False)
        handlers = logging.getLogger(LOGGER_NAME).handlers
        assert not any(isinstance(h.formatter, JsonFormatter) for h in handlers)

    def test_every_handler_carries_correlation_filter(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("LOG_FILE_PATH", str(tmp_path / "run.log"))
        configure_logging(level=logging.INFO, use_rich=False)
        handlers = logging.getLogger(LOGGER_NAME).handlers
        assert handlers
        for handler in handlers:
            assert any(isinstance(f, CorrelationIdFilter) for f in handler.filters)


class TestSuppressThirdPartyLoggers:
    def test_sets_noisy_loggers_to_warning(self) -> None:
        # Flip them to DEBUG first to verify the call actually changes them.
        for name in ("chromadb", "httpx", "sentence_transformers"):
            logging.getLogger(name).setLevel(logging.DEBUG)

        suppress_third_party_loggers()

        for name in ("chromadb", "httpx", "sentence_transformers"):
            assert logging.getLogger(name).level == logging.WARNING


# ---------------------------------------------------------------------------
# Research-identifier redaction (finding M3)
# ---------------------------------------------------------------------------

_ACCESSION = "0000320193-23-000077"


@pytest.mark.security
class TestAccessionRedactionFilter:
    """Accessions are scrubbed centrally, at the handler layer.

    Splitting the control is deliberate: the accession shape
    (``NNNNNNNNNN-NN-NNNNNN``) is unambiguous enough for a global regex,
    so one filter closes every call site — including ones nobody
    enumerated. Tickers are *not* regex-detectable (a generic
    uppercase-token pattern would mangle ``INFO`` / ``POST`` / ``API``),
    so they are redacted at the call sites and held by
    :class:`TestTickerCallSiteHygiene`.
    """

    def test_scrubs_an_accession_carried_in_record_args(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The trap: most sites log ``logger.info("... %s", accession)``,
        # so the value lives in ``record.args`` and never in
        # ``record.msg``. A filter that rewrites ``msg`` alone scrubs
        # nothing.
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")
        record = _make_record("Removed filing from registry: %s", _ACCESSION)
        assert AccessionRedactionFilter().filter(record) is True
        assert _ACCESSION not in record.getMessage()
        assert "<redacted:" in record.getMessage()

    def test_scrubs_an_accession_carried_in_the_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")
        record = _make_record(f"SECURITY_AUDIT: accession={_ACCESSION} form=10-K")
        AccessionRedactionFilter().filter(record)
        message = record.getMessage()
        assert _ACCESSION not in message
        assert "form=10-K" in message

    def test_scrubs_the_dash_free_accession_form(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")
        record = _make_record("accession=%s", _ACCESSION.replace("-", ""))
        AccessionRedactionFilter().filter(record)
        assert _ACCESSION.replace("-", "") not in record.getMessage()

    def test_placeholder_matches_redact_for_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Filter output and call-site output must agree for the same
        # string, so an operator can correlate a filtered line with a
        # deliberately-redacted one.
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")
        record = _make_record("accession=%s", _ACCESSION)
        AccessionRedactionFilter().filter(record)
        assert redact_for_log(_ACCESSION) in record.getMessage()

    def test_no_op_when_flag_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOG_REDACT_QUERIES", raising=False)
        record = _make_record("accession=%s", _ACCESSION)
        AccessionRedactionFilter().filter(record)
        assert record.getMessage() == f"accession={_ACCESSION}"

    def test_flag_is_read_per_record(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Caching the flag at construction time would make the control
        # untestable (and unreachable for a process that sets the env
        # after import).
        log_filter = AccessionRedactionFilter()

        monkeypatch.delenv("LOG_REDACT_QUERIES", raising=False)
        clear_record = _make_record("accession=%s", _ACCESSION)
        log_filter.filter(clear_record)
        assert _ACCESSION in clear_record.getMessage()

        monkeypatch.setenv("LOG_REDACT_QUERIES", "true")
        redacted_record = _make_record("accession=%s", _ACCESSION)
        log_filter.filter(redacted_record)
        assert _ACCESSION not in redacted_record.getMessage()

    def test_clears_args_so_a_literal_percent_survives(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # After substitution the record holds a fully-rendered message;
        # leaving ``args`` populated would re-run ``%``-interpolation
        # over text that may now contain a stray ``%``.
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")
        record = _make_record("accession=%s progress=100%% done", _ACCESSION)
        AccessionRedactionFilter().filter(record)
        assert record.args in ((), None)
        assert "100% done" in record.getMessage()

    def test_never_raises_on_a_malformed_record(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A raising filter drops the record. On a request path that
        # would turn a logging bug into missing audit evidence.
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")
        record = _make_record("two placeholders %s %s", "only-one")
        assert AccessionRedactionFilter().filter(record) is True

    def test_is_idempotent_across_handlers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The filter is attached per-handler, so a record with two
        # handlers is filtered twice.
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")
        record = _make_record("accession=%s", _ACCESSION)
        log_filter = AccessionRedactionFilter()
        log_filter.filter(record)
        first = record.getMessage()
        log_filter.filter(record)
        assert record.getMessage() == first

    def test_scrubs_the_exception_traceback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # ``logger.exception(...)`` renders the traceback separately from
        # the message; a domain error's ``details`` can carry an
        # accession into it.
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")
        import sys

        try:
            raise ValueError(f"storage failed for {_ACCESSION}")
        except ValueError:
            record = logging.LogRecord(
                name="sec_generative_search.test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="storage failure",
                args=(),
                exc_info=sys.exc_info(),
            )

        AccessionRedactionFilter().filter(record)
        payload = json.loads(JsonFormatter().format(record))
        assert _ACCESSION not in payload["exc"]
        assert "ValueError" in payload["exc"]

    def test_every_handler_carries_the_redaction_filter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Attaching to the logger instead of the handlers would miss
        # child-logger records (the package logger has
        # ``propagate = False``) — the same reason
        # ``CorrelationIdFilter`` sits on handlers.
        monkeypatch.setenv("LOG_FILE_PATH", str(tmp_path / "run.log"))
        configure_logging(level=logging.INFO, use_rich=False)
        handlers = logging.getLogger(LOGGER_NAME).handlers
        assert handlers
        for handler in handlers:
            assert any(isinstance(f, AccessionRedactionFilter) for f in handler.filters)


# --- Static call-site lock for the ticker half of the control --------------

_LOG_LEVELS = frozenset({"debug", "info", "warning", "error", "exception", "critical"})

# Wrapping any of these renders the identifier content-free, so the
# enclosed expression needs no further treatment.
_REDACTING_WRAPPERS = frozenset(
    {"redact_for_log", "redact_all_for_log", "mask_secret", "len", "sorted", "id"}
)

# Attribute / variable names that carry a ticker symbol (``chunk_id`` is
# ``{TICKER}_{FORM}_{DATE}_{INDEX}`` — see ``Chunk.chunk_id``).
_TICKER_CARRIER_ATTRS = frozenset({"ticker", "tickers", "chunk_id"})
_TICKER_CARRIER_NAMES = frozenset({"ticker", "tickers"})

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "sec_generative_search"


def _is_log_call(node: ast.Call) -> bool:
    """True when *node* is a logging or audit-log emission."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "audit_log"
    if isinstance(func, ast.Attribute):
        if func.attr == "audit_log":
            return True
        return func.attr in _LOG_LEVELS and "log" in ast.unparse(func.value).lower()
    return False


def _wrapper_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _unredacted_ticker_carriers(node: ast.AST) -> list[str]:
    """Return the ticker-carrying sub-expressions of *node* that are raw."""
    found: list[str] = []

    def walk(current: ast.AST) -> None:
        if isinstance(current, ast.Call) and _wrapper_name(current) in _REDACTING_WRAPPERS:
            return  # everything inside is rendered content-free
        if isinstance(current, ast.Attribute) and current.attr in _TICKER_CARRIER_ATTRS:
            found.append(ast.unparse(current))
            return
        if isinstance(current, ast.Name) and current.id in _TICKER_CARRIER_NAMES:
            found.append(ast.unparse(current))
            return
        if isinstance(current, ast.IfExp):
            # The condition is a truthiness test, not a rendered value.
            walk(current.body)
            walk(current.orelse)
            return
        for child in ast.iter_child_nodes(current):
            walk(child)

    walk(node)
    return found


@pytest.mark.security
class TestTickerCallSiteHygiene:
    """No production log call site may render a raw ticker.

    The central filter cannot close this half: a regex able to spot an
    arbitrary ticker would also match ``INFO``, ``POST``, ``API``,
    ``GPU`` and ``JSON``. So tickers are redacted where they are logged
    — and this lock is what keeps that true as new call sites land,
    which is exactly how the leak spread in the first place (the
    ``ingest.py`` docstring already claimed it emitted "the redacted
    ticker list" while the code did not).
    """

    def test_no_log_call_renders_a_raw_ticker(self) -> None:
        violations: list[str] = []

        for path in sorted(_SRC_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not _is_log_call(node):
                    continue
                for argument in [*node.args, *(kw.value for kw in node.keywords)]:
                    for carrier in _unredacted_ticker_carriers(argument):
                        violations.append(
                            f"{path.relative_to(_SRC_ROOT.parents[1])}:"
                            f"{node.lineno}: raw ticker expression {carrier!r}"
                        )

        assert not violations, (
            "log call sites render a raw ticker; wrap the value in "
            "redact_for_log()/redact_all_for_log() or log a count:\n  " + "\n  ".join(violations)
        )

    def test_the_scan_actually_finds_a_planted_violation(self) -> None:
        # Guards against the scan silently matching nothing (a rename of
        # the logger attribute or a change in the AST shape would make
        # the lock above vacuously green).
        planted = ast.parse('logger.info("fetched %s", filing_id.ticker)')
        call = planted.body[0].value  # type: ignore[attr-defined]
        assert _is_log_call(call)
        assert _unredacted_ticker_carriers(call.args[1]) == ["filing_id.ticker"]

        cleared = ast.parse('logger.info("fetched %s", redact_for_log(filing_id.ticker))')
        cleared_call = cleared.body[0].value  # type: ignore[attr-defined]
        assert _unredacted_ticker_carriers(cleared_call.args[1]) == []
