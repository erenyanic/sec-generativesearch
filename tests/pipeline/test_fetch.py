"""Tests for :mod:`sec_generative_search.pipeline.fetch`.

These tests cover the fetcher's pure helpers and the identity
configuration surface without touching EDGAR.  The edgartools
``set_identity``, ``Company.get_filings``, and ``Filing.html`` entry
points are patched so the suite runs offline and deterministically.

Security emphasis: the EDGAR identity (``name`` + ``email``) must never
appear in any log record — not even at DEBUG.  These are personal
credentials and the project contract forbids logging them.
"""

from __future__ import annotations

import io
import logging
from datetime import date
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from sec_generative_search.core.exceptions import FetchError
from sec_generative_search.core.logging import LOGGER_NAME, AccessionRedactionFilter
from sec_generative_search.pipeline import fetch as fetch_module
from sec_generative_search.pipeline.fetch import FilingFetcher, FilingInfo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_settings_stub(name: str | None = None, email: str | None = None) -> Any:
    """Build a minimal ``Settings``-like object for the fetcher.

    Avoids reading ``.env`` and keeps the test isolated from any
    credentials already present on the developer machine.
    """
    return SimpleNamespace(
        edgar=SimpleNamespace(identity_name=name, identity_email=email),
        database=SimpleNamespace(max_filings=500),
    )


@pytest.fixture
def fetcher(monkeypatch: pytest.MonkeyPatch) -> FilingFetcher:
    """Instantiate a fetcher with ``set_identity`` and settings stubbed out."""
    recorded: list[str] = []
    monkeypatch.setattr(fetch_module, "set_identity", lambda s: recorded.append(s))
    monkeypatch.setattr(fetch_module, "get_settings", lambda: _make_settings_stub())

    instance = FilingFetcher()
    instance._test_identity_log = recorded  # type: ignore[attr-defined]
    return instance


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestValidateFormType:
    def test_uppercases_input(self, fetcher: FilingFetcher) -> None:
        assert fetcher._validate_form_type("10-k") == "10-K"
        assert fetcher._validate_form_type("10-Q") == "10-Q"

    def test_accepts_amendment_forms(self, fetcher: FilingFetcher) -> None:
        assert fetcher._validate_form_type("10-K/A") == "10-K/A"

    def test_rejects_unsupported(self, fetcher: FilingFetcher) -> None:
        with pytest.raises(FetchError, match="Unsupported form type"):
            fetcher._validate_form_type("S-1")


class TestFormatDateFilter:
    def test_returns_none_when_both_empty(self, fetcher: FilingFetcher) -> None:
        assert fetcher._format_date_filter(None, None) is None

    def test_both_dates_as_strings(self, fetcher: FilingFetcher) -> None:
        assert fetcher._format_date_filter("2022-01-01", "2023-12-31") == "2022-01-01:2023-12-31"

    def test_both_dates_as_date_objects(self, fetcher: FilingFetcher) -> None:
        assert (
            fetcher._format_date_filter(date(2022, 1, 1), date(2023, 12, 31))
            == "2022-01-01:2023-12-31"
        )

    def test_open_start(self, fetcher: FilingFetcher) -> None:
        assert fetcher._format_date_filter(None, "2023-12-31") == ":2023-12-31"

    def test_open_end(self, fetcher: FilingFetcher) -> None:
        assert fetcher._format_date_filter("2022-01-01", None) == "2022-01-01:"


class TestParseFilingDate:
    def test_accepts_date_object(self, fetcher: FilingFetcher) -> None:
        d = date(2023, 11, 3)
        assert fetcher._parse_filing_date(d) is d

    def test_parses_iso_string(self, fetcher: FilingFetcher) -> None:
        assert fetcher._parse_filing_date("2023-11-03") == date(2023, 11, 3)


class TestAmendmentFiltering:
    def _filing(self, form: str) -> Any:
        return SimpleNamespace(form=form, accession_no="x")

    def test_is_amendment(self, fetcher: FilingFetcher) -> None:
        assert FilingFetcher._is_amendment(self._filing("10-K/A")) is True
        assert FilingFetcher._is_amendment(self._filing("10-K")) is False
        # Non-string form (pathological upstream data) must be handled
        # gracefully — False, not an exception.
        assert FilingFetcher._is_amendment(SimpleNamespace(form=None)) is False

    def test_should_skip_base_form_skips_amendments(self, fetcher: FilingFetcher) -> None:
        # Request 10-K → amendments (10-K/A) should be skipped so they
        # don't displace originals via the storage UNIQUE constraint.
        assert FilingFetcher._should_skip(self._filing("10-K/A"), "10-K") is True
        assert FilingFetcher._should_skip(self._filing("10-K"), "10-K") is False

    def test_should_skip_amendment_form_never_skips(self, fetcher: FilingFetcher) -> None:
        # Requesting 10-K/A explicitly → edgartools returns only amendments,
        # so the skip filter must be a no-op.
        assert FilingFetcher._should_skip(self._filing("10-K/A"), "10-K/A") is False
        assert FilingFetcher._should_skip(self._filing("10-K"), "10-K/A") is False


class TestFilingInfoToIdentifier:
    def test_round_trips_to_filing_identifier(self) -> None:
        info = FilingInfo(
            ticker="AAPL",
            form_type="10-K",
            filing_date=date(2023, 11, 3),
            accession_number="0000320193-23-000077",
            company_name="Apple Inc.",
        )
        fid = info.to_identifier()

        assert fid.ticker == "AAPL"
        assert fid.form_type == "10-K"
        assert fid.filing_date == date(2023, 11, 3)
        assert fid.accession_number == "0000320193-23-000077"


# ---------------------------------------------------------------------------
# Identity configuration — security critical
# ---------------------------------------------------------------------------


class TestIdentityConfiguration:
    def test_no_identity_on_empty_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(fetch_module, "set_identity", lambda s: calls.append(s))
        monkeypatch.setattr(fetch_module, "get_settings", lambda: _make_settings_stub())

        # No server-side credentials → constructor must NOT call
        # set_identity (Scenario B/C: per-session credentials required).
        FilingFetcher()

        assert calls == []

    def test_identity_applied_when_both_fields_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(fetch_module, "set_identity", lambda s: calls.append(s))
        monkeypatch.setattr(
            fetch_module,
            "get_settings",
            lambda: _make_settings_stub("Jane Doe", "jane@example.com"),
        )

        FilingFetcher()

        assert calls == ["Jane Doe jane@example.com"]

    def test_apply_identity_prefers_session_credentials(self, fetcher: FilingFetcher) -> None:
        recorded: list[str] = fetcher._test_identity_log  # type: ignore[attr-defined]
        recorded.clear()

        fetcher.apply_identity("Alice", "alice@example.com")

        assert recorded == ["Alice alice@example.com"]


@pytest.mark.security
class TestEdgarIdentityNotLogged:
    """The EDGAR identity is personal data — it must never hit the log.

    Covers the explicit privacy guarantee documented on
    :meth:`FilingFetcher.set_identity`.
    """

    def test_set_identity_does_not_log_credentials(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(fetch_module, "set_identity", lambda _s: None)
        monkeypatch.setattr(fetch_module, "get_settings", lambda: _make_settings_stub())

        fetcher = FilingFetcher()

        # Re-enable propagation so caplog (rooted at root) sees records
        # — the project logger normally sets propagate=False.
        fetcher_logger = logging.getLogger("sec_generative_search.pipeline.fetch")
        original_propagate = fetcher_logger.propagate
        fetcher_logger.propagate = True
        try:
            with caplog.at_level(logging.DEBUG):
                fetcher.set_identity("Jane Doe", "jane.doe@example.com")
        finally:
            fetcher_logger.propagate = original_propagate

        combined = "\n".join(r.getMessage() for r in caplog.records)
        assert "Jane Doe" not in combined
        assert "jane.doe@example.com" not in combined

    def test_configure_identity_does_not_log_credentials(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(fetch_module, "set_identity", lambda _s: None)
        monkeypatch.setattr(
            fetch_module,
            "get_settings",
            lambda: _make_settings_stub("Secret Name", "secret@example.com"),
        )

        fetcher_logger = logging.getLogger("sec_generative_search.pipeline.fetch")
        original_propagate = fetcher_logger.propagate
        fetcher_logger.propagate = True
        try:
            with caplog.at_level(logging.DEBUG):
                FilingFetcher()
        finally:
            fetcher_logger.propagate = original_propagate

        combined = "\n".join(r.getMessage() for r in caplog.records)
        assert "Secret Name" not in combined
        assert "secret@example.com" not in combined


@pytest.mark.security
class TestFetchLogStreamIsIdentifierFree:
    """The **real** fetcher's listing path logs no ticker or accession.

    ``fetch.py`` carries the largest concentration of wrapped ticker
    sites, so it gets a runtime companion to the static call-site lock
    in ``tests/core/test_logging.py``: the static lock proves an
    expression is *wrapped*, only a real call proves the wrapper
    survives the value. EDGAR is never touched — ``_get_company`` and
    ``_get_filings`` are stubbed.
    """

    def test_list_available_emits_no_raw_identifier(
        self,
        fetcher: FilingFetcher,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")

        ticker = "ZQXW"
        accession = "0000320193-23-000077"
        amended = "0000320193-23-000078"
        filings = [
            # An amendment first so the skip-branch (which logs the
            # accession) is exercised alongside the summary line.
            SimpleNamespace(
                accession_no=amended,
                form="10-K/A",
                filing_date=date(2023, 11, 4),
                company=ticker,
            ),
            SimpleNamespace(
                accession_no=accession,
                form="10-K",
                filing_date=date(2023, 11, 3),
                company=ticker,
            ),
        ]
        monkeypatch.setattr(fetcher, "_get_company", lambda t: SimpleNamespace())
        monkeypatch.setattr(fetcher, "_get_filings", lambda *a, **k: filings)

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.addFilter(AccessionRedactionFilter())
        package_logger = logging.getLogger(LOGGER_NAME)
        prior_level = package_logger.level
        package_logger.addHandler(handler)
        package_logger.setLevel(logging.DEBUG)
        try:
            assert fetcher.list_available(ticker, "10-K")
        finally:
            package_logger.removeHandler(handler)
            package_logger.setLevel(prior_level)

        emitted = stream.getvalue()
        assert emitted.strip(), "expected the fetcher to log something"
        assert ticker not in emitted
        assert accession not in emitted
        assert amended not in emitted
        assert "<redacted:" in emitted


# ---------------------------------------------------------------------------
# F22 — one edgartools ``Company`` per ticker inside a work-list build
# ---------------------------------------------------------------------------


class _CountingCompany:
    """Stands in for ``edgar.Company``; counts constructions per ticker."""

    constructed: ClassVar[list[str]] = []

    def __init__(self, ticker: str) -> None:
        type(self).constructed.append(ticker)
        self.ticker = ticker


@pytest.fixture
def counting_company(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    _CountingCompany.constructed = []
    monkeypatch.setattr(fetch_module, "Company", _CountingCompany)
    return _CountingCompany.constructed


class TestCompanyCache:
    def test_each_call_builds_a_company_outside_a_scope(
        self, fetcher: FilingFetcher, counting_company: list[str]
    ) -> None:
        fetcher._get_company("AAPL")
        fetcher._get_company("AAPL")
        assert counting_company == ["AAPL", "AAPL"]

    def test_scope_reuses_one_company_per_ticker(
        self, fetcher: FilingFetcher, counting_company: list[str]
    ) -> None:
        with fetcher.company_cache():
            first = fetcher._get_company("aapl")
            assert fetcher._get_company("AAPL") is first
            fetcher._get_company("MSFT")
        assert counting_company == ["AAPL", "MSFT"]

    def test_cache_ends_with_the_scope(
        self, fetcher: FilingFetcher, counting_company: list[str]
    ) -> None:
        """A cached ``Company`` must never outlive the build that loaded it."""
        with fetcher.company_cache():
            inside = fetcher._get_company("AAPL")
        assert fetcher._get_company("AAPL") is not inside
        with fetcher.company_cache():
            assert fetcher._get_company("AAPL") is not inside
        assert counting_company == ["AAPL"] * 3

    def test_nested_scope_reuses_the_outer_cache(
        self, fetcher: FilingFetcher, counting_company: list[str]
    ) -> None:
        with fetcher.company_cache():
            outer = fetcher._get_company("AAPL")
            with fetcher.company_cache():
                assert fetcher._get_company("AAPL") is outer
            # Leaving the inner block must not drop the outer cache.
            assert fetcher._get_company("AAPL") is outer
        assert counting_company == ["AAPL"]

    def test_threads_never_share_a_scope(
        self, fetcher: FilingFetcher, counting_company: list[str]
    ) -> None:
        import threading

        seen: list[object] = []
        with fetcher.company_cache():
            mine = fetcher._get_company("AAPL")
            worker = threading.Thread(target=lambda: seen.append(fetcher._get_company("AAPL")))
            worker.start()
            worker.join(timeout=5.0)
        assert seen and seen[0] is not mine
        assert counting_company == ["AAPL", "AAPL"]

    def test_a_failed_lookup_is_not_cached(
        self, fetcher: FilingFetcher, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts: list[str] = []

        def _flaky(ticker: str) -> object:
            attempts.append(ticker)
            if len(attempts) == 1:
                raise RuntimeError("EDGAR hiccup")
            return SimpleNamespace(ticker=ticker)

        monkeypatch.setattr(fetch_module, "Company", _flaky)
        with fetcher.company_cache():
            with pytest.raises(FetchError):
                fetcher._get_company("AAPL")
            assert fetcher._get_company("AAPL").ticker == "AAPL"
        assert attempts == ["AAPL", "AAPL"]

    def test_listing_across_forms_loads_the_company_once(
        self,
        fetcher: FilingFetcher,
        counting_company: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _filings(company: object, form_type: str, **_: Any) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    accession_no=f"0000320193-23-00000{i}",
                    form=form_type,
                    filing_date=date(2023, 11, i + 1),
                    company="Apple Inc.",
                )
                for i in range(2)
            ]

        monkeypatch.setattr(fetcher, "_get_filings", _filings)
        listed = fetcher.list_available_across_forms("AAPL", ("10-K", "10-Q", "8-K"), count=5)
        assert len(listed) == 5
        assert counting_company == ["AAPL"]


# ---------------------------------------------------------------------------
# A FetchError *message* never carries a ticker (callers log ``.message``)
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestFetchErrorMessagesCarryNoTicker:
    """``TaskManager._build_work_list`` logs ``exc.message`` beside an already
    redacted ticker, so a message that embeds the symbol re-leaks it under
    ``LOG_REDACT_QUERIES`` — out of reach of both the accession filter and
    the static ticker call-site lock (``exc.message`` is not a ticker
    carrier).  ``_get_company`` did exactly that ("Invalid ticker symbol:
    NVDA"), and a transient EDGAR failure raises it for a *valid* ticker."""

    def test_work_list_build_logs_no_raw_ticker_when_company_lookup_fails(
        self,
        fetcher: FilingFetcher,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sec_generative_search.api.tasks import TaskInfo, TaskManager

        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")
        ticker = "ZQXW"

        def _unreachable(_ticker: str) -> object:
            raise ConnectionError("EDGAR unreachable")

        monkeypatch.setattr(fetch_module, "Company", _unreachable)
        manager = TaskManager.__new__(TaskManager)  # only the build path is used
        manager._fetcher = fetcher
        info = TaskInfo(task_id="f" * 32, tickers=[ticker], form_types=["10-K", "10-Q"])

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.addFilter(AccessionRedactionFilter())
        package_logger = logging.getLogger(LOGGER_NAME)
        prior_level = package_logger.level
        package_logger.addHandler(handler)
        package_logger.setLevel(logging.DEBUG)
        try:
            assert manager._build_work_list(info) == []
        finally:
            package_logger.removeHandler(handler)
            package_logger.setLevel(prior_level)

        emitted = stream.getvalue()
        assert "fetch failed" in emitted  # the failure is still logged…
        assert ticker not in emitted  # …without the symbol

    def test_no_fetch_error_message_interpolates_a_ticker(self) -> None:
        """Static companion: every ``FetchError(...)`` built in ``fetch.py``
        keeps ``ticker`` out of its first (message) argument."""
        import ast
        from pathlib import Path

        source = Path(fetch_module.__file__).read_text(encoding="utf-8")
        offenders: list[int] = []
        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "FetchError"
                and node.args
            ):
                continue
            names = {n.id for n in ast.walk(node.args[0]) if isinstance(n, ast.Name)}
            attrs = {n.attr for n in ast.walk(node.args[0]) if isinstance(n, ast.Attribute)}
            if {"ticker", "tickers"} & (names | attrs):
                offenders.append(node.lineno)
        assert offenders == [], f"FetchError messages interpolating a ticker at lines {offenders}"


class TestLazyEdgarProxies:
    """F28: ``edgar`` is imported on the first EDGAR-bound call; the module-level
    ``Company`` / ``set_identity`` proxies must still reach edgartools."""

    def test_company_proxy_constructs_an_edgartools_company(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import edgar

        built: list[str] = []
        monkeypatch.setattr(edgar, "Company", lambda ticker: built.append(ticker) or "company")
        assert fetch_module.Company("AAPL") == "company"
        assert built == ["AAPL"]

    def test_set_identity_proxy_calls_edgartools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import edgar

        applied: list[str] = []
        monkeypatch.setattr(edgar, "set_identity", applied.append)
        fetch_module.set_identity("Jane Doe jane@example.invalid")
        assert applied == ["Jane Doe jane@example.invalid"]
