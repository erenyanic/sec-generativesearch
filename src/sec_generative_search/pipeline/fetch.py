"""
SEC filing fetcher using edgartools.

This module wraps the edgartools library to fetch SEC filings (8-K, 10-K, 10-Q
and amendments) from the EDGAR database.  Ingest is list-first: callers list
filing metadata (no HTML), drop known duplicates, then fetch each new
filing's HTML on demand — one ahead of processing (``pipeline/prefetch.py``).

Usage:
    from sec_generative_search.pipeline import FilingFetcher

    fetcher = FilingFetcher()

    # Newest 5 non-amendment 10-Ks (metadata only), optionally filtered by
    # year / date range
    available = fetcher.list_available("AAPL", "10-K", count=5)

    # The newest filings across several form types, merged by date
    available = fetcher.list_available_across_forms("AAPL", ("10-K", "10-Q"), count=4)

    # HTML for one listed filing
    filing_id, html = fetcher.fetch_filing_content(available[0])
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from sec_generative_search.config.constants import BASE_FORMS, SUPPORTED_FORMS
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import FetchError
from sec_generative_search.core.logging import get_logger, redact_for_log
from sec_generative_search.core.types import FilingIdentifier

if TYPE_CHECKING:
    from edgar import Company as EdgarCompany

logger = get_logger(__name__)


# edgartools is imported on the first EDGAR-bound call, not with this module
# (F28): ``import edgar`` costs ~0.8 s (its HTML/XBRL stack), and this module
# is reached by every ``sec_generative_search.pipeline`` import — the API and
# every ``sec-rag`` command, ``--help`` included.  The proxies keep the edgartools
# names at module level, which is also the seam the tests patch.
def Company(ticker: str) -> EdgarCompany:  # noqa: N802 — stands in for the edgartools class
    """Construct an ``edgar.Company`` (lazy import)."""
    from edgar import Company as _Company

    return _Company(ticker)


def set_identity(identity: str) -> None:
    """Call ``edgar.set_identity`` (lazy import)."""
    from edgar import set_identity as _set_identity

    _set_identity(identity)


@dataclass
class FilingInfo:
    """
    Summary information about an available filing (without content).

    This lightweight class is used by list_available() to preview
    filings before downloading their full HTML content.

    Attributes:
        ticker: Stock ticker symbol
        form_type: SEC form type (10-K, 10-Q)
        filing_date: Date filed with SEC
        accession_number: SEC-assigned unique identifier
        company_name: Full company name
        _filing_obj: Cached edgartools Filing object for direct content
            fetch (avoids redundant EDGAR API round-trips).  Not part
            of the public API — callers should use
            ``FilingFetcher.fetch_filing_content()`` instead.
    """

    ticker: str
    form_type: str
    filing_date: date
    accession_number: str
    company_name: str
    _filing_obj: Any = field(default=None, repr=False, compare=False)

    def to_identifier(self) -> FilingIdentifier:
        """Convert to FilingIdentifier for pipeline use."""
        return FilingIdentifier(
            ticker=self.ticker,
            form_type=self.form_type,
            filing_date=self.filing_date,
            accession_number=self.accession_number,
        )


class FilingFetcher:
    """
    Fetches SEC filings from EDGAR using edgartools.

    Lists filing metadata with optional filters and fetches one filing's
    HTML at a time.  It handles identity configuration automatically
    using credentials from settings.

    Methods:
        - list_available(): newest *count* filings of one form (metadata)
        - list_available_across_forms(): newest *count* across forms
        - fetch_filing_content(): HTML for one listed filing
        - fetch_by_accession(): fallback for a hand-built ``FilingInfo``

    Filter Options:
        - count: Maximum number of filings to list
        - year: Single year, list of years, or range
        - start_date/end_date: Date range filtering

    Attributes:
        settings: Application settings instance
        max_filings: Maximum filings limit from settings
    """

    def __init__(self) -> None:
        """Initialise the fetcher and configure EDGAR identity (if available)."""
        self.settings = get_settings()
        self.max_filings = self.settings.database.max_filings
        # Per-thread ``Company`` cache, live only inside ``company_cache()``.
        self._company_scope = threading.local()

        self._configure_identity()

    @contextmanager
    def company_cache(self) -> Iterator[None]:
        """Reuse one edgartools ``Company`` per ticker inside the block.

        ``Company.data`` — the parsed submissions index, hundreds of KB
        of JSON turned into Arrow tables for a large filer — is cached
        per *instance*, so listing three form types through three fresh
        ``Company`` objects loads and parses it three times (edgartools'
        own HTTP cache only saves the network round-trip, and only for
        30 s).  Inside this block the first lookup of a ticker is reused.

        Scope it to one work-list build: the cache dies with the block, so
        a ``Company`` never outlives the task that loaded it or carries a
        stale index into a later one.  Thread-local, so two concurrent
        builds never share an entry; a nested block reuses the outer
        cache.
        """
        if getattr(self._company_scope, "companies", None) is not None:
            yield
            return
        self._company_scope.companies = {}
        try:
            yield
        finally:
            self._company_scope.companies = None

    def apply_identity(self, name: str | None = None, email: str | None = None) -> None:
        """Apply the effective EDGAR identity for the current operation.

        ``edgar.set_identity()`` mutates process-global state, so callers must
        re-apply the intended identity before every EDGAR-bound operation.
        When per-session credentials are provided, they take precedence.
        Otherwise, the server-side defaults from settings are restored.
        """
        if name and email:
            self.set_identity(name, email)
            return

        self._configure_identity()

    def _configure_identity(self) -> None:
        """Configure SEC EDGAR identity from settings (if both fields are set).

        In web deployments (Scenarios B/C), server-side credentials may be
        unset — each user provides their own via ``set_identity()`` per
        request.  The fetcher still works; the caller is responsible for
        calling ``set_identity()`` before any EDGAR requests.
        """
        name = self.settings.edgar.identity_name
        email = self.settings.edgar.identity_email
        if name and email:
            try:
                set_identity(f"{name} {email}")
                logger.debug("EDGAR identity configured from server-side env vars")
            except Exception as e:
                raise FetchError(
                    "Failed to configure EDGAR identity",
                    details=str(e),
                ) from e
        else:
            logger.debug("EDGAR identity not configured — per-session credentials required")

    def set_identity(self, name: str, email: str) -> None:
        """Set EDGAR identity for the current request (per-session credentials).

        Called by the API layer when users supply credentials via HTTP
        headers (``X-Edgar-Name`` / ``X-Edgar-Email``).

        **Privacy:** EDGAR credentials are never logged — not even at
        DEBUG level.  They are personal identity data that must not
        appear in any log output, database, or file.
        """
        set_identity(f"{name} {email}")
        logger.debug("EDGAR identity set via per-session credentials")

    @staticmethod
    def _is_amendment(filing) -> bool:
        """Check whether a filing is an amendment (e.g. 10-K/A, 10-Q/A).

        edgartools returns amendments alongside their parent form when
        queried by a base form type (e.g. ``form='10-K'`` also returns
        ``10-K/A``).  We read the filing object's actual ``form``
        attribute and check for the ``/A`` suffix.

        Returns:
            True if the filing's actual form type ends with ``/A``.
        """
        actual_form = getattr(filing, "form", "")
        return isinstance(actual_form, str) and actual_form.endswith("/A")

    @staticmethod
    def _should_skip(filing, requested_form: str) -> bool:
        """Decide whether a filing should be skipped for the requested form.

        When the user requests a **base** form (e.g. ``10-K``), edgartools
        returns both originals and amendments.  We skip amendments so they
        don't displace the original via the UNIQUE constraint.

        When the user requests an **amendment** form (e.g. ``10-K/A``),
        edgartools returns only amendments, so nothing is skipped.

        Returns:
            True if the filing should be excluded from results.
        """
        if requested_form in BASE_FORMS:
            actual_form = getattr(filing, "form", "")
            return isinstance(actual_form, str) and actual_form.endswith("/A")
        return False

    def _validate_form_type(self, form_type: str) -> str:
        """
        Validate and normalise form type.

        Args:
            form_type: SEC form type (e.g., "10-K", "10-Q")

        Returns:
            Normalised form type in uppercase.

        Raises:
            FetchError: If form type is not supported.
        """
        normalised = form_type.upper()
        if normalised not in SUPPORTED_FORMS:
            raise FetchError(
                f"Unsupported form type: {form_type}",
                details=f"Supported forms: {', '.join(SUPPORTED_FORMS)}",
            )
        return normalised

    def _parse_filing_date(self, date_value: str | date) -> date:
        """
        Parse filing date from edgartools.

        edgartools may return dates as strings or date objects depending
        on the version. This method handles both cases.

        Args:
            date_value: Filing date as string (YYYY-MM-DD) or date object.

        Returns:
            Python date object.
        """
        if isinstance(date_value, date):
            return date_value
        return datetime.strptime(date_value, "%Y-%m-%d").date()

    def _format_date_filter(
        self,
        start_date: str | date | None,
        end_date: str | date | None,
    ) -> str | None:
        """
        Format date range for edgartools filing_date parameter.

        edgartools accepts date ranges in format: "YYYY-MM-DD:YYYY-MM-DD"
        For open-ended ranges: "YYYY-MM-DD:" or ":YYYY-MM-DD"

        Args:
            start_date: Range start (inclusive), or None for open start
            end_date: Range end (inclusive), or None for open end

        Returns:
            Formatted date range string, or None if no dates specified.
        """
        if start_date is None and end_date is None:
            return None

        start_str = ""
        end_str = ""

        if start_date is not None:
            start_str = start_date.isoformat() if isinstance(start_date, date) else start_date

        if end_date is not None:
            end_str = end_date.isoformat() if isinstance(end_date, date) else end_date

        return f"{start_str}:{end_str}"

    def _get_company(self, ticker: str) -> EdgarCompany:
        """
        Get Company object for ticker with error handling.

        Args:
            ticker: Stock ticker symbol

        Returns:
            edgartools Company object

        Raises:
            FetchError: If ticker is invalid

        Inside :meth:`company_cache` the instance is reused per ticker.
        """
        key = ticker.upper()
        companies = getattr(self._company_scope, "companies", None)
        if companies is not None and key in companies:
            return companies[key]
        try:
            company = Company(key)
        except Exception as e:
            # The message is logged by callers (``exc.message``), so it must
            # not name the ticker — that would bypass LOG_REDACT_QUERIES.  A
            # caller that shows the error to an operator names it itself.
            raise FetchError(
                "Invalid ticker symbol or EDGAR lookup failed",
                details=str(e),
            ) from e
        if companies is not None:
            companies[key] = company
        return company

    def _get_filings(
        self,
        company: EdgarCompany,
        form_type: str,
        year: int | list[int] | range | None = None,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ):
        """
        Get filings from company with optional filters.

        Args:
            company: edgartools Company object
            form_type: SEC form type
            year: Year filter (single, list, or range)
            start_date: Date range start
            end_date: Date range end

        Returns:
            edgartools Filings object

        Raises:
            FetchError: If no filings found
        """
        # Build filter arguments
        kwargs: dict = {"form": form_type}

        # Add year filter if specified
        if year is not None:
            if isinstance(year, range):
                kwargs["year"] = list(year)
            else:
                kwargs["year"] = year

        # Add date range filter if specified
        date_filter = self._format_date_filter(start_date, end_date)
        if date_filter is not None:
            kwargs["filing_date"] = date_filter

        logger.debug("Fetching filings with filters: %s", kwargs)

        try:
            filings = company.get_filings(**kwargs)

            if not filings or len(filings) == 0:
                filter_desc = []
                if year:
                    filter_desc.append(f"year={year}")
                if date_filter:
                    filter_desc.append(f"date={date_filter}")
                filter_str = ", ".join(filter_desc) if filter_desc else "no filters"

                raise FetchError(
                    f"No {form_type} filings found ({filter_str})",
                    details="Try adjusting your filter criteria.",
                )

            return filings

        except FetchError:
            raise
        except Exception as e:
            raise FetchError(
                "Failed to retrieve filings",
                details=str(e),
            ) from e

    def _fetch_filing_content(
        self,
        filing,
        ticker: str,
        form_type: str,
    ) -> tuple[FilingIdentifier, str]:
        """
        Fetch HTML content for a single filing.

        Args:
            filing: edgartools Filing object
            ticker: Stock ticker symbol
            form_type: SEC form type

        Returns:
            Tuple of (FilingIdentifier, html_content)

        Raises:
            FetchError: If content fetch fails
        """
        try:
            html_content = filing.html()

            if not html_content:
                raise FetchError(
                    "Empty HTML content received",
                    details=f"Filing {filing.accession_no} returned no content.",
                )

            filing_id = FilingIdentifier(
                ticker=ticker.upper(),
                form_type=form_type,
                filing_date=self._parse_filing_date(filing.filing_date),
                accession_number=filing.accession_no,
            )

            return filing_id, html_content

        except FetchError:
            raise
        except Exception as e:
            raise FetchError(
                f"Failed to fetch content for {filing.accession_no}",
                details=str(e),
            ) from e

    # =========================================================================
    # Public Methods (Content Fetch)
    # =========================================================================

    def fetch_filing_content(
        self,
        filing_info: FilingInfo,
    ) -> tuple[FilingIdentifier, str]:
        """
        Fetch HTML content for a filing using its cached edgartools object.

        When ``FilingInfo`` was created by ``list_available()``, the
        original edgartools ``Filing`` object is stored on
        ``_filing_obj``.  This method uses it directly to fetch HTML,
        avoiding the redundant EDGAR API round-trip that
        ``fetch_by_accession()`` would perform (which re-fetches ALL
        filings for the ticker and linear-scans for the accession
        number).

        Falls back to ``fetch_by_accession()`` when ``_filing_obj`` is
        ``None`` (e.g. when ``FilingInfo`` was constructed manually in
        tests).

        Args:
            filing_info: Filing metadata with optional cached filing object.

        Returns:
            Tuple of (FilingIdentifier, html_content).

        Raises:
            FetchError: If content fetch fails.
        """
        if filing_info._filing_obj is not None:
            logger.info(
                "Fetching %s %s content directly (accession: %s)",
                redact_for_log(filing_info.ticker),
                filing_info.form_type,
                filing_info.accession_number,
            )
            filing_id, html_content = self._fetch_filing_content(
                filing_info._filing_obj,
                filing_info.ticker,
                filing_info.form_type,
            )
            logger.info(
                "Fetched %s %s (%s): %s characters",
                redact_for_log(filing_info.ticker),
                filing_info.form_type,
                filing_id.date_str,
                f"{len(html_content):,}",
            )
            return filing_id, html_content

        # Fallback: no cached object — use the original linear scan.
        return self.fetch_by_accession(
            filing_info.ticker,
            filing_info.form_type,
            filing_info.accession_number,
        )

    # =========================================================================
    # Public Methods (Listing)
    # =========================================================================

    def list_available(
        self,
        ticker: str,
        form_type: str = "10-K",
        *,
        count: int | None = None,
        year: int | list[int] | range | None = None,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> list[FilingInfo]:
        """
        List available filings without downloading content.

        Use this method to preview what filings are available before
        downloading. This is useful for showing users what will be
        fetched or for selecting specific filings.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            form_type: SEC form type ("8-K", "10-K", or "10-Q")
            count: Maximum number of filings to list (default: max_filings)
            year: Filter by year (single int, list, or range)
            start_date: Filter by date range start (YYYY-MM-DD or date)
            end_date: Filter by date range end (YYYY-MM-DD or date)

        Returns:
            List of FilingInfo objects with filing metadata

        Raises:
            FetchError: If ticker is invalid or no filings found

        Example:
            >>> available = fetcher.list_available("AAPL", "10-K", count=10)
            >>> for info in available:
            ...     print(f"{info.filing_date}: {info.accession_number}")
        """
        form_type = self._validate_form_type(form_type)
        ticker = ticker.upper()

        # Default to max_filings if count not specified
        if count is None:
            count = self.max_filings

        company = self._get_company(ticker)
        filings = self._get_filings(
            company, form_type, year=year, start_date=start_date, end_date=end_date
        )

        # Limit results — islice stops iteration after count items,
        # avoiding materialising the entire filing list from EDGAR.
        # When a base form is requested, amendments are filtered out so
        # they do not displace the original via the UNIQUE constraint.
        result = []
        for filing in filings:
            if len(result) >= count:
                break
            if self._should_skip(filing, form_type):
                logger.debug(
                    "Skipping amendment %s (%s) — original filing preferred",
                    filing.accession_no,
                    getattr(filing, "form", "unknown"),
                )
                continue
            result.append(
                FilingInfo(
                    ticker=ticker,
                    form_type=form_type,
                    filing_date=self._parse_filing_date(filing.filing_date),
                    accession_number=filing.accession_no,
                    company_name=getattr(filing, "company", ticker),
                    _filing_obj=filing,
                )
            )

        logger.info(
            "Listed %d available %s filings for %s",
            len(result),
            form_type,
            redact_for_log(ticker),
        )

        return result

    def list_available_across_forms(
        self,
        ticker: str,
        form_types: tuple[str, ...],
        *,
        count: int,
        year: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[FilingInfo]:
        """
        List available filings across multiple form types, sorted by date.

        Calls ``list_available()`` per form type, merges all results, sorts
        by ``filing_date`` descending, and returns the top *count* entries.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            form_types: Form types to search across (e.g., ("10-K", "10-Q")).
            count: Maximum number of filings to return (newest first).
            year: Optional filing-year filter.
            start_date: Optional start-date filter (YYYY-MM-DD).
            end_date: Optional end-date filter (YYYY-MM-DD).

        Returns:
            List of ``FilingInfo`` objects, sorted by filing_date descending,
            truncated to *count*.
        """
        all_available: list[FilingInfo] = []
        with self.company_cache():
            for form_type in form_types:
                try:
                    available = self.list_available(
                        ticker,
                        form_type,
                        count=count,
                        year=year,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    all_available.extend(available)
                except FetchError:
                    continue
        all_available.sort(key=lambda fi: fi.filing_date, reverse=True)
        return all_available[:count]

    def fetch_by_accession(
        self,
        ticker: str,
        form_type: str,
        accession_number: str,
    ) -> tuple[FilingIdentifier, str]:
        """
        Fetch a specific filing by its accession number.

        Use this method when you know the exact accession number of the
        filing you want. This is useful for re-fetching a specific filing
        or when the user has selected from list_available().

        Args:
            ticker: Stock ticker symbol
            form_type: SEC form type ("8-K", "10-K", or "10-Q")
            accession_number: SEC accession number (e.g., "0000320193-23-000077")

        Returns:
            Tuple of (FilingIdentifier, html_content)

        Raises:
            FetchError: If filing not found or fetch fails

        Example:
            >>> filing_id, html = fetcher.fetch_by_accession(
            ...     "AAPL", "10-K", "0000320193-23-000077"
            ... )
        """
        form_type = self._validate_form_type(form_type)
        ticker = ticker.upper()

        logger.info(
            "Fetching %s %s by accession: %s",
            redact_for_log(ticker),
            form_type,
            accession_number,
        )

        company = self._get_company(ticker)
        filings = self._get_filings(company, form_type)

        # Search for matching accession number, skipping mismatched forms.
        for filing in filings:
            if self._should_skip(filing, form_type):
                continue
            if filing.accession_no == accession_number:
                filing_id, html_content = self._fetch_filing_content(filing, ticker, form_type)
                logger.info(
                    "Fetched %s %s (%s): %s characters",
                    redact_for_log(ticker),
                    form_type,
                    filing_id.date_str,
                    f"{len(html_content):,}",
                )
                return filing_id, html_content

        raise FetchError(
            f"Filing not found: {accession_number}",
            details=f"No {form_type} filing with this accession number for {ticker}.",
        )
