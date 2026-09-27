"""Tests for config/constants.py.

Covers the existing SEC filing constants and the RAG-specific
constants added for this project.
"""

from __future__ import annotations

import pytest

from sec_generative_search.config.constants import (
    BASE_FORMS,
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    DEFAULT_PROVIDER_MAX_RETRIES,
    DEFAULT_PROVIDER_TIMEOUT,
    SUPPORTED_FORMS,
    parse_form_types,
)


class TestSupportedForms:
    def test_contains_expected_forms(self) -> None:
        expected = {"8-K", "8-K/A", "10-K", "10-K/A", "10-Q", "10-Q/A"}
        assert set(SUPPORTED_FORMS) == expected

    def test_base_forms_are_subset(self) -> None:
        assert set(BASE_FORMS).issubset(SUPPORTED_FORMS)


class TestParseFormTypes:
    def test_single_form(self) -> None:
        assert parse_form_types("10-K") == ("10-K",)

    def test_lowercase_normalised(self) -> None:
        assert parse_form_types("10-k") == ("10-K",)

    def test_multiple_forms_sorted(self) -> None:
        assert parse_form_types("10-Q,10-K") == ("10-K", "10-Q")

    def test_whitespace_stripped(self) -> None:
        assert parse_form_types(" 10-K , 10-Q ") == ("10-K", "10-Q")

    def test_duplicates_removed(self) -> None:
        assert parse_form_types("10-K,10-K,10-Q") == ("10-K", "10-Q")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty form type"):
            parse_form_types("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty form type"):
            parse_form_types("   ,  ")

    def test_unknown_form_raises(self) -> None:
        # Security-relevant: unknown form types must be rejected, not
        # silently accepted — prevents injection of arbitrary form
        # strings into downstream EDGAR queries.
        with pytest.raises(ValueError, match="Unsupported form type"):
            parse_form_types("10-K,DEF 14A")


class TestRAGConstants:
    def test_context_token_budget_positive(self) -> None:
        assert DEFAULT_CONTEXT_TOKEN_BUDGET > 0

    def test_chunk_overlap_non_negative(self) -> None:
        assert DEFAULT_CHUNK_OVERLAP_TOKENS >= 0


class TestProviderConstants:
    def test_timeout_positive(self) -> None:
        assert DEFAULT_PROVIDER_TIMEOUT > 0

    def test_max_retries_non_negative(self) -> None:
        assert DEFAULT_PROVIDER_MAX_RETRIES >= 0
