"""Tests for :mod:`sec_generative_search.rag.citations`."""

from __future__ import annotations

import json
import logging

import pytest

from sec_generative_search.core.logging import LOGGER_NAME
from sec_generative_search.core.types import ContentType, RetrievalResult
from sec_generative_search.rag.citations import (
    _build_citations,
    extract_citations,
    extract_from_inline_markers,
    extract_from_json_envelope,
)


class TestInlineMarkerExtraction:
    def test_picks_up_simple_markers(self, sample_chunks) -> None:
        text = "Revenue grew [1] and margin compressed [2]."
        result = extract_from_inline_markers(text, sample_chunks)
        assert [c.chunk_id for c in result.citations] == [
            sample_chunks[0].chunk_id,
            sample_chunks[1].chunk_id,
        ]
        # Display indices are 1-based in mention order.
        assert [c.display_index for c in result.citations] == [1, 2]
        # The answer text is preserved verbatim — UI needs the markers.
        assert result.answer == text

    def test_first_mention_order_for_duplicates(self, sample_chunks) -> None:
        text = "X [2]. Y [2]. Z [1]."
        result = extract_from_inline_markers(text, sample_chunks)
        # First mention of [2] then [1].
        assert [c.chunk_id for c in result.citations] == [
            sample_chunks[1].chunk_id,
            sample_chunks[0].chunk_id,
        ]

    @pytest.mark.security
    def test_drops_out_of_range_markers(self, sample_chunks) -> None:
        """A model that fabricates [99] must not crash or produce a phantom citation."""
        text = "Real [1]. Fabricated [99]. Real [2]."
        result = extract_from_inline_markers(text, sample_chunks)
        assert len(result.citations) == 2
        valid_ids = {sample_chunks[0].chunk_id, sample_chunks[1].chunk_id}
        assert all(c.chunk_id in valid_ids for c in result.citations)

    def test_no_markers_yields_no_citations(self, sample_chunks) -> None:
        result = extract_from_inline_markers("Plain answer with no markers.", sample_chunks)
        assert result.citations == []
        assert result.answer == "Plain answer with no markers."

    def test_empty_chunks_returns_empty(self) -> None:
        result = extract_from_inline_markers("Answer [1].", [])
        assert result.citations == []

    @pytest.mark.security
    def test_inline_path_caps_at_max_citations(self, make_chunk) -> None:
        """>50 in-range markers must still yield at most 50 citations."""
        chunks = [make_chunk(index=i) for i in range(1, 61)]
        text = " ".join(f"[{i}]" for i in range(1, 61))
        result = extract_from_inline_markers(text, chunks)
        assert len(result.citations) == 50

    def test_drops_chunk_whose_to_citation_fails(self, make_chunk) -> None:
        """A retrieved chunk missing required citation metadata is dropped, not raised.

        ``_build_citations`` delegates to ``RetrievalResult.to_citation``,
        which raises ``CitationError`` when ``accession_number`` /
        ``filing_date`` are absent. That signals an upstream bug, not
        adversarial input, so the citation is logged-and-dropped — the
        answer must still come back.
        """
        from sec_generative_search.core.types import ContentType, RetrievalResult

        malformed = RetrievalResult(
            content="text",
            path="X",
            content_type=ContentType.TEXT,
            ticker="AAPL",
            form_type="10-K",
            similarity=0.9,
            chunk_id="AAPL_001",
            # accession_number / filing_date deliberately omitted.
        )
        good = make_chunk(index=2)
        result = extract_from_inline_markers("[1] then [2]", [malformed, good])
        # The malformed chunk is dropped; the well-formed one survives.
        assert [c.chunk_id for c in result.citations] == [good.chunk_id]


class TestJsonEnvelopeExtraction:
    def test_parses_minimal_envelope(self, sample_chunks) -> None:
        payload = json.dumps(
            {
                "answer": "Revenue grew.",
                "cited_chunk_ids": [sample_chunks[0].chunk_id],
            }
        )
        result = extract_from_json_envelope(payload, sample_chunks)
        assert result.answer == "Revenue grew."
        assert [c.chunk_id for c in result.citations] == [sample_chunks[0].chunk_id]

    def test_strips_markdown_fence(self, sample_chunks) -> None:
        payload = (
            "```json\n"
            + json.dumps(
                {
                    "answer": "Fenced.",
                    "cited_chunk_ids": [sample_chunks[1].chunk_id],
                }
            )
            + "\n```"
        )
        result = extract_from_json_envelope(payload, sample_chunks)
        assert result.answer == "Fenced."
        assert result.citations[0].chunk_id == sample_chunks[1].chunk_id

    def test_finds_object_amid_prose(self, sample_chunks) -> None:
        payload = (
            "Sure! Here is the JSON: "
            + json.dumps({"answer": "X.", "cited_chunk_ids": [sample_chunks[0].chunk_id]})
            + " Hope that helps."
        )
        result = extract_from_json_envelope(payload, sample_chunks)
        assert result.answer == "X."

    @pytest.mark.security
    def test_drops_unknown_chunk_ids(self, sample_chunks) -> None:
        """Model fabrication of a chunk_id must not produce a phantom citation."""
        payload = json.dumps(
            {
                "answer": "Answer.",
                "cited_chunk_ids": [
                    sample_chunks[0].chunk_id,
                    "FABRICATED_CHUNK_ID",
                ],
            }
        )
        result = extract_from_json_envelope(payload, sample_chunks)
        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == sample_chunks[0].chunk_id

    def test_raises_on_missing_answer(self, sample_chunks) -> None:
        from sec_generative_search.core.exceptions import CitationError

        payload = json.dumps({"cited_chunk_ids": []})
        with pytest.raises(CitationError):
            extract_from_json_envelope(payload, sample_chunks)

    def test_raises_on_no_object(self, sample_chunks) -> None:
        from sec_generative_search.core.exceptions import CitationError

        with pytest.raises(CitationError):
            extract_from_json_envelope("plain text no json", sample_chunks)

    def test_raises_on_malformed_json_object(self, sample_chunks) -> None:
        """An isolated ``{...}`` that is not valid JSON raises CitationError.

        Distinct from "no object": here a brace-balanced span *is* found
        but :func:`json.loads` rejects it (the model emitted broken JSON).
        """
        from sec_generative_search.core.exceptions import CitationError

        with pytest.raises(CitationError):
            # Two adjacent strings with no separator — JSONDecodeError.
            extract_from_json_envelope('{"answer": "x" "oops"}', sample_chunks)

    def test_raises_on_unbalanced_object(self, sample_chunks) -> None:
        """An opening brace with no matching close yields no object → CitationError."""
        from sec_generative_search.core.exceptions import CitationError

        with pytest.raises(CitationError):
            extract_from_json_envelope('{"answer": "x"', sample_chunks)

    def test_raises_when_cited_ids_not_a_list(self, sample_chunks) -> None:
        """``cited_chunk_ids`` of the wrong type is a parse failure, not silent drop."""
        from sec_generative_search.core.exceptions import CitationError

        payload = json.dumps({"answer": "x", "cited_chunk_ids": "not-a-list"})
        with pytest.raises(CitationError):
            extract_from_json_envelope(payload, sample_chunks)

    def test_dedupes_and_ignores_non_string_ids(self, sample_chunks) -> None:
        """Duplicate ids collapse; a non-string id is skipped, never crashes."""
        payload = json.dumps(
            {
                "answer": "x",
                "cited_chunk_ids": [
                    sample_chunks[0].chunk_id,
                    sample_chunks[0].chunk_id,  # duplicate → dropped
                    12345,  # non-string → skipped
                    sample_chunks[1].chunk_id,
                ],
            }
        )
        result = extract_from_json_envelope(payload, sample_chunks)
        assert [c.chunk_id for c in result.citations] == [
            sample_chunks[0].chunk_id,
            sample_chunks[1].chunk_id,
        ]

    @pytest.mark.security
    def test_json_path_caps_at_max_citations(self, make_chunk) -> None:
        """A pathological model emitting >50 ids must yield at most 50 citations."""
        chunks = [make_chunk(index=i) for i in range(1, 61)]
        payload = json.dumps({"answer": "x", "cited_chunk_ids": [c.chunk_id for c in chunks]})
        result = extract_from_json_envelope(payload, chunks)
        assert len(result.citations) == 50

    def test_isolates_object_with_nested_braces(self, sample_chunks) -> None:
        """A nested object inside the envelope must not confuse brace-matching."""
        payload = (
            '{"answer": "x", "meta": {"k": "v"}, '
            f'"cited_chunk_ids": ["{sample_chunks[0].chunk_id}"]}}'
        )
        result = extract_from_json_envelope(payload, sample_chunks)
        assert result.answer == "x"
        assert result.citations[0].chunk_id == sample_chunks[0].chunk_id

    def test_isolates_object_with_escaped_quote_and_brace_in_string(self, sample_chunks) -> None:
        """Braces/quotes inside a JSON string literal must not terminate the scan."""
        # answer holds an escaped quote AND a brace — both must be treated
        # as string content, not structural tokens.
        payload = json.dumps(
            {
                "answer": 'he said "hi" and wrote {x}',
                "cited_chunk_ids": [sample_chunks[0].chunk_id],
            }
        )
        result = extract_from_json_envelope(payload, sample_chunks)
        assert result.answer == 'he said "hi" and wrote {x}'
        assert result.citations[0].chunk_id == sample_chunks[0].chunk_id

    def test_strips_fence_without_newline(self, sample_chunks) -> None:
        """A single-line ```...``` wrapper (no newline) is still unwrapped."""
        payload = (
            "```"
            + json.dumps({"answer": "y", "cited_chunk_ids": [sample_chunks[0].chunk_id]})
            + "```"
        )
        result = extract_from_json_envelope(payload, sample_chunks)
        assert result.answer == "y"

    def test_handles_unterminated_fence(self, sample_chunks) -> None:
        """An opening ```json fence with no closing fence still parses."""
        payload = "```json\n" + json.dumps(
            {"answer": "z", "cited_chunk_ids": [sample_chunks[1].chunk_id]}
        )
        result = extract_from_json_envelope(payload, sample_chunks)
        assert result.answer == "z"
        assert result.citations[0].chunk_id == sample_chunks[1].chunk_id


class TestHybridDispatcher:
    def test_prefers_json_when_flag_set(self, sample_chunks) -> None:
        payload = json.dumps(
            {"answer": "JSON path.", "cited_chunk_ids": [sample_chunks[0].chunk_id]}
        )
        result = extract_citations(payload, sample_chunks, prefer_json=True)
        assert result.answer == "JSON path."
        assert result.citations[0].chunk_id == sample_chunks[0].chunk_id

    def test_falls_back_to_inline_when_json_path_set_but_payload_invalid(
        self, sample_chunks
    ) -> None:
        """JSON parse failure must not blank out citations — try inline markers."""
        payload = "Not JSON, just text [1]."
        result = extract_citations(payload, sample_chunks, prefer_json=True)
        assert result.answer == payload
        assert len(result.citations) == 1
        assert result.citations[0].chunk_id == sample_chunks[0].chunk_id

    def test_uses_inline_when_flag_unset(self, sample_chunks) -> None:
        payload = "Inline [1] [2]."
        result = extract_citations(payload, sample_chunks, prefer_json=False)
        assert len(result.citations) == 2


@pytest.mark.security
class TestCitationLoggingIsTotalAndContentFree:
    """The drop-and-log path must survive redaction being switched on.

    ``chunk_id`` is ``str | None`` and ``to_citation()`` raises precisely
    when it is falsy — so a *missing* id is the primary way execution
    reaches the warning below. Handing ``None`` to ``redact_for_log``
    there would raise ``AttributeError`` out of an ``except`` block and
    turn the documented "logged-and-dropped, never raised" contract into
    a raise, on the RAG generation path, only when
    ``LOG_REDACT_QUERIES`` is on.
    """

    @staticmethod
    def _malformed_chunk(*, chunk_id: str | None) -> RetrievalResult:
        return RetrievalResult(
            content="Segment revenue rose.",
            path="Part I > Item 1",
            content_type=ContentType.TEXT,
            ticker="ZQXW",
            form_type="10-K",
            similarity=0.5,
            # ``filing_date`` missing → ``to_citation`` raises even when
            # a chunk_id *is* present, so both branches are reachable.
            filing_date=None,
            accession_number="0000320193-23-000077",
            chunk_id=chunk_id,
            token_count=8,
        )

    @pytest.mark.parametrize("chunk_id", [None, "ZQXW_10-K_2023-09-30_001"])
    def test_drops_without_raising_under_redaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
        chunk_id: str | None,
    ) -> None:
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")
        assert _build_citations([self._malformed_chunk(chunk_id=chunk_id)]) == []

    def test_warning_carries_no_ticker(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")
        logger_name = "sec_generative_search.rag.citations"
        package_logger = logging.getLogger(LOGGER_NAME)
        prior_propagate = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger=logger_name):
                _build_citations([self._malformed_chunk(chunk_id="ZQXW_10-K_2023-09-30_001")])
        finally:
            package_logger.propagate = prior_propagate

        messages = [r.getMessage() for r in caplog.records if r.name == logger_name]
        assert messages, "expected a drop warning"
        # ``chunk_id`` is ``{TICKER}_{FORM}_{DATE}_{INDEX}`` — the whole
        # value is a ticker carrier, so it must not appear raw.
        assert all("ZQXW" not in message for message in messages)
