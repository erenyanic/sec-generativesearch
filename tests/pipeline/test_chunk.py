"""Tests for :mod:`sec_generative_search.pipeline.chunk`.

Covers the chunker's core guarantees:
    - Sentence-boundary aware splitting stays within the ``±``
      ``tolerance`` band around ``token_limit`` where possible, with a
      hard ceiling at ``token_limit + tolerance`` (the only exception
      is a single-sentence chunk that is itself larger than the
      ceiling — sentences are never mid-split).
    - Active-centring: once the running total crosses the lower edge
      of the band, the chunk is finalised at the next sentence
      boundary whose inclusion would push past ``token_limit`` — so
      chunks cluster around the target rather than drift to the upper
      edge.
    - ``ContentType.TABLE`` segments are emitted whole, even when they
      exceed ``token_limit`` — row alignment must survive chunking.
    - Sentence-level overlap (Strategy B, soft upper bound): always
      carries the trailing sentence of a finalised chunk into the next
      chunk — even when that single sentence alone exceeds
      ``overlap_tokens`` — to preserve grammatical continuity at the
      chunk boundary; prior sentences are added under a strict
      stop-before-overflow rule.  ``overlap_tokens=0`` disables the
      behaviour entirely.
    - Chunk boundaries never cross segments (section boundaries are
      preserved implicitly by chunking per segment).
    - Constructor rejects invalid overlap configuration eagerly rather
      than blowing up mid-chunk.
"""

from __future__ import annotations

import io
import logging
from datetime import date
from itertools import pairwise

import pytest

from sec_generative_search.core.exceptions import ChunkingError
from sec_generative_search.core.logging import LOGGER_NAME, AccessionRedactionFilter
from sec_generative_search.core.types import (
    ContentType,
    FilingIdentifier,
    Segment,
)
from sec_generative_search.pipeline.chunk import TextChunker


@pytest.fixture
def filing_id() -> FilingIdentifier:
    return FilingIdentifier(
        ticker="AAPL",
        form_type="10-K",
        filing_date=date(2023, 11, 3),
        accession_number="0000320193-23-000077",
    )


def _make_segment(
    content: str,
    filing_id: FilingIdentifier,
    path: str = "Part I > Item 1A",
    content_type: ContentType = ContentType.TEXT,
) -> Segment:
    return Segment(
        path=path,
        content_type=content_type,
        content=content,
        filing_id=filing_id,
    )


class TestChunkerInit:
    def test_defaults_from_settings(self, clean_env: pytest.MonkeyPatch) -> None:
        chunker = TextChunker()
        # Defaults match constants exposed via Settings — see settings.py.
        # token_limit / tolerance define the ±150 band around 1000.
        # overlap_tokens is 15 % of token_limit.
        assert chunker.token_limit == 1000
        assert chunker.tolerance == 150
        assert chunker.overlap_tokens == 150

    def test_explicit_arguments_override_settings(self, clean_env: pytest.MonkeyPatch) -> None:
        chunker = TextChunker(token_limit=100, tolerance=10, overlap_tokens=0)
        assert chunker.token_limit == 100
        assert chunker.tolerance == 10
        assert chunker.overlap_tokens == 0

    def test_negative_overlap_tokens_rejected(self, clean_env: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            TextChunker(overlap_tokens=-1)

    def test_overlap_not_less_than_limit_rejected(self, clean_env: pytest.MonkeyPatch) -> None:
        # Overlap must be strictly less than token_limit, otherwise the
        # chunker cannot make forward progress.
        with pytest.raises(ValueError, match="strictly less than"):
            TextChunker(token_limit=50, overlap_tokens=50)

    def test_overlap_equal_to_limit_rejected(self, clean_env: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValueError, match="strictly less than"):
            TextChunker(token_limit=100, overlap_tokens=100)


class TestChunkSegments:
    def test_empty_list_raises(self, clean_env: pytest.MonkeyPatch) -> None:
        chunker = TextChunker()
        with pytest.raises(ChunkingError, match="No segments"):
            chunker.chunk_segments([])

    def test_short_segment_kept_whole(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        chunker = TextChunker(token_limit=100, tolerance=10, overlap_tokens=0)
        seg = _make_segment("one two three four.", filing_id)
        chunks = chunker.chunk_segments([seg])

        assert len(chunks) == 1
        assert chunks[0].content == "one two three four."
        assert chunks[0].chunk_index == 0
        assert chunks[0].token_count == 4
        assert chunks[0].path == seg.path

    def test_long_segment_split_on_sentence_boundary(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        chunker = TextChunker(token_limit=10, tolerance=2, overlap_tokens=0)
        text = " ".join([f"Sentence {i} has five tokens." for i in range(6)])
        seg = _make_segment(text, filing_id)
        chunks = chunker.chunk_segments([seg])

        assert len(chunks) > 1
        for chunk in chunks:
            # Each chunk must end at a sentence boundary.
            assert chunk.content.rstrip().endswith(".")
            # Tokens stay within limit + tolerance (unless the single
            # sentence on its own exceeds it — not the case here).
            assert chunk.token_count <= chunker.token_limit + chunker.tolerance

    def test_sequential_chunk_indices(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        chunker = TextChunker(token_limit=10, tolerance=2, overlap_tokens=0)
        seg_a = _make_segment(
            " ".join([f"A{i} sentence filler here." for i in range(4)]),
            filing_id,
            path="Part I > Item 1",
        )
        seg_b = _make_segment(
            " ".join([f"B{i} sentence filler here." for i in range(4)]),
            filing_id,
            path="Part I > Item 2",
        )
        chunks = chunker.chunk_segments([seg_a, seg_b])

        # Indices must run 0..N-1 with no gaps — storage-layer IDs depend
        # on this.
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_chunks_do_not_span_segments(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        chunker = TextChunker(token_limit=500, tolerance=50, overlap_tokens=0)
        seg_a = _make_segment("Section A content.", filing_id, path="Part I > Item 1")
        seg_b = _make_segment("Section B content.", filing_id, path="Part II > Item 7")
        chunks = chunker.chunk_segments([seg_a, seg_b])

        # Each chunk's path must match exactly one segment — chunks must
        # never be stitched across section boundaries.
        paths = {c.path for c in chunks}
        assert paths == {"Part I > Item 1", "Part II > Item 7"}


class TestTableNeverSplit:
    def test_oversized_table_is_single_chunk(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        # 400 tokens — well over the 10-token limit.  Sentence-boundary
        # splitting would destroy row alignment; the chunker must emit
        # the whole table as one chunk.
        table_content = "\n".join(f"Row{i} | col_a | col_b | col_c" for i in range(100))
        chunker = TextChunker(token_limit=10, tolerance=2, overlap_tokens=0)
        seg = _make_segment(table_content, filing_id, content_type=ContentType.TABLE)

        chunks = chunker.chunk_segments([seg])

        assert len(chunks) == 1
        assert chunks[0].content == table_content
        assert chunks[0].content_type is ContentType.TABLE
        assert chunks[0].token_count > chunker.token_limit


class TestOverlap:
    def _build_segment(self, filing_id: FilingIdentifier) -> Segment:
        # 12 sentences * 5 tokens = 60 tokens — forces several splits at
        # token_limit=15.
        text = " ".join([f"Sentence {i} has five tokens." for i in range(12)])
        return _make_segment(text, filing_id)

    def test_overlap_zero_produces_disjoint_chunks(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        chunker = TextChunker(token_limit=15, tolerance=2, overlap_tokens=0)
        seg = self._build_segment(filing_id)
        chunks = chunker.chunk_segments([seg])

        assert len(chunks) >= 2
        total_tokens = sum(c.token_count for c in chunks)
        # No overlap → total tokens equals the segment's raw token count.
        assert total_tokens == len(seg.content.split())

    def test_overlap_repeats_trailing_sentences(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        chunker = TextChunker(token_limit=15, tolerance=2, overlap_tokens=10)
        seg = self._build_segment(filing_id)
        chunks = chunker.chunk_segments([seg])

        assert len(chunks) >= 2
        # The start of each chunk (except the first) must be a suffix of
        # the previous chunk — i.e. the overlap is real, not just padding.
        for prev, curr in pairwise(chunks):
            first_sentence = curr.content.split(".", 1)[0] + "."
            assert first_sentence in prev.content

    def test_overlap_respects_budget(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        # Sentences are 5 tokens each; overlap budget=8 → at most ONE
        # trailing sentence carries over.
        chunker = TextChunker(token_limit=15, tolerance=2, overlap_tokens=8)
        seg = self._build_segment(filing_id)
        chunks = chunker.chunk_segments([seg])

        for prev, curr in pairwise(chunks):
            carried = curr.content.split(".", 1)[0] + "."
            # Only the last sentence from prev should appear at the head
            # of curr — not the penultimate one.
            prev_sentences = [s.strip() + "." for s in prev.content.split(".") if s.strip()]
            assert carried.strip() == prev_sentences[-1]

    def test_overlap_drops_when_last_sentence_blocks_forward_progress(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        # Strategy B carves out the pathological case where the last
        # sentence is itself ``>= token_limit``: carrying it would
        # immediately re-trigger the band-check on the next chunk,
        # preventing forward progress.  Overlap is dropped instead.
        long_sentence = "word " * 40 + "tail."  # 41 tokens — alone exceeds token_limit
        short_sentences = " ".join([f"Short sentence number {i}." for i in range(10)])
        seg = _make_segment(long_sentence + " " + short_sentences, filing_id)
        chunker = TextChunker(token_limit=30, tolerance=5, overlap_tokens=10)

        chunks = chunker.chunk_segments([seg])

        # Chunker must still make forward progress and respect the token
        # limit band for the short-sentence tail.
        assert len(chunks) >= 2
        for chunk in chunks[1:]:
            assert chunk.token_count <= chunker.token_limit + chunker.tolerance

    def test_overlap_carries_last_sentence_even_when_alone_exceeds_budget(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        # Strategy B rule 1: the last sentence of a finalised chunk is
        # mandatory in the overlap — even when it alone exceeds
        # ``overlap_tokens`` — as long as it fits below ``token_limit``
        # (otherwise rule 1 backs off to preserve forward progress).
        # Build six 8-token sentences so the first chunk fills the band
        # and the trailing sentence (8 tokens) exceeds the 5-token
        # overlap budget without exceeding ``token_limit`` (20).
        sentences = " ".join(["alpha beta gamma delta one two three four." for _ in range(6)])
        seg = _make_segment(sentences, filing_id)
        chunker = TextChunker(token_limit=20, tolerance=2, overlap_tokens=5)

        chunks = chunker.chunk_segments([seg])

        assert len(chunks) >= 2
        # The trailing sentence of the first chunk MUST appear at the
        # start of the second chunk, even though 8 > overlap_tokens=5.
        prev_sentences = [s.strip() + "." for s in chunks[0].content.split(".") if s.strip()]
        assert chunks[1].content.split(".", 1)[0] + "." == prev_sentences[-1]

    def test_overlap_strict_stop_on_prior_sentences(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        # Strategy B rule 2: prior sentences are added under a strict
        # stop-before-overflow rule — no soft margin.  Each sentence is
        # 5 tokens; with overlap_tokens=11 a 5+5=10 carry-over fits but
        # the next +5=15 must not be added.
        text = " ".join([f"Phrase number {i} alpha beta." for i in range(10)])
        seg = _make_segment(text, filing_id)
        chunker = TextChunker(token_limit=20, tolerance=2, overlap_tokens=11)

        chunks = chunker.chunk_segments([seg])

        assert len(chunks) >= 2
        # First chunk has 4 sentences, then finalise; overlap carries
        # exactly the last 2 sentences (10 tokens) into chunk 1.
        for prev, curr in pairwise(chunks):
            prev_sentences = [s.strip() + "." for s in prev.content.split(".") if s.strip()]
            curr_sentences = [s.strip() + "." for s in curr.content.split(".") if s.strip()]
            # The two trailing sentences of prev must be the first two
            # of curr — no third sentence sneaks in via a soft margin.
            assert curr_sentences[:2] == prev_sentences[-2:]


class TestActiveCentringCut:
    """The cut is centred on ``token_limit``, not drifted to the upper edge."""

    def test_cut_inside_band_when_next_sentence_would_overshoot_target(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        # Each sentence is 6 tokens; token_limit=20, tolerance=5
        # → band [15, 25].  After 3 sentences (18 tokens) the running
        # total is inside the band; adding a 4th would push to 24 > 20,
        # so the chunk must finalise at 18 — not drift to 24.
        text = " ".join([f"alpha beta gamma delta epsilon zeta{i}." for i in range(8)])
        seg = _make_segment(text, filing_id)
        chunker = TextChunker(token_limit=20, tolerance=5, overlap_tokens=0)

        chunks = chunker.chunk_segments([seg])

        assert len(chunks) >= 2
        assert chunks[0].token_count == 18
        for chunk in chunks:
            # Every chunk lands inside [target - tolerance, target + tolerance],
            # except possibly the trailing flush.
            assert chunk.token_count <= chunker.token_limit + chunker.tolerance

    def test_below_lower_edge_keeps_accumulating(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        # While running_total is below the lower edge (token_limit -
        # tolerance), the chunker keeps adding sentences without
        # finalising — the active-centring branch only fires once
        # already inside the band.
        text = " ".join([f"alpha beta gamma {i}." for i in range(5)])  # 4 tokens each
        seg = _make_segment(text, filing_id)
        chunker = TextChunker(token_limit=20, tolerance=5, overlap_tokens=0)

        chunks = chunker.chunk_segments([seg])

        # 5 sentences x 4 tokens = 20 tokens; whole text fits inside
        # token_limit so a single chunk is emitted.
        assert len(chunks) == 1
        assert chunks[0].token_count == 20

    def test_hard_upper_bound_still_enforced(
        self, clean_env: pytest.MonkeyPatch, filing_id: FilingIdentifier
    ) -> None:
        # Even before reaching the lower edge of the band, the hard
        # ceiling at token_limit + tolerance forces a cut when the
        # next sentence is unusually large.
        sentences = ["short one two three.", "word " * 30 + "tail.", "short four five six."]
        seg = _make_segment(" ".join(sentences), filing_id)
        chunker = TextChunker(token_limit=20, tolerance=5, overlap_tokens=0)

        chunks = chunker.chunk_segments([seg])

        # The single oversized middle sentence (~31 tokens) must occupy
        # its own chunk — a sentence is never mid-split — and the
        # surrounding chunks must respect the ceiling.
        assert any(c.token_count > chunker.token_limit + chunker.tolerance for c in chunks)
        # The first short sentence is finalised before the long one is
        # appended, because adding the long sentence would overshoot
        # the upper bound.
        assert chunks[0].content.startswith("short one")
        assert "word word" in chunks[1].content


@pytest.mark.security
class TestChunkerLogStreamIsIdentifierFree:
    """The **real** chunker's progress line carries no ticker.

    Runtime companion to the static call-site lock in
    ``tests/core/test_logging.py``: that lock proves the expression is
    *wrapped*, this one proves the wrapper survives a real call. The
    handler carries :class:`AccessionRedactionFilter` because that is
    what ``configure_logging`` installs in production — ``caplog``
    attaches to root and would observe unfiltered records.
    """

    def test_chunking_emits_no_raw_ticker_or_accession(
        self,
        filing_id: FilingIdentifier,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.DEBUG)
        handler.addFilter(AccessionRedactionFilter())
        package_logger = logging.getLogger(LOGGER_NAME)
        prior_level = package_logger.level
        package_logger.addHandler(handler)
        package_logger.setLevel(logging.DEBUG)
        try:
            chunker = TextChunker(token_limit=50, tolerance=10, overlap_tokens=0)
            segment = _make_segment(
                f"Revenue rose in {filing_id.ticker}. " * 20,
                filing_id,
            )
            assert chunker.chunk_segments([segment])
        finally:
            package_logger.removeHandler(handler)
            package_logger.setLevel(prior_level)

        emitted = stream.getvalue()
        assert emitted.strip(), "expected the chunker to log something"
        assert filing_id.ticker not in emitted
        assert filing_id.accession_number not in emitted
