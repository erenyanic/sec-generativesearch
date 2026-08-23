"""
Text chunking for SEC filings.

This module splits long segments into smaller chunks suitable for embedding.
It uses sentence-boundary splitting to ensure chunks don't cut mid-sentence.

Section boundaries are preserved implicitly: the parser produces one
:class:`Segment` per section path (e.g. ``"Part I > Item 1A > Risk
Factors"``), and chunking is performed **per-segment** — chunks never
span across sections.

Usage:
    from sec_generative_search.pipeline import TextChunker

    chunker = TextChunker()
    chunks = chunker.chunk_segments(segments)
"""

import re

from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import ChunkingError
from sec_generative_search.core.logging import get_logger, redact_for_log
from sec_generative_search.core.types import Chunk, ContentType, Segment

logger = get_logger(__name__)


class TextChunker:
    """
    Splits segments into embedding-ready chunks.

    This class implements sentence-boundary aware chunking to ensure
    that text is split at natural boundaries rather than mid-sentence.

    The chunking algorithm:
        1. If segment fits within ``token_limit``, keep as-is.
        2. Otherwise, split on sentence boundaries (``. ! ?``).
        3. Accumulate sentences and finalise the chunk at the sentence
           boundary nearest to ``token_limit`` inside the bidirectional
           ``± tolerance`` band — i.e. once the running total is at or
           above ``token_limit - tolerance``, the next sentence whose
           inclusion would push past ``token_limit`` is the cut point.
        4. The hard upper bound is ``token_limit + tolerance``: a chunk
           never overshoots it (the only exception is a single-sentence
           chunk that is itself larger than the upper bound — sentences
           are never mid-split).
        5. Optional sentence-level overlap carries the tail of the
           previous chunk into the next one for context continuity.

    Table segments (``ContentType.TABLE``) are never split — table
    structure is preserved as a single chunk even when the token count
    exceeds ``token_limit``.  Sentence-boundary splitting would
    otherwise corrupt row alignment.

    Attributes:
        token_limit: Target tokens per chunk (from settings).
        tolerance: Bidirectional ``±`` band around ``token_limit``
            inside which the chunker may finalise a chunk on a sentence
            boundary (from settings).
        overlap_tokens: Target number of tokens of trailing context
            carried from one chunk into the next.  Zero disables
            overlap (from :class:`RAGSettings.chunk_overlap_tokens`).

    Example:
        >>> chunker = TextChunker()
        >>> chunks = chunker.chunk_segments(segments)
        >>> print(f"Created {len(chunks)} chunks")
    """

    # Sentence boundary pattern: split after . ! ? followed by whitespace
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        token_limit: int | None = None,
        tolerance: int | None = None,
        overlap_tokens: int | None = None,
    ) -> None:
        """
        Initialise the chunker with configurable limits.

        Args:
            token_limit: Max tokens per chunk. If None, uses settings.
            tolerance: Acceptable overrun. If None, uses settings.
            overlap_tokens: Sentence-level overlap tokens between
                adjacent chunks.  If ``None`` uses
                :attr:`RAGSettings.chunk_overlap_tokens`.  A negative
                value is rejected; zero disables overlap.
        """
        settings = get_settings()
        self.token_limit = token_limit or settings.chunking.token_limit
        self.tolerance = tolerance or settings.chunking.tolerance
        if overlap_tokens is None:
            overlap_tokens = settings.rag.chunk_overlap_tokens
        if overlap_tokens < 0:
            raise ValueError(f"overlap_tokens must be non-negative (got {overlap_tokens})")
        if overlap_tokens >= self.token_limit:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be strictly less than "
                f"token_limit ({self.token_limit}) to guarantee forward progress."
            )
        self.overlap_tokens = overlap_tokens

        logger.debug(
            "TextChunker initialised: limit=%d, tolerance=%d, overlap=%d",
            self.token_limit,
            self.tolerance,
            self.overlap_tokens,
        )

    def _count_tokens(self, text: str) -> int:
        """
        Approximate token count using whitespace splitting.

        This is a simple heuristic that works well for English text.
        More accurate tokenisation would require the actual model's
        tokeniser, but whitespace splitting is sufficient for chunking.

        Args:
            text: Text to count tokens in.

        Returns:
            Approximate token count.
        """
        return len(text.split())

    def _chunk_text(self, text: str) -> list[tuple[str, int]]:
        """
        Split text into chunks respecting sentence boundaries.

        Active-centring cut: once the running total reaches the lower
        edge of the ``±tolerance`` band (``token_limit - tolerance``),
        the chunk is finalised at the next sentence boundary whose
        inclusion would push past ``token_limit``.  This keeps chunks
        clustered around the target size rather than skewed toward the
        upper edge of the band.  The upper bound
        ``token_limit + tolerance`` remains a hard cap and forces a cut
        even before the lower edge is reached when the next sentence is
        unusually large.

        When ``overlap_tokens > 0``, the trailing sentences of a
        finalised chunk (per :meth:`_tail_for_overlap`) are retained as
        the seed of the next chunk so referent words like ``"the
        Company"`` / ``"such filing"`` survive embedding boundaries.

        Args:
            text: Text content to split.

        Returns:
            List of (chunk_text, token_count) tuples.
        """
        total_tokens = self._count_tokens(text)

        # If text already fits, return as single chunk
        if total_tokens <= self.token_limit:
            return [(text, total_tokens)]

        # Split on sentence boundaries
        sentences = self.SENTENCE_PATTERN.split(text)
        sentence_token_counts = [self._count_tokens(s) for s in sentences]

        chunks: list[tuple[str, int]] = []
        current_sentences: list[str] = []
        current_token_counts: list[int] = []
        current_tokens = 0

        for sentence, sentence_tokens in zip(sentences, sentence_token_counts, strict=True):
            next_total = current_tokens + sentence_tokens
            should_finalise = bool(current_sentences) and (
                # Hard upper bound — must cut, otherwise the chunk overshoots.
                next_total > self.token_limit + self.tolerance
                # Active-centring: we are already inside the ± band and
                # adding this sentence would push past the target.  Cut
                # at the current boundary so the chunk lands near
                # ``token_limit`` rather than drifting toward the upper edge.
                or (
                    current_tokens >= self.token_limit - self.tolerance
                    and next_total > self.token_limit
                )
            )
            if should_finalise:
                chunks.append((" ".join(current_sentences), current_tokens))
                if self.overlap_tokens > 0:
                    current_sentences, current_token_counts = self._tail_for_overlap(
                        current_sentences, current_token_counts
                    )
                    current_tokens = sum(current_token_counts)
                else:
                    current_sentences = []
                    current_token_counts = []
                    current_tokens = 0

            current_sentences.append(sentence)
            current_token_counts.append(sentence_tokens)
            current_tokens += sentence_tokens

        # Flush remaining sentences
        if current_sentences:
            chunks.append((" ".join(current_sentences), current_tokens))

        return chunks

    def _tail_for_overlap(
        self,
        sentences: list[str],
        token_counts: list[int],
    ) -> tuple[list[str], list[int]]:
        """Return the trailing sentences carried into the next chunk.

        Strategy B (soft upper bound, strict stop):

        1. **Always carry the last whole sentence**, even when it alone
           exceeds :attr:`overlap_tokens`.  Dropping the only sentence
           that survived a finalisation creates a silent context gap at
           exactly the wrong place — long sentences are usually the
           most information-dense in SEC filings.  The bloat is bounded
           by the chunker's own ``token_limit + tolerance`` band, so no
           other invariant breaks.

           The one absolute floor is forward progress: if the last
           sentence is itself ``≥ token_limit`` the next chunk would
           never accept further content (the band-check in
           :meth:`_chunk_text` would immediately re-finalise).  In that
           pathological case overlap is dropped entirely.

        2. **Otherwise**, walk backward over the preceding sentences and
           keep adding while ``running_total + next_count ≤
           overlap_tokens``.  Stop *before* the first overflow — there
           is no soft margin on prior sentences.
        """
        if not sentences:
            return [], []

        # Rule 1: the trailing sentence is mandatory unless it would
        # itself prevent forward progress on the next chunk.
        last_count = token_counts[-1]
        if last_count >= self.token_limit:
            return [], []

        tail: list[str] = [sentences[-1]]
        tail_counts: list[int] = [last_count]
        running = last_count

        if running >= self.overlap_tokens:
            # Last sentence already meets or exceeds the budget; do not
            # walk further backwards — would only bloat the seed.
            return tail, tail_counts

        # Rule 2: strict stop-before-overflow on prior sentences.
        for sentence, count in zip(
            reversed(sentences[:-1]),
            reversed(token_counts[:-1]),
            strict=True,
        ):
            if running + count > self.overlap_tokens:
                break
            tail.append(sentence)
            tail_counts.append(count)
            running += count
        tail.reverse()
        tail_counts.reverse()
        return tail, tail_counts

    def chunk_segment(self, segment: Segment, start_index: int = 0) -> list[Chunk]:
        """
        Split a single segment into chunks.

        Table segments are never split — their structure would be
        corrupted by sentence-boundary splitting, so they are emitted
        as a single chunk even when the token count exceeds the limit.
        Text segments are split per :meth:`_chunk_text`.

        Args:
            segment: Segment to chunk.
            start_index: Starting chunk index for this segment.

        Returns:
            List of Chunk objects with sequential indices.
        """
        if segment.content_type is ContentType.TABLE:
            token_count = self._count_tokens(segment.content)
            text_chunks: list[tuple[str, int]] = [(segment.content, token_count)]
        else:
            text_chunks = self._chunk_text(segment.content)

        return [
            Chunk(
                content=text,
                path=segment.path,
                content_type=segment.content_type,
                filing_id=segment.filing_id,
                chunk_index=start_index + i,
                token_count=tokens,
            )
            for i, (text, tokens) in enumerate(text_chunks)
        ]

    def chunk_segments(self, segments: list[Segment]) -> list[Chunk]:
        """
        Chunk all segments from a filing.

        This is the main entry point for chunking. It processes all
        segments and assigns sequential chunk indices across the
        entire filing.

        Args:
            segments: List of segments from FilingParser.

        Returns:
            List of Chunk objects ready for embedding.

        Raises:
            ChunkingError: If segments list is empty.

        Example:
            >>> chunks = chunker.chunk_segments(segments)
            >>> for chunk in chunks[:3]:
            ...     print(f"[{chunk.chunk_index}] {chunk.path[:50]}...")
        """
        if not segments:
            raise ChunkingError(
                "No segments to chunk",
                details="Received empty segments list.",
            )

        filing_id = segments[0].filing_id

        logger.info(
            "Chunking %d segments from %s %s",
            len(segments),
            redact_for_log(filing_id.ticker),
            filing_id.form_type,
        )

        chunks: list[Chunk] = []
        current_index = 0

        for segment in segments:
            segment_chunks = self.chunk_segment(segment, start_index=current_index)
            chunks.extend(segment_chunks)
            current_index += len(segment_chunks)

        # Log statistics — token counts are retained from chunking, no recount
        token_counts = [c.token_count for c in chunks]
        min_tokens = min(token_counts)
        max_tokens = max(token_counts)
        avg_tokens = sum(token_counts) / len(token_counts)
        over_limit = sum(1 for t in token_counts if t > self.token_limit)

        logger.info(
            "Created %d chunks from %d segments (tokens: %d-%d, avg %.0f, %d over limit)",
            len(chunks),
            len(segments),
            min_tokens,
            max_tokens,
            avg_tokens,
            over_limit,
        )

        return chunks
