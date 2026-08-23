"""Citation extraction from generated answers — hybrid strategy.

Two extraction paths share this module so the orchestrator can pick
the right one based on the active provider's
:class:`ProviderCapability.structured_output` flag without branching at
the call site.

Path A — :func:`extract_from_json_envelope`
    Used when ``structured_output`` is True. The model is instructed
    to return a JSON object of the form::

        {"answer": "...", "cited_chunk_ids": ["AAPL_10-K_2023-11-03_042", ...]}

    We parse the JSON, validate that every cited chunk_id appears in the
    retrieved set, and build :class:`Citation` records via
    :meth:`RetrievalResult.to_citation`.  Unknown ids are dropped with a
    log warning — the model occasionally fabricates ids despite the
    schema, and silently propagating those would corrupt the audit trail.

Path B — :func:`extract_from_inline_markers`
    Used when ``structured_output`` is False or the JSON parse fails.
    The model is instructed to insert inline ``[N]`` markers where
    ``N`` is the 1-based index of a chunk in the order the user message
    presented them. We regex-extract the markers, deduplicate them
    preserving order, and build :class:`Citation` records.

Both paths share :func:`_build_citations` so the validation behaviour
(drop-unknowns, dedupe, preserve order, assign ``display_index``) is
identical regardless of which extractor ran.

Why hybrid:
    Some providers in this project (Anthropic, OpenRouter routes,
    arbitrary slugs) advertise structured output unevenly. Falling
    back to inline markers keeps the orchestrator working on every
    registered LLM. The hybrid surface is owned by the RAG orchestrator.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from sec_generative_search.core.exceptions import CitationError
from sec_generative_search.core.logging import get_logger, redact_for_log

if TYPE_CHECKING:
    from sec_generative_search.core.types import Citation, RetrievalResult

__all__ = [
    "ExtractedAnswer",
    "extract_citations",
    "extract_from_inline_markers",
    "extract_from_json_envelope",
]


logger = get_logger(__name__)


# Inline ``[N]`` marker pattern — a single integer enclosed in square
# brackets.  Deliberately strict: ``[1, 2]``, ``[12-15]``, and other
# legal-text artefacts that occur in filings are NOT matched.  False
# negatives (a real citation rendered unusually) are preferred to false
# positives (filing language being parsed as a citation).
_INLINE_MARKER_PATTERN = re.compile(r"\[(\d{1,3})\]")


# Maximum number of citations we will attach to a single answer.  Hard
# upper bound to keep the audit trail bounded even if a model emits
# pathological output.  The retrieval layer's diversity caps already
# bound the candidate set well below this.
_MAX_CITATIONS = 50


class ExtractedAnswer:
    """Result of citation extraction — the cleaned answer plus citations.

    Plain class (not a dataclass) so the small surface area stays
    explicit and the citation list is a regular ``list`` callers can
    extend if they post-process.

    Attributes:
        answer: The model's text with any JSON-envelope wrapping
            stripped.  For inline-marker extraction this is the raw
            answer verbatim — markers are preserved so the UI can
            render them as clickable references.
        citations: Citations built from the chunks the model actually
            referenced, in first-mention order.  Each citation's
            ``display_index`` is assigned by mention order (1-based)
            so it lines up with the inline ``[N]`` markers.
    """

    __slots__ = ("answer", "citations")

    def __init__(self, answer: str, citations: list[Citation]) -> None:
        self.answer = answer
        self.citations = citations


def extract_citations(
    answer_text: str,
    retrieved_chunks: list[RetrievalResult],
    *,
    prefer_json: bool,
) -> ExtractedAnswer:
    """Hybrid entry point — pick JSON or inline-marker extraction.

    Args:
        answer_text: Raw model output.
        retrieved_chunks: The chunks fed to the model in the order the
            user-message context block presented them.  The 1-based
            position is the citation index.
        prefer_json: ``True`` when the active provider's
            :class:`ProviderCapability.structured_output` is True.
            Even when ``True``, a JSON parse failure falls through to
            inline-marker extraction so a one-off model glitch does not
            blank out the citation list.

    Returns:
        :class:`ExtractedAnswer` with citations in mention order.
    """
    if prefer_json:
        try:
            return extract_from_json_envelope(answer_text, retrieved_chunks)
        except CitationError as exc:
            logger.info(
                "JSON-envelope citation extraction failed (%s); falling back to inline markers",
                exc,
            )
    return extract_from_inline_markers(answer_text, retrieved_chunks)


def extract_from_json_envelope(
    answer_text: str,
    retrieved_chunks: list[RetrievalResult],
) -> ExtractedAnswer:
    """Parse a ``{"answer": ..., "cited_chunk_ids": [...]}`` envelope.

    The model is expected to emit ONLY the JSON object.  In practice
    some providers wrap it in markdown fences (```json ... ```) or
    surround it with explanatory prose; this function strips a
    leading/trailing fence and finds the first balanced JSON object in
    the text.

    Raises :class:`CitationError` when the envelope cannot be parsed,
    when ``answer`` is missing, or when ``cited_chunk_ids`` is not a
    list of strings.  The orchestrator catches this and falls back to
    inline-marker extraction.
    """
    payload = _isolate_json_object(answer_text)
    if payload is None:
        raise CitationError(
            "No JSON object found in model output",
            details=f"first 80 chars: {answer_text[:80]!r}",
        )

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CitationError(
            "Model output JSON envelope is malformed",
            details=str(exc),
        ) from exc

    if not isinstance(parsed, dict):
        raise CitationError(
            "Model output JSON envelope is not an object",
            details=f"got {type(parsed).__name__}",
        )

    answer = parsed.get("answer")
    if not isinstance(answer, str):
        raise CitationError(
            "Model output JSON envelope is missing 'answer'",
            details=f"keys: {sorted(parsed.keys())}",
        )

    raw_ids = parsed.get("cited_chunk_ids", [])
    if not isinstance(raw_ids, list):
        raise CitationError(
            "Model output JSON envelope 'cited_chunk_ids' is not a list",
            details=f"got {type(raw_ids).__name__}",
        )

    # Build a lookup from chunk_id to retrieved-chunk; any id the model
    # invented is dropped with a single log line.
    by_id = {c.chunk_id: c for c in retrieved_chunks if c.chunk_id}
    matched: list[RetrievalResult] = []
    seen: set[str] = set()
    for raw_id in raw_ids:
        if not isinstance(raw_id, str) or raw_id in seen:
            continue
        chunk = by_id.get(raw_id)
        if chunk is None:
            logger.info(
                "Model emitted unknown citation chunk_id=%r; dropping",
                redact_for_log(raw_id),
            )
            continue
        matched.append(chunk)
        seen.add(raw_id)
        if len(matched) >= _MAX_CITATIONS:
            break

    return ExtractedAnswer(
        answer=answer,
        citations=_build_citations(matched),
    )


def extract_from_inline_markers(
    answer_text: str,
    retrieved_chunks: list[RetrievalResult],
) -> ExtractedAnswer:
    """Regex-extract ``[N]`` markers and validate against the retrieved set.

    Markers referencing an index outside ``[1, len(retrieved_chunks)]``
    are dropped with a log warning. Duplicates are deduplicated
    preserving first-mention order, so ``"X [2]. Y [2]. Z [1]."``
    yields citations for chunk 2 then chunk 1.

    The answer text is returned verbatim (markers preserved) so the UI
    can render them as clickable references that line up with the
    citation list's ``display_index``.
    """
    if not retrieved_chunks:
        return ExtractedAnswer(answer=answer_text, citations=[])

    matched: list[RetrievalResult] = []
    seen: set[int] = set()
    for match in _INLINE_MARKER_PATTERN.finditer(answer_text):
        index = int(match.group(1))
        if index < 1 or index > len(retrieved_chunks):
            logger.info("Model emitted out-of-range inline marker [%d]; dropping", index)
            continue
        if index in seen:
            continue
        seen.add(index)
        matched.append(retrieved_chunks[index - 1])
        if len(matched) >= _MAX_CITATIONS:
            break

    return ExtractedAnswer(
        answer=answer_text,
        citations=_build_citations(matched),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_citations(chunks: list[RetrievalResult]) -> list[Citation]:
    """Turn an ordered list of retrieved chunks into Citation records.

    ``display_index`` is assigned by position in *chunks* (1-based),
    matching the inline-marker convention.
    Citation construction itself is delegated to
    :meth:`RetrievalResult.to_citation` so the validation rules
    (chunk_id / accession_number / filing_date present and parseable)
    live in one place.
    """
    out: list[Citation] = []
    for display_index, chunk in enumerate(chunks, start=1):
        try:
            out.append(chunk.to_citation(display_index=display_index))
        except CitationError as exc:
            # Malformed retrieval result reaching this point indicates
            # an upstream bug, not adversarial input — log and drop.
            # A *missing* ``chunk_id`` is the single most common way to
            # land here (``to_citation`` raises on a falsy id), and it is
            # itself the diagnostic — so report it as missing rather than
            # handing ``None`` to ``redact_for_log``, which would raise
            # out of this handler and break the never-raise contract.
            logger.warning(
                "Skipping citation for chunk_id=%s due to validation error: %s",
                redact_for_log(chunk.chunk_id) if chunk.chunk_id else "<missing>",
                exc,
            )
    return out


def _isolate_json_object(text: str) -> str | None:
    """Find the first balanced JSON object in *text*, or return ``None``.

    Strips markdown code fences first (``"```json\\n...\\n```"`` →
    ``"..."``) because some providers wrap structured-output payloads in
    fences despite the system prompt asking for raw JSON.
    """
    stripped = text.strip()

    # Strip a leading ```json (or just ```) fence.
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        stripped = stripped.strip()

    # Scan for the first ``{`` and walk the brace depth to find its match.
    start = stripped.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(stripped)):
        char = stripped[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : i + 1]
    return None
