"""Retrieval service for query embedding, vector search, and ranking.

:class:`RetrievalService` is the single-query primitive used by the RAG
orchestrator. It takes a natural language query plus optional metadata
filters and returns a ranked list of
:class:`~sec_generative_search.core.types.RetrievalResult` ready to be
packed into a prompt. Multi-query fan-out belongs in the orchestration
layer.

Composition rules:

- The embedder and the ChromaDB client are passed in pre-built. The
    service never instantiates an embedder itself; that contract is
    enforced project-wide via :mod:`sec_generative_search.providers.factory`.
- The token counter is injectable as a ``Callable[[str], int]``. The
    default lazily loads ``tiktoken``'s ``cl100k_base`` encoding so the
    retrieval layer stays provider-neutral.
- The reranker is optional. When supplied, the service over-fetches,
    reranks the candidate set, attaches the rerank score to each
    :class:`RetrievalResult`, and slices back to ``top_k``. Cosine
    ``similarity`` is preserved on the result so the cosine signal stays
    interpretable in the UI.
- ``sanitize_retrieved_context()`` is not applied here; retrieval stays
    content-neutral and the prompt template owns the trust boundary.

Logging discipline:

- The query string is Tier 3 user-generated data. Every emission goes
    through :func:`~sec_generative_search.core.logging.redact_for_log`
    so a deployment that sets ``LOG_REDACT_QUERIES=1`` never lands the
    raw query in operator logs.
- Filter values are Tier 1 (public filing identifiers) and are logged
    verbatim; redacting a ticker would blind operators to traffic shape
    for no security gain.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
from typing import TYPE_CHECKING

from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import (
    DatabaseError,
    ProviderError,
    SearchError,
)
from sec_generative_search.core.logging import get_logger, redact_for_log
from sec_generative_search.core.metrics import get_metrics
from sec_generative_search.core.types import RetrievalResult, SearchResult

if TYPE_CHECKING:
    from sec_generative_search.database.client import ChromaDBClient
    from sec_generative_search.providers.base import (
        BaseEmbeddingProvider,
        BaseRerankerProvider,
    )

__all__ = ["RetrievalService", "TokenCounter"]


logger = get_logger(__name__)


TokenCounter = Callable[[str], int]
"""Callable signature for the injectable token counter.

Returns the token count for a piece of text under whatever tokeniser the
caller chose.  ``cl100k_base`` is the project default; passing a
provider-native counter is supported for callers that already hold one.
"""


# Lazy module-level cache for the default tiktoken encoder.  ``tiktoken``
# is already a hard dependency (used by every OpenAI-compatible provider
# for exact counts and as the offline approximation for Anthropic and
# Gemini), so importing it costs nothing extra; caching the encoder
# instance keeps per-call overhead at zero.
_default_token_counter: TokenCounter | None = None


def _get_default_token_counter() -> TokenCounter:
    """Return a process-wide cached ``cl100k_base`` token counter."""
    global _default_token_counter
    if _default_token_counter is None:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")

        def _count(text: str) -> int:
            return len(encoder.encode(text))

        _default_token_counter = _count
    return _default_token_counter


def _validate_iso_date(value: str | None, field_name: str) -> None:
    """Reject malformed ISO dates before they reach the storage layer.

    ChromaDB will silently coerce a malformed date string to garbage in
    the integer ``filing_date_int`` filter (e.g. ``"2024-13-99"`` becomes
    ``20241399`` and matches nothing).  Failing fast at the service
    boundary makes the error attributable to the caller, not to a
    "no results" mystery.
    """
    if value is None:
        return
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise SearchError(
            f"Invalid {field_name}: must be YYYY-MM-DD",
            details=f"got {value!r}",
        ) from exc


class RetrievalService:
    """Embedding-based retrieval over the sealed ``sec_filings`` collection.

    Single-query primitive: one call to :meth:`retrieve` does
    ``embed → vector search → dedupe → rerank? → diversity → packing``
    and returns a list of :class:`RetrievalResult`. Multi-query fan-out
    (comparative analysis) lives in the RAG orchestrator.

    Construction:

        >>> from sec_generative_search.providers.factory import build_embedder
        >>> from sec_generative_search.config import get_settings
        >>> from sec_generative_search.database import ChromaDBClient
        >>> settings = get_settings()
        >>> embedder = build_embedder(settings.embedding)
        >>> stamp = ...  # built from the registry
        >>> chroma = ChromaDBClient(stamp)
        >>> svc = RetrievalService(embedder=embedder, chroma_client=chroma)
        >>> hits = svc.retrieve("revenue concentration risk", top_k=5)
    """

    def __init__(
        self,
        embedder: BaseEmbeddingProvider,
        chroma_client: ChromaDBClient,
        *,
        reranker: BaseRerankerProvider | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """Bind pre-built dependencies; never instantiates an embedder.

        Args:
            embedder: Concrete provider built via
                :func:`~sec_generative_search.providers.factory.build_embedder`.
                The embedder's ``provider``/``model``/``dimension`` must
                match the stamp on ``chroma_client`` — that invariant is
                enforced at storage open time, not here.
            chroma_client: Pre-built and stamp-verified.
            reranker: Optional provider-neutral reranker. When ``None``
                the service falls back to cosine-similarity ranking only.
            token_counter: Optional ``Callable[[str], int]``.  Defaults to
                a cached ``tiktoken cl100k_base`` counter — matches the
                offline approximation used elsewhere in the project.
        """
        self._embedder = embedder
        self._chroma = chroma_client
        self._reranker = reranker
        self._token_counter = token_counter or _get_default_token_counter()

        settings = get_settings()
        self._default_top_k = settings.search.top_k
        self._default_min_similarity = settings.search.min_similarity
        self._default_context_budget = settings.rag.context_token_budget
        # Diversity caps and rerank over-fetch are operator defaults
        # resolved here, overridable per-call. ``None`` at the call
        # boundary means "use the configured default".
        self._default_max_per_section = settings.search.max_per_section
        self._default_max_per_filing = settings.search.max_per_filing
        self._default_rerank_over_fetch_factor = settings.search.rerank_over_fetch_factor

        logger.debug(
            "RetrievalService ready: top_k=%d, min_similarity=%.2f, context_budget=%d, reranker=%s",
            self._default_top_k,
            self._default_min_similarity,
            self._default_context_budget,
            type(reranker).__name__ if reranker is not None else "none",
        )

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        ticker: str | list[str] | None = None,
        form_type: str | list[str] | None = None,
        accession_number: str | list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        min_similarity: float | None = None,
        max_per_section: int | None = None,
        max_per_filing: int | None = None,
        rerank_over_fetch_factor: int | None = None,
        context_token_budget: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve up to ``top_k`` chunks ranked for the given query.

        The pipeline is:

            1. Validate query and date filters.
            2. Embed query → vector.
            3. Vector search with metadata filters (``ticker``,
               ``form_type``, ``accession_number``, ``start_date``,
               ``end_date``).  When a reranker is bound the service
               over-fetches by ``rerank_over_fetch_factor`` so the
               reranker has candidates to choose from.
            4. Drop results below ``min_similarity``.
            5. Lift each :class:`SearchResult` to a
               :class:`RetrievalResult` and populate ``token_count`` via
               the injected counter.
                6. Deduplicate by ``chunk_id`` (defensive — Chroma should
                    not emit duplicates, but multi-query fan-out may).
            7. If a reranker is bound, rerank the candidate set,
               attach ``rerank_score``, and re-sort.
            8. Apply per-section / per-filing diversity caps.
            9. Pack to ``context_token_budget`` (drop-tail).
            10. Slice to ``top_k`` and return.

        Args:
            query: Natural language question (Tier 3 — never logged
                without :func:`redact_for_log`).
            top_k: Maximum number of results.  Defaults to
                ``settings.search.top_k``.
            ticker: Single ticker or list (case-insensitive — Chroma
                uppercases internally).
            form_type: SEC form type filter (e.g. ``"10-K"``).
            accession_number: SEC accession number filter.
            start_date: Inclusive lower bound, ``YYYY-MM-DD``.
            end_date: Inclusive upper bound, ``YYYY-MM-DD``.
            min_similarity: Cosine threshold; chunks below are dropped
                before any further processing.  Defaults to
                ``settings.search.min_similarity``.
            max_per_section: Maximum chunks per ``section_path``.  ``0``
                disables the cap.  ``None`` uses
                ``settings.search.max_per_section``.
            max_per_filing: Maximum chunks per ``accession_number``.
                ``0`` disables the cap.  ``None`` uses
                ``settings.search.max_per_filing``.
            rerank_over_fetch_factor: When a reranker is bound, fetch
                ``top_k * factor`` candidates so the reranker has a
                pool to re-order.  Ignored when no reranker is bound.
                ``None`` uses ``settings.search.rerank_over_fetch_factor``.
            context_token_budget: Token budget the returned list must
                fit under.  Defaults to ``settings.rag.context_token_budget``.

        Returns:
            List of :class:`RetrievalResult`, ordered by rerank score
            when a reranker ran (cosine similarity otherwise),
            descending.  Length is at most ``top_k`` and the cumulative
            ``token_count`` is at most ``context_token_budget``.

        Raises:
            SearchError: Empty query, malformed date filter, invalid
                ``top_k``, or an otherwise-untyped downstream failure.
            DatabaseError: Storage-layer fault raised by
                ``ChromaDBClient.query`` — propagated verbatim, never
                re-typed (see :meth:`_fetch_candidates`).
            ProviderError: Embedding-side fault raised by
                ``embed_query`` — likewise propagated verbatim so the
                caller keeps the upstream-outage retry signal.
        """
        if not query or not query.strip():
            raise SearchError(
                "Empty retrieval query",
                details="Cannot retrieve with an empty or whitespace-only query.",
            )

        _validate_iso_date(start_date, "start_date")
        _validate_iso_date(end_date, "end_date")

        effective_top_k = top_k if top_k is not None else self._default_top_k
        if effective_top_k <= 0:
            raise SearchError(
                "Invalid top_k",
                details=f"top_k must be a positive integer; got {effective_top_k}.",
            )

        effective_min_sim = (
            min_similarity if min_similarity is not None else self._default_min_similarity
        )
        effective_budget = (
            context_token_budget
            if context_token_budget is not None
            else self._default_context_budget
        )
        effective_max_per_section = (
            max_per_section if max_per_section is not None else self._default_max_per_section
        )
        effective_max_per_filing = (
            max_per_filing if max_per_filing is not None else self._default_max_per_filing
        )
        effective_over_fetch = (
            rerank_over_fetch_factor
            if rerank_over_fetch_factor is not None
            else self._default_rerank_over_fetch_factor
        )

        fetch_count = effective_top_k
        if self._reranker is not None and effective_over_fetch > 1:
            fetch_count = effective_top_k * effective_over_fetch

        logger.info(
            "Retrieving: query=%r top_k=%d fetch=%d min_sim=%.2f ticker=%s form_type=%s rerank=%s",
            redact_for_log(query[:80]),
            effective_top_k,
            fetch_count,
            effective_min_sim,
            redact_for_log(ticker) if ticker else "any",
            form_type if form_type else "any",
            self._reranker is not None,
        )

        # Time only the real retrieval work (embed → vector search →
        # rank → pack), not the cheap validation above. ``finally``
        # records even when ``_fetch_candidates`` raises, so a backend
        # outage still shows up as latency in the histogram. The label
        # set is empty — retrieval carries no content-free axis worth
        # the cardinality.
        start = time.monotonic()
        try:
            raw = self._fetch_candidates(
                query=query,
                n_results=fetch_count,
                ticker=ticker,
                form_type=form_type,
                accession_number=accession_number,
                start_date=start_date,
                end_date=end_date,
            )

            if effective_min_sim > 0.0:
                raw = [r for r in raw if r.similarity >= effective_min_sim]

            candidates = self._lift_to_retrieval_results(raw)
            candidates = _dedupe_by_chunk_id(candidates)

            if self._reranker is not None and candidates:
                candidates = self._apply_reranker(query, candidates)

            if effective_max_per_section > 0 or effective_max_per_filing > 0:
                candidates = _apply_diversity_caps(
                    candidates,
                    max_per_section=effective_max_per_section,
                    max_per_filing=effective_max_per_filing,
                )

            if effective_budget > 0:
                candidates = _pack_to_budget(candidates, budget=effective_budget)

            final = candidates[:effective_top_k]
        finally:
            get_metrics().observe_retrieval(time.monotonic() - start)

        logger.info("Retrieval returned %d result(s)", len(final))
        return final

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_candidates(
        self,
        *,
        query: str,
        n_results: int,
        ticker: str | list[str] | None,
        form_type: str | list[str] | None,
        accession_number: str | list[str] | None,
        start_date: str | None,
        end_date: str | None,
    ) -> list[SearchResult]:
        """Embed the query and call ``ChromaDBClient.query``.

        Two steps run under one ``try``: the embed call (raises
        :class:`ProviderError` when a hosted embedder is unreachable or
        rejects the key) and the vector query (raises
        :class:`DatabaseError` on a storage fault).  Both are siblings of
        :class:`SearchError` under ``SECGenerativeSearchError`` — *not*
        subclasses — so each needs its own re-raise arm.  Re-typing them
        to ``SearchError`` would collapse a storage outage and an
        upstream outage onto the caller-fault category, sending the
        wrong HTTP status and retry signal and pushing the driver text
        into a route body.  Only genuinely untyped failures are wrapped.
        """
        try:
            vector = self._embedder.embed_query(query)
            # ``embed_query`` returns a 1-D ``np.ndarray``; ChromaDB
            # expects ``list[list[float]]``.
            query_embeddings = [vector.tolist()]
            return self._chroma.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                ticker=ticker,
                form_type=form_type,
                accession_number=accession_number,
                start_date=start_date,
                end_date=end_date,
            )
        except (SearchError, DatabaseError, ProviderError):
            # Already typed and content-safe: let each reach the route
            # arm that owns its status code (400 / 500 / 502).
            raise
        except Exception as exc:
            # Genuinely unexpected — no domain class fits, so the caller
            # sees the uniform retrieval-failure category.
            raise SearchError(
                "Retrieval failed",
                details=str(exc),
            ) from exc

    def _lift_to_retrieval_results(self, hits: list[SearchResult]) -> list[RetrievalResult]:
        """Lift raw search results and pre-compute token counts.

        Token counts are computed once here, not in the packer, so that
        a caller who skips packing (``context_token_budget=0``) still
        receives populated counts — useful for evaluation and for
        downstream consumers that own their own packing strategy.
        """
        out: list[RetrievalResult] = []
        for hit in hits:
            out.append(
                RetrievalResult.from_search_result(
                    hit,
                    token_count=self._token_counter(hit.content),
                )
            )
        return out

    def _apply_reranker(
        self, query: str, candidates: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Run the bound reranker, attach scores, and sort by score.

        The reranker is consulted once with the full candidate document
        list (its own ``top_k`` argument is left unset — the caller
        slices to ``top_k`` later, after diversity and packing).  Any
        candidate the reranker omits keeps its cosine similarity but
        receives ``rerank_score=None``; those candidates fall to the
        bottom of the returned list (deterministic tie-break by
        cosine similarity).
        """
        documents = [c.content for c in candidates]
        rerank_results = self._reranker.rerank(query, documents)  # type: ignore[union-attr]

        scored: dict[int, float] = {r.index: r.score for r in rerank_results}
        for i, candidate in enumerate(candidates):
            candidate.rerank_score = scored.get(i)

        # Sort: rerank_score desc (None last), then cosine similarity desc.
        # Two-key sort on a stable algorithm gives the right ordering for
        # candidates the reranker omitted.
        candidates.sort(
            key=lambda c: (
                -(c.rerank_score if c.rerank_score is not None else float("-inf")),
                -c.similarity,
            )
        )
        return candidates


# ---------------------------------------------------------------------------
# Pure helpers — exposed only via RetrievalService
# ---------------------------------------------------------------------------


def _dedupe_by_chunk_id(results: list[RetrievalResult]) -> list[RetrievalResult]:
    """Drop duplicate chunks while preserving input order.

    Defensive: ChromaDB will not emit the same ``chunk_id`` twice from a
    single query, but multi-query fan-out can merge N candidate lists
    where the same chunk appears repeatedly. Centralising the rule here
    keeps the merge step trivial.

    Results without a ``chunk_id`` (should not happen for ChromaDB-
    sourced results, but the type allows it) are kept verbatim — the
    fallback is conservative because we cannot prove they are
    duplicates.
    """
    seen: set[str] = set()
    out: list[RetrievalResult] = []
    for r in results:
        if r.chunk_id is None:
            out.append(r)
            continue
        if r.chunk_id in seen:
            continue
        seen.add(r.chunk_id)
        out.append(r)
    return out


def _apply_diversity_caps(
    results: list[RetrievalResult],
    *,
    max_per_section: int,
    max_per_filing: int,
) -> list[RetrievalResult]:
    """Cap chunks per section path and per filing.

    Iterates in the input order (which is already similarity- or
    rerank-sorted by the caller) and drops any chunk that would push a
    bucket past its cap.  Stable: the relative order of kept chunks
    matches the input order.

    A cap of ``0`` on a dimension disables that dimension; both ``0``
    means "no caps applied" (and the caller should typically skip this
    helper entirely for that case).

    Section bucket key uses ``path`` rather than the parsed
    ``section_boundaries`` tuple — operators authoring filters in the UI
    will think in terms of the displayed path string, and the two
    representations are 1:1 anyway via :meth:`from_search_result`.
    """
    section_counts: dict[str, int] = {}
    filing_counts: dict[str, int] = {}
    out: list[RetrievalResult] = []
    for r in results:
        if max_per_section > 0:
            sect_key = r.path
            if section_counts.get(sect_key, 0) >= max_per_section:
                continue
        if max_per_filing > 0:
            filing_key = r.accession_number or ""
            if filing_key and filing_counts.get(filing_key, 0) >= max_per_filing:
                continue
        out.append(r)
        if max_per_section > 0:
            section_counts[r.path] = section_counts.get(r.path, 0) + 1
        if max_per_filing > 0 and r.accession_number:
            filing_counts[r.accession_number] = filing_counts.get(r.accession_number, 0) + 1
    return out


def _pack_to_budget(results: list[RetrievalResult], *, budget: int) -> list[RetrievalResult]:
    """Greedy drop-tail packer over the token budget.

    Walks ``results`` in order, accumulating ``token_count``.  As soon
    as adding the next chunk would exceed ``budget``, the walk stops and
    everything from that point forward is dropped.

    Mid-chunk clipping is intentionally NOT supported: SEC filings have
    semantically meaningful chunk boundaries (sentence-aware splits
    around section paths), and clipping mid-sentence would corrupt the
    citation text-span that the orchestrator hands to the answer.  When
    a future strategy *does* clip, it must set
    :attr:`RetrievalResult.truncated` on the affected chunk; that flag
    is reserved for that use and stays untouched here.

    Chunks with ``token_count == 0`` (caller did not populate counts)
    are treated as cost-free and always kept; this is consistent with
    "0 means not yet counted" in the type docstring and avoids dropping
    legitimate content because of an unwired counter.
    """
    if budget <= 0:
        return results
    used = 0
    out: list[RetrievalResult] = []
    for r in results:
        if r.token_count <= 0:
            out.append(r)
            continue
        if used + r.token_count > budget:
            break
        out.append(r)
        used += r.token_count
    return out
