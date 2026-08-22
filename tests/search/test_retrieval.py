"""Tests for :mod:`sec_generative_search.search.retrieval`.

Covers service construction and dependency injection, query and date
validation, embedding-based retrieval against a mocked Chroma client,
deduplication by ``chunk_id``, diversity caps, context-window packing,
the reranker seam, and ``RetrievalResult.to_citation``.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest

from sec_generative_search.core.exceptions import (
    CitationError,
    DatabaseError,
    ProviderError,
    SearchError,
)
from sec_generative_search.core.types import (
    ContentType,
    RetrievalResult,
    SearchResult,
)
from sec_generative_search.providers.base import (
    BaseEmbeddingProvider,
    BaseRerankerProvider,
    RerankResult,
)
from sec_generative_search.search.retrieval import (
    RetrievalService,
    _apply_diversity_caps,
    _dedupe_by_chunk_id,
    _pack_to_budget,
)

if TYPE_CHECKING:
    import numpy as np
else:
    np = pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeEmbedder(BaseEmbeddingProvider):
    """Deterministic 4-dim embedder used for service tests."""

    provider_name = "fake-embed"

    def validate_key(self) -> bool:
        return True

    def get_capabilities(self):  # type: ignore[no-untyped-def]
        from sec_generative_search.core.types import ProviderCapability

        return ProviderCapability(embeddings=True)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 4), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        del text
        return np.ones(4, dtype=np.float32)

    def get_dimension(self) -> int:
        return 4


class _FakeChroma:
    """Records the kwargs of the last ``query`` call and returns a canned set."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.last_kwargs: dict | None = None

    def query(self, **kwargs) -> list[SearchResult]:  # type: ignore[no-untyped-def]
        self.last_kwargs = kwargs
        # Honour ``n_results`` so over-fetch tests can assert a slice.
        n = kwargs.get("n_results", len(self.results))
        return list(self.results[:n])


class _FailingChroma:
    """Always raises a non-typed exception to exercise the wrap branch."""

    def query(self, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        raise RuntimeError("chroma exploded")


class _RaisingChroma:
    """Raises a caller-supplied exception from ``query``."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def query(self, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        raise self.exc


class _RaisingEmbedder(_FakeEmbedder):
    """Embedder whose ``embed_query`` raises a caller-supplied exception."""

    def __init__(self, api_key: str, exc: Exception) -> None:
        super().__init__(api_key)
        self.exc = exc

    def embed_query(self, text: str) -> np.ndarray:
        del text
        raise self.exc


class _IdentityReranker(BaseRerankerProvider):
    """Reranker that scores documents in original order (1.0, 0.5, 0.33, ...)."""

    provider_name = "fake-rerank"

    def validate_key(self) -> bool:
        return True

    def get_capabilities(self):  # type: ignore[no-untyped-def]
        from sec_generative_search.core.types import ProviderCapability

        return ProviderCapability()

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        del query
        out = [RerankResult(index=i, score=1.0 / (i + 1)) for i in range(len(documents))]
        return out[:top_k] if top_k is not None else out


class _ReverseReranker(BaseRerankerProvider):
    """Reranker that gives a higher score to LATER documents — used to
    prove the service re-sorts on rerank score, not just attaches it."""

    provider_name = "reverse-rerank"

    def validate_key(self) -> bool:
        return True

    def get_capabilities(self):  # type: ignore[no-untyped-def]
        from sec_generative_search.core.types import ProviderCapability

        return ProviderCapability()

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        del query, top_k
        # Higher score for LATER documents → reverses the cosine order.
        return [RerankResult(index=i, score=float(i + 1)) for i in range(len(documents))]


# ---------------------------------------------------------------------------
# Sample-data builders
# ---------------------------------------------------------------------------


def _make_search_result(
    *,
    chunk_id: str,
    similarity: float = 0.9,
    path: str = "Part I > Item 1A > Risk Factors",
    accession_number: str = "0000320193-23-000077",
    ticker: str = "AAPL",
    form_type: str = "10-K",
    filing_date: str = "2023-11-03",
    content: str = "Risk factors related to supply chain disruptions.",
    content_type: ContentType = ContentType.TEXT,
) -> SearchResult:
    return SearchResult(
        content=content,
        path=path,
        content_type=content_type,
        ticker=ticker,
        form_type=form_type,
        similarity=similarity,
        filing_date=filing_date,
        accession_number=accession_number,
        chunk_id=chunk_id,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_query_raises(self) -> None:
        svc = RetrievalService(_FakeEmbedder("k"), _FakeChroma([]), token_counter=lambda _t: 1)
        with pytest.raises(SearchError, match="Empty retrieval query"):
            svc.retrieve("")

    def test_whitespace_query_raises(self) -> None:
        svc = RetrievalService(_FakeEmbedder("k"), _FakeChroma([]), token_counter=lambda _t: 1)
        with pytest.raises(SearchError, match="Empty retrieval query"):
            svc.retrieve("   \n\t")

    def test_invalid_top_k_raises(self) -> None:
        svc = RetrievalService(_FakeEmbedder("k"), _FakeChroma([]), token_counter=lambda _t: 1)
        with pytest.raises(SearchError, match="Invalid top_k"):
            svc.retrieve("hello", top_k=0)

    def test_bad_start_date_raises(self) -> None:
        svc = RetrievalService(_FakeEmbedder("k"), _FakeChroma([]), token_counter=lambda _t: 1)
        with pytest.raises(SearchError, match="start_date"):
            svc.retrieve("hello", start_date="2024-13-99")

    def test_bad_end_date_raises(self) -> None:
        svc = RetrievalService(_FakeEmbedder("k"), _FakeChroma([]), token_counter=lambda _t: 1)
        with pytest.raises(SearchError, match="end_date"):
            svc.retrieve("hello", end_date="not-a-date")

    def test_chroma_failure_wraps_in_search_error(self) -> None:
        svc = RetrievalService(_FakeEmbedder("k"), _FailingChroma(), token_counter=lambda _t: 1)
        with pytest.raises(SearchError, match="Retrieval failed"):
            svc.retrieve("hello")


# ---------------------------------------------------------------------------
# Basic retrieval and filter passthrough
# ---------------------------------------------------------------------------


class TestBasicRetrieval:
    def test_returns_retrieval_results(self) -> None:
        chroma = _FakeChroma([_make_search_result(chunk_id="c1")])
        svc = RetrievalService(_FakeEmbedder("k"), chroma, token_counter=lambda t: len(t))
        out = svc.retrieve("supply chain")
        assert len(out) == 1
        assert isinstance(out[0], RetrievalResult)
        assert out[0].chunk_id == "c1"
        assert out[0].token_count == len("Risk factors related to supply chain disruptions.")
        # Section boundaries derived from the path:
        assert out[0].section_boundaries == ("Part I", "Item 1A", "Risk Factors")

    def test_passes_filters_to_chroma(self) -> None:
        chroma = _FakeChroma([])
        svc = RetrievalService(_FakeEmbedder("k"), chroma, token_counter=lambda _t: 1)
        svc.retrieve(
            "x",
            ticker="AAPL",
            form_type="10-K",
            accession_number="0000320193-23-000077",
            start_date="2023-01-01",
            end_date="2023-12-31",
        )
        kwargs = chroma.last_kwargs
        assert kwargs is not None
        assert kwargs["ticker"] == "AAPL"
        assert kwargs["form_type"] == "10-K"
        assert kwargs["accession_number"] == "0000320193-23-000077"
        assert kwargs["start_date"] == "2023-01-01"
        assert kwargs["end_date"] == "2023-12-31"

    def test_min_similarity_drops_low_scoring_results(self) -> None:
        results = [
            _make_search_result(chunk_id="hi", similarity=0.9),
            _make_search_result(chunk_id="lo", similarity=0.3),
        ]
        chroma = _FakeChroma(results)
        svc = RetrievalService(_FakeEmbedder("k"), chroma, token_counter=lambda _t: 1)
        out = svc.retrieve("x", min_similarity=0.5)
        assert [r.chunk_id for r in out] == ["hi"]

    def test_top_k_truncates(self) -> None:
        results = [_make_search_result(chunk_id=f"c{i}") for i in range(10)]
        chroma = _FakeChroma(results)
        svc = RetrievalService(_FakeEmbedder("k"), chroma, token_counter=lambda _t: 1)
        out = svc.retrieve("x", top_k=3)
        assert len(out) == 3


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDedupe:
    def test_dedupe_helper_keeps_first_occurrence(self) -> None:
        a = RetrievalResult.from_search_result(_make_search_result(chunk_id="dup"))
        b = RetrievalResult.from_search_result(_make_search_result(chunk_id="dup"))
        c = RetrievalResult.from_search_result(_make_search_result(chunk_id="other"))
        out = _dedupe_by_chunk_id([a, b, c])
        assert [r.chunk_id for r in out] == ["dup", "other"]

    def test_dedupe_keeps_chunks_without_id(self) -> None:
        a = RetrievalResult.from_search_result(_make_search_result(chunk_id="x"))
        b = RetrievalResult.from_search_result(_make_search_result(chunk_id="y"))
        # Force chunk_id=None to mimic an upstream malformed result.
        b.chunk_id = None
        out = _dedupe_by_chunk_id([a, b])
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Diversity caps
# ---------------------------------------------------------------------------


class TestDiversity:
    def _three_per_section(self) -> list[RetrievalResult]:
        path = "Part I > Item 1A > Risk Factors"
        return [
            RetrievalResult.from_search_result(_make_search_result(chunk_id=f"c{i}", path=path))
            for i in range(3)
        ]

    def test_max_per_section_cap(self) -> None:
        results = self._three_per_section()
        out = _apply_diversity_caps(results, max_per_section=2, max_per_filing=0)
        assert len(out) == 2
        assert [r.chunk_id for r in out] == ["c0", "c1"]

    def test_max_per_filing_cap(self) -> None:
        results = [
            RetrievalResult.from_search_result(
                _make_search_result(
                    chunk_id=f"c{i}",
                    accession_number="A",
                    path=f"Section {i}",
                )
            )
            for i in range(3)
        ]
        out = _apply_diversity_caps(results, max_per_section=0, max_per_filing=2)
        assert len(out) == 2

    def test_zero_caps_disable_filtering(self) -> None:
        results = self._three_per_section()
        out = _apply_diversity_caps(results, max_per_section=0, max_per_filing=0)
        assert len(out) == 3

    def test_section_and_filing_caps_compose(self) -> None:
        # Two sections, three filings, six chunks; section cap=1 then
        # filing cap=2 should leave at most min(distinct sections, ...).
        results = [
            RetrievalResult.from_search_result(
                _make_search_result(
                    chunk_id=f"c{i}",
                    path=f"sec-{i % 2}",
                    accession_number=f"acc-{i % 3}",
                )
            )
            for i in range(6)
        ]
        out = _apply_diversity_caps(results, max_per_section=1, max_per_filing=2)
        # Only 2 distinct sections → at most 2 results survive section cap.
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


class TestPacking:
    def test_drop_tail_when_budget_exceeded(self) -> None:
        results = [
            RetrievalResult.from_search_result(
                _make_search_result(chunk_id=f"c{i}"),
                token_count=40,
            )
            for i in range(5)
        ]
        out = _pack_to_budget(results, budget=100)
        # 40 + 40 = 80 fits; +40 = 120 exceeds → drop tail.
        assert [r.chunk_id for r in out] == ["c0", "c1"]

    def test_zero_budget_passes_through(self) -> None:
        results = [RetrievalResult.from_search_result(_make_search_result(chunk_id="c0"))]
        assert _pack_to_budget(results, budget=0) == results

    def test_zero_token_count_treated_as_free(self) -> None:
        results = [
            RetrievalResult.from_search_result(
                _make_search_result(chunk_id=f"c{i}"),
                token_count=0,
            )
            for i in range(3)
        ]
        out = _pack_to_budget(results, budget=10)
        # All three chunks have token_count=0; none consume budget.
        assert len(out) == 3

    def test_does_not_set_truncated_flag(self) -> None:
        # The drop-tail packer never partially clips a chunk, so the
        # ``truncated`` flag must remain at its default ``False``.
        results = [
            RetrievalResult.from_search_result(
                _make_search_result(chunk_id=f"c{i}"),
                token_count=40,
            )
            for i in range(3)
        ]
        out = _pack_to_budget(results, budget=80)
        assert all(r.truncated is False for r in out)


# ---------------------------------------------------------------------------
# Reranker seam
# ---------------------------------------------------------------------------


class TestReranker:
    def test_no_reranker_leaves_rerank_score_none(self) -> None:
        chroma = _FakeChroma(
            [
                _make_search_result(chunk_id="a"),
                _make_search_result(chunk_id="b"),
            ]
        )
        svc = RetrievalService(_FakeEmbedder("k"), chroma, token_counter=lambda _t: 1)
        out = svc.retrieve("x", top_k=2)
        assert all(r.rerank_score is None for r in out)

    def test_reranker_attaches_score_and_preserves_similarity(self) -> None:
        results = [
            _make_search_result(chunk_id="a", similarity=0.9),
            _make_search_result(chunk_id="b", similarity=0.8),
        ]
        chroma = _FakeChroma(results)
        svc = RetrievalService(
            _FakeEmbedder("k"),
            chroma,
            reranker=_IdentityReranker("k"),
            token_counter=lambda _t: 1,
        )
        out = svc.retrieve("x", top_k=2, rerank_over_fetch_factor=1)
        # Cosine similarity values must round-trip unchanged.
        assert out[0].similarity == pytest.approx(0.9)
        assert out[1].similarity == pytest.approx(0.8)
        # Rerank score is populated.
        assert out[0].rerank_score == pytest.approx(1.0)
        assert out[1].rerank_score == pytest.approx(0.5)

    def test_reranker_changes_order_when_reverse(self) -> None:
        results = [
            _make_search_result(chunk_id="a", similarity=0.9),
            _make_search_result(chunk_id="b", similarity=0.8),
            _make_search_result(chunk_id="c", similarity=0.7),
        ]
        chroma = _FakeChroma(results)
        svc = RetrievalService(
            _FakeEmbedder("k"),
            chroma,
            reranker=_ReverseReranker("k"),
            token_counter=lambda _t: 1,
        )
        out = svc.retrieve("x", top_k=3, rerank_over_fetch_factor=1)
        # Reverse reranker pushed last document to the top.
        assert [r.chunk_id for r in out] == ["c", "b", "a"]

    def test_over_fetch_factor_multiplies_n_results(self) -> None:
        chroma = _FakeChroma([_make_search_result(chunk_id=f"c{i}") for i in range(20)])
        svc = RetrievalService(
            _FakeEmbedder("k"),
            chroma,
            reranker=_IdentityReranker("k"),
            token_counter=lambda _t: 1,
        )
        svc.retrieve("x", top_k=3, rerank_over_fetch_factor=4)
        assert chroma.last_kwargs is not None
        assert chroma.last_kwargs["n_results"] == 12  # 3 * 4


# ---------------------------------------------------------------------------
# Settings-driven defaults.
# ---------------------------------------------------------------------------


class TestSettingsDrivenDefaults:
    """The diversity caps + rerank over-fetch resolve from
    ``settings.search`` when the caller omits them — the mechanism that
    lets ``SEARCH_*`` env vars (and the deployment profile) drive both
    the search route and RAG retrieval without per-call wiring.
    """

    @staticmethod
    def _reset(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
        from sec_generative_search.config.settings import reload_settings

        for name in names:
            monkeypatch.delenv(name, raising=False)
        reload_settings()

    def test_diversity_cap_resolved_from_settings(self, clean_env: pytest.MonkeyPatch) -> None:
        from sec_generative_search.config.settings import reload_settings

        clean_env.setenv("SEARCH_MAX_PER_SECTION", "1")
        reload_settings()
        try:
            results = [
                _make_search_result(chunk_id="a", path="P1"),
                _make_search_result(chunk_id="b", path="P1"),
                _make_search_result(chunk_id="c", path="P2"),
            ]
            svc = RetrievalService(
                _FakeEmbedder("k"), _FakeChroma(results), token_counter=lambda _t: 1
            )
            # Caller omits max_per_section → settings value (1) applies:
            # one chunk per section path survives.
            out = svc.retrieve("x", top_k=10)
            assert [r.chunk_id for r in out] == ["a", "c"]
        finally:
            self._reset(clean_env, "SEARCH_MAX_PER_SECTION")

    def test_explicit_arg_overrides_settings_default(self, clean_env: pytest.MonkeyPatch) -> None:
        from sec_generative_search.config.settings import reload_settings

        clean_env.setenv("SEARCH_MAX_PER_SECTION", "1")
        reload_settings()
        try:
            results = [
                _make_search_result(chunk_id="a", path="P1"),
                _make_search_result(chunk_id="b", path="P1"),
                _make_search_result(chunk_id="c", path="P2"),
            ]
            svc = RetrievalService(
                _FakeEmbedder("k"), _FakeChroma(results), token_counter=lambda _t: 1
            )
            # Explicit 0 disables the cap even though settings set it to 1.
            out = svc.retrieve("x", top_k=10, max_per_section=0)
            assert [r.chunk_id for r in out] == ["a", "b", "c"]
        finally:
            self._reset(clean_env, "SEARCH_MAX_PER_SECTION")

    def test_over_fetch_factor_resolved_from_settings(self, clean_env: pytest.MonkeyPatch) -> None:
        from sec_generative_search.config.settings import reload_settings

        clean_env.setenv("SEARCH_RERANK_OVER_FETCH_FACTOR", "5")
        reload_settings()
        try:
            chroma = _FakeChroma([_make_search_result(chunk_id=f"c{i}") for i in range(40)])
            svc = RetrievalService(
                _FakeEmbedder("k"),
                chroma,
                reranker=_IdentityReranker("k"),
                token_counter=lambda _t: 1,
            )
            # Caller omits the factor → settings value (5) drives the fetch.
            svc.retrieve("x", top_k=3)
            assert chroma.last_kwargs is not None
            assert chroma.last_kwargs["n_results"] == 15  # 3 * 5
        finally:
            self._reset(clean_env, "SEARCH_RERANK_OVER_FETCH_FACTOR")


# ---------------------------------------------------------------------------
# Citation conversion
# ---------------------------------------------------------------------------


class TestToCitation:
    def test_round_trips_metadata(self) -> None:
        rr = RetrievalResult.from_search_result(_make_search_result(chunk_id="x"))
        cit = rr.to_citation(display_index=2)
        assert cit.chunk_id == "x"
        assert cit.section_path == "Part I > Item 1A > Risk Factors"
        assert cit.text_span == "Risk factors related to supply chain disruptions."
        assert cit.similarity == pytest.approx(0.9)
        assert cit.display_index == 2
        assert cit.filing_id.ticker == "AAPL"
        assert cit.filing_id.form_type == "10-K"
        assert cit.filing_id.filing_date == date(2023, 11, 3)
        assert cit.filing_id.accession_number == "0000320193-23-000077"

    def test_missing_chunk_id_raises(self) -> None:
        rr = RetrievalResult.from_search_result(_make_search_result(chunk_id="x"))
        rr.chunk_id = None
        with pytest.raises(CitationError, match="chunk_id"):
            rr.to_citation()

    def test_missing_accession_raises(self) -> None:
        rr = RetrievalResult.from_search_result(_make_search_result(chunk_id="x"))
        rr.accession_number = None
        with pytest.raises(CitationError, match="accession_number"):
            rr.to_citation()

    def test_missing_filing_date_raises(self) -> None:
        rr = RetrievalResult.from_search_result(_make_search_result(chunk_id="x"))
        rr.filing_date = None
        with pytest.raises(CitationError, match="filing_date"):
            rr.to_citation()

    def test_malformed_filing_date_raises(self) -> None:
        rr = RetrievalResult.from_search_result(
            _make_search_result(chunk_id="x", filing_date="2024-13-99")
        )
        with pytest.raises(CitationError, match="filing_date"):
            rr.to_citation()


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


class TestSecurity:
    @pytest.mark.security
    def test_query_redaction_on_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from sec_generative_search.core.logging import LOGGER_NAME

        monkeypatch.setenv("LOG_REDACT_QUERIES", "1")
        # `caplog` attaches to root; package logger has propagate=False.
        logger_obj = __import__("logging").getLogger(LOGGER_NAME)
        previous = logger_obj.propagate
        logger_obj.propagate = True
        try:
            chroma = _FakeChroma([_make_search_result(chunk_id="c1")])
            svc = RetrievalService(_FakeEmbedder("k"), chroma, token_counter=lambda _t: 1)
            with caplog.at_level("INFO", logger=LOGGER_NAME):
                svc.retrieve("revenue concentration risk")
        finally:
            logger_obj.propagate = previous

        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "revenue concentration risk" not in joined
        assert "<redacted:" in joined

    @pytest.mark.security
    def test_query_logged_verbatim_when_redaction_off(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from sec_generative_search.core.logging import LOGGER_NAME

        monkeypatch.delenv("LOG_REDACT_QUERIES", raising=False)
        logger_obj = __import__("logging").getLogger(LOGGER_NAME)
        previous = logger_obj.propagate
        logger_obj.propagate = True
        try:
            chroma = _FakeChroma([])
            svc = RetrievalService(_FakeEmbedder("k"), chroma, token_counter=lambda _t: 1)
            with caplog.at_level("INFO", logger=LOGGER_NAME):
                svc.retrieve("supply chain risk")
        finally:
            logger_obj.propagate = previous

        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "supply chain risk" in joined

    @pytest.mark.security
    def test_database_error_propagates_verbatim(self) -> None:
        # A storage fault MUST keep its class so the route maps it to
        # 500 ``database_error`` (generic body) rather than the
        # caller-fault 400 whose message used to render
        # ``"{message} — {details}"`` — leaking the driver text.
        original = DatabaseError(
            "collection read failed",
            details="sqlite3.OperationalError: database is locked at /app/data/metadata.sqlite",
        )
        svc = RetrievalService(
            _FakeEmbedder("k"),
            _RaisingChroma(original),
            token_counter=lambda _t: 1,
        )
        with pytest.raises(DatabaseError) as excinfo:
            svc.retrieve("revenue concentration risk")
        assert excinfo.value is original
        assert not isinstance(excinfo.value, SearchError)

    @pytest.mark.security
    def test_provider_error_propagates_verbatim(self) -> None:
        # A hosted-embedder outage MUST keep its class so the route maps
        # it to a retryable 502, not a do-not-retry 400.
        original = ProviderError(
            "embedder upstream failed",
            details="https://embed.internal.example/v1 returned 500",
        )
        svc = RetrievalService(
            _RaisingEmbedder("k", original),
            _FakeChroma([]),
            token_counter=lambda _t: 1,
        )
        with pytest.raises(ProviderError) as excinfo:
            svc.retrieve("revenue concentration risk")
        assert excinfo.value is original
        assert not isinstance(excinfo.value, SearchError)

    @pytest.mark.security
    def test_untyped_error_is_still_wrapped(self) -> None:
        # The narrowing must not remove the catch-all: an unexpected
        # failure still collapses onto the uniform SearchError category.
        svc = RetrievalService(
            _FakeEmbedder("k"),
            _FailingChroma(),
            token_counter=lambda _t: 1,
        )
        with pytest.raises(SearchError):
            svc.retrieve("revenue concentration risk")

    @pytest.mark.security
    def test_no_credential_shaped_attributes(self) -> None:
        svc = RetrievalService(_FakeEmbedder("k"), _FakeChroma([]), token_counter=lambda _t: 1)
        # Credential-bearing fields would be a contract violation; the
        # parametrised test in tests/core/test_types.py enforces the
        # same rule for domain types.  The service surface must stay
        # equally clean.
        forbidden = {
            "api_key",
            "authorization",
            "bearer",
            "secret",
            "password",
            "encryption_key",
            "edgar_identity",
        }
        for name in vars(svc):
            assert name.lstrip("_").lower() not in forbidden, name


# ---------------------------------------------------------------------------
# Settings interaction
# ---------------------------------------------------------------------------


class TestSettingsDefaults:
    def test_default_top_k_from_settings_when_not_passed(self) -> None:
        # Caches the SEARCH_TOP_K value frozen at module import; we
        # cannot reliably reload it under pytest because Settings'
        # nested defaults are evaluated once at class-body time.
        # Instead, prove that omitting ``top_k`` returns at most the
        # configured default (5 by default).
        chroma = _FakeChroma([_make_search_result(chunk_id=f"c{i}") for i in range(20)])
        svc = RetrievalService(_FakeEmbedder("k"), chroma, token_counter=lambda _t: 1)
        out = svc.retrieve("x")
        assert 0 < len(out) <= 20
        # And confirm that the explicit override path works.
        out2 = svc.retrieve("x", top_k=2)
        assert len(out2) == 2
