"""Tests for core domain types.

Covers the carried-over ingestion types (:class:`FilingIdentifier`,
:class:`Chunk`, etc.) and the RAG-specific additions
(:class:`RetrievalResult`, :class:`Citation`, :class:`TokenUsage`,
:class:`GenerationResult`, :class:`ConversationTurn`,
:class:`ProviderCapability`).

Security tests live alongside functional ones and are marked with
``@pytest.mark.security`` so they can be selected independently.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from sec_generative_search.core.types import (
    BackupReport,
    CatalogueRefreshReport,
    Chunk,
    Citation,
    ContentType,
    ConversationTurn,
    DeploymentProfile,
    EmbedderStamp,
    EvictionReport,
    ExportReport,
    FilingIdentifier,
    GenerationResult,
    ImportReport,
    IngestResult,
    PricingTier,
    ProviderCapability,
    ReindexReport,
    RestoreReport,
    RetrievalResult,
    SearchResult,
    Segment,
    TokenUsage,
    estimate_cost,
)


@pytest.fixture
def filing_id() -> FilingIdentifier:
    return FilingIdentifier(
        ticker="aapl",
        form_type="10-k",
        filing_date=date(2023, 11, 3),
        accession_number="0000320193-23-000077",
    )


class TestFilingIdentifier:
    def test_auto_uppercases_ticker_and_form(self, filing_id: FilingIdentifier) -> None:
        assert filing_id.ticker == "AAPL"
        assert filing_id.form_type == "10-K"

    def test_frozen(self, filing_id: FilingIdentifier) -> None:
        with pytest.raises(FrozenInstanceError):
            filing_id.ticker = "MSFT"  # type: ignore[misc]

    def test_date_str_iso_format(self, filing_id: FilingIdentifier) -> None:
        assert filing_id.date_str == "2023-11-03"

    def test_equality_matches_field_tuple(self) -> None:
        a = FilingIdentifier("AAPL", "10-K", date(2023, 1, 1), "acc-1")
        b = FilingIdentifier("aapl", "10-k", date(2023, 1, 1), "acc-1")
        assert a == b
        assert hash(a) == hash(b)


class TestChunk:
    def test_chunk_id_format(self, filing_id: FilingIdentifier) -> None:
        chunk = Chunk(
            content="Our business is subject to...",
            path="Part I > Item 1A > Risk Factors",
            content_type=ContentType.TEXT,
            filing_id=filing_id,
            chunk_index=42,
        )
        assert chunk.chunk_id == "AAPL_10-K_2023-11-03_042"

    def test_to_metadata(self, filing_id: FilingIdentifier) -> None:
        chunk = Chunk(
            content="text",
            path="Part I > Item 1",
            content_type=ContentType.TABLE,
            filing_id=filing_id,
            chunk_index=0,
        )
        md = chunk.to_metadata()
        assert md["ticker"] == "AAPL"
        assert md["form_type"] == "10-K"
        assert md["filing_date"] == "2023-11-03"
        # Range-query-friendly integer form of the date.
        assert md["filing_date_int"] == 20231103
        assert md["content_type"] == "table"
        assert md["accession_number"] == "0000320193-23-000077"


class TestSegment:
    def test_roundtrip(self, filing_id: FilingIdentifier) -> None:
        seg = Segment(
            path="Part I > Item 1A",
            content_type=ContentType.TEXT,
            content="risk factors content",
            filing_id=filing_id,
        )
        assert seg.filing_id is filing_id
        assert seg.content_type is ContentType.TEXT


class TestSearchResult:
    def test_from_chromadb_result_converts_distance_to_similarity(self) -> None:
        result = SearchResult.from_chromadb_result(
            document="revenue grew 5%",
            metadata={
                "path": "Part II > Item 7 > MD&A",
                "content_type": "text",
                "ticker": "MSFT",
                "form_type": "10-K",
                "filing_date": "2023-07-27",
                "accession_number": "acc-1",
            },
            distance=0.2,
            chunk_id="MSFT_10-K_2023-07-27_001",
        )
        assert result.ticker == "MSFT"
        assert result.similarity == pytest.approx(0.8)
        assert result.chunk_id == "MSFT_10-K_2023-07-27_001"
        assert result.content_type is ContentType.TEXT


class TestIngestResult:
    def test_basic_fields(self, filing_id: FilingIdentifier) -> None:
        r = IngestResult(
            filing_id=filing_id,
            segment_count=5,
            chunk_count=12,
            duration_seconds=1.23,
        )
        assert r.filing_id.ticker == "AAPL"
        assert r.chunk_count == 12


# ---------------------------------------------------------------------------
# RAG domain types
# ---------------------------------------------------------------------------


def _make_search_result(path: str = "Part I > Item 1A > Risk Factors") -> SearchResult:
    """Build a minimal :class:`SearchResult` for retrieval-result tests."""
    return SearchResult(
        content="Our supply chain is exposed to geopolitical risk.",
        path=path,
        content_type=ContentType.TEXT,
        ticker="AAPL",
        form_type="10-K",
        similarity=0.82,
        filing_date="2023-11-03",
        accession_number="0000320193-23-000077",
        chunk_id="AAPL_10-K_2023-11-03_042",
    )


class TestRetrievalResult:
    def test_is_a_search_result(self) -> None:
        """Subclassing keeps ``isinstance`` checks cheap for existing APIs."""
        retrieval = RetrievalResult.from_search_result(_make_search_result())
        assert isinstance(retrieval, SearchResult)

    def test_defaults_are_safe_zero_values(self) -> None:
        retrieval = RetrievalResult.from_search_result(_make_search_result())
        assert retrieval.token_count == 0
        assert retrieval.truncated is False
        # Path has three parts separated by ' > '.
        assert retrieval.section_boundaries == (
            "Part I",
            "Item 1A",
            "Risk Factors",
        )

    def test_from_search_result_honours_explicit_boundaries(self) -> None:
        """Explicit ``section_boundaries`` overrides the path-split default."""
        override = ("Business", "Segment Reporting")
        retrieval = RetrievalResult.from_search_result(
            _make_search_result(),
            token_count=187,
            truncated=True,
            section_boundaries=override,
        )
        assert retrieval.token_count == 187
        assert retrieval.truncated is True
        assert retrieval.section_boundaries == override

    def test_from_search_result_handles_unknown_path(self) -> None:
        """A parser-provided ``'(unknown)'`` path should not crash parsing."""
        retrieval = RetrievalResult.from_search_result(
            _make_search_result(path="(unknown)"),
        )
        # Single-component path yields a single-element tuple.
        assert retrieval.section_boundaries == ("(unknown)",)

    def test_preserves_original_search_fields(self) -> None:
        base = _make_search_result()
        retrieval = RetrievalResult.from_search_result(base, token_count=42)
        assert retrieval.similarity == base.similarity
        assert retrieval.chunk_id == base.chunk_id
        assert retrieval.content == base.content


class TestCitation:
    def test_frozen(self, filing_id: FilingIdentifier) -> None:
        citation = Citation(
            chunk_id="AAPL_10-K_2023-11-03_042",
            filing_id=filing_id,
            section_path="Part I > Item 1A",
            text_span="exposed to geopolitical risk",
            similarity=0.82,
            display_index=1,
        )
        with pytest.raises(FrozenInstanceError):
            citation.display_index = 2  # type: ignore[misc]

    def test_display_index_defaults_to_zero(self, filing_id: FilingIdentifier) -> None:
        """``0`` reads as 'not yet assigned' — the orchestrator fills this in."""
        citation = Citation(
            chunk_id="AAPL_10-K_2023-11-03_042",
            filing_id=filing_id,
            section_path="Part I > Item 1A",
            text_span="...",
            similarity=0.5,
        )
        assert citation.display_index == 0

    def test_hashable_so_citations_can_dedupe(self, filing_id: FilingIdentifier) -> None:
        """Frozen dataclasses are hashable — enables set-based dedup in the orchestrator."""
        citation = Citation(
            chunk_id="AAPL_10-K_2023-11-03_042",
            filing_id=filing_id,
            section_path="Part I > Item 1A",
            text_span="...",
            similarity=0.5,
            display_index=1,
        )
        assert hash(citation) == hash(citation)
        # Two equal citations are set-equivalent.
        assert len({citation, citation}) == 1


class TestTokenUsage:
    def test_defaults_zero(self) -> None:
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_total_tokens_sums(self) -> None:
        usage = TokenUsage(input_tokens=512, output_tokens=128)
        assert usage.total_tokens == 640

    def test_addition_aggregates(self) -> None:
        a = TokenUsage(input_tokens=100, output_tokens=50)
        b = TokenUsage(input_tokens=200, output_tokens=25)
        combined = a + b
        assert combined.input_tokens == 300
        assert combined.output_tokens == 75
        assert combined.total_tokens == 375

    def test_addition_with_wrong_type_returns_notimplemented(self) -> None:
        """Adding to a non-TokenUsage must not silently coerce."""
        with pytest.raises(TypeError):
            _ = TokenUsage(1, 1) + 5  # type: ignore[operator]


class TestGenerationResult:
    def test_separates_retrieved_and_cited(self, filing_id: FilingIdentifier) -> None:
        """Retrieved chunks and citations are distinct collections."""
        chunk_a = RetrievalResult.from_search_result(_make_search_result())
        citation = Citation(
            chunk_id="AAPL_10-K_2023-11-03_042",
            filing_id=filing_id,
            section_path="Part I > Item 1A",
            text_span="exposed to geopolitical risk",
            similarity=0.82,
            display_index=1,
        )
        result = GenerationResult(
            answer="Apple flags geopolitical exposure [1].",
            provider="openai",
            model="gpt-4o-mini",
            prompt_version="v1",
            citations=[citation],
            retrieved_chunks=[chunk_a],
        )
        # Distinct collections — the orchestrator may feed more chunks than
        # the model actually cites.
        assert result.citations is not result.retrieved_chunks
        assert len(result.citations) == 1
        assert len(result.retrieved_chunks) == 1

    def test_default_lists_are_independent_across_instances(self) -> None:
        """``field(default_factory=list)`` guards against the classic mutable-default bug."""
        a = GenerationResult(answer="A", provider="openai", model="gpt-4o", prompt_version="v1")
        b = GenerationResult(answer="B", provider="openai", model="gpt-4o", prompt_version="v1")
        a.citations.append(
            Citation(
                chunk_id="x",
                filing_id=FilingIdentifier("T", "10-K", date(2024, 1, 1), "acc"),
                section_path="p",
                text_span="t",
                similarity=0.1,
            )
        )
        assert b.citations == []

    def test_token_usage_defaults_to_zero_record(self) -> None:
        result = GenerationResult(
            answer="", provider="anthropic", model="claude-haiku", prompt_version="v0"
        )
        assert result.token_usage.total_tokens == 0
        assert result.streamed is False
        assert result.latency_seconds == 0.0


class TestConversationTurn:
    def test_roundtrip(self, filing_id: FilingIdentifier) -> None:
        retrieval = RetrievalResult.from_search_result(_make_search_result())
        gen = GenerationResult(
            answer="Apple flags geopolitical exposure.",
            provider="openai",
            model="gpt-4o-mini",
            prompt_version="v1",
            retrieved_chunks=[retrieval],
        )
        turn = ConversationTurn(
            query="What risks does Apple disclose?",
            retrieval_results=[retrieval],
            generation_result=gen,
            timestamp=datetime(2026, 4, 15, 12, 0, tzinfo=UTC),
        )
        assert turn.query.startswith("What risks")
        assert turn.timestamp.tzinfo is UTC
        assert turn.generation_result.answer.startswith("Apple")


class TestProviderCapability:
    def test_defaults_are_nothing_supported(self) -> None:
        """Safe default: an unregistered provider advertises no features."""
        caps = ProviderCapability()
        assert caps.chat is False
        assert caps.embeddings is False
        assert caps.streaming is False
        assert caps.tool_use is False
        assert caps.structured_output is False
        assert caps.prompt_caching is False
        assert caps.vision is False
        assert caps.context_window_tokens == 0
        assert caps.max_output_tokens == 0
        assert caps.pricing_tier is PricingTier.UNKNOWN

    def test_frozen(self) -> None:
        caps = ProviderCapability(chat=True)
        with pytest.raises(FrozenInstanceError):
            caps.chat = False  # type: ignore[misc]

    def test_pricing_tier_enum_values(self) -> None:
        # Value set is stable — UIs and CLI filters depend on it.
        assert {t.value for t in PricingTier} == {
            "free",
            "low",
            "standard",
            "high",
            "premium",
            "unknown",
        }

    def test_fully_populated_instance(self) -> None:
        caps = ProviderCapability(
            chat=True,
            embeddings=False,
            streaming=True,
            tool_use=True,
            structured_output=True,
            prompt_caching=True,
            vision=False,
            context_window_tokens=200_000,
            max_output_tokens=8_192,
            pricing_tier=PricingTier.STANDARD,
        )
        assert caps.context_window_tokens == 200_000
        assert caps.pricing_tier is PricingTier.STANDARD


# ---------------------------------------------------------------------------
# estimate_cost — canonical per-request USD cost primitive
# ---------------------------------------------------------------------------


class TestEstimateCost:
    """The single sanctioned token-usage → USD estimate.

    Mirrors :func:`derive_pricing_tier`'s role for tiers: pure, total, and the
    only place per-MTok cost is turned into a dollar figure.
    """

    def test_known_cost_is_token_weighted_sum(self) -> None:
        # in=0.6, out=2.4 USD/MTok (gpt-5.4-mini-shaped). 120 input + 42
        # output tokens → 120/1e6*0.6 + 42/1e6*2.4 = 0.0001728.
        cap = ProviderCapability(chat=True, input_cost_per_mtok=0.6, output_cost_per_mtok=2.4)
        assert estimate_cost(TokenUsage(input_tokens=120, output_tokens=42), cap) == pytest.approx(
            0.0001728
        )

    def test_zero_tokens_is_exactly_zero(self) -> None:
        # A refusal short-circuits the LLM with a zero usage record; the
        # estimate is a clean 0.0 (not None) when the model's cost is known.
        cap = ProviderCapability(chat=True, input_cost_per_mtok=1.0, output_cost_per_mtok=1.0)
        assert estimate_cost(TokenUsage(), cap) == 0.0

    def test_free_model_is_zero(self) -> None:
        cap = ProviderCapability(chat=True, input_cost_per_mtok=0.0, output_cost_per_mtok=0.0)
        assert cap.pricing_tier is PricingTier.FREE
        assert estimate_cost(TokenUsage(input_tokens=1000, output_tokens=1000), cap) == 0.0

    def test_both_costs_unknown_returns_none(self) -> None:
        # The UNKNOWN case — OpenRouter / overlay-only rows whose tier is
        # also UNKNOWN. The honest answer is "unknown", not a partial figure.
        cap = ProviderCapability(chat=True)
        assert cap.pricing_tier is PricingTier.UNKNOWN
        assert estimate_cost(TokenUsage(input_tokens=120, output_tokens=42), cap) is None

    def test_one_sided_cost_returns_none(self) -> None:
        # A one-sided estimate would understate the true price; refuse it.
        in_only = ProviderCapability(chat=True, input_cost_per_mtok=0.6)
        out_only = ProviderCapability(chat=True, output_cost_per_mtok=2.4)
        usage = TokenUsage(input_tokens=120, output_tokens=42)
        assert estimate_cost(usage, in_only) is None
        assert estimate_cost(usage, out_only) is None

    def test_estimate_is_pure_no_mutation(self) -> None:
        # Neither argument is mutated — the helper is side-effect-free.
        cap = ProviderCapability(chat=True, input_cost_per_mtok=1.0, output_cost_per_mtok=2.0)
        usage = TokenUsage(input_tokens=10, output_tokens=20)
        estimate_cost(usage, cap)
        assert usage.input_tokens == 10
        assert usage.output_tokens == 20
        assert cap.input_cost_per_mtok == 1.0


# ---------------------------------------------------------------------------
# EmbedderStamp — digital seal on every ChromaDB collection
# ---------------------------------------------------------------------------


class TestEmbedderStamp:
    def test_frozen(self) -> None:
        stamp = EmbedderStamp(provider="openai", model="text-embedding-3-small", dimension=1536)
        with pytest.raises(FrozenInstanceError):
            stamp.provider = "anthropic"  # type: ignore[misc]

    def test_to_metadata_keys_and_coercion(self) -> None:
        stamp = EmbedderStamp(provider="openai", model="text-embedding-3-small", dimension=1536)
        metadata = stamp.to_metadata()
        assert metadata == {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": "1536",
        }
        # ChromaDB treats every metadata value as JSON; the dimension
        # MUST be serialised as a string so backends do not disagree
        # about integer representation.
        assert isinstance(metadata["embedding_dimension"], str)

    def test_round_trip(self) -> None:
        stamp = EmbedderStamp(provider="local", model="google/embeddinggemma-300m", dimension=768)
        assert EmbedderStamp.from_metadata(stamp.to_metadata()) == stamp

    @pytest.mark.parametrize(
        "missing_key",
        ["embedding_provider", "embedding_model", "embedding_dimension"],
    )
    def test_from_metadata_rejects_missing_key(self, missing_key: str) -> None:
        good = EmbedderStamp(
            provider="openai", model="text-embedding-3-small", dimension=1536
        ).to_metadata()
        del good[missing_key]
        with pytest.raises(ValueError, match="missing required key"):
            EmbedderStamp.from_metadata(good)

    def test_from_metadata_rejects_non_integer_dimension(self) -> None:
        metadata = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": "not-a-number",
        }
        with pytest.raises(ValueError, match="non-integer dimension"):
            EmbedderStamp.from_metadata(metadata)

    def test_from_metadata_rejects_non_positive_dimension(self) -> None:
        metadata = EmbedderStamp(
            provider="openai", model="text-embedding-3-small", dimension=1536
        ).to_metadata()
        metadata["embedding_dimension"] = "0"
        with pytest.raises(ValueError, match="non-positive dimension"):
            EmbedderStamp.from_metadata(metadata)

    def test_from_metadata_rejects_non_string_provider(self) -> None:
        metadata = {
            "embedding_provider": 42,
            "embedding_model": "text-embedding-3-small",
            "embedding_dimension": "1536",
        }
        with pytest.raises(ValueError, match="non-string"):
            EmbedderStamp.from_metadata(metadata)

    def test_hashability(self) -> None:
        """Frozen dataclasses hash by field values — useful for
        deduping collection stamps across registries."""
        a = EmbedderStamp(provider="openai", model="text-embedding-3-small", dimension=1536)
        b = EmbedderStamp(provider="openai", model="text-embedding-3-small", dimension=1536)
        assert hash(a) == hash(b)
        assert {a, b} == {a}


# ---------------------------------------------------------------------------
# Deployment profile + eviction report
# ---------------------------------------------------------------------------


class TestDeploymentProfile:
    """``DeploymentProfile`` is a ``StrEnum`` so its values double as
    settings strings and JSON / log-friendly identifiers without an
    extra ``.value`` access on every read.
    """

    def test_string_values(self) -> None:
        assert DeploymentProfile.LOCAL == "local"
        assert DeploymentProfile.TEAM == "team"
        assert DeploymentProfile.CLOUD == "cloud"

    def test_membership(self) -> None:
        valid = {profile.value for profile in DeploymentProfile}
        assert valid == {"local", "team", "cloud"}


class TestEvictionReport:
    """Audit shape returned by :meth:`FilingStore.evict_expired`."""

    def test_basic_construction(self) -> None:
        report = EvictionReport(
            filings_evicted=4,
            chunks_evicted=237,
            max_age_days=90,
        )
        assert report.filings_evicted == 4
        assert report.chunks_evicted == 237
        assert report.max_age_days == 90

    def test_frozen(self) -> None:
        report = EvictionReport(
            filings_evicted=0,
            chunks_evicted=0,
            max_age_days=30,
        )
        with pytest.raises(FrozenInstanceError):
            report.filings_evicted = 99  # type: ignore[misc]

    def test_zero_eviction_is_a_valid_report(self) -> None:
        """A successful sweep that found nothing returns this shape;
        it must not be confused with a failure (no exception is raised)."""
        report = EvictionReport(
            filings_evicted=0,
            chunks_evicted=0,
            max_age_days=7,
        )
        assert report.filings_evicted == 0


# ---------------------------------------------------------------------------
# Security — no domain type may carry a credential
# ---------------------------------------------------------------------------


_SECRET_FIELD_HINTS = (
    "api_key",
    "api-key",
    "apikey",
    "secret",
    "password",
    "credential",
    "private_key",
    "auth_token",
    "bearer",
)


@pytest.mark.security
class TestNoCredentialFieldsOnDomainTypes:
    """Secrets must never travel through core domain types.

    This test fails fast if a future change introduces a field whose name
    looks like it carries a credential.  It is intentionally noisy — a
    benign rename (``api_key`` → ``key_id``) will trip it and force a
    reviewer to think about whether the field really should be there.
    """

    @pytest.mark.parametrize(
        "cls",
        [
            FilingIdentifier,
            Segment,
            Chunk,
            SearchResult,
            RetrievalResult,
            Citation,
            TokenUsage,
            GenerationResult,
            ConversationTurn,
            ProviderCapability,
            IngestResult,
            EmbedderStamp,
            ReindexReport,
            EvictionReport,
            BackupReport,
            RestoreReport,
            ExportReport,
            ImportReport,
            CatalogueRefreshReport,
        ],
    )
    def test_no_secret_looking_fields(self, cls: type) -> None:
        for f in fields(cls):
            lowered = f.name.lower()
            for hint in _SECRET_FIELD_HINTS:
                assert hint not in lowered, (
                    f"{cls.__name__}.{f.name} looks credential-bearing; "
                    "domain types must not carry secrets."
                )


class TestPricingTierHelpTextMatchesTheTierFunction:
    """F37: the web model-picker explains the tiers to users.  It said "Under
    $2 / $5 / $10 per 1M output tokens" while ``derive_pricing_tier`` buckets
    the *blended* mean at 1 / 4 / 15 USD — and nothing failed when the two
    disagreed.  The legend now derives from one named constant that must
    equal ``_PRICING_TIER_BLENDED_BOUNDS``; this lock reads it from the TSX."""

    _PICKER = (
        Path(__file__).resolve().parents[2] / "frontend" / "src" / "components" / "model-picker.tsx"
    )

    def _frontend_bounds(self) -> list[float]:
        import re

        source = self._PICKER.read_text(encoding="utf-8")
        match = re.search(
            r"export const PRICING_TIER_BLENDED_BOUNDS_USD = \[([^\]]*)\] as const;", source
        )
        assert match, "PRICING_TIER_BLENDED_BOUNDS_USD not found in model-picker.tsx"
        return [float(v) for v in match.group(1).split(",")]

    def test_frontend_bounds_equal_the_backend_bounds(self) -> None:
        from sec_generative_search.core.types import _PRICING_TIER_BLENDED_BOUNDS

        assert self._frontend_bounds() == [bound for bound, _ in _PRICING_TIER_BLENDED_BOUNDS]

    def test_bounds_split_the_tiers_the_legend_names(self) -> None:
        """Each legend band agrees with the function at its edges."""
        from sec_generative_search.core.types import derive_pricing_tier

        low, standard, high = self._frontend_bounds()
        assert derive_pricing_tier(0.0, 0.0) is PricingTier.FREE
        assert derive_pricing_tier(low - 0.01, low - 0.01) is PricingTier.LOW
        assert derive_pricing_tier(low, low) is PricingTier.STANDARD
        assert derive_pricing_tier(standard, standard) is PricingTier.HIGH
        assert derive_pricing_tier(high, high) is PricingTier.PREMIUM
        # Blended, not output-only: cheap input pulls an expensive output down.
        assert derive_pricing_tier(0.5, 7.0) is PricingTier.STANDARD
