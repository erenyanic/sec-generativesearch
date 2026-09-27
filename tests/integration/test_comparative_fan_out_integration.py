"""Comparative fan-out over the real retrieval stack (F14).

Drives the real :class:`RAGOrchestrator` + :class:`RetrievalService` over an
on-disk ChromaDB seeded for three tickers.  Pins that the fan-out embeds the
question once — not once per ticker — and that every leg returns exactly
what the old per-leg embed path returned (same vector, same hits, same
merge order).
"""

from __future__ import annotations

from datetime import date

import pytest

from sec_generative_search.database import FilingStore
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.orchestrator import RAGOrchestrator
from sec_generative_search.rag.query_understanding import QueryPlan

from .conftest import KeywordEmbedder, ScriptedLLM, build_processed_filing, make_filing_id

pytestmark = pytest.mark.integration

_FILINGS = (
    ("AAPL", "0000320193-23-000077", date(2023, 11, 3)),
    ("MSFT", "0000789019-23-000095", date(2023, 7, 27)),
    ("NVDA", "0001045810-24-000029", date(2024, 2, 21)),
)
_QUERY = "Compare revenue growth and margin"


@pytest.fixture
def corpus(store: FilingStore, embedder: KeywordEmbedder) -> FilingStore:
    for ticker, accession, filed in _FILINGS:
        filing = build_processed_filing(
            make_filing_id(ticker=ticker, accession_number=accession, filing_date=filed),
            [
                ("Part II > Item 7 > MD&A", f"{ticker} revenue growth was strong."),
                ("Part II > Item 7 > Margin", f"{ticker} margin expanded on growth."),
                ("Part I > Item 1A > Risk Factors", f"{ticker} litigation and debt risk."),
            ],
            embedder,
        )
        store.store_filing(filing, register_if_new=True)
    return store


def test_three_ticker_fan_out_embeds_once_with_identical_legs(
    corpus: FilingStore,
    embedder: KeywordEmbedder,
    retrieval,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedded: list[str] = []
    real_embed_query = embedder.embed_query

    def counting_embed_query(text: str):
        embedded.append(text)
        return real_embed_query(text)

    monkeypatch.setattr(embedder, "embed_query", counting_embed_query)

    legs: list[tuple[str, dict, list[str | None]]] = []
    real_retrieve = retrieval.retrieve

    def recording_retrieve(query: str, **kwargs):
        hits = real_retrieve(query, **kwargs)
        legs.append((query, kwargs, [h.chunk_id for h in hits]))
        return hits

    monkeypatch.setattr(retrieval, "retrieve", recording_retrieve)

    plan = QueryPlan(
        raw_query=_QUERY,
        query_en=_QUERY,
        tickers=[t for t, _, _ in _FILINGS],
        suggested_answer_mode=AnswerMode.COMPARATIVE,
    )
    result = RAGOrchestrator(retrieval=retrieval, llm=ScriptedLLM(reply="Growth [1].")).generate(
        plan
    )

    # One embed for three legs (was three).
    assert embedded == [_QUERY]
    assert [kwargs["ticker"] for _, kwargs, _ in legs] == ["AAPL", "MSFT", "NVDA"]

    # Replaying each leg through the per-leg embed path returns the same hits.
    embedded.clear()
    for query, kwargs, ids in legs:
        assert ids, "every ticker should contribute hits"
        per_leg = {k: v for k, v in kwargs.items() if k != "query_embedding"}
        assert [h.chunk_id for h in real_retrieve(query, **per_leg)] == ids
    assert len(embedded) == 3

    # Merge order and dedupe are unchanged: legs in ticker order.
    assert [c.chunk_id for c in result.retrieved_chunks] == [i for _, _, ids in legs for i in ids]
