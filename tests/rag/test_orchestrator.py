"""Tests for :mod:`sec_generative_search.rag.orchestrator`.

Covers the full pipeline against fake retrieval and LLM doubles —
construction, single-query happy path, refusal short-circuit, streaming,
comparative fan-out, conversation history, structured-output preference,
multilingual handling, and traceability.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from sec_generative_search.core.types import (
    ConversationTurn,
    GenerationResult,
    ProviderCapability,
    TokenUsage,
)
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.orchestrator import RAGOrchestrator
from sec_generative_search.rag.query_understanding import QueryPlan
from tests.rag.conftest import FakeLLMProvider, FakeRetrievalService


def _build_orchestrator(
    *,
    retrieval: FakeRetrievalService,
    llm: FakeLLMProvider,
) -> RAGOrchestrator:
    """Construct an orchestrator with a stable token counter."""

    def counter(text: str) -> int:
        return max(1, len(text) // 4)

    return RAGOrchestrator(retrieval=retrieval, llm=llm, token_counter=counter)


class TestSingleQueryHappyPath:
    def test_returns_generation_result_with_inline_citations(
        self, fake_retrieval, fake_llm, sample_chunks
    ) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="What about revenue?")

        result = orch.generate(plan)

        assert isinstance(result, GenerationResult)
        assert result.provider == "fake-llm"
        assert result.prompt_version == "v1.0.0"
        assert result.streamed is False
        # Reply includes [1] [2] markers; both present in chunks.
        assert len(result.citations) == 2
        assert {c.chunk_id for c in result.citations} == {
            sample_chunks[0].chunk_id,
            sample_chunks[1].chunk_id,
        }
        # Retrieved chunks are recorded in full (superset of citations).
        assert len(result.retrieved_chunks) == 3
        # Token usage recorded.
        assert result.token_usage.total_tokens == 20

    def test_passes_query_en_to_retrieval(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(
            raw_query="AAPL gelirleri nasil?",
            detected_language="tr",
            query_en="How is AAPL revenue?",
        )
        orch.generate(plan)
        # Retrieval saw the English rendering, not the raw Turkish query.
        assert fake_retrieval.calls[0]["query"] == "How is AAPL revenue?"

    def test_user_prompt_carries_raw_query(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(
            raw_query="AAPL gelirleri nasil?",
            detected_language="tr",
            query_en="How is AAPL revenue?",
        )
        orch.generate(plan)
        # The model still sees the user's verbatim question — keeps the
        # answer faithful to the user's wording.
        assert "AAPL gelirleri nasil?" in fake_llm.last_request.prompt
        # And the system prompt asks for the user's language.
        assert "tr" in fake_llm.last_request.system

    def test_filters_from_plan_propagate_to_retrieval(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(
            raw_query="Q",
            tickers=["AAPL"],
            form_types=["10-K"],
            date_range=("2023-01-01", "2023-12-31"),
        )
        orch.generate(plan)
        call = fake_retrieval.calls[0]
        assert call["ticker"] == "AAPL"
        assert call["form_type"] == "10-K"
        assert call["start_date"] == "2023-01-01"
        assert call["end_date"] == "2023-12-31"

    def test_extra_filters_override_plan(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q", tickers=["AAPL"])
        orch.generate(plan, extra_filters={"ticker": "MSFT"})
        assert fake_retrieval.calls[0]["ticker"] == "MSFT"


class TestRefusalPath:
    def test_short_circuits_when_no_chunks(self, fake_llm) -> None:
        empty_retrieval = FakeRetrievalService(results=[])
        orch = _build_orchestrator(retrieval=empty_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q")

        result = orch.generate(plan)

        assert "cannot answer" in result.answer.lower()
        assert result.citations == []
        assert result.retrieved_chunks == []
        # LLM was never called.
        assert fake_llm.call_count == 0
        # Token usage stays at zero — we did not pretend to consume any.
        assert result.token_usage.total_tokens == 0

    def test_refusal_disabled_calls_llm_with_empty_context(self, fake_llm) -> None:
        # Pydantic-settings v2 evaluates nested-settings defaults at class
        # definition time; ``reload_settings()`` does not flow env-var
        # overrides into nested ``settings.rag`` (see
        # tests/config/test_settings.py for the direct-RAGSettings()
        # pattern).  Patch the orchestrator's captured settings instead.
        empty_retrieval = FakeRetrievalService(results=[])
        orch = _build_orchestrator(retrieval=empty_retrieval, llm=fake_llm)
        orch._rag_settings.refusal_enabled = False  # type: ignore[attr-defined]

        plan = QueryPlan(raw_query="Q")
        result = orch.generate(plan)
        # Provider was called.
        assert fake_llm.call_count == 1
        # And the request shows an empty context block.
        assert "(no chunks retrieved)" in fake_llm.last_request.prompt
        # Result is the model's reply, not a refusal.
        assert "cannot answer" not in result.answer.lower()


class TestStreaming:
    def test_yields_deltas_then_final(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q")

        events = list(orch.generate_stream(plan))

        deltas = [e for e in events if e.delta is not None]
        finals = [e for e in events if e.final is not None]
        assert len(deltas) >= 1
        assert len(finals) == 1
        # Deltas must reconstruct the model's reply.
        assert "".join(d.delta or "" for d in deltas) == fake_llm.reply

    def test_final_carries_assembled_result(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q")

        events = list(orch.generate_stream(plan))
        final = next(e.final for e in events if e.final is not None)
        assert final.streamed is True
        assert len(final.citations) == 2  # [1] and [2] in the canned reply
        assert final.token_usage.total_tokens > 0

    def test_streams_refusal(self, fake_llm) -> None:
        empty_retrieval = FakeRetrievalService(results=[])
        orch = _build_orchestrator(retrieval=empty_retrieval, llm=fake_llm)

        events = list(orch.generate_stream(QueryPlan(raw_query="Q")))
        assert any(e.delta is not None and "cannot answer" in e.delta.lower() for e in events)
        final = next(e.final for e in events if e.final is not None)
        assert final.citations == []
        assert fake_llm.call_count == 0


class TestComparativeFanOut:
    def test_runs_one_retrieval_per_ticker(self, sample_chunks, fake_llm) -> None:
        # Configure per-call results so we can see fan-out happen.
        retrieval = FakeRetrievalService(
            per_call_results=[
                [sample_chunks[0]],
                [sample_chunks[1]],
            ]
        )
        orch = _build_orchestrator(retrieval=retrieval, llm=fake_llm)
        plan = QueryPlan(
            raw_query="Compare AAPL vs MSFT revenue",
            tickers=["AAPL", "MSFT"],
            suggested_answer_mode=AnswerMode.COMPARATIVE,
        )

        orch.generate(plan)

        assert len(retrieval.calls) == 2
        tickers_seen = {call["ticker"] for call in retrieval.calls}
        assert tickers_seen == {"AAPL", "MSFT"}

    def test_dedupes_overlapping_results(self, sample_chunks, fake_llm) -> None:
        # Same chunk returned twice — must collapse to one.
        retrieval = FakeRetrievalService(
            per_call_results=[
                [sample_chunks[0]],
                [sample_chunks[0]],
            ]
        )
        orch = _build_orchestrator(retrieval=retrieval, llm=fake_llm)
        plan = QueryPlan(
            raw_query="Compare",
            tickers=["AAPL", "MSFT"],
            suggested_answer_mode=AnswerMode.COMPARATIVE,
        )
        result = orch.generate(plan)
        assert len(result.retrieved_chunks) == 1

    def test_single_ticker_skips_fan_out(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(
            raw_query="One ticker",
            tickers=["AAPL"],
            suggested_answer_mode=AnswerMode.COMPARATIVE,
        )
        orch.generate(plan)
        assert len(fake_retrieval.calls) == 1
        # The single-query path embeds inside ``retrieve`` as before.
        assert fake_retrieval.embed_calls == []
        assert "query_embedding" not in fake_retrieval.calls[0]

    @pytest.mark.parametrize("streaming", [False, True])
    def test_fan_out_embeds_the_query_once(self, sample_chunks, fake_llm, streaming) -> None:
        """F14: three legs, one embed; every leg reuses the same vector."""
        retrieval = FakeRetrievalService(per_call_results=[[c] for c in sample_chunks[:3]])
        orch = _build_orchestrator(retrieval=retrieval, llm=fake_llm)
        plan = QueryPlan(
            raw_query="Compare AAPL, MSFT and NVDA revenue",
            query_en="Compare AAPL, MSFT and NVDA revenue",
            tickers=["AAPL", "MSFT", "NVDA"],
            suggested_answer_mode=AnswerMode.COMPARATIVE,
        )

        if streaming:
            list(orch.generate_stream(plan))
        else:
            orch.generate(plan)

        assert retrieval.embed_calls == ["Compare AAPL, MSFT and NVDA revenue"]
        assert [call["ticker"] for call in retrieval.calls] == ["AAPL", "MSFT", "NVDA"]
        vectors = [call["query_embedding"] for call in retrieval.calls]
        assert all(v is vectors[0] for v in vectors)
        assert vectors[0] == ("fake-vector", "Compare AAPL, MSFT and NVDA revenue")
        # The query text still travels with every leg.
        assert all(call["query"] == plan.query_en for call in retrieval.calls)

    @pytest.mark.security
    @pytest.mark.parametrize("streaming", [False, True])
    def test_embed_failure_propagates_unchanged(self, fake_llm, streaming) -> None:
        """A hosted-embedder outage in the pre-embed keeps its type (M1 contract).

        The ``/api/rag/*`` ladder classifies the ``ProviderError`` exactly as
        when the embed ran inside the first ``retrieve``; no leg and no LLM
        call happen.
        """
        from sec_generative_search.core.exceptions import ProviderError

        original = ProviderError("embedding upstream down", provider="openai")

        class _FailingEmbed(FakeRetrievalService):
            def embed_query(self, query: str) -> tuple[str, str]:
                raise original

        retrieval = _FailingEmbed()
        orch = _build_orchestrator(retrieval=retrieval, llm=fake_llm)
        plan = QueryPlan(
            raw_query="Compare AAPL and MSFT",
            tickers=["AAPL", "MSFT"],
            suggested_answer_mode=AnswerMode.COMPARATIVE,
        )
        with pytest.raises(ProviderError) as excinfo:
            if streaming:
                list(orch.generate_stream(plan))
            else:
                orch.generate(plan)
        assert excinfo.value is original
        assert retrieval.calls == []
        assert fake_llm.last_request is None


class TestConversationHistory:
    def _make_turn(self, q: str, a: str) -> ConversationTurn:
        return ConversationTurn(
            query=q,
            retrieval_results=[],
            generation_result=GenerationResult(
                answer=a,
                provider="fake-llm",
                model="fake-model",
                prompt_version="v1.0.0",
                token_usage=TokenUsage(input_tokens=5, output_tokens=3),
            ),
            timestamp=datetime(2026, 5, 1, 12, 0, 0),
        )

    def test_history_disabled_by_default(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        history = [self._make_turn("Prior Q?", "Prior A.")]
        orch.generate(QueryPlan(raw_query="Q"), history=history)
        assert "Prior Q?" not in fake_llm.last_request.prompt

    def test_history_enabled_renders_q_a_pairs(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        orch._rag_settings.chat_history_enabled = True  # type: ignore[attr-defined]

        history = [self._make_turn("Prior Q?", "Prior A.")]
        orch.generate(QueryPlan(raw_query="Q"), history=history)
        assert "Prior Q?" in fake_llm.last_request.prompt
        assert "Prior A." in fake_llm.last_request.prompt

    def test_history_trimmed_to_max_turns(self, fake_retrieval, fake_llm) -> None:
        """Turns beyond ``chat_history_max_turns`` are dropped before rendering.

        The budget is generous (8000-token window → ~1200-token history
        slice) so the token packer would keep both turns; only the
        ``max_turns`` cap removes the oldest, isolating that trim path.
        """
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        orch._rag_settings.chat_history_enabled = True  # type: ignore[attr-defined]
        orch._rag_settings.chat_history_max_turns = 1  # type: ignore[attr-defined]

        history = [
            self._make_turn("Oldest Q?", "Oldest A."),
            self._make_turn("Newest Q?", "Newest A."),
        ]
        orch.generate(QueryPlan(raw_query="Q"), history=history)
        prompt = fake_llm.last_request.prompt
        assert "Newest Q?" in prompt
        assert "Oldest Q?" not in prompt

    @pytest.mark.security
    def test_history_does_not_leak_prior_chunks(self, fake_retrieval, fake_llm, make_chunk) -> None:
        """Prior chunks must never be carried into a follow-up prompt."""
        # Build a fresh chunk distinct from anything the current retrieval
        # returns — using ``sample_chunks[0]`` would let the sentinel leak
        # via the *current* turn's retrieval, masking the test.
        prior_chunk = make_chunk(
            index=99,
            ticker="TSLA",
            content="TIER1-CHUNK-SENTINEL-XYZ",
        )
        prior = ConversationTurn(
            query="Prior Q?",
            retrieval_results=[prior_chunk],
            generation_result=GenerationResult(
                answer="Prior A.",
                provider="fake-llm",
                model="fake-model",
                prompt_version="v1.0.0",
            ),
            timestamp=datetime(2026, 5, 1, 12, 0, 0),
        )

        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        orch._rag_settings.chat_history_enabled = True  # type: ignore[attr-defined]
        orch.generate(QueryPlan(raw_query="Q"), history=[prior])
        assert "TIER1-CHUNK-SENTINEL-XYZ" not in fake_llm.last_request.prompt


class TestStructuredOutputPreference:
    def test_sets_response_format_when_preferred(
        self, fake_retrieval, fake_llm, sample_chunks
    ) -> None:
        # Reply must be a JSON envelope so the JSON path succeeds.
        fake_llm.reply = json.dumps(
            {
                "answer": "Revenue grew.",
                "cited_chunk_ids": [sample_chunks[0].chunk_id],
            }
        )
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q")
        result = orch.generate(plan, prefer_structured_output=True)
        assert fake_llm.last_request.response_format == "json"
        assert fake_llm.last_request.response_schema is not None
        assert result.answer == "Revenue grew."
        assert result.citations[0].chunk_id == sample_chunks[0].chunk_id


class TestOutputLanguageOverride:
    def test_auto_uses_detected_language(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q", detected_language="de")
        orch.generate(plan)
        # German appears in the system prompt.
        assert "de" in fake_llm.last_request.system

    def test_explicit_override_locks_language(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        orch._rag_settings.output_language = "en"  # type: ignore[attr-defined]

        plan = QueryPlan(raw_query="Q", detected_language="tr")
        orch.generate(plan)
        # Operator override pins English regardless of detection.
        sys_prompt = fake_llm.last_request.system
        # The directive line includes the chosen language; we want
        # English explicitly mentioned, and we want Turkish NOT to be
        # the one passed.
        assert " en" in sys_prompt or "in en" in sys_prompt


class TestTraceability:
    def test_records_provider_model_prompt_version(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q")
        result = orch.generate(plan, model="explicit-model")
        assert result.provider == "fake-llm"
        # Provider echoes the model slug back via GenerationResponse.model.
        assert result.model == "explicit-model"
        assert result.prompt_version == "v1.0.0"
        assert result.latency_seconds >= 0.0
        # retrieved_chunks ⊇ citations — diagnoses overshoot vs. ignore-context.
        cited_ids = {c.chunk_id for c in result.citations}
        retrieved_ids = {c.chunk_id for c in result.retrieved_chunks}
        assert cited_ids.issubset(retrieved_ids)


class TestBudgetIntegration:
    def test_passes_chunks_token_budget_to_retrieval(self, fake_retrieval, fake_llm) -> None:
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=fake_llm)
        plan = QueryPlan(raw_query="Q")
        orch.generate(plan, max_output_tokens=500)
        budget_arg = fake_retrieval.calls[0]["context_token_budget"]
        # Capability says window=8000; default history fraction = 1200;
        # answer = 500; plus a small system slice — chunks must be the
        # remainder, well above zero.
        assert budget_arg > 1000

    def test_unknown_window_falls_back_to_settings_budget(self, fake_retrieval) -> None:
        # LLM that advertises an unknown window.
        unknown_window_llm = FakeLLMProvider(
            reply="Reply.",
            capability=ProviderCapability(chat=True, streaming=True, context_window_tokens=0),
        )
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=unknown_window_llm)
        orch._rag_settings.context_token_budget = 1234  # type: ignore[attr-defined]
        orch.generate(QueryPlan(raw_query="Q"))
        assert fake_retrieval.calls[0]["context_token_budget"] == 1234

    def test_capability_discovery_failure_degrades_to_fallback_budget(self, fake_retrieval) -> None:
        """A provider whose ``get_capabilities`` raises must not break generation.

        The allocator treats a capability-discovery error as an unknown
        window (``total_window == 0``) and falls back to the settings
        chunks budget — an instrumentation/capability hiccup can never
        abort a live generate call.
        """

        class ExplodingCapabilityLLM(FakeLLMProvider):
            def get_capabilities(self):  # type: ignore[override]
                raise RuntimeError("capability matrix unavailable")

        llm = ExplodingCapabilityLLM(reply="Reply [1].")
        orch = _build_orchestrator(retrieval=fake_retrieval, llm=llm)
        orch._rag_settings.context_token_budget = 4321  # type: ignore[attr-defined]
        result = orch.generate(QueryPlan(raw_query="Q"))
        # Generation still succeeded …
        assert result.answer == "Reply [1]."
        # … on the unknown-window fallback budget.
        assert fake_retrieval.calls[0]["context_token_budget"] == 4321
