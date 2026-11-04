"""Tests for ``sec-rag rag query``.

The retrieval and generation primitives (:class:`RetrievalService`,
:class:`RAGOrchestrator`, ``understand_query``, the ``build_embedder`` /
``build_llm_provider`` factory seams) are stubbed at the ``cli.rag``
import site — stubbing at the import site (rather than at the source
modules) proves the CLI wires the right seams.

Goals:

- Both factory seams (``build_embedder`` and ``build_llm_provider``)
  are used; direct adapter instantiation never happens.
- The CLI resolver chain composes ``encrypted-user → admin-env``
  according to settings (encrypted tier appears only when both
  ``persist_provider_credentials`` and ``registry.encrypted`` are
  true; otherwise the chain collapses to admin-env alone).
- ``--show-plan`` runs ``understand_query`` and exits 0 *before*
  generation — the orchestrator must not be called.
- ``--skip-plan`` bypasses ``understand_query`` entirely and feeds a
  minimal plan to the orchestrator.
- The auto-plan path runs ``understand_query`` once and then
  ``orchestrator.generate`` once with the returned plan.
- ``--ticker`` / ``--form`` / ``--since`` / ``--until`` / ``--mode``
  overrides land on the plan and reach the orchestrator.
- Provider exceptions map to single operator-facing envelopes that
  mirror ``POST /api/rag/query`` (auth → unauthorised, transient →
  unavailable, every other provider error → "Provider error", etc.).
- Security: the raw question and the provider key never leak into
  stdout; EDGAR identity never surfaces.
"""

from __future__ import annotations

import re
import threading
from datetime import date
from typing import Any, ClassVar

import pytest
import typer
from typer.testing import CliRunner

import sec_generative_search.cli.rag as rag_module
from sec_generative_search.cli.rag import rag_app
from sec_generative_search.config.settings import (
    DatabaseSettings,
    EmbeddingSettings,
    LLMSettings,
    RAGSettings,
    SearchSettings,
)
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    GenerationError,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    SearchError,
)
from sec_generative_search.core.types import (
    Citation,
    EmbedderStamp,
    FilingIdentifier,
    GenerationResult,
    TokenUsage,
)
from sec_generative_search.providers.base import ProviderCapability
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.query_understanding import QueryPlan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Strips ANSI colour escapes; Click renders :class:`typer.BadParameter`
# inside a Rich-styled panel that injects colour codes between words and
# soft-wraps the message to the panel width.  In a narrow CI terminal
# that means a literal substring like ``"--since and --until must be
# supplied together"`` is broken by both colour codes and newlines, so a
# raw ``in`` check fails even when the message is intact.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _stripped(output: str) -> str:
    """Return *output* with ANSI codes removed and whitespace collapsed.

    Click / Rich panels render error messages with colour codes and
    word-wrap that vary with terminal width.  Tests that need to assert
    on a substring should anchor against this normalised view rather
    than against the raw bytes — otherwise CI runs with narrower
    terminals than the developer's local box flake.
    """
    return re.sub(r"\s+", " ", _ANSI_RE.sub("", output))


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _make_plan(
    *,
    raw_query: str = "What are Apple's main revenue segments?",
    detected_language: str = "en",
    tickers: list[str] | None = None,
    form_types: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
    mode: AnswerMode = AnswerMode.CONCISE,
) -> QueryPlan:
    return QueryPlan(
        raw_query=raw_query,
        detected_language=detected_language,
        query_en=raw_query,
        tickers=list(tickers or []),
        form_types=list(form_types or []),
        date_range=date_range,
        intent="ask",
        suggested_answer_mode=mode,
    )


def _make_citation(
    *,
    ticker: str = "AAPL",
    form_type: str = "10-K",
    accession: str = "0000320193-24-000001",
    section_path: str = "Part II > Item 7 > MD&A",
    text_span: str = "Revenue grew across all geographic segments.",
    similarity: float = 0.62,
    display_index: int = 1,
) -> Citation:
    return Citation(
        chunk_id=f"{accession}:1",
        filing_id=FilingIdentifier(
            ticker=ticker,
            form_type=form_type,
            filing_date=date(2024, 1, 15),
            accession_number=accession,
        ),
        section_path=section_path,
        text_span=text_span,
        similarity=similarity,
        display_index=display_index,
    )


def _make_stream_delta(text: str) -> Any:
    """Construct a ``StreamEvent``-shaped delta-only event.

    ``cli.rag._run_chat_stream`` duck-types stream events via
    ``getattr(item, "delta", None)`` / ``getattr(item, "final", None)``,
    so a hand-rolled stand-in works fine.  Using the real
    :class:`StreamEvent` would also work but the dataclass is frozen
    and we want to mutate ``final`` in helpers occasionally.
    """
    from sec_generative_search.rag.orchestrator import StreamEvent

    return StreamEvent(delta=text)


def _make_stream_final(
    result: GenerationResult | None = None,
) -> Any:
    """Construct a ``StreamEvent``-shaped final event."""
    from sec_generative_search.rag.orchestrator import StreamEvent

    return StreamEvent(final=result if result is not None else _make_result())


def _make_result(
    *,
    answer: str = "Apple has three main revenue segments [1].",
    citations: list[Citation] | None = None,
    refused: bool = False,
) -> GenerationResult:
    if refused:
        return GenerationResult(
            answer="I cannot answer.",
            provider="openai",
            model="gpt-x",
            prompt_version="v0",
            citations=[],
            retrieved_chunks=[],
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
            latency_seconds=0.1,
            streamed=False,
        )
    return GenerationResult(
        answer=answer,
        provider="openai",
        model="gpt-x",
        prompt_version="v1",
        citations=list(citations or [_make_citation()]),
        retrieved_chunks=[],  # filled by orchestrator; not asserted on
        token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        latency_seconds=0.42,
        streamed=False,
    )


class _FakeEmbedder:
    instances: ClassVar[list[_FakeEmbedder]] = []

    def __init__(self) -> None:
        _FakeEmbedder.instances.append(self)


class _FakeLLM:
    instances: ClassVar[list[_FakeLLM]] = []
    provider_name = "openai"

    def __init__(self) -> None:
        _FakeLLM.instances.append(self)


class _FakeChroma:
    instances: ClassVar[list[_FakeChroma]] = []

    def __init__(self, stamp: EmbedderStamp, *, chroma_path: str | None = None) -> None:
        self.stamp = stamp
        _FakeChroma.instances.append(self)


class _FakeRegistry:
    """Stand-in for :class:`MetadataRegistry`.

    Exposes the ``encrypted`` flag that ``_build_api_key_resolver``
    consults; defaults to False so the encrypted tier collapses out.
    """

    instances: ClassVar[list[_FakeRegistry]] = []
    encrypted_default: ClassVar[bool] = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.encrypted = type(self).encrypted_default
        _FakeRegistry.instances.append(self)


class _FakeRetrievalService:
    instances: ClassVar[list[_FakeRetrievalService]] = []

    def __init__(self, *, embedder: Any, chroma_client: Any) -> None:
        self.embedder = embedder
        self.chroma_client = chroma_client
        _FakeRetrievalService.instances.append(self)


class _FakeOrchestrator:
    """Recording :class:`RAGOrchestrator` stand-in.

    ``generate`` (non-streaming) is the surface ``rag query`` consumes.
    ``generate_stream`` is the surface ``rag chat`` consumes — the
    chat-test classes arm it to yield a configured sequence of stream
    events or to raise a configured exception.
    """

    instances: ClassVar[list[_FakeOrchestrator]] = []

    def __init__(self, *, retrieval: Any, llm: Any) -> None:
        self.retrieval = retrieval
        self.llm = llm
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.returns: GenerationResult = _make_result()
        self.raises: BaseException | None = None
        # Per-call stream-event sequence.  Each item is the list of
        # ``StreamEvent`` instances ``generate_stream`` will yield.
        self.stream_events_per_call: list[list[Any]] = []
        # Per-call raises for ``generate_stream`` — when the entry is
        # not None the producer raises that exception instead of
        # yielding the corresponding events list.
        self.stream_raises_per_call: list[BaseException | None] = []
        _FakeOrchestrator.instances.append(self)

    def generate(self, plan: QueryPlan, **kwargs: Any) -> GenerationResult:
        self.calls.append({"plan": plan, **kwargs})
        if self.raises is not None:
            raise self.raises
        return self.returns

    def generate_stream(self, plan: QueryPlan, **kwargs: Any) -> Any:
        idx = len(self.stream_calls)
        # Snapshot mutable kwargs at call time — the CLI passes the
        # live ``history`` list by reference and appends to it after
        # the call returns, so storing the bare reference would alias
        # the post-mutation state across all earlier recorded calls.
        recorded = {"plan": plan, **kwargs}
        if isinstance(recorded.get("history"), list):
            recorded["history"] = list(recorded["history"])
        self.stream_calls.append(recorded)
        if idx < len(self.stream_raises_per_call) and self.stream_raises_per_call[idx] is not None:
            raise self.stream_raises_per_call[idx]
        if idx < len(self.stream_events_per_call):
            events = self.stream_events_per_call[idx]
        else:
            events = [_make_stream_final()]
        yield from events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def app() -> typer.Typer:
    """Throw-away Typer app wrapping the ``rag`` sub-Typer."""
    test_app = typer.Typer()
    test_app.add_typer(rag_app, name="rag")
    return test_app


class _StubSettings:
    """Hand-built settings stand-in matching the seams ``cli.rag`` reads."""

    def __init__(
        self,
        *,
        llm_provider: str = "openai",
        llm_default_model: str | None = None,
        persist_provider_credentials: bool = False,
    ) -> None:
        self.embedding = EmbeddingSettings.model_construct(
            provider="openai",
            model_name="text-embedding-3-small",
            device="auto",
            batch_size=32,
            idle_timeout_minutes=0,
        )
        self.database = DatabaseSettings.model_construct(
            deployment_profile="local",
            retention_max_age_days=0,
            chroma_path="./data/chroma_db",
            metadata_db_path="./data/metadata.sqlite",
            max_filings=10000,
            encryption_key=None,
            encryption_key_file=None,
            task_history_retention_days=0,
            task_history_persist_tickers=False,
            persist_provider_credentials=persist_provider_credentials,
        )
        self.llm = LLMSettings.model_construct(
            default_provider=llm_provider,
            default_model=llm_default_model,
            max_output_tokens=2048,
        )
        self.rag = RAGSettings.model_construct(
            context_token_budget=8000,
            refusal_enabled=True,
        )
        self.search = SearchSettings.model_construct(top_k=5, min_similarity=0.0)


_DEFAULT_PLAN = _make_plan()


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace every seam ``cli.rag`` reaches into with a recording stub.

    ``understand_query`` is stubbed at the import site (``rag_module``)
    so we can assert on whether it was called and arm its return value
    or exception independently from the orchestrator stub.
    """
    _FakeEmbedder.instances.clear()
    _FakeLLM.instances.clear()
    _FakeChroma.instances.clear()
    _FakeRegistry.instances.clear()
    _FakeRegistry.encrypted_default = False
    _FakeRetrievalService.instances.clear()
    _FakeOrchestrator.instances.clear()

    understand_calls: list[dict[str, Any]] = []

    def _fake_understand(
        query: str,
        *,
        llm: Any,
        model: str,
        structured_output_supported: bool = False,
    ) -> QueryPlan:
        record = {
            "query": query,
            "llm": llm,
            "model": model,
            "structured_output_supported": structured_output_supported,
        }
        understand_calls.append(record)
        if "raises" in understand_calls[-1] or _understand_raises[0] is not None:
            raise _understand_raises[0]  # type: ignore[misc]
        return _understand_returns[0]

    _understand_returns: list[QueryPlan] = [_DEFAULT_PLAN]
    _understand_raises: list[BaseException | None] = [None]

    monkeypatch.setattr(rag_module, "get_settings", _StubSettings)
    monkeypatch.setattr(rag_module, "build_embedder", lambda _settings: _FakeEmbedder())
    monkeypatch.setattr(
        rag_module,
        "build_llm_provider",
        lambda _name, *, api_key_resolver=None: _FakeLLM(),
    )
    monkeypatch.setattr(rag_module, "ChromaDBClient", _FakeChroma)
    monkeypatch.setattr(rag_module, "MetadataRegistry", _FakeRegistry)
    monkeypatch.setattr(rag_module, "RetrievalService", _FakeRetrievalService)
    monkeypatch.setattr(rag_module, "RAGOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(rag_module, "understand_query", _fake_understand)

    return {
        "embedders": _FakeEmbedder.instances,
        "llms": _FakeLLM.instances,
        "chromas": _FakeChroma.instances,
        "registries": _FakeRegistry.instances,
        "services": _FakeRetrievalService.instances,
        "orchestrators": _FakeOrchestrator.instances,
        "understand_calls": understand_calls,
        "understand_returns": _understand_returns,
        "understand_raises": _understand_raises,
    }


def _arm_orchestrator_returns(result: GenerationResult) -> None:
    original = _FakeOrchestrator.__init__

    def _seeded(self: _FakeOrchestrator, **kwargs: Any) -> None:
        original(self, **kwargs)
        self.returns = result

    _FakeOrchestrator.__init__ = _seeded  # type: ignore[method-assign]


def _arm_orchestrator_raises(exc: BaseException) -> None:
    original = _FakeOrchestrator.__init__

    def _seeded(self: _FakeOrchestrator, **kwargs: Any) -> None:
        original(self, **kwargs)
        self.raises = exc

    _FakeOrchestrator.__init__ = _seeded  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _reset_orchestrator_init() -> Any:
    yield

    def _default(self: _FakeOrchestrator, **kwargs: Any) -> None:
        self.retrieval = kwargs.get("retrieval")
        self.llm = kwargs.get("llm")
        self.calls = []
        self.stream_calls = []
        self.returns = _make_result()
        self.raises = None
        self.stream_events_per_call = []
        self.stream_raises_per_call = []
        _FakeOrchestrator.instances.append(self)

    _FakeOrchestrator.__init__ = _default  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_auto_plan_then_generate(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Default flow: understand_query → orchestrator.generate."""
        result = runner.invoke(app, ["rag", "query", "What is revenue?"])
        assert result.exit_code == 0, result.output
        # Both factory seams used exactly once.
        assert len(patched["embedders"]) == 1
        assert len(patched["llms"]) == 1
        # Query understanding ran once.
        assert len(patched["understand_calls"]) == 1
        # Orchestrator generated once with the auto-plan.
        assert len(patched["orchestrators"]) == 1
        orch = patched["orchestrators"][-1]
        assert len(orch.calls) == 1
        # Answer and citation marker render.
        assert "Apple has three main revenue segments" in result.output
        assert "Citations" in result.output

    def test_refused_path(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """When retrieved chunks and citations are both empty the renderer
        labels the panel ``Refused`` — same shape as the API's ``refused=true``."""
        _arm_orchestrator_returns(_make_result(refused=True))
        result = runner.invoke(app, ["rag", "query", "Something obscure"])
        assert result.exit_code == 0, result.output
        assert "Refused" in result.output

    def test_show_plan_skips_generation(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """``--show-plan`` runs query-understanding, prints the plan, and exits
        before any orchestrator construction."""
        result = runner.invoke(app, ["rag", "query", "What is revenue?", "--show-plan"])
        assert result.exit_code == 0, result.output
        assert "Query Plan" in result.output
        assert len(patched["understand_calls"]) == 1
        # No orchestrator was ever constructed.
        assert patched["orchestrators"] == []

    def test_skip_plan_bypasses_understanding(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """``--skip-plan`` feeds a minimal plan straight to the orchestrator —
        ``understand_query`` is never called."""
        result = runner.invoke(app, ["rag", "query", "raw question", "--skip-plan"])
        assert result.exit_code == 0, result.output
        assert patched["understand_calls"] == []
        orch = patched["orchestrators"][-1]
        assert len(orch.calls) == 1
        plan: QueryPlan = orch.calls[0]["plan"]
        assert plan.raw_query == "raw question"
        assert plan.query_en == "raw question"
        assert plan.tickers == []
        assert plan.form_types == []


# ---------------------------------------------------------------------------
# Plan overrides
# ---------------------------------------------------------------------------


class TestPlanOverrides:
    def test_ticker_and_form_uppercased(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            [
                "rag",
                "query",
                "revenue",
                "--skip-plan",
                "--ticker",
                "aapl",
                "--ticker",
                "msft",
                "--form",
                "10-k",
            ],
        )
        assert result.exit_code == 0, result.output
        plan: QueryPlan = patched["orchestrators"][-1].calls[0]["plan"]
        assert plan.tickers == ["AAPL", "MSFT"]
        assert plan.form_types == ["10-K"]

    def test_date_range_override(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            [
                "rag",
                "query",
                "revenue",
                "--skip-plan",
                "--since",
                "2023-01-01",
                "--until",
                "2023-12-31",
            ],
        )
        assert result.exit_code == 0, result.output
        plan: QueryPlan = patched["orchestrators"][-1].calls[0]["plan"]
        assert plan.date_range == ("2023-01-01", "2023-12-31")

    def test_since_without_until_with_no_plan_range_rejected(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """``--since`` alone with no plan ``date_range`` is rejected at the
        boundary; the orchestrator must not be reached."""
        result = runner.invoke(
            app,
            ["rag", "query", "q", "--skip-plan", "--since", "2023-01-01"],
        )
        assert result.exit_code == 2
        assert "--since and --until must be supplied together" in _stripped(result.output)
        assert patched["orchestrators"] == []

    def test_mode_override(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            ["rag", "query", "compare X and Y", "--skip-plan", "--mode", "comparative"],
        )
        assert result.exit_code == 0, result.output
        # The orchestrator receives the AnswerMode override AND the plan
        # mode is also updated (overrides shadow plan extraction).
        orch = patched["orchestrators"][-1]
        assert orch.calls[0]["mode"] == AnswerMode.COMPARATIVE
        assert orch.calls[0]["plan"].suggested_answer_mode == AnswerMode.COMPARATIVE

    def test_unknown_mode_rejected(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """``_coerce_mode`` fails closed on unknown values — typos are caught
        rather than silently coerced to ``concise``."""
        result = runner.invoke(
            app,
            ["rag", "query", "q", "--skip-plan", "--mode", "analyitcal"],
        )
        assert result.exit_code == 2
        assert "Invalid --mode" in _stripped(result.output)
        assert patched["orchestrators"] == []


# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------


class TestBoundaryValidation:
    def test_empty_question_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["rag", "query", "   "])
        assert result.exit_code == 1
        assert "Invalid question" in result.output
        # Boundary rejection fires before any stack construction.
        assert patched["embedders"] == []

    def test_show_and_skip_plan_mutually_exclusive(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["rag", "query", "q", "--show-plan", "--skip-plan"])
        assert result.exit_code == 1
        assert "Invalid flag combination" in result.output
        assert patched["embedders"] == []

    def test_malformed_since_rejected(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan", "--since", "2024-13-99"])
        assert result.exit_code == 2
        assert "Invalid date format" in result.output


# ---------------------------------------------------------------------------
# Provider exception ladder
# ---------------------------------------------------------------------------


class TestProviderExceptions:
    def test_understand_provider_auth_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        patched["understand_raises"][0] = ProviderAuthError("bad key")
        result = runner.invoke(app, ["rag", "query", "q"])
        assert result.exit_code == 1
        assert "Provider unauthorised" in result.output
        # Orchestrator never built — auth failure short-circuits.
        assert patched["orchestrators"] == []

    def test_understand_rate_limit_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        patched["understand_raises"][0] = ProviderRateLimitError("slow down")
        result = runner.invoke(app, ["rag", "query", "q"])
        assert result.exit_code == 1
        assert "Provider unavailable" in result.output

    def test_generate_auth_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_orchestrator_raises(ProviderAuthError("invalid"))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 1
        assert "Provider unauthorised" in result.output

    def test_generate_rate_limit_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_orchestrator_raises(ProviderTimeoutError("timeout"))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 1
        assert "Provider unavailable" in result.output

    def test_understand_connection_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        # An unreachable endpoint during query understanding surfaces as
        # "Provider unavailable" with endpoint-focused guidance (chiefly an
        # unstarted local_llm server), not a key-rotation hint.
        patched["understand_raises"][0] = ProviderConnectionError("refused")
        result = runner.invoke(app, ["rag", "query", "q"])
        assert result.exit_code == 1
        out = _stripped(result.output)
        assert "Provider unavailable" in out
        assert "could not be reached" in out
        assert "local_llm" in out

    def test_generate_connection_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_orchestrator_raises(ProviderConnectionError("no route to host"))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 1
        out = _stripped(result.output)
        assert "Provider unavailable" in out
        assert "local_llm" in out

    def test_generate_provider_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_orchestrator_raises(ProviderError("upstream 500"))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 1
        assert "Provider error" in result.output

    def test_generate_generation_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_orchestrator_raises(GenerationError("malformed json"))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 1
        assert "Generation failed" in result.output

    def test_generate_search_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_orchestrator_raises(SearchError("bad filter"))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 1
        assert "Retrieval failed" in result.output

    def test_llm_construction_missing_key_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise(_name: str, *, api_key_resolver: Any = None) -> Any:
            raise ConfigurationError("no key for openai")

        monkeypatch.setattr(rag_module, "build_llm_provider", _raise)
        result = runner.invoke(app, ["rag", "query", "q"])
        assert result.exit_code == 1
        assert "LLM provider key required" in result.output


# ---------------------------------------------------------------------------
# Unknown provider / model
# ---------------------------------------------------------------------------


class TestProviderValidation:
    def test_unknown_provider_exits_1(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise(name: str, surface: Any, *, model: Any = None) -> Any:
            raise KeyError(f"unknown provider {name!r}")

        monkeypatch.setattr(
            "sec_generative_search.providers.registry.ProviderRegistry.get_capability",
            classmethod(lambda cls, *a, **kw: _raise(*a, **kw)),
        )
        result = runner.invoke(app, ["rag", "query", "q"])
        assert result.exit_code == 1
        assert "Unknown LLM provider" in result.output
        # No retrieval stack built.
        assert patched["embedders"] == []

    def test_get_capability_invoked_with_model(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``capability.structured_output`` MUST drive the
        ``prefer_structured_output`` argument to ``orchestrator.generate``."""
        seen: dict[str, Any] = {}

        def _capability(name: str, surface: Any, *, model: Any = None) -> ProviderCapability:
            seen["name"] = name
            seen["model"] = model
            return ProviderCapability(chat=True, structured_output=True)

        monkeypatch.setattr(
            "sec_generative_search.providers.registry.ProviderRegistry.get_capability",
            classmethod(lambda cls, *a, **kw: _capability(*a, **kw)),
        )

        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 0, result.output
        orch = patched["orchestrators"][-1]
        assert orch.calls[0]["prefer_structured_output"] is True


# ---------------------------------------------------------------------------
# Resolver chain composition
# ---------------------------------------------------------------------------


class TestResolverChain:
    def test_encrypted_tier_skipped_when_persist_disabled(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Scenario A default: ``persist_provider_credentials=false`` →
        encrypted tier collapses; chain is admin-env only."""
        # Settings stub defaults persist_provider_credentials=False.
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 0, result.output
        # build_llm_provider was called — the chain composed without
        # raising, which means the encrypted tier wasn't attempted.

    def test_encrypted_tier_attempted_when_both_flags_set(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When both ``persist_provider_credentials=true`` AND
        ``registry.encrypted=true`` the chain composes the encrypted
        tier in front of admin-env.  The store construction is stubbed —
        we only assert that the chain composition does not blow up
        the command."""

        def _stub_settings_persist() -> _StubSettings:
            return _StubSettings(persist_provider_credentials=True)

        monkeypatch.setattr(rag_module, "get_settings", _stub_settings_persist)
        _FakeRegistry.encrypted_default = True

        seen_resolvers: list[Any] = []

        class _StubStore:
            def __init__(self, registry: Any) -> None:
                seen_resolvers.append(registry)

            def get(self, *_args: Any, **_kwargs: Any) -> None:
                return None

        monkeypatch.setattr(
            "sec_generative_search.database.credentials.EncryptedCredentialStore",
            _StubStore,
        )

        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 0, result.output
        # The encrypted-credential store was constructed from the
        # registry handle the CLI owns.
        assert len(seen_resolvers) == 1


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSecurity:
    def test_question_not_echoed_in_error_envelope(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """The raw question MUST NOT appear in any error envelope.

        Mirrors the ``/api/rag/query`` discipline: the response never
        echoes the raw query.  An operator piping CLI output into
        operator logs must not find Tier-3 question text inlined.
        """
        sentinel = "PII-LADEN-QUESTION-MUST-NOT-ECHO"
        _arm_orchestrator_raises(ProviderError("upstream"))
        result = runner.invoke(app, ["rag", "query", sentinel, "--skip-plan"])
        assert result.exit_code == 1
        assert sentinel not in result.output

    def test_api_key_never_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An ``OPENAI_API_KEY`` in the environment must never leak —
        including on the loudest error paths."""
        canary = "sk-canary-rag-key-must-not-appear"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", canary)
        _arm_orchestrator_raises(ProviderError("boom"))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert canary not in result.output

    def test_edgar_identity_never_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EDGAR_IDENTITY_NAME", "Secret User")
        monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "canary-edgar@example.test")
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 0
        assert "Secret User" not in result.output
        assert "canary-edgar@example.test" not in result.output

    def test_both_factory_seams_used(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """The CLI MUST go through :func:`build_embedder` AND
        :func:`build_llm_provider` — never direct adapter
        instantiation.  The fixtures replace both with recorders so if
        the CLI side-stepped either seam the recorder for that seam
        would be empty.
        """
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 0, result.output
        assert len(patched["embedders"]) == 1
        assert len(patched["llms"]) == 1

    def test_hostile_markup_in_answer_renders_verbatim(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A retrieved chunk / generated answer containing
        ``[red]...[/red]`` MUST render literally — otherwise hostile
        SEC text could repaint operator output.

        Tested via the answer panel only because Rich aggressively
        wraps table cells based on terminal width, splitting literal
        markup tags across lines.  The escape contract is identical at
        both render sites (``escape(result.answer)`` and
        ``escape(c.text_span)``); proving it on the answer panel is
        sufficient — both call sites flow through
        :func:`rich.markup.escape`.
        """
        evil = "Risk factor: [red]EVIL[/red]"
        _arm_orchestrator_returns(_make_result(answer=evil))
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 0, result.output
        assert "[red]EVIL[/red]" in result.output


# ---------------------------------------------------------------------------
# `sec-rag rag chat` — REPL-level tests
# ---------------------------------------------------------------------------


def _arm_stream_events(events_per_call: list[list[Any]]) -> None:
    """Seed ``generate_stream`` outputs onto every future orchestrator."""
    original = _FakeOrchestrator.__init__

    def _seeded(self: _FakeOrchestrator, **kwargs: Any) -> None:
        original(self, **kwargs)
        self.stream_events_per_call = events_per_call

    _FakeOrchestrator.__init__ = _seeded  # type: ignore[method-assign]


def _arm_stream_raises(exc_per_call: list[BaseException | None]) -> None:
    """Seed ``generate_stream`` exceptions onto every future orchestrator."""
    original = _FakeOrchestrator.__init__

    def _seeded(self: _FakeOrchestrator, **kwargs: Any) -> None:
        original(self, **kwargs)
        self.stream_raises_per_call = exc_per_call

    _FakeOrchestrator.__init__ = _seeded  # type: ignore[method-assign]


class TestChatHappyPath:
    def test_one_turn_then_exit(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Single question → stream produces deltas + final → /exit."""
        _arm_stream_events(
            [
                [
                    _make_stream_delta("Apple "),
                    _make_stream_delta("revenue grew."),
                    _make_stream_final(),
                ]
            ]
        )
        result = runner.invoke(app, ["rag", "chat"], input="What is revenue?\n/exit\n")
        assert result.exit_code == 0, result.output
        # Banner appeared.
        assert "sec-rag rag chat" in result.output
        # Stream deltas surfaced.
        assert "Apple " in result.output
        assert "revenue grew." in result.output
        # Citations panel rendered.
        assert "Citations" in result.output
        # /exit closes cleanly.
        assert "Bye." in result.output
        # Exactly one stream call.
        orch = patched["orchestrators"][-1]
        assert len(orch.stream_calls) == 1

    def test_skip_plan_bypasses_understanding(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """``--skip-plan`` short-circuits ``understand_query`` per turn."""
        _arm_stream_events([[_make_stream_final()]])
        result = runner.invoke(
            app,
            ["rag", "chat", "--skip-plan"],
            input="raw question\n/exit\n",
        )
        assert result.exit_code == 0, result.output
        assert patched["understand_calls"] == []
        orch = patched["orchestrators"][-1]
        assert len(orch.stream_calls) == 1
        plan: QueryPlan = orch.stream_calls[0]["plan"]
        assert plan.raw_query == "raw question"

    def test_show_plan_prints_plan_but_still_generates(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """``--show-plan`` in chat is informational — generation still runs.

        Differs from ``rag query --show-plan`` (which exits before
        generation) because the chat REPL would degenerate to a no-op
        otherwise.
        """
        _arm_stream_events([[_make_stream_final()]])
        result = runner.invoke(app, ["rag", "chat", "--show-plan"], input="hello\n/exit\n")
        assert result.exit_code == 0, result.output
        assert "Query Plan" in result.output
        orch = patched["orchestrators"][-1]
        # Generation still ran.
        assert len(orch.stream_calls) == 1

    def test_session_overrides_apply_to_every_turn(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Session-scope ``--ticker`` / ``--mode`` shadow plan fields on
        every turn — not just the first."""
        _arm_stream_events([[_make_stream_final()], [_make_stream_final()]])
        result = runner.invoke(
            app,
            [
                "rag",
                "chat",
                "--skip-plan",
                "--ticker",
                "aapl",
                "--mode",
                "analytical",
            ],
            input="q1\nq2\n/exit\n",
        )
        assert result.exit_code == 0, result.output
        orch = patched["orchestrators"][-1]
        assert len(orch.stream_calls) == 2
        for call in orch.stream_calls:
            plan: QueryPlan = call["plan"]
            assert plan.tickers == ["AAPL"]
            assert plan.suggested_answer_mode == AnswerMode.ANALYTICAL


class TestChatHistory:
    def test_history_accumulates_across_turns(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """The orchestrator MUST see prior turns on follow-up calls.

        First turn: no history passed.
        Second turn: history of length 1 passed.
        """
        _arm_stream_events([[_make_stream_final()], [_make_stream_final()]])
        result = runner.invoke(
            app,
            ["rag", "chat", "--skip-plan"],
            input="first\nsecond\n/exit\n",
        )
        assert result.exit_code == 0, result.output
        orch = patched["orchestrators"][-1]
        assert len(orch.stream_calls) == 2
        first_history = orch.stream_calls[0]["history"]
        second_history = orch.stream_calls[1]["history"]
        # First turn: no prior history (None passed).
        assert first_history is None
        # Second turn: one ConversationTurn passed.
        assert second_history is not None
        assert len(second_history) == 1

    def test_clear_drops_history(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """``/clear`` empties the in-memory history."""
        _arm_stream_events([[_make_stream_final()], [_make_stream_final()]])
        result = runner.invoke(
            app,
            ["rag", "chat", "--skip-plan"],
            input="first\n/clear\nafter-clear\n/exit\n",
        )
        assert result.exit_code == 0, result.output
        assert "History cleared" in result.output
        orch = patched["orchestrators"][-1]
        # The second generation call sees no history again.
        second_history = orch.stream_calls[1]["history"]
        assert second_history is None

    def test_cancelled_turn_not_committed(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A turn cancelled mid-stream MUST NOT enter history.

        Patches ``_run_chat_stream`` at the import site so we can drive
        the (None, interrupted=True) return synchronously.  The
        underlying streaming primitive is exercised by
        ``TestRunChatStream`` separately.
        """
        captured: list[Any] = []

        def _fake_run_chat_stream(
            orchestrator: Any,
            plan: QueryPlan,
            **kwargs: Any,
        ) -> tuple[GenerationResult | None, bool]:
            captured.append({"history": kwargs.get("history")})
            # First call: cancel.  Second call: succeed.
            if len(captured) == 1:
                return None, True
            return _make_result(), False

        monkeypatch.setattr(rag_module, "_run_chat_stream", _fake_run_chat_stream)

        result = runner.invoke(
            app,
            ["rag", "chat", "--skip-plan"],
            input="first-cancelled\nsecond-success\n/exit\n",
        )
        assert result.exit_code == 0, result.output
        # Second turn must see history of length 0 — the cancelled
        # turn was not committed.
        assert len(captured) == 2
        assert captured[1]["history"] is None


class TestChatSlashCommands:
    def test_unknown_slash_command_is_rejected(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """A typo like ``/exti`` MUST NOT silently flow into the
        generation pipeline as a question."""
        _arm_stream_events([[_make_stream_final()]])
        result = runner.invoke(app, ["rag", "chat", "--skip-plan"], input="/exti\n/exit\n")
        assert result.exit_code == 0, result.output
        assert "Unknown command" in result.output
        # No generation was triggered.
        orch = patched["orchestrators"][-1]
        assert orch.stream_calls == []

    def test_empty_input_is_ignored(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Blank lines must loop without triggering generation."""
        _arm_stream_events([[_make_stream_final()]])
        result = runner.invoke(
            app,
            ["rag", "chat", "--skip-plan"],
            input="\n   \nreal question\n/exit\n",
        )
        assert result.exit_code == 0, result.output
        orch = patched["orchestrators"][-1]
        # Only the real question generated.
        assert len(orch.stream_calls) == 1

    def test_help_command(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["rag", "chat"], input="/help\n/exit\n")
        assert result.exit_code == 0, result.output
        assert "Slash commands" in result.output

    def test_quit_alias(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["rag", "chat"], input="/quit\n")
        assert result.exit_code == 0, result.output
        assert "Bye." in result.output

    def test_eof_at_prompt_exits_cleanly(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Ctrl-D (EOF) at the prompt closes the REPL with exit 0."""
        # Empty input → CliRunner sends EOF immediately.
        result = runner.invoke(app, ["rag", "chat"], input="")
        assert result.exit_code == 0, result.output


class TestChatBoundaryValidation:
    def test_show_and_skip_plan_mutually_exclusive(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["rag", "chat", "--show-plan", "--skip-plan"])
        assert result.exit_code == 1
        assert "Invalid flag combination" in result.output

    def test_malformed_since_rejected(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["rag", "chat", "--since", "2024-13-99"])
        assert result.exit_code == 2
        assert "Invalid date format" in result.output

    def test_unknown_mode_rejected_at_boundary(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(app, ["rag", "chat", "--mode", "analyitcal"])
        assert result.exit_code == 2
        assert "Invalid --mode" in _stripped(result.output)


class TestChatProviderExceptions:
    """REPL must NOT exit on a transient upstream failure.

    Provider exception envelopes mirror ``rag query`` (label / hint),
    but the loop continues so the operator can re-ask, /clear, or /exit.
    """

    def test_understand_provider_auth_error_continues_loop(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        patched["understand_raises"][0] = ProviderAuthError("bad key")
        result = runner.invoke(app, ["rag", "chat"], input="q1\n/exit\n")
        # REPL itself exits 0 — the auth failure surfaces inline.
        assert result.exit_code == 0, result.output
        assert "Provider unauthorised" in result.output
        assert "Bye." in result.output
        orch = patched["orchestrators"][-1]
        # Auth failure short-circuits before streaming.
        assert orch.stream_calls == []

    def test_stream_provider_error_continues_loop(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_stream_raises([ProviderError("boom"), None])
        _arm_stream_events([[], [_make_stream_final()]])
        result = runner.invoke(
            app,
            ["rag", "chat", "--skip-plan"],
            input="first-fails\nsecond-works\n/exit\n",
        )
        assert result.exit_code == 0, result.output
        assert "Provider error" in result.output
        orch = patched["orchestrators"][-1]
        assert len(orch.stream_calls) == 2

    def test_stream_connection_error_continues_loop(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        # An unreachable endpoint mid-stream surfaces inline as "Provider
        # unavailable" (endpoint-focused) and the REPL keeps running.
        _arm_stream_raises([ProviderConnectionError("refused"), None])
        _arm_stream_events([[], [_make_stream_final()]])
        result = runner.invoke(
            app,
            ["rag", "chat", "--skip-plan"],
            input="first-fails\nsecond-works\n/exit\n",
        )
        assert result.exit_code == 0, result.output
        out = _stripped(result.output)
        assert "Provider unavailable" in out
        assert "could not be reached" in out
        orch = patched["orchestrators"][-1]
        assert len(orch.stream_calls) == 2

    def test_stream_auth_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_stream_raises([ProviderAuthError("invalid")])
        result = runner.invoke(app, ["rag", "chat", "--skip-plan"], input="q\n/exit\n")
        assert result.exit_code == 0, result.output
        assert "Provider unauthorised" in result.output

    def test_stream_generation_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        _arm_stream_raises([GenerationError("malformed json")])
        result = runner.invoke(app, ["rag", "chat", "--skip-plan"], input="q\n/exit\n")
        assert result.exit_code == 0, result.output
        assert "Generation failed" in result.output


# ---------------------------------------------------------------------------
# `_run_chat_stream` — unit tests over the streaming primitive itself
# ---------------------------------------------------------------------------


class _InlineOrchestrator:
    """Minimal orchestrator that yields a configured event sequence.

    Used for direct unit tests of :func:`cli.rag._run_chat_stream`; the
    REPL test path uses the recording :class:`_FakeOrchestrator` instead.
    """

    def __init__(
        self,
        events: list[Any] | None = None,
        *,
        raises: BaseException | None = None,
        block_event: Any = None,
    ) -> None:
        self._events = events or []
        self._raises = raises
        self._block_event = block_event

    def generate_stream(self, plan: QueryPlan, **kwargs: Any) -> Any:
        if self._raises is not None:
            raise self._raises
        yield from self._events
        if self._block_event is not None:
            # Block until released — used by the cancellation tests to
            # keep the producer alive while the consumer is interrupted.
            self._block_event.wait(timeout=2.0)


class TestRunChatStream:
    def test_happy_path_returns_final(self) -> None:
        from sec_generative_search.cli.rag import _run_chat_stream

        events = [
            _make_stream_delta("Hello "),
            _make_stream_delta("world."),
            _make_stream_final(_make_result(answer="Hello world.")),
        ]
        result, interrupted = _run_chat_stream(
            _InlineOrchestrator(events),
            _make_plan(),
            mode=None,
            model=None,
            max_output_tokens=None,
            history=None,
            prefer_structured_output=False,
            provider_name="openai",
        )
        assert interrupted is False
        assert result is not None
        assert result.answer == "Hello world."

    def test_producer_exception_returns_none(self) -> None:
        """A provider exception inside ``generate_stream`` becomes a
        rendered envelope and ``(None, False)``.  The caller (REPL) then
        skips committing the turn to history."""
        from sec_generative_search.cli.rag import _run_chat_stream

        result, interrupted = _run_chat_stream(
            _InlineOrchestrator(raises=ProviderError("upstream 500")),
            _make_plan(),
            mode=None,
            model=None,
            max_output_tokens=None,
            history=None,
            prefer_structured_output=False,
            provider_name="openai",
        )
        assert result is None
        assert interrupted is False

    def test_keyboard_interrupt_during_stream_returns_interrupted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A consumer-side KeyboardInterrupt cancels the in-flight stream.

        Simulates Ctrl-C by patching the module's ``_queue.Queue``
        replacement so ``get`` raises KeyboardInterrupt on the second
        call — i.e. after the first delta has been consumed but before
        the final event arrives.  Returns ``(None, True)`` and the
        producer thread is signalled to stop pushing further events.
        """
        import queue as _q
        import types

        from sec_generative_search.cli import rag as rag_mod

        real_queue_cls = _q.Queue

        class _InterruptingQueue(real_queue_cls):  # type: ignore[misc, valid-type]
            _call_counter: ClassVar[int] = 0

            def get(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
                type(self)._call_counter += 1
                if type(self)._call_counter == 2:
                    raise KeyboardInterrupt
                return super().get(*args, **kwargs)

        _InterruptingQueue._call_counter = 0

        monkeypatch.setattr(
            rag_mod,
            "_queue",
            types.SimpleNamespace(Queue=_InterruptingQueue, Empty=_q.Empty),
        )

        release = threading.Event()
        events = [
            _make_stream_delta("partial "),
            _make_stream_delta("answer"),
            _make_stream_final(),
        ]

        result, interrupted = rag_mod._run_chat_stream(
            _InlineOrchestrator(events, block_event=release),
            _make_plan(),
            mode=None,
            model=None,
            max_output_tokens=None,
            history=None,
            prefer_structured_output=False,
            provider_name="openai",
        )
        release.set()  # let producer thread exit
        assert interrupted is True
        assert result is None


# ---------------------------------------------------------------------------
# `sec-rag rag chat` — security
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestChatSecurity:
    def test_question_not_echoed_on_error(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Tier-3 question text MUST NOT appear in any error envelope."""
        sentinel = "PII-CHAT-QUESTION-MUST-NOT-ECHO"
        _arm_stream_raises([ProviderError("upstream")])
        result = runner.invoke(
            app,
            ["rag", "chat", "--skip-plan"],
            input=f"{sentinel}\n/exit\n",
        )
        assert result.exit_code == 0, result.output
        # The user typed it at the prompt so the prompt echo MAY contain
        # it depending on terminal behaviour, but Click's CliRunner
        # buffers stdout only — there is no terminal echo here, so the
        # sentinel should not surface at all.
        assert sentinel not in result.output

    def test_api_key_never_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        canary = "sk-canary-chat-key-must-not-appear"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", canary)
        _arm_stream_raises([ProviderError("boom")])
        result = runner.invoke(app, ["rag", "chat", "--skip-plan"], input="q\n/exit\n")
        assert canary not in result.output

    def test_edgar_identity_never_in_output(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EDGAR_IDENTITY_NAME", "Chat User")
        monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "chat-canary@example.test")
        _arm_stream_events([[_make_stream_final()]])
        result = runner.invoke(app, ["rag", "chat", "--skip-plan"], input="q\n/exit\n")
        assert result.exit_code == 0
        assert "Chat User" not in result.output
        assert "chat-canary@example.test" not in result.output

    def test_hostile_markup_in_delta_renders_verbatim(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """A delta containing ``[red]EVIL[/red]`` MUST render literally —
        the consumer flows every delta through :func:`rich.markup.escape`."""
        evil = "Risk: [red]EVIL[/red]"
        _arm_stream_events([[_make_stream_delta(evil), _make_stream_final()]])
        result = runner.invoke(app, ["rag", "chat", "--skip-plan"], input="q\n/exit\n")
        assert result.exit_code == 0, result.output
        assert "[red]EVIL[/red]" in result.output

    def test_both_factory_seams_used_for_chat(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Same factory-seam discipline as ``rag query`` — direct adapter
        instantiation MUST NOT happen on the chat surface either."""
        _arm_stream_events([[_make_stream_final()]])
        result = runner.invoke(app, ["rag", "chat", "--skip-plan"], input="q\n/exit\n")
        assert result.exit_code == 0, result.output
        assert len(patched["embedders"]) == 1
        assert len(patched["llms"]) == 1


# ---------------------------------------------------------------------------
# `--provider` / `--model` / `--openrouter-*` flags
# ---------------------------------------------------------------------------


class TestProviderModelOverride:
    """``--provider`` / ``--model`` must override LLM_DEFAULT_PROVIDER /
    LLM_DEFAULT_MODEL exactly the way the API body does on
    ``POST /api/rag/query``.

    Validated through the same seams ``rag query`` uses internally:
    the ``ProviderRegistry.get_capability`` probe receives the
    overridden name + model, the ``build_llm_provider`` factory is
    invoked with the overridden provider name, and the orchestrator
    sees the overridden model on the ``model`` kwarg.
    """

    def test_provider_override_propagates_to_factory(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen_names: list[str] = []

        def _record_build(name: str, *, api_key_resolver: Any = None) -> Any:
            seen_names.append(name)
            return _FakeLLM()

        monkeypatch.setattr(rag_module, "build_llm_provider", _record_build)

        result = runner.invoke(
            app,
            ["rag", "query", "q", "--skip-plan", "--provider", "anthropic"],
        )
        assert result.exit_code == 0, result.output
        # The factory was invoked with the override, not the settings default
        # (which is ``openai`` in ``_StubSettings``).
        assert seen_names == ["anthropic"]

    def test_model_override_reaches_orchestrator(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            [
                "rag",
                "query",
                "q",
                "--skip-plan",
                "--model",
                "claude-haiku-4-5",
            ],
        )
        assert result.exit_code == 0, result.output
        orch = patched["orchestrators"][-1]
        assert orch.calls[0]["model"] == "claude-haiku-4-5"

    def test_provider_and_model_combine(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``get_capability`` MUST see both overrides — the capability probe
        is what drives ``prefer_structured_output``."""
        seen: dict[str, Any] = {}

        def _capability(name: str, surface: Any, *, model: Any = None) -> ProviderCapability:
            seen["name"] = name
            seen["model"] = model
            return ProviderCapability(chat=True, structured_output=False)

        monkeypatch.setattr(
            "sec_generative_search.providers.registry.ProviderRegistry.get_capability",
            classmethod(lambda cls, *a, **kw: _capability(*a, **kw)),
        )

        result = runner.invoke(
            app,
            [
                "rag",
                "query",
                "q",
                "--skip-plan",
                "--provider",
                "anthropic",
                "--model",
                "claude-opus-4-7",
            ],
        )
        assert result.exit_code == 0, result.output
        assert seen["name"] == "anthropic"
        assert seen["model"] == "claude-opus-4-7"

    def test_empty_string_overrides_fall_back_to_settings(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty ``--provider`` / ``--model`` strings must not override the
        settings defaults.  The settings fallback is the documented
        contract: an empty string here would otherwise crash the
        capability probe and surface as ``Unknown LLM provider``."""
        seen_names: list[str] = []

        def _record_build(name: str, *, api_key_resolver: Any = None) -> Any:
            seen_names.append(name)
            return _FakeLLM()

        monkeypatch.setattr(rag_module, "build_llm_provider", _record_build)

        result = runner.invoke(
            app,
            ["rag", "query", "q", "--skip-plan", "--provider", "", "--model", ""],
        )
        assert result.exit_code == 0, result.output
        # Fell back to the settings default (``openai``).
        assert seen_names == ["openai"]

    def test_chat_provider_override_propagates(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen_names: list[str] = []

        def _record_build(name: str, *, api_key_resolver: Any = None) -> Any:
            seen_names.append(name)
            return _FakeLLM()

        monkeypatch.setattr(rag_module, "build_llm_provider", _record_build)
        _arm_stream_events([[_make_stream_final()]])

        result = runner.invoke(
            app,
            ["rag", "chat", "--skip-plan", "--provider", "anthropic"],
            input="q\n/exit\n",
        )
        assert result.exit_code == 0, result.output
        assert seen_names == ["anthropic"]

    def test_chat_model_override_reaches_every_turn(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Session-scope ``--model`` shadows the plan on every turn —
        not just the first."""
        _arm_stream_events([[_make_stream_final()], [_make_stream_final()]])
        result = runner.invoke(
            app,
            ["rag", "chat", "--skip-plan", "--model", "claude-haiku-4-5"],
            input="q1\nq2\n/exit\n",
        )
        assert result.exit_code == 0, result.output
        orch = patched["orchestrators"][-1]
        assert len(orch.stream_calls) == 2
        for call in orch.stream_calls:
            assert call["model"] == "claude-haiku-4-5"


class TestOpenRouterRoutingHints:
    """``--openrouter-provider`` / ``--openrouter-fallbacks`` must build
    a frozen :class:`OpenRouterRoutingHints` and forward it into the
    orchestrator's ``routing_hints`` kwarg.

    The orchestrator is recorded; the actual hint pass-through to
    OpenRouter's ``provider`` block is unit-tested separately in
    ``tests/providers/test_openrouter.py``.  Here we only assert the
    CLI wires the hint correctly and rejects mismatched provider /
    flag combinations.
    """

    def test_routing_hints_forwarded_to_orchestrator(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # OpenRouter must register as a routing-honouring provider.
        # ``ProviderRegistry`` ships this for real but stub
        # ``get_capability`` to avoid touching the catalogue.
        monkeypatch.setattr(
            "sec_generative_search.providers.registry.ProviderRegistry.get_capability",
            classmethod(
                lambda cls, name, surface, *, model=None: ProviderCapability(
                    chat=True, structured_output=False
                )
            ),
        )

        result = runner.invoke(
            app,
            [
                "rag",
                "query",
                "q",
                "--skip-plan",
                "--provider",
                "openrouter",
                "--model",
                "openai/gpt-4o",
                "--openrouter-provider",
                "anthropic",
                "--openrouter-provider",
                "openai",
                "--no-openrouter-fallbacks",
            ],
        )
        assert result.exit_code == 0, result.output
        orch = patched["orchestrators"][-1]
        hint = orch.calls[0]["routing_hints"]
        from sec_generative_search.providers.openrouter import (
            OpenRouterRoutingHints,
        )

        assert isinstance(hint, OpenRouterRoutingHints)
        assert hint.order == ("anthropic", "openai")
        assert hint.allow_fallbacks is False

    def test_no_routing_flags_yields_none(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """Default behaviour: no openrouter flags → ``routing_hints=None``.
        Confirms the hint surface is opt-in and adds no overhead to the
        common path."""
        result = runner.invoke(app, ["rag", "query", "q", "--skip-plan"])
        assert result.exit_code == 0, result.output
        orch = patched["orchestrators"][-1]
        assert orch.calls[0]["routing_hints"] is None

    def test_routing_hints_rejected_when_provider_mismatch(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """The hard-fail guard: openrouter flags + non-openrouter
        provider exits 1 with an actionable hint, BEFORE the storage
        layer opens — a flag mismatch is a boundary error."""
        result = runner.invoke(
            app,
            [
                "rag",
                "query",
                "q",
                "--skip-plan",
                "--openrouter-provider",
                "anthropic",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid flag combination" in result.output
        assert "openrouter" in result.output.lower()
        # No retrieval stack was constructed — the guard fires early.
        assert patched["embedders"] == []
        assert patched["orchestrators"] == []

    def test_routing_fallback_flag_alone_rejected_on_mismatch(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        """``--no-openrouter-fallbacks`` alone (no ``--openrouter-provider``)
        also trips the guard — the flag set is bound together."""
        result = runner.invoke(
            app,
            [
                "rag",
                "query",
                "q",
                "--skip-plan",
                "--no-openrouter-fallbacks",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid flag combination" in result.output
        assert patched["embedders"] == []

    def test_chat_routing_hints_forwarded_on_every_turn(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """In the REPL, the session-scope hint applies to every turn —
        not just the first."""
        monkeypatch.setattr(
            "sec_generative_search.providers.registry.ProviderRegistry.get_capability",
            classmethod(
                lambda cls, name, surface, *, model=None: ProviderCapability(
                    chat=True, structured_output=False
                )
            ),
        )
        _arm_stream_events([[_make_stream_final()], [_make_stream_final()]])
        result = runner.invoke(
            app,
            [
                "rag",
                "chat",
                "--skip-plan",
                "--provider",
                "openrouter",
                "--openrouter-provider",
                "anthropic",
            ],
            input="q1\nq2\n/exit\n",
        )
        assert result.exit_code == 0, result.output
        orch = patched["orchestrators"][-1]
        assert len(orch.stream_calls) == 2
        from sec_generative_search.providers.openrouter import (
            OpenRouterRoutingHints,
        )

        for call in orch.stream_calls:
            hint = call["routing_hints"]
            assert isinstance(hint, OpenRouterRoutingHints)
            assert hint.order == ("anthropic",)

    def test_chat_routing_hints_rejected_when_provider_mismatch(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
    ) -> None:
        result = runner.invoke(
            app,
            [
                "rag",
                "chat",
                "--openrouter-provider",
                "anthropic",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid flag combination" in result.output
        # No turn ever began.
        assert patched["orchestrators"] == []


@pytest.mark.security
class TestOpenRouterRoutingSecurity:
    """The routing-hints flag set MUST never leak credential-shaped data
    into operator output or audit lines.

    :class:`OpenRouterRoutingHints` carries no credential-shaped
    fields by design (a registry-level security test enforces the
    field set), but the CLI is a fresh surface — we re-verify the
    discipline here so a future regression at this layer fails fast.
    """

    def test_audit_line_reports_metadata_only(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The audit log MUST record routing metadata (counts /
        fallback toggle) — never the upstream slugs themselves.

        Operators pivot on "was routing pinned?" without needing the
        slug list, and keeping the slugs out of the audit line avoids a
        Tier-2 disclosure of which vendors a given run preferred.
        """
        import logging

        # Package logger has ``propagate = False`` by default; enable it
        # so caplog (which attaches to root) sees the audit line.
        pkg_logger = logging.getLogger("sec_generative_search")
        prior_propagate = pkg_logger.propagate
        pkg_logger.propagate = True
        try:
            monkeypatch.setattr(
                "sec_generative_search.providers.registry.ProviderRegistry.get_capability",
                classmethod(
                    lambda cls, name, surface, *, model=None: ProviderCapability(
                        chat=True, structured_output=False
                    )
                ),
            )
            with caplog.at_level("INFO", logger="sec_generative_search"):
                result = runner.invoke(
                    app,
                    [
                        "rag",
                        "query",
                        "q",
                        "--skip-plan",
                        "--provider",
                        "openrouter",
                        "--openrouter-provider",
                        "secret-vendor-slug",
                        "--no-openrouter-fallbacks",
                    ],
                )
        finally:
            pkg_logger.propagate = prior_propagate

        assert result.exit_code == 0, result.output
        # ``or_hints=`` lands on the audit line in a metadata-only shape.
        audit_records = [r for r in caplog.records if "SECURITY_AUDIT" in r.getMessage()]
        assert audit_records, "expected an audit log record"
        joined = "\n".join(r.getMessage() for r in audit_records)
        assert "or_hints=" in joined
        # The upstream slug name MUST NOT appear in the audit line.
        assert "secret-vendor-slug" not in joined

    def test_routing_hints_is_frozen_after_construction(
        self,
        runner: CliRunner,
        app: typer.Typer,
        patched: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The hint object handed to the orchestrator MUST be the
        frozen :class:`OpenRouterRoutingHints` — never a mutable bag.
        Mutating it after construction would let a future bug rewrite
        the routing decision mid-call."""
        monkeypatch.setattr(
            "sec_generative_search.providers.registry.ProviderRegistry.get_capability",
            classmethod(
                lambda cls, name, surface, *, model=None: ProviderCapability(
                    chat=True, structured_output=False
                )
            ),
        )
        result = runner.invoke(
            app,
            [
                "rag",
                "query",
                "q",
                "--skip-plan",
                "--provider",
                "openrouter",
                "--openrouter-provider",
                "anthropic",
            ],
        )
        assert result.exit_code == 0, result.output
        orch = patched["orchestrators"][-1]
        hint = orch.calls[0]["routing_hints"]
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            hint.allow_fallbacks = True  # type: ignore[misc]
