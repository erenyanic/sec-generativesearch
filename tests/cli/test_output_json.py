"""Tests for the shared ``--output json`` flag.

Scope:

- ``sec-rag search --output json`` — allow-list lift of
  :class:`api.schemas.SearchResponse` (``{hits, total}``); failure paths
  render ``{error, message, hint}`` envelopes.
- ``sec-rag rag query --output json`` — allow-list lift of
  :class:`api.schemas.RagQueryResponse`; ``--show-plan --output json``
  emits :class:`api.schemas.RagPlanResponse`-shape.
- ``sec-rag manage list --output json`` — allow-list lift of
  :class:`api.schemas.FilingListResponse`.
- ``sec-rag manage status --output json`` — operator-relevant snapshot
  ``{filing_count, max_filings, chunk_count, tickers, form_breakdown}``.
- ``sec-rag provider list --output json`` — registered providers with
  a boolean ``key_resolves`` flag only.  The masked tail rendered in
  text mode is NEVER serialised.
- ``sec-rag provider validate --output json`` — allow-list lift of
  :class:`api.schemas.ProviderValidateResponse`; failure paths render
  ``{error, message, hint}`` envelopes.

Security goals enforced by parametrised tests:

- The raw key never appears in any JSON document (``list`` rendered
  tail, ``validate`` envelope on rejection / no-credential / transient).
- The raw query never appears in any error envelope across ``search``,
  ``rag query``.
- EDGAR identity never reaches stdout via JSON output.
- The ``rag chat`` REPL deliberately has no ``--output`` flag —
    the flag is limited to the read paths.
- The destructive ``manage`` paths (``remove`` / ``clear``) and the
  write path ``provider set`` deliberately have no ``--output`` flag
  for the same reason.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, ClassVar

import pytest
import typer
from typer.testing import CliRunner

import sec_generative_search.cli.manage as manage_module
import sec_generative_search.cli.provider as provider_module
import sec_generative_search.cli.rag as rag_module
import sec_generative_search.cli.search as search_module
from sec_generative_search.cli._json import (
    OutputFormat,
    coerce_output_format,
    error_envelope,
    is_json,
    print_json,
)
from sec_generative_search.cli.manage import manage_app
from sec_generative_search.cli.provider import provider_app
from sec_generative_search.cli.rag import rag_app
from sec_generative_search.cli.search import search
from sec_generative_search.config.settings import (
    DatabaseSettings,
    EmbeddingSettings,
    LLMSettings,
    RAGSettings,
    SearchSettings,
)
from sec_generative_search.core.exceptions import (
    DatabaseError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    SearchError,
)
from sec_generative_search.core.types import (
    Citation,
    ContentType,
    EmbedderStamp,
    FilingIdentifier,
    GenerationResult,
    RetrievalResult,
    TokenUsage,
)
from sec_generative_search.database import FilingRecord
from sec_generative_search.database.metadata import DatabaseStatistics
from sec_generative_search.rag.modes import AnswerMode
from sec_generative_search.rag.query_understanding import QueryPlan

# Click renders :class:`typer.BadParameter` inside a Rich-styled panel
# that injects ANSI colour codes between words and soft-wraps the
# message to the terminal width.  In a narrow CI terminal a literal
# substring like ``"Invalid --output"`` is broken by both colour codes
# and newlines, so a raw ``in`` check fails even when the message is
# intact.  Same pattern as ``tests/cli/test_rag.py::_stripped``.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _stripped(output: str) -> str:
    """Strip ANSI codes and collapse whitespace for substring assertions."""
    return re.sub(r"\s+", " ", _ANSI_RE.sub("", output))


# ---------------------------------------------------------------------------
# JSON helper smoke tests
# ---------------------------------------------------------------------------


class TestJsonHelpers:
    """Direct tests for the shared :mod:`cli._json` plumbing.

    A bug in these helpers would propagate to every command; pinning
    them at the helper-module level keeps regression diagnoses cheap.
    """

    def test_coerce_output_format_accepts_known_values(self) -> None:
        assert coerce_output_format("text") is OutputFormat.TEXT
        assert coerce_output_format("json") is OutputFormat.JSON
        # Whitespace and case insensitivity — operators routinely
        # type ``--output JSON``.
        assert coerce_output_format(" JSON ") is OutputFormat.JSON

    def test_coerce_output_format_rejects_unknown_values(self) -> None:
        with pytest.raises(typer.BadParameter):
            coerce_output_format("yaml")

    def test_is_json_predicate(self) -> None:
        assert is_json(OutputFormat.JSON) is True
        assert is_json(OutputFormat.TEXT) is False

    def test_error_envelope_omits_none_fields(self) -> None:
        env = error_envelope("boom", "something failed")
        assert env == {"error": "boom", "message": "something failed"}
        # ``hint`` and ``details`` MUST be absent when not supplied —
        # a ``null`` value would force every parser to special-case it.
        assert "hint" not in env
        assert "details" not in env

    def test_error_envelope_includes_hint_and_details(self) -> None:
        env = error_envelope("boom", "msg", hint="try X", details="root cause")
        assert env == {
            "error": "boom",
            "message": "msg",
            "hint": "try X",
            "details": "root cause",
        }

    def test_print_json_emits_single_line_with_newline(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        print_json({"hello": "world"})
        captured = capsys.readouterr()
        assert captured.out == '{"hello": "world"}\n'

    def test_print_json_preserves_unicode_verbatim(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``ensure_ascii=False`` keeps non-ASCII bytes intact for log shippers."""
        print_json({"ticker": "ŞKR", "note": "tüm"})
        captured = capsys.readouterr()
        assert "ŞKR" in captured.out
        assert "tüm" in captured.out


# ---------------------------------------------------------------------------
# Test doubles shared by command tests
# ---------------------------------------------------------------------------


def _stub_db_settings(*, persist_provider_credentials: bool = False) -> DatabaseSettings:
    return DatabaseSettings.model_construct(
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


def _stub_embedding_settings() -> EmbeddingSettings:
    return EmbeddingSettings.model_construct(
        provider="openai",
        model_name="text-embedding-3-small",
        device="auto",
        batch_size=32,
        idle_timeout_minutes=0,
    )


class _SearchStubSettings:
    def __init__(self) -> None:
        self.embedding = _stub_embedding_settings()
        self.database = _stub_db_settings()


class _RagStubSettings:
    def __init__(self) -> None:
        self.embedding = _stub_embedding_settings()
        self.database = _stub_db_settings()
        self.llm = LLMSettings.model_construct(
            default_provider="openai",
            default_model=None,
            max_output_tokens=2048,
        )
        self.rag = RAGSettings.model_construct(
            context_token_budget=8000,
            refusal_enabled=True,
        )
        self.search = SearchSettings.model_construct(top_k=5, min_similarity=0.0)


class _ProviderStubSettings:
    def __init__(self, *, persist_provider_credentials: bool = False) -> None:
        self.embedding = _stub_embedding_settings()
        self.database = _stub_db_settings(persist_provider_credentials=persist_provider_credentials)
        self.llm = LLMSettings.model_construct(
            default_provider="openai",
            default_model=None,
            max_output_tokens=2048,
        )
        self.rag = RAGSettings.model_construct(
            context_token_budget=8000,
            refusal_enabled=True,
        )
        self.search = SearchSettings.model_construct(top_k=5, min_similarity=0.0)


# ---------------------------------------------------------------------------
# ``sec-rag search --output json``
# ---------------------------------------------------------------------------


def _make_hit(
    *,
    ticker: str = "AAPL",
    accession: str = "0000320193-24-000001",
    similarity: float = 0.55,
    content: str = "Sample content describing risk factors.",
) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        path="Part I > Item 1A > Risk Factors",
        content_type=ContentType.TEXT,
        ticker=ticker,
        form_type="10-K",
        similarity=similarity,
        filing_date="2024-01-15",
        accession_number=accession,
        chunk_id=f"{accession}:1",
        token_count=12,
    )


class _FakeRetrievalService:
    instances: ClassVar[list[_FakeRetrievalService]] = []

    def __init__(self, *, embedder: Any, chroma_client: Any) -> None:
        self.embedder = embedder
        self.chroma_client = chroma_client
        self.returns: list[RetrievalResult] = []
        self.raises: BaseException | None = None
        _FakeRetrievalService.instances.append(self)

    def retrieve(self, query: str, **kwargs: Any) -> list[RetrievalResult]:
        if self.raises is not None:
            raise self.raises
        return list(self.returns)


class _RecordingChroma:
    instances: ClassVar[list[_RecordingChroma]] = []

    def __init__(self, stamp: EmbedderStamp, *, chroma_path: str | None = None) -> None:
        self.stamp = stamp
        _RecordingChroma.instances.append(self)


class _RecordingEmbedder:
    instances: ClassVar[list[_RecordingEmbedder]] = []

    def __init__(self) -> None:
        _RecordingEmbedder.instances.append(self)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def search_app() -> typer.Typer:
    test_app = typer.Typer()
    test_app.command(name="search")(search)

    @test_app.command(name="_n", hidden=True)
    def _n() -> None:  # pragma: no cover — keeps Typer in group mode
        ...

    return test_app


@pytest.fixture
def patched_search(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    _FakeRetrievalService.instances.clear()
    _RecordingChroma.instances.clear()
    _RecordingEmbedder.instances.clear()

    monkeypatch.setattr(search_module, "get_settings", _SearchStubSettings)
    monkeypatch.setattr(search_module, "build_embedder", lambda _s: _RecordingEmbedder())
    monkeypatch.setattr(search_module, "ChromaDBClient", _RecordingChroma)
    monkeypatch.setattr(search_module, "RetrievalService", _FakeRetrievalService)
    return {
        "services": _FakeRetrievalService.instances,
        "chromas": _RecordingChroma.instances,
        "embedders": _RecordingEmbedder.instances,
    }


def _arm_search_returns(results: list[RetrievalResult]) -> None:
    original = _FakeRetrievalService.__init__

    def _seeded(self: _FakeRetrievalService, **kwargs: Any) -> None:
        original(self, **kwargs)
        self.returns = list(results)

    _FakeRetrievalService.__init__ = _seeded  # type: ignore[method-assign]


def _arm_search_raises(exc: BaseException) -> None:
    original = _FakeRetrievalService.__init__

    def _seeded(self: _FakeRetrievalService, **kwargs: Any) -> None:
        original(self, **kwargs)
        self.raises = exc

    _FakeRetrievalService.__init__ = _seeded  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _reset_search_init() -> Any:
    yield

    def _default(self: _FakeRetrievalService, **kwargs: Any) -> None:
        self.embedder = kwargs.get("embedder")
        self.chroma_client = kwargs.get("chroma_client")
        self.returns = []
        self.raises = None
        _FakeRetrievalService.instances.append(self)

    _FakeRetrievalService.__init__ = _default  # type: ignore[method-assign]


class TestSearchJson:
    def test_emits_search_response_shape(
        self,
        runner: CliRunner,
        search_app: typer.Typer,
        patched_search: dict[str, Any],
    ) -> None:
        _arm_search_returns([_make_hit()])
        result = runner.invoke(search_app, ["search", "risk", "--output", "json"])
        assert result.exit_code == 0, result.output
        # Output is a single JSON document (well-formed).
        payload = json.loads(result.output)
        assert set(payload.keys()) == {"hits", "total"}
        assert payload["total"] == 1
        # Hit shape mirrors SearchHit exactly — fail loudly if a future
        # RetrievalResult field accidentally leaks through.
        expected_keys = {
            "chunk_id",
            "content",
            "path",
            "content_type",
            "ticker",
            "form_type",
            "filing_date",
            "accession_number",
            "similarity",
            "rerank_score",
            "token_count",
            "section_boundaries",
        }
        assert set(payload["hits"][0].keys()) == expected_keys
        assert payload["hits"][0]["ticker"] == "AAPL"
        # ``content_type`` is the enum value, not the enum repr.
        assert payload["hits"][0]["content_type"] == "text"

    def test_empty_result_set_emits_zero_total(
        self,
        runner: CliRunner,
        search_app: typer.Typer,
        patched_search: dict[str, Any],
    ) -> None:
        _arm_search_returns([])
        result = runner.invoke(search_app, ["search", "no matches", "-o", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload == {"hits": [], "total": 0}

    def test_search_failure_emits_error_envelope(
        self,
        runner: CliRunner,
        search_app: typer.Typer,
        patched_search: dict[str, Any],
    ) -> None:
        _arm_search_raises(SearchError("bad filter", details="ticker has no value"))
        result = runner.invoke(search_app, ["search", "q", "--output", "json"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["error"] == "search_failed"
        assert "hint" in payload

    def test_invalid_output_rejected(
        self,
        runner: CliRunner,
        search_app: typer.Typer,
        patched_search: dict[str, Any],
    ) -> None:
        result = runner.invoke(search_app, ["search", "q", "--output", "yaml"])
        assert result.exit_code == 2
        # Click wraps the BadParameter message in a Rich panel; strip
        # ANSI / collapse whitespace before the substring check so a
        # narrow CI terminal does not flake.
        assert "Invalid --output" in _stripped(result.output)


@pytest.mark.security
class TestSearchJsonSecurity:
    def test_raw_query_not_in_error_envelope(
        self,
        runner: CliRunner,
        search_app: typer.Typer,
        patched_search: dict[str, Any],
    ) -> None:
        sentinel = "PII-LADEN-QUERY-DO-NOT-ECHO"
        _arm_search_raises(SearchError("boom"))
        result = runner.invoke(search_app, ["search", sentinel, "-o", "json"])
        assert result.exit_code == 1
        assert sentinel not in result.output

    def test_raw_query_not_in_success_payload(
        self,
        runner: CliRunner,
        search_app: typer.Typer,
        patched_search: dict[str, Any],
    ) -> None:
        """The ``SearchResponse`` API surface deliberately drops ``query``;
        the CLI JSON must do the same — operators piping JSON into a log
        shipper should not find queries inlined into a per-result document."""
        sentinel = "ANOTHER-QUERY-CANARY"
        _arm_search_returns([_make_hit()])
        result = runner.invoke(search_app, ["search", sentinel, "-o", "json"])
        assert result.exit_code == 0
        assert sentinel not in result.output

    def test_api_key_never_in_json_output(
        self,
        runner: CliRunner,
        search_app: typer.Typer,
        patched_search: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        canary = "sk-canary-must-not-leak-to-json"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", canary)
        _arm_search_raises(SearchError("boom"))
        result = runner.invoke(search_app, ["search", "q", "-o", "json"])
        assert canary not in result.output


# ---------------------------------------------------------------------------
# ``sec-rag manage status / list --output json``
# ---------------------------------------------------------------------------


def _make_record(
    *,
    ticker: str = "AAPL",
    form_type: str = "10-K",
    filing_date: str = "2024-01-15",
    accession: str = "0000320193-24-000001",
    chunk_count: int = 10,
    ingested_at: str = "2024-02-01T00:00:00+00:00",
    rec_id: int = 1,
) -> FilingRecord:
    return FilingRecord(
        id=rec_id,
        ticker=ticker,
        form_type=form_type,
        filing_date=filing_date,
        accession_number=accession,
        chunk_count=chunk_count,
        ingested_at=ingested_at,
    )


class _ManageRegistry:
    instances: ClassVar[list[_ManageRegistry]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.filings: list[FilingRecord] = []
        self.statistics: DatabaseStatistics | None = None
        _ManageRegistry.instances.append(self)

    def list_filings(
        self,
        ticker: str | None = None,
        form_type: str | None = None,
    ) -> list[FilingRecord]:
        rows = list(self.filings)
        if ticker:
            rows = [r for r in rows if r.ticker == ticker]
        if form_type:
            rows = [r for r in rows if r.form_type == form_type]
        return rows

    def get_statistics(self) -> DatabaseStatistics:
        return self.statistics or DatabaseStatistics(
            filing_count=0,
            tickers=[],
            form_breakdown={},
            ticker_breakdown=[],
        )


class _ManageChroma:
    instances: ClassVar[list[_ManageChroma]] = []

    def __init__(self, stamp: EmbedderStamp, *, chroma_path: str | None = None) -> None:
        self.stamp = stamp
        self.chunk_count = 0
        _ManageChroma.instances.append(self)

    def collection_count(self) -> int:
        return self.chunk_count


class _ManageStore:
    """:class:`FilingStore` stand-in — read paths don't touch writes."""

    def __init__(self, chroma: Any, registry: Any) -> None:
        self.chroma = chroma
        self.registry = registry


@pytest.fixture
def manage_test_app() -> typer.Typer:
    test_app = typer.Typer()
    test_app.add_typer(manage_app, name="manage")
    return test_app


@pytest.fixture
def patched_manage(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    _ManageRegistry.instances.clear()
    _ManageChroma.instances.clear()

    class _Settings:
        def __init__(self) -> None:
            self.embedding = _stub_embedding_settings()
            self.database = _stub_db_settings()

    monkeypatch.setattr(manage_module, "get_settings", _Settings)
    monkeypatch.setattr(manage_module, "ChromaDBClient", _ManageChroma)
    monkeypatch.setattr(manage_module, "MetadataRegistry", _ManageRegistry)
    monkeypatch.setattr(manage_module, "FilingStore", _ManageStore)
    return {
        "registries": _ManageRegistry.instances,
        "chromas": _ManageChroma.instances,
    }


def _seed_manage_registry(
    filings: list[FilingRecord], *, stats: DatabaseStatistics | None = None
) -> None:
    original = _ManageRegistry.__init__

    def _seeded(self: _ManageRegistry, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        self.filings = list(filings)
        if stats is not None:
            self.statistics = stats

    _ManageRegistry.__init__ = _seeded  # type: ignore[method-assign]


def _seed_manage_chroma_chunks(chunks: int) -> None:
    original = _ManageChroma.__init__

    def _seeded(self: _ManageChroma, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        self.chunk_count = chunks

    _ManageChroma.__init__ = _seeded  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _reset_manage_init() -> Any:
    yield

    def _default_reg(self: _ManageRegistry, *args: Any, **kwargs: Any) -> None:
        self.filings = []
        self.statistics = None
        _ManageRegistry.instances.append(self)

    def _default_chroma(
        self: _ManageChroma, stamp: EmbedderStamp, *, chroma_path: str | None = None
    ) -> None:
        self.stamp = stamp
        self.chunk_count = 0
        _ManageChroma.instances.append(self)

    _ManageRegistry.__init__ = _default_reg  # type: ignore[method-assign]
    _ManageChroma.__init__ = _default_chroma  # type: ignore[method-assign]


class TestManageStatusJson:
    def test_emits_operator_snapshot(
        self,
        runner: CliRunner,
        manage_test_app: typer.Typer,
        patched_manage: dict[str, Any],
    ) -> None:
        stats = DatabaseStatistics(
            filing_count=2,
            tickers=["AAPL", "MSFT"],
            form_breakdown={"10-K": 1, "10-Q": 1},
            ticker_breakdown=[],
        )
        _seed_manage_registry([_make_record()], stats=stats)
        _seed_manage_chroma_chunks(42)
        result = runner.invoke(manage_test_app, ["manage", "status", "--output", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert set(payload.keys()) == {
            "filing_count",
            "max_filings",
            "chunk_count",
            "tickers",
            "form_breakdown",
        }
        assert payload["filing_count"] == 2
        assert payload["max_filings"] == 10000
        assert payload["chunk_count"] == 42
        assert payload["tickers"] == ["AAPL", "MSFT"]
        assert payload["form_breakdown"] == {"10-K": 1, "10-Q": 1}

    def test_empty_database_renders_zero_counts(
        self,
        runner: CliRunner,
        manage_test_app: typer.Typer,
        patched_manage: dict[str, Any],
    ) -> None:
        result = runner.invoke(manage_test_app, ["manage", "status", "-o", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["filing_count"] == 0
        assert payload["chunk_count"] == 0
        assert payload["tickers"] == []
        assert payload["form_breakdown"] == {}

    def test_storage_failure_emits_envelope(
        self,
        runner: CliRunner,
        manage_test_app: typer.Typer,
        patched_manage: dict[str, Any],
    ) -> None:
        def _boom(*_a: Any, **_kw: Any) -> None:
            raise DatabaseError("stamp mismatch")

        original = _ManageChroma.__init__

        def _seeded(self: _ManageChroma, *a: Any, **kw: Any) -> None:
            _boom()
            original(self, *a, **kw)

        _ManageChroma.__init__ = _seeded  # type: ignore[method-assign]
        result = runner.invoke(manage_test_app, ["manage", "status", "-o", "json"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["error"] == "storage_initialisation_failed"


class TestManageListJson:
    def test_emits_filing_list_response(
        self,
        runner: CliRunner,
        manage_test_app: typer.Typer,
        patched_manage: dict[str, Any],
    ) -> None:
        _seed_manage_registry(
            [
                _make_record(),
                _make_record(
                    rec_id=2,
                    ticker="MSFT",
                    accession="0000789019-24-000002",
                ),
            ]
        )
        result = runner.invoke(manage_test_app, ["manage", "list", "-o", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert set(payload.keys()) == {"filings", "total"}
        assert payload["total"] == 2
        expected_keys = {
            "ticker",
            "form_type",
            "filing_date",
            "accession_number",
            "chunk_count",
            "ingested_at",
        }
        assert set(payload["filings"][0].keys()) == expected_keys
        # The internal SQLite auto-increment ``id`` is dropped.
        assert "id" not in payload["filings"][0]

    def test_empty_list_uniform_shape(
        self,
        runner: CliRunner,
        manage_test_app: typer.Typer,
        patched_manage: dict[str, Any],
    ) -> None:
        result = runner.invoke(manage_test_app, ["manage", "list", "-o", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        # Empty list is a valid result — the shape is uniform.
        assert payload == {"filings": [], "total": 0}

    def test_filter_propagates_in_json_mode(
        self,
        runner: CliRunner,
        manage_test_app: typer.Typer,
        patched_manage: dict[str, Any],
    ) -> None:
        _seed_manage_registry(
            [
                _make_record(),
                _make_record(
                    rec_id=2,
                    ticker="MSFT",
                    accession="0000789019-24-000002",
                ),
            ]
        )
        result = runner.invoke(manage_test_app, ["manage", "list", "-k", "aapl", "-o", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["total"] == 1
        assert payload["filings"][0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# ``sec-rag provider list / validate --output json``
# ---------------------------------------------------------------------------


class _ProviderFakeRegistry:
    instances: ClassVar[list[_ProviderFakeRegistry]] = []
    encrypted_default: ClassVar[bool] = False
    raise_on_construct: ClassVar[BaseException | None] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if type(self).raise_on_construct is not None:
            raise type(self).raise_on_construct
        self.encrypted = type(self).encrypted_default
        self.closed = False
        _ProviderFakeRegistry.instances.append(self)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def provider_test_app() -> typer.Typer:
    test_app = typer.Typer()
    test_app.add_typer(provider_app, name="provider")
    return test_app


@pytest.fixture(autouse=True)
def _reset_provider_doubles() -> Any:
    _ProviderFakeRegistry.instances.clear()
    _ProviderFakeRegistry.encrypted_default = False
    _ProviderFakeRegistry.raise_on_construct = None
    yield


@pytest.fixture
def patched_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_module, "get_settings", lambda: _ProviderStubSettings())
    monkeypatch.setattr(provider_module, "MetadataRegistry", _ProviderFakeRegistry)


class TestProviderListJson:
    def test_emits_providers_payload(
        self,
        runner: CliRunner,
        provider_test_app: typer.Typer,
        patched_provider: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        result = runner.invoke(provider_test_app, ["provider", "list", "-o", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert set(payload.keys()) == {"providers", "total"}
        # OpenAI is always registered; assert on at least one entry.
        names = {row["name"] for row in payload["providers"]}
        assert "openai" in names
        # Per-row shape is the allow-list — no extra fields creep in.
        expected_keys = {
            "name",
            "surface",
            "default_model",
            "pricing_tier",
            "admin_env_var",
            "key_resolves",
            "requires_extras",
            "supports_arbitrary_models",
            "supports_upstream_routing",
        }
        for row in payload["providers"]:
            assert set(row.keys()) == expected_keys
            # ``key_resolves`` is a strict boolean — never a tail or
            # mask string.
            assert isinstance(row["key_resolves"], bool)

    def test_local_llm_surfaces_openrouter_style(
        self,
        runner: CliRunner,
        provider_test_app: typer.Typer,
        patched_provider: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The self-hosted ``local_llm`` provider renders as a free-text-slug
        LLM row — an empty catalogue advertised via
        ``supports_arbitrary_models`` (OpenRouter-style), with routing hints
        dropped (``supports_upstream_routing`` False). Unlike OpenRouter it
        carries no admin-env var (a self-hosted endpoint needs no
        credential) and prices FREE (cost-derived from the 0.0 default-model
        cost), never UNKNOWN."""
        for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        result = runner.invoke(
            provider_test_app,
            ["provider", "list", "--surface", "llm", "-o", "json"],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        by_name = {row["name"]: row for row in payload["providers"]}
        assert "local_llm" in by_name, "local_llm must surface on the LLM list"
        row = by_name["local_llm"]
        assert row["surface"] == "llm"
        assert row["supports_arbitrary_models"] is True
        assert row["supports_upstream_routing"] is False
        # A self-hosted endpoint has no admin-default env var and no
        # optional-extras gate (the ``openai`` SDK is a core dependency).
        assert row["admin_env_var"] is None
        assert row["requires_extras"] == []
        # FREE is derived from the 0.0-cost default-model capability —
        # never the honest-UNKNOWN an arbitrary-slug OpenRouter row reports.
        assert row["pricing_tier"] == "free"
        assert by_name["openrouter"]["pricing_tier"] == "unknown"

    def test_surface_filter_in_json_mode(
        self,
        runner: CliRunner,
        provider_test_app: typer.Typer,
        patched_provider: None,
    ) -> None:
        result = runner.invoke(
            provider_test_app,
            ["provider", "list", "--surface", "embedding", "-o", "json"],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        # Every returned row is an embedding surface.
        assert all(row["surface"] == "embedding" for row in payload["providers"])
        names = {row["name"] for row in payload["providers"]}
        # OpenAI ships both; the embedding row must be present.
        assert "openai" in names
        # LLM-only providers MUST NOT appear.
        assert "anthropic" not in names
        assert "deepseek" not in names

    def test_registry_unavailable_still_emits_json(
        self,
        runner: CliRunner,
        provider_test_app: typer.Typer,
        patched_provider: None,
    ) -> None:
        """``provider list`` MUST work on a fresh deployment where the
        SQLite file does not yet exist."""
        _ProviderFakeRegistry.raise_on_construct = DatabaseError("no db yet")
        result = runner.invoke(provider_test_app, ["provider", "list", "-o", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["total"] >= 1


@pytest.mark.security
class TestProviderListJsonSecurity:
    def test_resolved_key_not_in_json_output(
        self,
        runner: CliRunner,
        provider_test_app: typer.Typer,
        patched_provider: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A resolvable admin-env key must NEVER appear in the JSON
        output — only the boolean ``key_resolves`` flag.  This is the
        crux of the JSON-mode discipline: the Rich-mode tail visualisation
        is for *operator glance*; piping JSON into a log shipper would
        otherwise persist masked tails into long-term storage."""
        canary = "sk-canary-list-json-must-not-leak"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", canary)
        result = runner.invoke(provider_test_app, ["provider", "list", "-o", "json"])
        assert result.exit_code == 0, result.output
        # The canary must not appear at all — and crucially the masked
        # tail (last 4 chars) must also be absent.  ``mask_secret``
        # generates a fixed-format string; we check the most identifying
        # 4-char tail of the canary.
        assert canary not in result.output
        assert canary[-4:] not in result.output
        # ``key_resolves`` flag should be True for at least one row
        # (openai LLM surface).
        payload = json.loads(result.output)
        openai_llm = next(
            r for r in payload["providers"] if r["name"] == "openai" and r["surface"] == "llm"
        )
        assert openai_llm["key_resolves"] is True


class TestProviderValidateJson:
    def test_validate_accepted_emits_valid_true(
        self,
        runner: CliRunner,
        provider_test_app: typer.Typer,
        patched_provider: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-canary-accept")  # pragma: allowlist secret
        monkeypatch.setattr(
            provider_module,
            "validate_credential",
            lambda provider, surface, api_key, model=None: True,
        )
        result = runner.invoke(provider_test_app, ["provider", "validate", "openai", "-o", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload == {"valid": True, "provider": "openai", "surface": "llm"}

    def test_validate_rejected_emits_valid_false_and_exits_1(
        self,
        runner: CliRunner,
        provider_test_app: typer.Typer,
        patched_provider: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-canary-reject")  # pragma: allowlist secret
        monkeypatch.setattr(
            provider_module,
            "validate_credential",
            lambda provider, surface, api_key, model=None: False,
        )
        result = runner.invoke(provider_test_app, ["provider", "validate", "openai", "-o", "json"])
        # Rejection is exit 1 even in JSON mode — same shape as the API
        # discipline: a 'false' verdict is still a 1xx-ish 200.
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["valid"] is False

    def test_validate_no_credential_emits_envelope(
        self,
        runner: CliRunner,
        provider_test_app: typer.Typer,
        patched_provider: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = runner.invoke(provider_test_app, ["provider", "validate", "openai", "-o", "json"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["error"] == "no_credential"

    def test_validate_transient_exits_2(
        self,
        runner: CliRunner,
        provider_test_app: typer.Typer,
        patched_provider: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-canary-rate")  # pragma: allowlist secret

        def _raise(*_a: Any, **_kw: Any) -> bool:
            raise ProviderRateLimitError("slow down")

        monkeypatch.setattr(provider_module, "validate_credential", _raise)
        result = runner.invoke(provider_test_app, ["provider", "validate", "openai", "-o", "json"])
        # Transient → exit 2; same code as text mode.
        assert result.exit_code == 2
        payload = json.loads(result.output)
        assert payload["error"] == "provider_unavailable"

    def test_validate_unknown_provider_emits_envelope(
        self,
        runner: CliRunner,
        provider_test_app: typer.Typer,
        patched_provider: None,
    ) -> None:
        result = runner.invoke(
            provider_test_app, ["provider", "validate", "doesnotexist", "-o", "json"]
        )
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["error"] == "unknown_provider"


@pytest.mark.security
class TestProviderValidateJsonSecurity:
    def test_api_key_never_in_validate_json(
        self,
        runner: CliRunner,
        provider_test_app: typer.Typer,
        patched_provider: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        canary = "sk-canary-validate-json-secret"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", canary)
        # Cycle through accepted / rejected / transient / unknown — none
        # of the four paths is permitted to surface the key.
        monkeypatch.setattr(
            provider_module,
            "validate_credential",
            lambda provider, surface, api_key, model=None: True,
        )
        result = runner.invoke(provider_test_app, ["provider", "validate", "openai", "-o", "json"])
        assert result.exit_code == 0
        assert canary not in result.output

        monkeypatch.setattr(
            provider_module,
            "validate_credential",
            lambda provider, surface, api_key, model=None: False,
        )
        result = runner.invoke(provider_test_app, ["provider", "validate", "openai", "-o", "json"])
        assert result.exit_code == 1
        assert canary not in result.output


# ---------------------------------------------------------------------------
# ``sec-rag rag query --output json``
# ---------------------------------------------------------------------------


def _make_rag_plan(
    *,
    raw_query: str = "What are Apple's revenue segments?",
    tickers: list[str] | None = None,
) -> QueryPlan:
    return QueryPlan(
        raw_query=raw_query,
        detected_language="en",
        query_en=raw_query,
        tickers=list(tickers or []),
        form_types=[],
        date_range=None,
        intent="ask",
        suggested_answer_mode=AnswerMode.CONCISE,
    )


def _make_rag_citation() -> Citation:
    return Citation(
        chunk_id="0000320193-24-000001:1",
        filing_id=FilingIdentifier(
            ticker="AAPL",
            form_type="10-K",
            filing_date=date(2024, 1, 15),
            accession_number="0000320193-24-000001",
        ),
        section_path="Part II > Item 7 > MD&A",
        text_span="Revenue grew across segments.",
        similarity=0.62,
        display_index=1,
    )


def _make_rag_result(*, refused: bool = False) -> GenerationResult:
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
        answer="Apple has three main revenue segments [1].",
        provider="openai",
        model="gpt-x",
        prompt_version="v1",
        citations=[_make_rag_citation()],
        retrieved_chunks=[],
        token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        latency_seconds=0.42,
        streamed=False,
    )


class _RagFakeEmbedder:
    pass


class _RagFakeLLM:
    pass


class _RagFakeChroma:
    def __init__(self, stamp: EmbedderStamp, *, chroma_path: str | None = None) -> None:
        self.stamp = stamp


class _RagFakeRegistry:
    encrypted_default: ClassVar[bool] = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.encrypted = type(self).encrypted_default


class _RagFakeRetrievalService:
    def __init__(self, *, embedder: Any, chroma_client: Any) -> None:
        self.embedder = embedder
        self.chroma_client = chroma_client


class _RagFakeOrchestrator:
    def __init__(self, *, retrieval: Any, llm: Any) -> None:
        self.retrieval = retrieval
        self.llm = llm
        self.returns: GenerationResult = _make_rag_result()
        self.raises: BaseException | None = None

    def generate(self, plan: QueryPlan, **kwargs: Any) -> GenerationResult:
        if self.raises is not None:
            raise self.raises
        return self.returns


@pytest.fixture
def rag_test_app() -> typer.Typer:
    test_app = typer.Typer()
    test_app.add_typer(rag_app, name="rag")
    return test_app


@pytest.fixture
def patched_rag(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "understand_returns": _make_rag_plan(),
        "understand_raises": None,
        "orchestrator_returns": _make_rag_result(),
        "orchestrator_raises": None,
        "understand_call_count": 0,
        "generate_call_count": 0,
    }

    def _fake_understand(
        query: str,
        *,
        llm: Any,
        model: str,
        structured_output_supported: bool = False,
    ) -> QueryPlan:
        state["understand_call_count"] += 1
        if state["understand_raises"] is not None:
            raise state["understand_raises"]
        plan = state["understand_returns"]
        if isinstance(plan, QueryPlan):
            # Refresh raw_query so the test can assert on the input.
            plan.raw_query = query
            if not plan.query_en:
                plan.query_en = query
        return plan

    class _Orchestrator(_RagFakeOrchestrator):
        def generate(self, plan: QueryPlan, **kwargs: Any) -> GenerationResult:
            state["generate_call_count"] += 1
            if state["orchestrator_raises"] is not None:
                raise state["orchestrator_raises"]
            return state["orchestrator_returns"]

    monkeypatch.setattr(rag_module, "get_settings", _RagStubSettings)
    monkeypatch.setattr(rag_module, "build_embedder", lambda _settings: _RagFakeEmbedder())
    monkeypatch.setattr(
        rag_module,
        "build_llm_provider",
        lambda _name, *, api_key_resolver=None: _RagFakeLLM(),
    )
    monkeypatch.setattr(rag_module, "ChromaDBClient", _RagFakeChroma)
    monkeypatch.setattr(rag_module, "MetadataRegistry", _RagFakeRegistry)
    monkeypatch.setattr(rag_module, "RetrievalService", _RagFakeRetrievalService)
    monkeypatch.setattr(rag_module, "RAGOrchestrator", _Orchestrator)
    monkeypatch.setattr(rag_module, "understand_query", _fake_understand)
    return state


class TestRagQueryJson:
    def test_emits_rag_query_response_shape(
        self,
        runner: CliRunner,
        rag_test_app: typer.Typer,
        patched_rag: dict[str, Any],
    ) -> None:
        result = runner.invoke(rag_test_app, ["rag", "query", "What is revenue?", "-o", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        expected_keys = {
            "answer",
            "citations",
            "provider",
            "model",
            "prompt_version",
            "token_usage",
            "estimated_cost_usd",
            "latency_seconds",
            "streamed",
            "refused",
        }
        assert set(payload.keys()) == expected_keys
        # ``retrieved_chunks`` MUST NOT leak — same discipline as the
        # API surface (citations are the audit trail).
        assert "retrieved_chunks" not in payload
        # The CLI mirrors the API's per-request estimate.
        # The resolved provider/model (openai default → gpt-5.4-mini,
        # in=0.6 / out=2.4 USD/MTok) over 100+50 tokens → 0.00018.
        assert payload["estimated_cost_usd"] == pytest.approx(0.00018)
        # Citation shape mirrors CitationSchema.
        citation = payload["citations"][0]
        cite_keys = {
            "chunk_id",
            "ticker",
            "form_type",
            "filing_date",
            "accession_number",
            "section_path",
            "text_span",
            "similarity",
            "display_index",
        }
        assert set(citation.keys()) == cite_keys
        assert citation["filing_date"] == "2024-01-15"
        # ``token_usage`` is a nested dict, not a flat ``input_tokens``.
        assert payload["token_usage"] == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

    def test_refused_path_emits_refused_true(
        self,
        runner: CliRunner,
        rag_test_app: typer.Typer,
        patched_rag: dict[str, Any],
    ) -> None:
        patched_rag["orchestrator_returns"] = _make_rag_result(refused=True)
        result = runner.invoke(rag_test_app, ["rag", "query", "obscure?", "-o", "json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["refused"] is True
        assert payload["citations"] == []

    def test_show_plan_emits_plan_response(
        self,
        runner: CliRunner,
        rag_test_app: typer.Typer,
        patched_rag: dict[str, Any],
    ) -> None:
        patched_rag["understand_returns"] = _make_rag_plan(tickers=["AAPL"])
        result = runner.invoke(
            rag_test_app, ["rag", "query", "Apple revenue?", "--show-plan", "-o", "json"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert set(payload.keys()) == {"plan", "provider", "model"}
        plan = payload["plan"]
        plan_keys = {
            "raw_query",
            "detected_language",
            "query_en",
            "tickers",
            "form_types",
            "date_range",
            "intent",
            "suggested_answer_mode",
        }
        assert set(plan.keys()) == plan_keys
        # ``date_range`` is a list on the wire (JSON has no tuple type).
        assert plan["date_range"] is None
        assert plan["tickers"] == ["AAPL"]
        # Generation MUST NOT run on the show-plan path.
        assert patched_rag["generate_call_count"] == 0

    def test_provider_auth_error_emits_envelope(
        self,
        runner: CliRunner,
        rag_test_app: typer.Typer,
        patched_rag: dict[str, Any],
    ) -> None:
        patched_rag["orchestrator_raises"] = ProviderAuthError("bad key")
        result = runner.invoke(rag_test_app, ["rag", "query", "q", "--skip-plan", "-o", "json"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["error"] == "provider_unauthorized"

    def test_retrieval_failure_emits_envelope(
        self,
        runner: CliRunner,
        rag_test_app: typer.Typer,
        patched_rag: dict[str, Any],
    ) -> None:
        patched_rag["orchestrator_raises"] = SearchError("retrieval broken")
        result = runner.invoke(rag_test_app, ["rag", "query", "q", "--skip-plan", "-o", "json"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["error"] == "retrieval_failed"

    def test_invalid_flag_combination_emits_envelope(
        self,
        runner: CliRunner,
        rag_test_app: typer.Typer,
        patched_rag: dict[str, Any],
    ) -> None:
        """``--show-plan`` + ``--skip-plan`` mutually exclusive."""
        result = runner.invoke(
            rag_test_app,
            ["rag", "query", "q", "--show-plan", "--skip-plan", "-o", "json"],
        )
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["error"] == "invalid_flag_combination"


@pytest.mark.security
class TestRagQueryJsonSecurity:
    def test_raw_question_never_in_error_envelope(
        self,
        runner: CliRunner,
        rag_test_app: typer.Typer,
        patched_rag: dict[str, Any],
    ) -> None:
        sentinel = "PII-LADEN-RAG-QUESTION"
        patched_rag["orchestrator_raises"] = ProviderError("upstream 500")
        result = runner.invoke(
            rag_test_app, ["rag", "query", sentinel, "--skip-plan", "-o", "json"]
        )
        assert result.exit_code == 1
        assert sentinel not in result.output

    def test_api_key_never_in_json_output(
        self,
        runner: CliRunner,
        rag_test_app: typer.Typer,
        patched_rag: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        canary = "sk-rag-json-canary"  # pragma: allowlist secret
        monkeypatch.setenv("OPENAI_API_KEY", canary)
        result = runner.invoke(rag_test_app, ["rag", "query", "q", "--skip-plan", "-o", "json"])
        assert result.exit_code == 0
        assert canary not in result.output

    def test_edgar_identity_never_in_json_output(
        self,
        runner: CliRunner,
        rag_test_app: typer.Typer,
        patched_rag: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("EDGAR_IDENTITY_NAME", "Secret User")
        monkeypatch.setenv("EDGAR_IDENTITY_EMAIL", "canary-edgar@example.test")
        result = runner.invoke(rag_test_app, ["rag", "query", "q", "--skip-plan", "-o", "json"])
        assert "Secret User" not in result.output
        assert "canary-edgar@example.test" not in result.output

    def test_chat_does_not_expose_output_flag(
        self,
        runner: CliRunner,
        rag_test_app: typer.Typer,
        patched_rag: dict[str, Any],
    ) -> None:
        """``rag chat`` is an interactive REPL and deliberately has no
        JSON surface.  Adding one would conflate the streaming-event
        sequence with a single-document JSON payload."""
        result = runner.invoke(rag_test_app, ["rag", "chat", "--help"])
        assert result.exit_code == 0
        # ``rag chat`` MUST NOT advertise an ``--output json`` flag.
        # Help text covers every supported option, so a missing
        # ``--output`` confirms scoping.
        assert "--output" not in result.output


# ---------------------------------------------------------------------------
# Surface scoping — write paths MUST NOT expose --output
# ---------------------------------------------------------------------------


class TestSurfaceScoping:
    """The ``--output json`` flag is limited to the read paths
    (``search`` / ``rag query`` / ``manage list`` / ``manage status`` /
    ``provider list`` / ``provider validate``).  Destructive / write
    commands (``manage remove`` / ``manage clear`` / ``provider set`` /
    ``rag chat``) do NOT expose the flag.  Surfacing it on a destructive
    path would conflate the success / abort envelope with a structured
    result.
    """

    def test_manage_remove_has_no_output_flag(
        self,
        runner: CliRunner,
        manage_test_app: typer.Typer,
    ) -> None:
        result = runner.invoke(manage_test_app, ["manage", "remove", "--help"])
        assert result.exit_code == 0
        assert "--output" not in result.output

    def test_manage_clear_has_no_output_flag(
        self,
        runner: CliRunner,
        manage_test_app: typer.Typer,
    ) -> None:
        result = runner.invoke(manage_test_app, ["manage", "clear", "--help"])
        assert result.exit_code == 0
        assert "--output" not in result.output

    def test_provider_set_has_no_output_flag(
        self,
        runner: CliRunner,
        provider_test_app: typer.Typer,
    ) -> None:
        result = runner.invoke(provider_test_app, ["provider", "set", "--help"])
        assert result.exit_code == 0
        assert "--output" not in result.output
