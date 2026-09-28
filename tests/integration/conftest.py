"""Shared fixtures and deterministic doubles for the integration suite.

The doubles here replace only the three genuine external boundaries:

* :class:`KeywordEmbedder` — a deterministic bag-of-keywords embedder.
    It produces L2-normalised vectors over a fixed vocabulary so cosine
    similarity is meaningful and reproducible.

* :class:`ScriptedLLM` — a :class:`BaseLLMProvider` that returns a
    canned answer (optionally derived from the request) and supports both
    the non-streaming and streaming paths.

Everything else in the stack — :class:`ChromaDBClient`,
:class:`MetadataRegistry`, :class:`FilingStore`,
:class:`RetrievalService`, :class:`RAGOrchestrator`, the real
:class:`TextChunker` and :class:`FilingParser` — is the production class.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from sec_generative_search.core.types import (
    Chunk,
    ContentType,
    EmbedderStamp,
    FilingIdentifier,
    IngestResult,
    ProviderCapability,
    TokenUsage,
)
from sec_generative_search.database import (
    ChromaDBClient,
    FilingStore,
    MetadataRegistry,
)
from sec_generative_search.pipeline.orchestrator import ProcessedFiling
from sec_generative_search.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    GenerationRequest,
    GenerationResponse,
)
from sec_generative_search.search.retrieval import RetrievalService

# ---------------------------------------------------------------------------
# Deterministic keyword-bag embedder
# ---------------------------------------------------------------------------

# Fixed vocabulary — one dimension per term.  Every test chunk and query
# is written to contain at least one of these so no vector is all-zero
# (an all-zero vector yields a NaN cosine distance in ChromaDB).
EMBED_VOCAB: tuple[str, ...] = (
    "revenue",
    "growth",
    "margin",
    "litigation",
    "lawsuit",
    "risk",
    "cybersecurity",
    "dividend",
    "debt",
    "competition",
    "supply",
    "chain",
    "guidance",
    "currency",
    "acquisition",
    "tax",
)
EMBED_DIM = len(EMBED_VOCAB)


def _vectorise(text: str) -> np.ndarray:
    """Map text to an L2-normalised term-count vector over ``EMBED_VOCAB``."""
    lowered = text.lower()
    vec = np.array(
        [float(lowered.count(term)) for term in EMBED_VOCAB],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        # Defensive: a text with no vocabulary term gets a uniform
        # vector so cosine distance stays finite.  Test content is
        # written to avoid this branch, but a stray query should not
        # blow up ChromaDB.
        return np.full(EMBED_DIM, 1.0 / np.sqrt(EMBED_DIM), dtype=np.float32)
    return (vec / norm).astype(np.float32)


class KeywordEmbedder(BaseEmbeddingProvider):
    """Deterministic embedder used across the integration suite.

    ``provider``/``model``/``dimension`` deliberately match the
    :func:`stamp` fixture so the storage layer's stamp seal is coherent.
    """

    provider_name = "keyword-test"

    def __init__(self) -> None:
        super().__init__(api_key="local")  # pragma: allowlist secret
        self.embed_calls = 0

    def validate_key(self) -> bool:  # pragma: no cover - never exercised
        return True

    def get_capabilities(self) -> ProviderCapability:
        return ProviderCapability(embeddings=True)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.embed_calls += 1
        if not texts:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        return np.vstack([_vectorise(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return _vectorise(text)

    def get_dimension(self) -> int:
        return EMBED_DIM


# ---------------------------------------------------------------------------
# Scripted LLM provider
# ---------------------------------------------------------------------------


class ScriptedLLM(BaseLLMProvider):
    """A canned-reply LLM provider.

    The reply is either a fixed string or, when ``reply_fn`` is supplied,
    derived from the :class:`GenerationRequest` so a test can echo back
    markers the orchestrator placed in the prompt.  Streaming splits the
    reply into two deltas plus a closing usage-only frame, mirroring the
    real OpenAI-compatible streaming shape.
    """

    provider_name = "scripted-llm"

    def __init__(
        self,
        *,
        reply: str = "Answer.",
        reply_fn=None,
        structured_output: bool = False,
    ) -> None:
        super().__init__(api_key="scripted-key-padding-1234")  # pragma: allowlist secret
        self._reply = reply
        self._reply_fn = reply_fn
        self.last_request: GenerationRequest | None = None
        self._capability = ProviderCapability(
            chat=True,
            streaming=True,
            structured_output=structured_output,
            context_window_tokens=8000,
            max_output_tokens=2048,
        )

    def _resolve_reply(self, request: GenerationRequest) -> str:
        if self._reply_fn is not None:
            return self._reply_fn(request)
        return self._reply

    def validate_key(self) -> bool:  # pragma: no cover - never exercised
        return True

    def get_capabilities(self) -> ProviderCapability:
        return self._capability

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.last_request = request
        return GenerationResponse(
            text=self._resolve_reply(request),
            model=request.model or "scripted-model",
            token_usage=TokenUsage(input_tokens=20, output_tokens=10),
            finish_reason="stop",
        )

    def generate_stream(self, request: GenerationRequest) -> Iterator[GenerationResponse]:
        self.last_request = request
        reply = self._resolve_reply(request)
        midpoint = len(reply) // 2 or 1
        model = request.model or "scripted-model"
        for piece in (reply[:midpoint], reply[midpoint:]):
            if piece:
                yield GenerationResponse(text=piece, model=model, token_usage=TokenUsage())
        yield GenerationResponse(
            text="",
            model=model,
            token_usage=TokenUsage(input_tokens=20, output_tokens=10),
        )


# ---------------------------------------------------------------------------
# Real-stack fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def stamp() -> EmbedderStamp:
    """Stamp matching :class:`KeywordEmbedder` (provider/model/dimension)."""
    return EmbedderStamp(
        provider="keyword-test",
        model="keyword-bag",
        dimension=EMBED_DIM,
    )


@pytest.fixture
def embedder() -> KeywordEmbedder:
    return KeywordEmbedder()


@pytest.fixture
def chroma(stamp: EmbedderStamp, tmp_path: Path) -> ChromaDBClient:
    """Stamped ChromaDB client over a per-test on-disk directory."""
    return ChromaDBClient(stamp, chroma_path=str(tmp_path / "chroma"))


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[MetadataRegistry]:
    reg = MetadataRegistry(db_path=str(tmp_path / "metadata.sqlite"))
    yield reg
    reg.close()


@pytest.fixture
def store(chroma: ChromaDBClient, registry: MetadataRegistry) -> FilingStore:
    return FilingStore(chroma, registry)


@pytest.fixture
def retrieval(embedder: KeywordEmbedder, chroma: ChromaDBClient) -> RetrievalService:
    return RetrievalService(embedder, chroma)


# ---------------------------------------------------------------------------
# Settings isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_settings_singleton() -> Iterator[None]:
    """Keep the settings singleton clean across integration tests."""
    from sec_generative_search.config.settings import reload_settings

    reload_settings()
    yield
    reload_settings()


# ---------------------------------------------------------------------------
# Helpers — build ProcessedFiling objects with precise chunk control
# ---------------------------------------------------------------------------


def make_filing_id(
    *,
    ticker: str = "AAPL",
    form_type: str = "10-K",
    filing_date: date = date(2023, 11, 3),
    accession_number: str = "0000320193-23-000077",
) -> FilingIdentifier:
    return FilingIdentifier(
        ticker=ticker,
        form_type=form_type,
        filing_date=filing_date,
        accession_number=accession_number,
    )


def build_processed_filing(
    filing_id: FilingIdentifier,
    sections: list[tuple[str, str]],
    embedder: KeywordEmbedder,
) -> ProcessedFiling:
    """Build a :class:`ProcessedFiling` from ``(section_path, content)`` pairs.

    Chunks are embedded with the real :class:`KeywordEmbedder` so the
    stored vectors are coherent with what the retrieval service will
    embed the query into.
    """
    chunks = [
        Chunk(
            content=content,
            path=path,
            content_type=ContentType.TEXT,
            filing_id=filing_id,
            chunk_index=index,
            token_count=max(1, len(content) // 4),
        )
        for index, (path, content) in enumerate(sections)
    ]
    embeddings = embedder.embed_chunks(chunks)
    return ProcessedFiling(
        filing_id=filing_id,
        chunks=chunks,
        embeddings=embeddings,
        ingest_result=IngestResult(
            filing_id=filing_id,
            segment_count=len(chunks),
            chunk_count=len(chunks),
            duration_seconds=0.0,
        ),
    )


__all__ = [
    "EMBED_DIM",
    "EMBED_VOCAB",
    "KeywordEmbedder",
    "ScriptedLLM",
    "build_processed_filing",
    "make_filing_id",
]
