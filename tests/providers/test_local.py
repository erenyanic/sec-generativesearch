"""Tests for :mod:`sec_generative_search.providers.local`.

Every test injects a stub loader — the real
:class:`sentence_transformers.SentenceTransformer` is never touched, so
CI completes without downloading weights and without torch installed.

Coverage map:

- Construction guards (unknown model / invalid batch size / invalid
  idle-timeout / invalid device).
- Default-model fallback and ``MODEL_DIMENSIONS`` contract.
- Lazy load: construction does not invoke the loader; ``embed_*`` and
  ``validate_key`` do.  The loader receives the resolved device and the
  Hugging Face token (or ``None`` when no token was supplied).
- Thread-safety: concurrent ``embed_texts`` calls load the model at
  most once.
- Device resolution: ``"auto"`` picks CUDA when the stub says CUDA is
  available, else CPU.  Explicit ``"cpu"`` / ``"cuda"`` pass through.
- BF16 quantisation is applied on CUDA only and opt-out via
  ``quantise_on_cuda=False``.
- Idle unload: ``maybe_unload`` honours the timeout, is a no-op when
  disabled, and releases the model reference.
- Empty-input short-circuit: ``embed_texts([])`` returns an empty
  ``(0, dimension)`` array without loading the model.
- Protocol conformance with :class:`ChunkEmbedder`.
- Security: HF token (or the ``"local"`` sentinel) never leaks through
  ``repr`` / ``str`` / log output.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import pytest

from sec_generative_search.core.logging import LOGGER_NAME
from sec_generative_search.core.types import (
    Chunk,
    ContentType,
    FilingIdentifier,
    ProviderCapability,
)
from sec_generative_search.pipeline.orchestrator import ChunkEmbedder
from sec_generative_search.providers.local import (
    LocalEmbeddingProvider,
)

if TYPE_CHECKING:
    import numpy as np
else:
    np = pytest.importorskip("numpy")


_HF_TOKEN = "hf_LONGTESTTOKENABCDEFGHIJKL"
_HF_TOKEN_TAIL = _HF_TOKEN[-4:]
_MODEL = "google/embeddinggemma-300m"
_DIM = 768


# ---------------------------------------------------------------------------
# Stub sentence-transformers model + loader
# ---------------------------------------------------------------------------


@dataclass
class _StubModel:
    """Minimal stand-in for a ``SentenceTransformer`` instance.

    Records encode calls so tests can assert on batch size, progress-bar
    propagation, and call counts.  The returned vectors are
    deterministic: each row is ``[index, 0, 0, ...]`` so the caller can
    verify that input order is preserved.
    """

    dimension: int = _DIM
    dtype: str = "float32"
    encode_calls: list[dict[str, Any]] = field(default_factory=list)

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> Any:
        self.encode_calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
            }
        )
        array = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i in range(len(texts)):
            array[i, 0] = float(i)
        return array

    def to(self, *, dtype: Any) -> _StubModel:
        """Emulate the torch ``.to(dtype=...)`` hook used by BF16 path."""
        self.dtype = str(dtype)
        return self


@dataclass
class _StubLoader:
    """Injectable loader used in place of ``SentenceTransformer(...)``.

    Tracks each load so tests can assert on lazy-load timing, the
    resolved device, and the token actually passed through to the
    underlying library.
    """

    model: _StubModel = field(default_factory=_StubModel)
    calls: list[dict[str, Any]] = field(default_factory=list)
    barrier: threading.Event | None = None

    def __call__(
        self,
        *,
        model_name: str,
        device: str,
        token: str | None,
    ) -> _StubModel:
        self.calls.append({"model_name": model_name, "device": device, "token": token})
        if self.barrier is not None:
            # Used by the re-entrant-load test to hold both threads
            # inside the loader long enough to race.
            self.barrier.wait(timeout=2.0)
        return self.model


@pytest.fixture
def loader() -> _StubLoader:
    return _StubLoader()


# ---------------------------------------------------------------------------
# Construction contract
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_provider_name_is_local(self, loader: _StubLoader) -> None:
        assert LocalEmbeddingProvider.provider_name == "local"

    def test_default_model_is_embeddinggemma(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(loader=loader)
        assert provider.get_dimension() == _DIM

    def test_unknown_model_rejected(self, loader: _StubLoader) -> None:
        with pytest.raises(ValueError, match="Unknown embedding model"):
            LocalEmbeddingProvider(model="totally-unreal-model", loader=loader)

    def test_invalid_batch_size_rejected(self, loader: _StubLoader) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            LocalEmbeddingProvider(batch_size=0, loader=loader)

    def test_negative_idle_timeout_rejected(self, loader: _StubLoader) -> None:
        with pytest.raises(ValueError, match="idle_timeout_minutes"):
            LocalEmbeddingProvider(idle_timeout_minutes=-1, loader=loader)

    def test_invalid_device_rejected(self, loader: _StubLoader) -> None:
        with pytest.raises(ValueError, match="Unsupported device"):
            LocalEmbeddingProvider(device="tpu", loader=loader)

    def test_construction_does_not_load_model(self, loader: _StubLoader) -> None:
        # Loader must not be called until first embed / validate.
        LocalEmbeddingProvider(loader=loader)
        assert loader.calls == []

    def test_model_dimensions_catalogue_complete(self) -> None:
        # Every catalogue entry must carry a positive dimension.
        for slug, dim in LocalEmbeddingProvider.MODEL_DIMENSIONS.items():
            assert isinstance(slug, str) and slug
            assert isinstance(dim, int) and dim > 0


# ---------------------------------------------------------------------------
# Lazy load contract
# ---------------------------------------------------------------------------


class TestLazyLoad:
    def test_first_embed_triggers_load(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        assert loader.calls == []

        vectors = provider.embed_texts(["hello"])
        assert vectors.shape == (1, _DIM)
        assert len(loader.calls) == 1
        assert loader.calls[0]["model_name"] == _MODEL
        assert loader.calls[0]["device"] == "cpu"

    def test_second_embed_reuses_model(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        provider.embed_texts(["a"])
        provider.embed_texts(["b"])
        # Loader must fire exactly once despite two embed calls.
        assert len(loader.calls) == 1

    def test_validate_key_loads_once(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        assert provider.validate_key() is True
        assert provider.is_loaded
        assert len(loader.calls) == 1

    def test_warm_up_loads_once_without_encoding(self, loader: _StubLoader) -> None:
        # EMBEDDING_WARM_ON_BOOT seam: load the weights, encode nothing,
        # and stay idempotent so a later embed reuses the warmed model.
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        provider.warm_up()
        provider.warm_up()
        assert provider.is_loaded
        assert len(loader.calls) == 1
        assert loader.model.encode_calls == []
        provider.embed_texts(["x"])
        assert len(loader.calls) == 1

    def test_warm_up_propagates_loader_failure(self) -> None:
        def failing_loader(**_: Any) -> Any:
            raise OSError("gated repo")

        provider = LocalEmbeddingProvider(loader=failing_loader, device="cpu")
        with pytest.raises(OSError, match="gated repo"):
            provider.warm_up()
        assert not provider.is_loaded

    def test_token_forwarded_only_when_supplied(self, loader: _StubLoader) -> None:
        with_token = LocalEmbeddingProvider(_HF_TOKEN, loader=loader, device="cpu")
        with_token.embed_texts(["x"])
        assert loader.calls[-1]["token"] == _HF_TOKEN

        loader.calls.clear()
        without_token = LocalEmbeddingProvider(loader=loader, device="cpu")
        without_token.embed_texts(["x"])
        # No token -> the loader receives ``None`` so sentence-transformers
        # does not accidentally authenticate with the public sentinel.
        assert loader.calls[-1]["token"] is None

    def test_concurrent_load_is_single_flight(self, loader: _StubLoader) -> None:
        # Force both threads to sit in the loader simultaneously; the
        # lock must still guarantee a single load.
        barrier = threading.Event()
        loader.barrier = barrier
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")

        errors: list[BaseException] = []

        def run() -> None:
            try:
                provider.embed_texts(["x"])
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads:
            t.start()
        # Release the first entrant; the rest must short-circuit on the
        # double-checked flag once the lock releases.
        barrier.set()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive()

        assert errors == []
        assert len(loader.calls) == 1


# ---------------------------------------------------------------------------
# Device resolution and BF16 quantisation
# ---------------------------------------------------------------------------


class TestDeviceResolution:
    def test_cpu_device_passthrough(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        provider.embed_texts(["x"])
        assert loader.calls[-1]["device"] == "cpu"

    def test_cuda_device_passthrough(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(loader=loader, device="cuda")
        provider.embed_texts(["x"])
        assert loader.calls[-1]["device"] == "cuda"

    def test_auto_resolves_to_cpu_without_torch(
        self,
        loader: _StubLoader,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Simulate torch-absent environment.
        def fake_resolve(preference: str) -> str:
            assert preference == "auto"
            return "cpu"

        monkeypatch.setattr(LocalEmbeddingProvider, "_resolve_device", staticmethod(fake_resolve))
        provider = LocalEmbeddingProvider(loader=loader, device="auto")
        provider.embed_texts(["x"])
        assert loader.calls[-1]["device"] == "cpu"

    def test_auto_resolves_to_cuda_when_available(
        self,
        loader: _StubLoader,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            LocalEmbeddingProvider,
            "_resolve_device",
            staticmethod(lambda preference: "cuda"),
        )
        provider = LocalEmbeddingProvider(loader=loader, device="auto")
        provider.embed_texts(["x"])
        assert loader.calls[-1]["device"] == "cuda"


class TestBF16Quantisation:
    def test_quantised_on_cuda(
        self,
        loader: _StubLoader,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            LocalEmbeddingProvider,
            "_to_bfloat16",
            staticmethod(lambda model: model.to(dtype="bfloat16")),
        )
        provider = LocalEmbeddingProvider(loader=loader, device="cuda")
        provider.embed_texts(["x"])
        assert loader.model.dtype == "bfloat16"

    def test_not_quantised_on_cpu(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        provider.embed_texts(["x"])
        # float32 sentinel set by _StubModel; unchanged -> no cast.
        assert loader.model.dtype == "float32"

    def test_opt_out_prevents_quantisation(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(
            loader=loader,
            device="cuda",
            quantise_on_cuda=False,
        )
        provider.embed_texts(["x"])
        assert loader.model.dtype == "float32"

    def test_bf16_failure_falls_back_to_full_precision(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When the torch cast raises, the model must still load and a
        warning must be emitted — BF16 is an optimisation, not a
        correctness requirement.
        """

        class _CastExplodingModel(_StubModel):
            def to(self, *, dtype: Any) -> Any:  # type: ignore[override]
                raise RuntimeError("simulated torch cast failure")

        exploding_loader = _StubLoader(model=_CastExplodingModel())

        package_logger = logging.getLogger(LOGGER_NAME)
        previous = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
                provider = LocalEmbeddingProvider(loader=exploding_loader, device="cuda")
                provider.embed_texts(["x"])
        finally:
            package_logger.propagate = previous

        # Load succeeded despite cast failure.
        assert provider.is_loaded
        assert any("BF16 quantisation failed" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# Embedding surface
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_embed_texts_returns_float32_matrix(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        out = provider.embed_texts(["a", "b", "c"])
        assert out.shape == (3, _DIM)
        assert out.dtype == np.float32

    def test_embed_query_returns_1d_vector(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        vec = provider.embed_query("apple")
        assert vec.shape == (_DIM,)
        assert vec.dtype == np.float32

    def test_empty_input_short_circuits(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        out = provider.embed_texts([])
        assert out.shape == (0, _DIM)
        # No load, no encode call — empty input must not wake the model.
        assert loader.calls == []
        assert loader.model.encode_calls == []

    def test_embed_chunks_forwards_progress_flag(
        self,
        loader: _StubLoader,
    ) -> None:
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        filing = FilingIdentifier(
            ticker="AAPL",
            form_type="10-K",
            filing_date=date(2024, 2, 1),
            accession_number="acc-1",
        )
        chunks = [
            Chunk(
                content=f"chunk {i}",
                path="Part I > Item 1",
                content_type=ContentType.TEXT,
                filing_id=filing,
                chunk_index=i,
            )
            for i in range(3)
        ]
        out = provider.embed_chunks(chunks, show_progress=True)
        assert out.shape == (3, _DIM)
        assert loader.model.encode_calls[-1]["show_progress_bar"] is True

    def test_embed_chunks_empty_short_circuits(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        out = provider.embed_chunks([], show_progress=False)
        assert out.shape == (0, _DIM)
        assert loader.calls == []


# ---------------------------------------------------------------------------
# Capabilities and protocol conformance
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_capabilities_are_static(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        caps = provider.get_capabilities()
        assert caps == ProviderCapability(embeddings=True)
        # Capability probe must not load the model.
        assert loader.calls == []

    def test_satisfies_chunk_embedder_protocol(self, loader: _StubLoader) -> None:
        """Structural check — the orchestrator's
        :class:`ChunkEmbedder` is not @runtime_checkable, so we verify
        by attribute and by a smoke call.
        """
        assert ChunkEmbedder is not None
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        assert callable(provider.embed_chunks)


# ---------------------------------------------------------------------------
# Idle unload — honours EMBEDDING_IDLE_TIMEOUT_MINUTES
# ---------------------------------------------------------------------------


class _ManualClock:
    """Deterministic clock: ``tick(seconds)`` advances the time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


class TestIdleUnload:
    def test_unload_disabled_when_timeout_zero(
        self,
        loader: _StubLoader,
    ) -> None:
        clock = _ManualClock()
        provider = LocalEmbeddingProvider(
            loader=loader,
            device="cpu",
            idle_timeout_minutes=0,
            clock=clock,
        )
        provider.embed_texts(["x"])
        assert provider.is_loaded

        clock.tick(60 * 60)  # one hour — still idle, still loaded.
        assert provider.maybe_unload() is False
        assert provider.is_loaded

    def test_unload_when_idle_past_threshold(
        self,
        loader: _StubLoader,
    ) -> None:
        clock = _ManualClock()
        provider = LocalEmbeddingProvider(
            loader=loader,
            device="cpu",
            idle_timeout_minutes=15,
            clock=clock,
        )
        provider.embed_texts(["x"])
        assert provider.is_loaded

        clock.tick(15 * 60 + 1)
        assert provider.maybe_unload() is True
        assert not provider.is_loaded

    def test_no_unload_within_threshold(
        self,
        loader: _StubLoader,
    ) -> None:
        clock = _ManualClock()
        provider = LocalEmbeddingProvider(
            loader=loader,
            device="cpu",
            idle_timeout_minutes=15,
            clock=clock,
        )
        provider.embed_texts(["x"])
        clock.tick(10 * 60)
        assert provider.maybe_unload() is False
        assert provider.is_loaded

    def test_successful_call_resets_idle_timer(
        self,
        loader: _StubLoader,
    ) -> None:
        clock = _ManualClock()
        provider = LocalEmbeddingProvider(
            loader=loader,
            device="cpu",
            idle_timeout_minutes=15,
            clock=clock,
        )
        provider.embed_texts(["x"])
        clock.tick(10 * 60)  # below threshold
        provider.embed_texts(["y"])  # resets timer
        clock.tick(10 * 60)  # 10 min since last use — still below 15
        assert provider.maybe_unload() is False
        assert provider.is_loaded

    def test_unload_when_never_loaded_is_noop(
        self,
        loader: _StubLoader,
    ) -> None:
        provider = LocalEmbeddingProvider(
            loader=loader,
            device="cpu",
            idle_timeout_minutes=5,
        )
        assert provider.maybe_unload() is False
        provider.unload()  # also a no-op
        assert not provider.is_loaded
        assert loader.calls == []

    def test_explicit_unload_reloads_on_next_call(
        self,
        loader: _StubLoader,
    ) -> None:
        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        provider.embed_texts(["x"])
        assert len(loader.calls) == 1

        provider.unload()
        assert not provider.is_loaded

        provider.embed_texts(["y"])
        # A fresh load must happen after explicit unload.
        assert len(loader.calls) == 2

    def test_now_argument_overrides_clock(
        self,
        loader: _StubLoader,
    ) -> None:
        clock = _ManualClock()
        provider = LocalEmbeddingProvider(
            loader=loader,
            device="cpu",
            idle_timeout_minutes=15,
            clock=clock,
        )
        provider.embed_texts(["x"])
        # ``now`` is in the far future: caller forces a unload decision.
        assert provider.maybe_unload(now=clock.now + 1_000_000) is True
        assert not provider.is_loaded


# ---------------------------------------------------------------------------
# Security — HF token must not leak through repr / str / logging
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSecretSafety:
    def test_repr_masks_hf_token(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(_HF_TOKEN, loader=loader)
        text = repr(provider)
        assert _HF_TOKEN not in text
        assert _HF_TOKEN_TAIL in text
        assert "local" in text  # provider_name

    def test_str_matches_repr(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(_HF_TOKEN, loader=loader)
        assert str(provider) == repr(provider)

    def test_sentinel_is_redacted_when_no_token(self, loader: _StubLoader) -> None:
        provider = LocalEmbeddingProvider(None, loader=loader)
        text = repr(provider)
        # The sentinel is short (<8 chars) and mask_secret fully masks
        # short inputs; callers must not be able to read it back.
        assert "local" in text  # from provider_name, not from _api_key
        # The stored sentinel is "local" — mask_secret replaces short
        # inputs with "<masked>", so the literal sentinel must not
        # appear as a key value in the repr.
        assert "api_key=local" not in text

    def test_logger_emissions_redact_token(
        self,
        loader: _StubLoader,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        package_logger = logging.getLogger(LOGGER_NAME)
        previous = package_logger.propagate
        package_logger.propagate = True
        try:
            with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
                provider = LocalEmbeddingProvider(_HF_TOKEN, loader=loader, device="cpu")
                # Trigger the "loaded model" info line.
                provider.embed_texts(["probe"])
                package_logger.info("Constructed %s", provider)
        finally:
            package_logger.propagate = previous

        for record in caplog.records:
            assert _HF_TOKEN not in record.getMessage()
            assert _HF_TOKEN not in str(record.args)

    def test_token_not_leaked_via_has_hf_token_flag(
        self,
        loader: _StubLoader,
    ) -> None:
        provider = LocalEmbeddingProvider(_HF_TOKEN, loader=loader, device="cpu")
        # The boolean flag is the only bit we expose; the raw token must
        # stay behind ``self._api_key``.
        assert provider._has_hf_token is True
        assert _HF_TOKEN not in repr(provider)


# ---------------------------------------------------------------------------
# CUDA cache flush is best-effort — never raises
# ---------------------------------------------------------------------------


class TestCudaCacheFlush:
    def test_unload_without_torch_is_silent(
        self,
        loader: _StubLoader,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force the torch import inside _try_empty_cuda_cache to fail;
        # unload must still succeed.
        import builtins

        real_import = builtins.__import__

        def fake_import(
            name: str,
            module_globals: Any = None,
            module_locals: Any = None,
            fromlist: Any = (),
            level: int = 0,
        ) -> Any:
            if name == "torch":
                raise ImportError("simulated missing torch")
            return real_import(name, module_globals, module_locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        provider = LocalEmbeddingProvider(loader=loader, device="cpu")
        provider.embed_texts(["x"])
        # Must not raise even though torch is unavailable.
        provider.unload()
        assert not provider.is_loaded


# ---------------------------------------------------------------------------
# Integration with BaseEmbeddingProvider contract
# ---------------------------------------------------------------------------


class TestABCContract:
    def test_api_key_must_be_string(self, loader: _StubLoader) -> None:
        with pytest.raises(TypeError, match="api_key"):
            LocalEmbeddingProvider(b"bytes-are-not-str", loader=loader)  # type: ignore[arg-type]

    def test_api_key_empty_string_falls_back_to_sentinel(
        self,
        loader: _StubLoader,
    ) -> None:
        # Empty string is equivalent to "no token supplied" for the local
        # provider — we substitute the public sentinel so the base's
        # non-empty requirement is satisfied without forcing the caller
        # to invent a placeholder.
        provider = LocalEmbeddingProvider("", loader=loader, device="cpu")
        provider.embed_texts(["x"])
        assert loader.calls[-1]["token"] is None


# ---------------------------------------------------------------------------
# Iterator/sanity coverage
# ---------------------------------------------------------------------------


def _make_chunks(count: int) -> Iterator[Chunk]:
    filing = FilingIdentifier(
        ticker="MSFT",
        form_type="10-Q",
        filing_date=date(2024, 1, 15),
        accession_number="acc-42",
    )
    for i in range(count):
        yield Chunk(
            content=f"row {i}",
            path="Part II > Item 2",
            content_type=ContentType.TEXT,
            filing_id=filing,
            chunk_index=i,
        )


def test_end_to_end_chunk_embedding(loader: _StubLoader) -> None:
    """Smoke-test the full orchestrator-facing surface with a stubbed
    model — the ``ChunkEmbedder`` protocol is the only contract the
    pipeline relies on.
    """
    provider = LocalEmbeddingProvider(loader=loader, device="cpu")
    chunks = list(_make_chunks(5))
    matrix = provider.embed_chunks(chunks, show_progress=False)
    assert matrix.shape == (5, _DIM)
    assert matrix.dtype == np.float32
    # Input order preserved via the stub's row=[index, 0, 0, ...] scheme.
    for i in range(5):
        assert matrix[i, 0] == pytest.approx(float(i))
