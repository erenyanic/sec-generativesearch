"""Local embedding provider backed by ``sentence-transformers``.

Runs entirely on the user's machine — no outbound network call once the
model weights are cached locally.  Suits the Scenario A (local) and
Scenario B (team) profiles where operators would rather pay a one-off
model-download cost than stream SEC filings through a hosted embedding
endpoint.

Key design choices:

- ``sentence-transformers`` and :mod:`torch` are **optional** runtime
  dependencies (`[local-embeddings]` extra).  Every import of them is
  lazy so that a deployment without the extra never pays the import
  cost and the unit tests can stub the loader without a GPU.

- The Hugging Face access token is carried in the standard ``api_key``
  slot defined by :class:`_ProviderBase`, so the existing redaction and
  no-leak guarantees apply unchanged.  When no token is supplied we
  fall back to a non-secret sentinel — it is <8 characters long and is
  therefore fully masked by :func:`mask_secret`.

- Model load is deferred to first use (:meth:`_ensure_model`) and the
  loader is an injectable callable so CI can avoid downloading a real
  model.  Re-entrant loads are serialised by ``self._load_lock``.

- BF16 quantisation is opt-in on CUDA (halves VRAM) and skipped on CPU
  where BF16 is poorly supported and often slower than float32.

- ``maybe_unload()`` implements the ``EMBEDDING_IDLE_TIMEOUT_MINUTES``
  contract by letting the orchestrator / lifespan layer release the
  model when it has been idle for the configured window.  The clock
  is injectable for deterministic tests; we deliberately do *not*
  spawn a background ``Timer`` thread — that would complicate teardown
  and fixture state without any operational benefit.

The provider is stateless with respect to per-call input: nothing from
:meth:`embed_texts` is cached across calls.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from sec_generative_search.core.logging import get_logger
from sec_generative_search.core.types import Chunk, ProviderCapability
from sec_generative_search.providers.base import BaseEmbeddingProvider

if TYPE_CHECKING:
    import numpy as np


__all__ = [
    "LocalEmbeddingProvider",
]


logger = get_logger(__name__)


# Public sentinel for "no Hugging Face token supplied".  NOT a secret —
# short enough that :func:`mask_secret` renders it as ``"<masked>"``
# rather than leaking a recognisable tail.  Using a sentinel keeps the
# base-class contract (``api_key`` must be a non-empty string) intact
# while still allowing public, un-gated models to load without any
# credential at all.
_NO_TOKEN_SENTINEL = "local"


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """On-device embedding provider using ``sentence-transformers``.

    The provider owns the model handle, the resolved device, and the
    timestamp of the last successful call.  None of these travel with
    the :class:`GenerationRequest` / :class:`GenerationResponse`
    dataclasses, which remain credential-free.

    The public surface mirrors the other embedding providers so that
    callers can swap one in via the :class:`ChunkEmbedder` protocol
    without any adapter:

    >>> provider = LocalEmbeddingProvider(
    ...     hf_token=None,
    ...     model="google/embeddinggemma-300m",
    ...     device="auto",
    ...     idle_timeout_minutes=15,
    ... )
    >>> provider.get_dimension()
    768
    """

    provider_name = "local"
    default_model: ClassVar[str] = "google/embeddinggemma-300m"

    # Native output dimension for each supported model.  Models must be
    # listed here so the collection dimension can be stamped before the
    # first embed call (ChromaDB collections are dimension-locked on
    # creation).
    #
    # Adding a model is a one-line change; unknown models are rejected
    # at construction rather than silently permitted.
    MODEL_DIMENSIONS: ClassVar[dict[str, int]] = {
        "google/embeddinggemma-300m": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
    }

    def __init__(
        self,
        hf_token: str | None = None,
        *,
        model: str | None = None,
        device: str = "auto",
        idle_timeout_minutes: int = 0,
        batch_size: int = 32,
        quantise_on_cuda: bool = True,
        loader: Callable[..., Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Construct the provider without loading the model.

        Args:
            hf_token: Hugging Face access token for gated models (e.g.
                ``google/embeddinggemma-300m``).  ``None`` is fine for
                public models.  When supplied it is validated and
                redacted via the :class:`_ProviderBase` contract.
            model: Model slug; must appear in :attr:`MODEL_DIMENSIONS`.
                ``None`` falls back to :attr:`default_model`.
            device: ``"auto"`` / ``"cuda"`` / ``"cpu"``.  ``"auto"``
                resolves to CUDA when :mod:`torch` reports it
                available, else CPU.  Resolution is deferred to
                :meth:`_ensure_model` so construction never imports
                torch.
            idle_timeout_minutes: When > 0, :meth:`maybe_unload`
                releases the model after this many minutes without a
                successful call.  ``0`` disables auto-unload.
            batch_size: Encode batch size — tune for low-VRAM GPUs.
            quantise_on_cuda: When ``True`` (default), cast the model
                to BF16 once it is on a CUDA device.  Ignored on CPU.
            loader: Injectable callable with signature
                ``(model_name, device, token) -> SentenceTransformer``.
                Defaults to the real :class:`sentence_transformers.SentenceTransformer`
                constructor.  The tests pass a stub so CI never
                downloads weights.
            clock: Injectable time source, defaults to
                :func:`time.monotonic`.  The idle-unload tests inject a
                controllable clock.
        """
        token = hf_token if hf_token else _NO_TOKEN_SENTINEL
        super().__init__(token)
        self._has_hf_token = bool(hf_token)

        resolved_model = model or self.default_model
        if resolved_model not in self.MODEL_DIMENSIONS:
            raise ValueError(
                f"Unknown embedding model '{resolved_model}' for "
                f"{self.provider_name}. Add it to "
                f"LocalEmbeddingProvider.MODEL_DIMENSIONS."
            )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if idle_timeout_minutes < 0:
            raise ValueError("idle_timeout_minutes must be >= 0")
        if device not in {"auto", "cuda", "cpu"} and not device.startswith("cuda:"):
            raise ValueError(
                f"Unsupported device '{device}'. Use 'auto', 'cpu', 'cuda', or 'cuda:<index>'."
            )

        self._model_slug = resolved_model
        self._device_preference = device
        self._idle_timeout_seconds = idle_timeout_minutes * 60
        self._batch_size = batch_size
        self._quantise_on_cuda = quantise_on_cuda
        self._loader = loader  # None -> real sentence-transformers loader
        self._clock = clock or time.monotonic

        # Lazily populated — see ``_ensure_model``.
        self._model: Any = None
        self._resolved_device: str | None = None
        self._last_used: float | None = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API — BaseEmbeddingProvider
    # ------------------------------------------------------------------

    def validate_key(self) -> bool:
        """Load the model as the cheapest authenticated round-trip.

        For a public model this confirms the weights are reachable; for
        a gated model it confirms the Hugging Face token is accepted.
        Failure surfaces as the underlying loader's exception — the
        caller is responsible for mapping it if required.  Returns
        ``True`` on success.
        """
        self._ensure_model()
        return True

    def get_capabilities(self) -> ProviderCapability:
        """Static capability probe — never touches the network."""
        return ProviderCapability(embeddings=True)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed *texts* as a ``(len(texts), dimension)`` float32 array.

        Empty input short-circuits without loading the model — matches
        the behaviour of every hosted embedding adapter and avoids a
        cold-start for trivial calls.
        """
        import numpy as np

        if not texts:
            return np.zeros((0, self.get_dimension()), dtype=np.float32)

        self._ensure_model()
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self._mark_used()
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query as a ``(dimension,)`` float32 array."""
        return self.embed_texts([text])[0]

    def embed_chunks(
        self,
        chunks: list[Chunk],
        *,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Override the base default so the progress flag reaches the
        ``sentence-transformers`` encoder.  Large corpora benefit from
        the inline progress bar when the CLI drives ingestion.
        """
        import numpy as np

        if not chunks:
            return np.zeros((0, self.get_dimension()), dtype=np.float32)

        self._ensure_model()
        vectors = self._model.encode(
            [c.content for c in chunks],
            batch_size=self._batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        self._mark_used()
        return np.asarray(vectors, dtype=np.float32)

    def get_dimension(self) -> int:
        return self.MODEL_DIMENSIONS[self._model_slug]

    # ------------------------------------------------------------------
    # Idle unload — honours EMBEDDING_IDLE_TIMEOUT_MINUTES
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """``True`` when the model is currently resident in memory."""
        return self._model is not None

    def warm_up(self) -> None:
        """Load the model now instead of on the first embed call.

        Backs ``EMBEDDING_WARM_ON_BOOT``: the API lifespan calls it
        before the server binds, so a cold weight download finishes
        before the startup probe passes rather than inside a user
        request.  Idempotent; loader errors propagate unchanged so the
        caller can fail the boot.
        """
        self._ensure_model()

    def maybe_unload(self, now: float | None = None) -> bool:
        """Unload the model when it has been idle past the threshold.

        Called by the orchestrator / API lifespan at natural
        checkpoints (after each filing, on session close, etc.).
        Returns ``True`` when an unload actually happened.

        Disabled when ``idle_timeout_minutes == 0`` or when the model
        has never been loaded.  The clock source is injectable via the
        constructor so tests do not need to sleep.
        """
        if self._idle_timeout_seconds == 0:
            return False
        if self._model is None or self._last_used is None:
            return False
        current = now if now is not None else self._clock()
        if current - self._last_used < self._idle_timeout_seconds:
            return False
        self.unload()
        return True

    def unload(self) -> None:
        """Drop the model reference and empty the CUDA cache if present.

        Safe to call when the model was never loaded.  The cache flush
        is best-effort — failures to import torch or talk to the driver
        are logged and swallowed so ingestion teardown never raises
        from a cleanup hook.
        """
        with self._load_lock:
            if self._model is None:
                return
            logger.info(
                "LocalEmbeddingProvider: unloading model=%s (device=%s)",
                self._model_slug,
                self._resolved_device,
            )
            self._model = None
            self._last_used = None
            self._resolved_device = None
            self._try_empty_cuda_cache()

    # ------------------------------------------------------------------
    # Internals — model load, device resolution, quantisation
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Lazy, thread-safe model load.

        The double-checked pattern avoids acquiring the lock on the hot
        path once the model is loaded.  All network / filesystem I/O is
        owned by the injected loader; this method is responsible only
        for device resolution and optional quantisation.
        """
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return

            resolved_device = self._resolve_device(self._device_preference)
            loader = self._loader or self._default_loader
            token = self._api_key if self._has_hf_token else None

            model = loader(
                model_name=self._model_slug,
                device=resolved_device,
                token=token,
            )

            if self._quantise_on_cuda and self._is_cuda(resolved_device):
                model = self._to_bfloat16(model)

            self._model = model
            self._resolved_device = resolved_device
            self._mark_used()

            logger.info(
                "LocalEmbeddingProvider: loaded model=%s (device=%s, quantised=%s)",
                self._model_slug,
                resolved_device,
                self._quantise_on_cuda and self._is_cuda(resolved_device),
            )

    def _mark_used(self) -> None:
        """Refresh the idle timer.  Called on every successful embed."""
        self._last_used = self._clock()

    @staticmethod
    def _resolve_device(preference: str) -> str:
        """Resolve ``"auto"`` into the best available concrete device.

        ``"cuda"`` and ``"cpu"`` pass through unchanged.  Torch import
        failures are treated as "no CUDA" — the provider still runs on
        CPU even if a torch installation is partially broken.
        """
        if preference != "auto":
            return preference
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @staticmethod
    def _is_cuda(device: str | None) -> bool:
        return bool(device and device.startswith("cuda"))

    @staticmethod
    def _to_bfloat16(model: Any) -> Any:
        """Cast a loaded model to BF16, falling back to full precision.

        ``.to(dtype=...)`` is the portable way across the
        ``sentence-transformers`` versions we support.  If torch or the
        model rejects the cast we log a warning and return the original
        — BF16 is a nice-to-have, not a correctness requirement.
        """
        try:
            import torch

            return model.to(dtype=torch.bfloat16)
        except Exception as exc:
            logger.warning(
                "LocalEmbeddingProvider: BF16 quantisation failed, "
                "continuing at full precision: %s",
                exc,
            )
            return model

    @staticmethod
    def _try_empty_cuda_cache() -> None:
        """Best-effort CUDA cache flush after unload."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            # torch may not be installed in profiles that never use the
            # local provider — that is fine, no cache to flush.
            return
        except Exception as exc:
            logger.warning(
                "LocalEmbeddingProvider: CUDA cache flush failed: %s",
                exc,
            )

    @staticmethod
    def _default_loader(
        *,
        model_name: str,
        device: str,
        token: str | None,
    ) -> Any:
        """Production loader — imports ``sentence-transformers`` lazily.

        The import is inside the function so modules that never
        instantiate the local provider (API-only, cloud embedding
        profiles) do not pay the ~2 GB torch import cost.
        """
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            model_name,
            device=device,
            token=token,
        )
