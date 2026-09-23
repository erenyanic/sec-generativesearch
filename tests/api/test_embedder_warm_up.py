"""Lifespan embedder warm-up (``EMBEDDING_WARM_ON_BOOT``, F25).

The production lifespan is driven for real, but only as far as the storage
chain: ``build_embedder`` / ``get_settings`` / the registry dimension lookup
are patched to stubs and
``ChromaDBClient`` is replaced by a sentinel that records its construction and
then aborts the boot. Warm-up runs *before* any storage opens, so that is
enough to prove the three properties that matter:

- the warm-up runs before the server can accept traffic (ahead of storage);
- the flag's default (off) keeps the historical lazy boot;
- a failed warm-up refuses the boot and neither the log stream nor the raised
  message carries the underlying exception text.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI

from sec_generative_search.core.exceptions import ConfigurationError
from sec_generative_search.core.logging import LOGGER_NAME

# ``sec_generative_search.api`` re-exports the FastAPI ``app`` instance, which
# shadows the submodule on attribute import — resolve the module explicitly.
app_module = importlib.import_module("sec_generative_search.api.app")

_MODEL = "google/embeddinggemma-300m"
# Stand-in for whatever a loader might put in its exception text (a URL, a
# path, a token fragment). It must never reach a log record or the message.
_LEAKY_DETAIL = "hf_LEAKYDETAILSHOULDNOTAPPEAR /home/op/.cache/secret-path"


class _StorageReachedError(Exception):
    """Raised by the sentinel ChromaDBClient to stop the lifespan early."""


class _WarmableEmbedder:
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        self._events = events
        self._fail = fail

    def warm_up(self) -> None:
        self._events.append("warm_up")
        if self._fail:
            raise OSError(_LEAKY_DETAIL)


class _HostedLikeEmbedder:
    """No ``warm_up`` attribute — the duck-typed seam must skip it."""


def _settings(*, warm_on_boot: bool) -> Any:
    return SimpleNamespace(
        embedding=SimpleNamespace(provider="local", model_name=_MODEL, warm_on_boot=warm_on_boot),
    )


def _boot(
    monkeypatch: pytest.MonkeyPatch,
    *,
    embedder: object,
    warm_on_boot: bool,
    events: list[str],
) -> None:
    def chroma_sentinel(*_: Any, **__: Any) -> None:
        events.append("storage")
        raise _StorageReachedError

    monkeypatch.setattr(app_module, "get_settings", lambda: _settings(warm_on_boot=warm_on_boot))
    monkeypatch.setattr(app_module, "build_embedder", lambda _cfg: embedder)
    # The real dimension lookup gates on the [local-embeddings] extra, which
    # CI does not install.
    monkeypatch.setattr(
        app_module, "ProviderRegistry", SimpleNamespace(get_dimension=lambda _p, _m: 768)
    )
    monkeypatch.setattr(app_module, "ChromaDBClient", chroma_sentinel)

    async def enter() -> None:
        async with app_module.lifespan(FastAPI()):
            pass  # pragma: no cover — the sentinel always aborts first

    asyncio.run(enter())


class TestEmbedderWarmUp:
    def test_warm_up_runs_before_storage_opens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events: list[str] = []
        with pytest.raises(_StorageReachedError):
            _boot(
                monkeypatch,
                embedder=_WarmableEmbedder(events),
                warm_on_boot=True,
                events=events,
            )
        assert events == ["warm_up", "storage"]

    def test_flag_off_keeps_the_lazy_boot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events: list[str] = []
        with pytest.raises(_StorageReachedError):
            _boot(
                monkeypatch,
                embedder=_WarmableEmbedder(events),
                warm_on_boot=False,
                events=events,
            )
        assert events == ["storage"]

    def test_embedder_without_warm_up_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events: list[str] = []
        with pytest.raises(_StorageReachedError):
            _boot(
                monkeypatch,
                embedder=_HostedLikeEmbedder(),
                warm_on_boot=True,
                events=events,
            )
        assert events == ["storage"]


@pytest.mark.security
class TestEmbedderWarmUpFailure:
    def test_failed_warm_up_refuses_boot_without_leaking_detail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``configure_logging`` sets ``propagate = False`` on the package
        # logger, so attach a capturing handler to it directly.
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Capture(level=logging.DEBUG)
        package_logger = logging.getLogger(LOGGER_NAME)
        package_logger.addHandler(handler)
        previous_level = package_logger.level
        package_logger.setLevel(logging.DEBUG)
        events: list[str] = []
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                _boot(
                    monkeypatch,
                    embedder=_WarmableEmbedder(events, fail=True),
                    warm_on_boot=True,
                    events=events,
                )
        finally:
            package_logger.removeHandler(handler)
            package_logger.setLevel(previous_level)

        # Fail-fast: the boot stops at the warm-up; storage never opens.
        assert events == ["warm_up"]
        assert "EMBEDDING_WARM_ON_BOOT" in str(exc_info.value)
        assert _LEAKY_DETAIL not in str(exc_info.value)

        rendered = [r.getMessage() for r in records]
        failure_lines = [m for m in rendered if "warm-up failed" in m]
        assert failure_lines, f"no warm-up failure line was logged: {rendered}"
        # Exception *type* only — never its text.
        assert all("OSError" in m for m in failure_lines)
        for message in rendered:
            assert _LEAKY_DETAIL not in message
            assert "hf_LEAKY" not in message
