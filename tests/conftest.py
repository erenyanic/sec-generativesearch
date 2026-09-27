"""Shared pytest fixtures for SEC-GenerativeSearch.

Currently focused on foundation tests. As more subsystems land,
fixtures for HTTP clients, mocked
providers, and temporary ChromaDB/SQLite stores will be added here.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _restore_settings_singleton() -> Iterator[None]:
    """Put the settings singleton back to the object it was before the test.

    Fixtures that call ``reload_settings()`` in teardown run *before*
    ``monkeypatch`` restores the environment (teardown is reverse setup
    order), so the rebuilt singleton can carry a test's patched env — e.g.
    ``DB_MAX_FILINGS=1`` from ``tests/api/test_task_manager.py`` — into the
    next test.  Serially the alphabetical file order hid it; under
    ``pytest -n auto`` a worker ran ``tests/database`` straight after and
    failed (OPTIMIZATIONS.md F31).  Restoring the pre-test object instead
    of rebuilding costs nothing per test.
    """
    from sec_generative_search.config import settings as settings_module

    before = settings_module._settings_instance
    yield
    settings_module._settings_instance = before


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Remove all env vars that influence Settings defaults.

    Used by settings tests so that one test's env var does not leak into
    another.  Pydantic Settings reads os.environ at instantiation time,
    so isolating that environment keeps tests deterministic.
    """
    prefixes = (
        "EDGAR_",
        "EMBEDDING_",
        "CHUNKING_",
        "DB_",
        "LLM_",
        "LOCAL_LLM_",
        "PROVIDER_",
        "RAG_",
        "SEARCH_",
        "LOG_FILE_",
        "LOG_LEVEL",
        "LOG_REDACT_QUERIES",
        "HUGGING_FACE_",
        "API_",
    )
    for key in list(os.environ.keys()):
        if key.startswith(prefixes):
            monkeypatch.delenv(key, raising=False)
    yield monkeypatch
