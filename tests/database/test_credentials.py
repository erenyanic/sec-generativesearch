"""Tests for :mod:`sec_generative_search.database.credentials` + migration v2.

Coverage:

- Migration v2 creates ``provider_credentials`` and is idempotent.
- ``EncryptedCredentialStore`` refuses construction without
  ``DB_PERSIST_PROVIDER_CREDENTIALS=true`` or without the registry's
  ``encrypted=True`` invariant.
- CRUD round-trips: set / get / upsert / delete / clear / list /
  count, with per-``user_id`` and per-``provider`` isolation.
- Security: raw key never appears in audit log, ``__repr__``,
  serialised :class:`CredentialRecord` (which deliberately omits
  ``api_key`` from its frozen fields), or any read API except the
  explicit ``get`` path.
- Schema-version stamping after migration v2.

Notes on the SQLCipher-skipped path: ``pysqlcipher3`` is not part of the
CI install (``[encryption]`` extra).  Tests
that exercise SQL paths force-set ``MetadataRegistry._encrypted = True``
under a ``# noqa: SLF001`` so the store's invariant check passes; the
store is only validated against plain-sqlite3 SQL semantics, which is
sufficient because SQLCipher's wire-level differences are confined to
the ``PRAGMA key`` setup that the registry already owns.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sec_generative_search.config.settings import DatabaseSettings
from sec_generative_search.core.exceptions import ConfigurationError
from sec_generative_search.core.logging import configure_logging
from sec_generative_search.database.credentials import (
    CredentialRecord,
    EncryptedCredentialStore,
)
from sec_generative_search.database.metadata import MetadataRegistry

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def persist_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> DatabaseSettings:
    """A ``DatabaseSettings`` configured for credential persistence.

    The path validator on :class:`DatabaseSettings` requires the
    ``chroma_path`` / ``metadata_db_path`` to live inside cwd, so the
    fixture ``chdir``s to ``tmp_path`` and uses ``./``-rooted paths
    that resolve there.  This mirrors the established pattern in
    ``tests/config/test_settings.py``.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chroma").mkdir()
    return DatabaseSettings(
        chroma_path="./chroma",
        metadata_db_path="./meta.sqlite",
        encryption_key="unit-test-key-not-a-secret",
        persist_provider_credentials=True,
    )


@pytest.fixture
def encrypted_registry(persist_settings: DatabaseSettings) -> Iterator[MetadataRegistry]:
    """A registry with ``encrypted`` forced ``True`` for SQL-path tests.

    pysqlcipher3 is not installed in CI; the registry would honestly
    report ``encrypted=False`` and the credential store would refuse to
    construct.  Force the flag so the SQL paths can be exercised on
    plain sqlite3 — the wire-level encryption is the registry's
    responsibility, validated separately in ``test_metadata.py``.

    ``persist_settings`` already ``chdir``'d to ``tmp_path``, so the
    relative ``metadata_db_path`` resolves correctly.
    """
    registry = MetadataRegistry(
        db_path=persist_settings.metadata_db_path,
        encryption_key=persist_settings.encryption_key,
    )
    registry._encrypted = True
    try:
        yield registry
    finally:
        registry.close()


@pytest.fixture
def encrypted_store(
    encrypted_registry: MetadataRegistry,
    persist_settings: DatabaseSettings,
) -> EncryptedCredentialStore:
    return EncryptedCredentialStore(encrypted_registry, settings=persist_settings)


@pytest.fixture
def audit_caplog(
    caplog: pytest.LogCaptureFixture,
) -> Iterator[pytest.LogCaptureFixture]:
    """See ``tests/core/test_credentials.py::audit_caplog`` for rationale.

    ``configure_logging()`` is forced first so the propagate flip below is not
    silently undone: ``audit_log`` → ``get_logger`` lazily configures logging
    on first use, and ``configure_logging`` resets ``propagate = False``.  If
    the module-level ``_logging_configured`` flag is ``False`` at the moment
    the audited call runs (e.g. after ``tests/core/test_logging.py`` resets it
    in teardown), that lazy reconfigure fires *after* this fixture sets
    ``propagate = True`` and flips it straight back to ``False`` — the audit
    record then reaches the package handler but never propagates to caplog's
    root handler.  Configuring up-front makes the lazy call an idempotent
    no-op, so ``propagate`` stays ``True`` through the emit.
    """
    configure_logging()
    pkg_logger = logging.getLogger("sec_generative_search")
    previous = pkg_logger.propagate
    pkg_logger.propagate = True
    caplog.set_level(logging.WARNING, logger="sec_generative_search.security.audit")
    try:
        yield caplog
    finally:
        pkg_logger.propagate = previous


# ---------------------------------------------------------------------------
# Migration v2 — provider_credentials table creation + stamping
# ---------------------------------------------------------------------------


class TestMigrationV2:
    def test_creates_provider_credentials_table(self, tmp_path: Path) -> None:
        """First-time open runs migration v2 and creates the table."""
        registry = MetadataRegistry(db_path=str(tmp_path / "meta.sqlite"))
        try:
            row = registry._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='provider_credentials'"
            ).fetchone()
            assert row is not None
        finally:
            registry.close()

    def test_stamps_schema_version_at_v2(self, tmp_path: Path) -> None:
        registry = MetadataRegistry(db_path=str(tmp_path / "meta.sqlite"))
        try:
            versions = [
                row[0]
                for row in registry._conn.execute(
                    "SELECT version FROM schema_version ORDER BY version"
                ).fetchall()
            ]
            assert 1 in versions
            assert 2 in versions
        finally:
            registry.close()

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        """Re-opening the database does not re-apply v2 (which would
        ``INSERT`` a duplicate version row and violate the PK)."""
        path = str(tmp_path / "meta.sqlite")
        MetadataRegistry(db_path=path).close()
        # Second open must succeed; the schema_version PK would reject
        # any duplicate stamp attempt.
        registry = MetadataRegistry(db_path=path)
        try:
            stamps = registry._conn.execute(
                "SELECT COUNT(*) FROM schema_version WHERE version = 2"
            ).fetchone()[0]
            assert stamps == 1
        finally:
            registry.close()


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


class TestConstructionGuards:
    def test_refuses_when_persist_disabled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "chroma").mkdir()
        registry = MetadataRegistry(db_path="./meta.sqlite")
        try:
            settings = DatabaseSettings(
                chroma_path="./chroma",
                metadata_db_path="./meta.sqlite",
                persist_provider_credentials=False,
            )
            with pytest.raises(ConfigurationError, match="DB_PERSIST_PROVIDER_CREDENTIALS"):
                EncryptedCredentialStore(registry, settings=settings)
        finally:
            registry.close()

    def test_refuses_when_registry_not_encrypted(
        self,
        persist_settings: DatabaseSettings,
        tmp_path: Path,
    ) -> None:
        """An unencrypted registry MUST be refused even when persistence
        is on — settings validation catches this at load, but the store
        re-checks defensively because a hand-crafted registry could
        bypass the settings layer entirely."""
        # cwd is already tmp_path (via the persist_settings fixture).
        registry = MetadataRegistry(db_path="./meta_alt.sqlite")
        # registry.encrypted is False (no key, no driver)
        try:
            with pytest.raises(ConfigurationError, match="without SQLCipher"):
                EncryptedCredentialStore(registry, settings=persist_settings)
        finally:
            registry.close()


# ---------------------------------------------------------------------------
# CRUD round-trips
# ---------------------------------------------------------------------------


class TestCRUD:
    def test_set_and_get(self, encrypted_store: EncryptedCredentialStore) -> None:
        encrypted_store.set("user-1", "openai", "sk-test-1234567890")
        assert encrypted_store.get("user-1", "openai") == "sk-test-1234567890"

    def test_get_unknown_returns_none(
        self,
        encrypted_store: EncryptedCredentialStore,
    ) -> None:
        assert encrypted_store.get("nope", "openai") is None

    def test_upsert_overwrites(
        self,
        encrypted_store: EncryptedCredentialStore,
    ) -> None:
        encrypted_store.set("u", "openai", "sk-old-1234567890")
        encrypted_store.set("u", "openai", "sk-new-ABCDEFGHIJ")
        assert encrypted_store.get("u", "openai") == "sk-new-ABCDEFGHIJ"

    def test_per_user_isolation(
        self,
        encrypted_store: EncryptedCredentialStore,
    ) -> None:
        encrypted_store.set("user-A", "openai", "sk-aaaaaaaaaaaa")
        encrypted_store.set("user-B", "openai", "sk-bbbbbbbbbbbb")
        assert encrypted_store.get("user-A", "openai") == "sk-aaaaaaaaaaaa"
        assert encrypted_store.get("user-B", "openai") == "sk-bbbbbbbbbbbb"

    def test_per_provider_isolation(
        self,
        encrypted_store: EncryptedCredentialStore,
    ) -> None:
        encrypted_store.set("u", "openai", "sk-openai-1234567890")
        encrypted_store.set("u", "gemini", "gem-key-1234567890")
        assert encrypted_store.list_providers("u") == {"openai", "gemini"}
        assert encrypted_store.get("u", "openai") == "sk-openai-1234567890"
        assert encrypted_store.get("u", "gemini") == "gem-key-1234567890"

    def test_delete_returns_true_when_removed(
        self,
        encrypted_store: EncryptedCredentialStore,
    ) -> None:
        encrypted_store.set("u", "openai", "sk-test-1234567890")
        assert encrypted_store.delete("u", "openai") is True
        assert encrypted_store.get("u", "openai") is None

    def test_delete_returns_false_when_absent(
        self,
        encrypted_store: EncryptedCredentialStore,
    ) -> None:
        assert encrypted_store.delete("u", "openai") is False

    def test_clear_removes_all_for_user(
        self,
        encrypted_store: EncryptedCredentialStore,
    ) -> None:
        encrypted_store.set("u", "openai", "sk-1111111111")
        encrypted_store.set("u", "gemini", "gem-2222222")
        encrypted_store.set("other", "openai", "sk-3333333333")
        assert encrypted_store.clear("u") == 2
        assert encrypted_store.list_providers("u") == set()
        # Other user's credentials are untouched.
        assert encrypted_store.get("other", "openai") == "sk-3333333333"

    def test_count_returns_total_rows(
        self,
        encrypted_store: EncryptedCredentialStore,
    ) -> None:
        assert encrypted_store.count() == 0
        encrypted_store.set("u1", "openai", "sk-1111111111")
        encrypted_store.set("u1", "gemini", "gem-2222222")
        encrypted_store.set("u2", "openai", "sk-3333333333")
        assert encrypted_store.count() == 3

    def test_set_rejects_empty_api_key(
        self,
        encrypted_store: EncryptedCredentialStore,
    ) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            encrypted_store.set("u", "openai", "")

    def test_rotation_preserves_created_at_and_bumps_updated_at(
        self,
        encrypted_store: EncryptedCredentialStore,
        encrypted_registry: MetadataRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One row per ``(user_id, provider)``; a rotation keeps its history."""
        stamps = iter(
            [
                datetime(2026, 1, 1, tzinfo=UTC),
                datetime(2026, 2, 1, tzinfo=UTC),
            ]
        )

        class _Clock:
            @staticmethod
            def now(tz: object = None) -> datetime:
                return next(stamps)

        monkeypatch.setattr("sec_generative_search.database.credentials.datetime", _Clock)

        encrypted_store.set("u", "openai", "sk-old-1234567890")
        encrypted_store.set("u", "openai", "sk-new-ABCDEFGHIJ")

        rows = encrypted_registry._conn.execute(
            "SELECT api_key, created_at, updated_at FROM provider_credentials"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            (
                "sk-new-ABCDEFGHIJ",
                "2026-01-01T00:00:00+00:00",
                "2026-02-01T00:00:00+00:00",
            )
        ]


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSecurity:
    def test_audit_log_on_set_carries_only_masked_tail(
        self,
        encrypted_store: EncryptedCredentialStore,
        audit_caplog: pytest.LogCaptureFixture,
    ) -> None:
        audit_caplog.clear()
        encrypted_store.set("u-NEEDLE", "openai", "sk-NEEDLE-1234567890")
        joined = "\n".join(r.getMessage() for r in audit_caplog.records)
        assert "sk-NEEDLE-1234567890" not in joined
        assert "u-NEEDLE" not in joined
        assert "store=encrypted_persistent" in joined
        # Masked tails appear in the audit detail.
        assert "***7890" in joined

    def test_audit_log_on_delete(
        self,
        encrypted_store: EncryptedCredentialStore,
        audit_caplog: pytest.LogCaptureFixture,
    ) -> None:
        encrypted_store.set("u-NEEDLE", "openai", "sk-NEEDLE-1234567890")
        audit_caplog.clear()
        encrypted_store.delete("u-NEEDLE", "openai")
        joined = "\n".join(r.getMessage() for r in audit_caplog.records)
        assert "sk-NEEDLE-1234567890" not in joined
        assert "credential_delete" in joined

    def test_repr_does_not_carry_credentials_or_user_ids(
        self,
        encrypted_store: EncryptedCredentialStore,
    ) -> None:
        encrypted_store.set("u-NEEDLE", "openai", "sk-NEEDLE-1234567890")
        rendered = repr(encrypted_store)
        assert "sk-" not in rendered.lower()
        assert "NEEDLE" not in rendered
        assert "openai" not in rendered

    def test_credential_record_dataclass_has_no_api_key_field(self) -> None:
        """The introspection dataclass MUST omit the secret.  A frozen
        record with the secret would survive serialisation, repr, and
        accidental log statements; refuse the field at the type level.
        """
        from dataclasses import fields

        names = {f.name for f in fields(CredentialRecord)}
        assert "api_key" not in names

    def test_list_providers_does_not_return_keys(
        self,
        encrypted_store: EncryptedCredentialStore,
    ) -> None:
        encrypted_store.set("u", "openai", "sk-NEEDLE-1234567890")
        out = encrypted_store.list_providers("u")
        assert out == {"openai"}
        # Defensive: the set must be string provider names only.
        for value in out:
            assert "sk-" not in value

    def test_count_returns_integer_no_payload_leak(
        self,
        encrypted_store: EncryptedCredentialStore,
    ) -> None:
        encrypted_store.set("u", "openai", "sk-NEEDLE-1234567890")
        assert isinstance(encrypted_store.count(), int)
