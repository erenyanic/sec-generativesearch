"""Tests for :class:`sec_generative_search.database.BackupService`.

Drives a real :class:`chromadb.PersistentClient` and a real
:class:`MetadataRegistry` under ``tmp_path`` so the backup / restore
round-trip exercises the actual filesystem and SQLite contracts.  No
embedder is needed — backup is a pure storage-layer concern, the
service never embeds.
"""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import sqlite3
import stat
import tarfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from sec_generative_search.core.exceptions import (
    DatabaseError,
    EmbeddingCollectionMismatchError,
)
from sec_generative_search.core.types import (
    BackupReport,
    Chunk,
    ContentType,
    EmbedderStamp,
    FilingIdentifier,
    IngestResult,
    RestoreReport,
)
from sec_generative_search.database import (
    BackupService,
    ChromaDBClient,
    MetadataRegistry,
)
from sec_generative_search.database.migrations import MIGRATIONS
from sec_generative_search.pipeline.orchestrator import ProcessedFiling

# The current schema version — v1 baseline plus every declared migration.
# Computed once so individual test assertions stay tracking the source of
# truth as new migrations land in production.
_CURRENT_SCHEMA_VERSION = max((1, *(v for v, _ in MIGRATIONS)))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_filing_id(
    *,
    ticker: str = "AAPL",
    accession_number: str = "0000320193-23-000077",
) -> FilingIdentifier:
    return FilingIdentifier(
        ticker=ticker,
        form_type="10-K",
        filing_date=date(2023, 11, 3),
        accession_number=accession_number,
    )


def _make_processed_filing(stamp: EmbedderStamp) -> ProcessedFiling:
    filing_id = _make_filing_id()
    chunks = [
        Chunk(
            content="Backup test chunk",
            path="Part I > Item 1 > Business",
            content_type=ContentType.TEXT,
            filing_id=filing_id,
            chunk_index=0,
            token_count=3,
        )
    ]
    embeddings = np.arange(stamp.dimension, dtype=np.float32).reshape(1, stamp.dimension)
    return ProcessedFiling(
        filing_id=filing_id,
        chunks=chunks,
        embeddings=embeddings,
        ingest_result=IngestResult(
            filing_id=filing_id,
            segment_count=1,
            chunk_count=1,
            duration_seconds=0.0,
        ),
    )


def _seed_storage(
    chroma_path: str,
    metadata_db_path: str,
    stamp: EmbedderStamp,
) -> None:
    """Seed both stores with one filing then close handles."""
    chroma = ChromaDBClient(stamp, chroma_path=chroma_path)
    pf = _make_processed_filing(stamp)
    chroma.store_filing(pf)
    del chroma  # release Chroma's file handles for the upcoming move

    registry = MetadataRegistry(db_path=metadata_db_path)
    registry.register_filing(pf.filing_id, pf.ingest_result.chunk_count)
    registry.close()


def _rewrite_manifest(
    src_tarball: Path,
    dst_tarball: Path,
    *,
    overrides: dict,
) -> None:
    """Copy *src_tarball* to *dst_tarball* with MANIFEST.json patched.

    Used by the negative tests that need a tarball whose manifest
    declares a forward-only schema, a bogus format version, or an
    encryption flag that does not match the source storage.  Body
    members (``metadata.sqlite`` and ``chroma/*``) are passed through
    unchanged.
    """
    with tarfile.open(src_tarball, "r:gz") as src, tarfile.open(dst_tarball, "w:gz") as dst:
        for member in src.getmembers():
            if member.name == "MANIFEST.json":
                fobj = src.extractfile(member)
                assert fobj is not None
                manifest = json.loads(fobj.read())
                manifest.update(overrides)
                payload = json.dumps(manifest, indent=2).encode("utf-8")
                new = tarfile.TarInfo("MANIFEST.json")
                new.size = len(payload)
                new.mode = 0o644
                dst.addfile(new, io.BytesIO(payload))
            else:
                fobj = src.extractfile(member) if member.isfile() else None
                dst.addfile(member, fobj)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chroma_path(tmp_path: Path) -> str:
    return str(tmp_path / "chroma")


@pytest.fixture
def metadata_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "metadata.sqlite")


@pytest.fixture
def stamp() -> EmbedderStamp:
    return EmbedderStamp(
        provider="local",
        model="google/embeddinggemma-300m",
        dimension=4,
    )


@pytest.fixture
def seeded(
    chroma_path: str,
    metadata_db_path: str,
    stamp: EmbedderStamp,
) -> tuple[str, str, EmbedderStamp]:
    _seed_storage(chroma_path, metadata_db_path, stamp)
    return chroma_path, metadata_db_path, stamp


@pytest.fixture
def service(
    chroma_path: str,
    metadata_db_path: str,
) -> BackupService:
    return BackupService(
        chroma_path=chroma_path,
        metadata_db_path=metadata_db_path,
    )


# ---------------------------------------------------------------------------
# Backup happy path
# ---------------------------------------------------------------------------


class TestBackupHappyPath:
    def test_backup_produces_tarball_with_manifest_sqlite_and_chroma(
        self,
        seeded: tuple[str, str, EmbedderStamp],
        service: BackupService,
        tmp_path: Path,
    ) -> None:
        _, _, stamp = seeded
        output = tmp_path / "backup.tar.gz"

        report = service.backup(output)

        assert isinstance(report, BackupReport)
        assert report.embedder_stamp == stamp
        assert report.schema_version == _CURRENT_SCHEMA_VERSION
        assert report.sqlcipher_encrypted is False
        assert report.size_bytes > 0
        assert report.duration_seconds >= 0.0
        assert output.exists() and output.is_file()

        with tarfile.open(output) as tar:
            names = set(tar.getnames())
        assert "MANIFEST.json" in names
        assert "metadata.sqlite" in names
        assert any(n.startswith("chroma") for n in names)

    def test_manifest_contains_full_stamp_and_metadata(
        self,
        seeded: tuple[str, str, EmbedderStamp],
        service: BackupService,
        tmp_path: Path,
    ) -> None:
        _, _, stamp = seeded
        output = tmp_path / "backup.tar.gz"
        service.backup(output)

        with tarfile.open(output) as tar:
            member = tar.getmember("MANIFEST.json")
            fobj = tar.extractfile(member)
            assert fobj is not None
            manifest = json.loads(fobj.read())

        assert manifest["format_version"] == 1
        assert manifest["embedder_stamp"] == {
            "provider": stamp.provider,
            "model": stamp.model,
            "dimension": stamp.dimension,
        }
        assert manifest["schema_version"] == _CURRENT_SCHEMA_VERSION
        assert manifest["sqlcipher_encrypted"] is False
        assert "created_at_utc" in manifest

    def test_progress_callback_emits_expected_steps(
        self,
        seeded: tuple[str, str, EmbedderStamp],
        service: BackupService,
        tmp_path: Path,
    ) -> None:
        events: list[tuple[str, int, int]] = []
        service.backup(
            tmp_path / "backup.tar.gz",
            progress_callback=lambda s, c, t: events.append((s, c, t)),
        )
        steps = {s for s, _, _ in events}
        assert {"backup-sqlite", "backup-chroma", "backup-archive"} <= steps


# ---------------------------------------------------------------------------
# Backup refuse paths
# ---------------------------------------------------------------------------


class TestBackupRefuse:
    def test_refuses_existing_output_without_force(
        self,
        seeded: tuple[str, str, EmbedderStamp],
        service: BackupService,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "exists.tar.gz"
        output.write_bytes(b"sentinel")
        with pytest.raises(DatabaseError, match="already exists"):
            service.backup(output)
        # Existing artefact must not be overwritten on refusal.
        assert output.read_bytes() == b"sentinel"

    def test_force_overwrites_existing_output(
        self,
        seeded: tuple[str, str, EmbedderStamp],
        service: BackupService,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "exists.tar.gz"
        output.write_bytes(b"sentinel")
        service.backup(output, force=True)
        # Real gzip header (1f 8b) — not the sentinel any more.
        assert output.read_bytes()[:2] == b"\x1f\x8b"

    def test_refuses_directory_as_output(
        self,
        seeded: tuple[str, str, EmbedderStamp],
        service: BackupService,
        tmp_path: Path,
    ) -> None:
        outdir = tmp_path / "outdir"
        outdir.mkdir()
        with pytest.raises(DatabaseError, match="directory"):
            service.backup(outdir)

    def test_refuses_when_chroma_path_missing(
        self,
        metadata_db_path: str,
        tmp_path: Path,
    ) -> None:
        # Bootstrap only the SQLite side so chroma_path is genuinely absent.
        registry = MetadataRegistry(db_path=metadata_db_path)
        registry.close()
        bare = BackupService(
            chroma_path=str(tmp_path / "absent_chroma"),
            metadata_db_path=metadata_db_path,
        )
        with pytest.raises(DatabaseError, match="ChromaDB path not found"):
            bare.backup(tmp_path / "out.tar.gz")

    def test_refuses_when_sqlite_path_missing(
        self,
        chroma_path: str,
        stamp: EmbedderStamp,
        tmp_path: Path,
    ) -> None:
        # Open + close a Chroma collection so the path exists.
        ChromaDBClient(stamp, chroma_path=chroma_path)
        bare = BackupService(
            chroma_path=chroma_path,
            metadata_db_path=str(tmp_path / "absent.sqlite"),
        )
        with pytest.raises(DatabaseError, match="Metadata SQLite file not found"):
            bare.backup(tmp_path / "out.tar.gz")

    def test_refuses_unstamped_chroma_collection(
        self,
        chroma_path: str,
        metadata_db_path: str,
        tmp_path: Path,
    ) -> None:
        # Build a Chroma collection without any stamp, then ask to back it up.
        import chromadb

        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        raw = chromadb.PersistentClient(path=chroma_path)
        raw.create_collection(
            name="sec_filings",
            metadata={"hnsw:space": "cosine"},
        )
        del raw

        registry = MetadataRegistry(db_path=metadata_db_path)
        registry.close()

        bare = BackupService(
            chroma_path=chroma_path,
            metadata_db_path=metadata_db_path,
        )
        with pytest.raises(DatabaseError, match="no valid embedder stamp"):
            bare.backup(tmp_path / "out.tar.gz")


# ---------------------------------------------------------------------------
# Restore happy path
# ---------------------------------------------------------------------------


class TestRestoreHappyPath:
    def test_round_trip_restores_chroma_and_sqlite(
        self,
        seeded: tuple[str, str, EmbedderStamp],
        service: BackupService,
        tmp_path: Path,
    ) -> None:
        chroma_path, metadata_db_path, stamp = seeded
        output = tmp_path / "backup.tar.gz"
        service.backup(output)

        # Wipe live storage so restore is the only path that recreates it.
        shutil.rmtree(chroma_path)
        Path(metadata_db_path).unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{metadata_db_path}{suffix}")
            if sidecar.exists():
                sidecar.unlink()

        report = service.restore(output, expected_stamp=stamp)
        assert isinstance(report, RestoreReport)
        assert report.embedder_stamp == stamp
        assert report.schema_version == _CURRENT_SCHEMA_VERSION
        assert report.sqlcipher_encrypted is False

        # ChromaDB has the seeded chunk back.
        chroma = ChromaDBClient(stamp, chroma_path=chroma_path)
        assert chroma.collection_count() == 1
        del chroma

        # SQLite has the seeded filing row back.
        registry = MetadataRegistry(db_path=metadata_db_path)
        try:
            assert registry.count() == 1
        finally:
            registry.close()

    def test_restore_does_not_touch_state_when_chroma_path_is_new(
        self,
        seeded: tuple[str, str, EmbedderStamp],
        service: BackupService,
        tmp_path: Path,
    ) -> None:
        """Restore creates the target ChromaDB directory if it does not exist."""
        chroma_path, metadata_db_path, stamp = seeded
        output = tmp_path / "backup.tar.gz"
        service.backup(output)

        # Drop both targets so the parent dirs need re-creating during restore.
        shutil.rmtree(chroma_path)
        Path(metadata_db_path).unlink()

        service.restore(output, expected_stamp=stamp)
        assert Path(chroma_path).is_dir()
        assert Path(metadata_db_path).is_file()


# ---------------------------------------------------------------------------
# Restore refuse paths
# ---------------------------------------------------------------------------


class TestRestoreRefuse:
    def test_refuses_missing_input(
        self,
        service: BackupService,
        stamp: EmbedderStamp,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(DatabaseError, match="not found"):
            service.restore(tmp_path / "missing.tar.gz", expected_stamp=stamp)

    def test_refuses_stamp_mismatch_with_typed_exception(
        self,
        seeded: tuple[str, str, EmbedderStamp],
        service: BackupService,
        tmp_path: Path,
    ) -> None:
        _, _, stamp = seeded
        output = tmp_path / "backup.tar.gz"
        service.backup(output)

        wrong = EmbedderStamp(
            provider="openai",
            model="text-embedding-3-small",
            dimension=1536,
        )
        with pytest.raises(EmbeddingCollectionMismatchError) as exc_info:
            service.restore(output, expected_stamp=wrong)
        assert exc_info.value.expected == wrong
        assert exc_info.value.actual == stamp

    def test_refuses_forward_only_schema(
        self,
        seeded: tuple[str, str, EmbedderStamp],
        service: BackupService,
        tmp_path: Path,
    ) -> None:
        _, _, stamp = seeded
        output = tmp_path / "backup.tar.gz"
        forged = tmp_path / "forged.tar.gz"
        service.backup(output)
        _rewrite_manifest(output, forged, overrides={"schema_version": 999})
        with pytest.raises(DatabaseError, match="newer than this build"):
            service.restore(forged, expected_stamp=stamp)

    def test_refuses_wrong_format_version(
        self,
        seeded: tuple[str, str, EmbedderStamp],
        service: BackupService,
        tmp_path: Path,
    ) -> None:
        _, _, stamp = seeded
        output = tmp_path / "backup.tar.gz"
        forged = tmp_path / "forged.tar.gz"
        service.backup(output)
        _rewrite_manifest(output, forged, overrides={"format_version": 99})
        with pytest.raises(DatabaseError, match="format_version"):
            service.restore(forged, expected_stamp=stamp)

    def test_refuses_encryption_state_mismatch(
        self,
        seeded: tuple[str, str, EmbedderStamp],
        service: BackupService,
        tmp_path: Path,
    ) -> None:
        _, _, stamp = seeded
        output = tmp_path / "backup.tar.gz"
        forged = tmp_path / "forged.tar.gz"
        service.backup(output)
        # Host has no encryption configured; pretend the artefact is encrypted.
        _rewrite_manifest(output, forged, overrides={"sqlcipher_encrypted": True})
        with pytest.raises(DatabaseError, match="Encryption mismatch"):
            service.restore(forged, expected_stamp=stamp)

    def test_refuses_archive_missing_manifest(
        self,
        service: BackupService,
        stamp: EmbedderStamp,
        tmp_path: Path,
    ) -> None:
        bogus = tmp_path / "bogus.tar.gz"
        payload = b"not a manifest"
        with tarfile.open(bogus, "w:gz") as tar:
            info = tarfile.TarInfo("other.txt")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        with pytest.raises(DatabaseError, match=r"missing 'MANIFEST\.json'"):
            service.restore(bogus, expected_stamp=stamp)

    def test_refuses_malformed_manifest_json(
        self,
        service: BackupService,
        stamp: EmbedderStamp,
        tmp_path: Path,
    ) -> None:
        bogus = tmp_path / "bogus.tar.gz"
        payload = b"this is not json"
        with tarfile.open(bogus, "w:gz") as tar:
            info = tarfile.TarInfo("MANIFEST.json")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        with pytest.raises(DatabaseError, match="not valid JSON"):
            service.restore(bogus, expected_stamp=stamp)

    def test_refuses_corrupt_tarball(
        self,
        service: BackupService,
        stamp: EmbedderStamp,
        tmp_path: Path,
    ) -> None:
        bogus = tmp_path / "bogus.tar.gz"
        bogus.write_bytes(b"not even gzip")
        with pytest.raises(DatabaseError, match="Failed to read backup archive"):
            service.restore(bogus, expected_stamp=stamp)

    def test_refuses_archive_missing_chroma_subdir(
        self,
        service: BackupService,
        seeded: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        """Manifest passes validation but body is missing chroma/."""
        _, _, stamp = seeded
        forged = tmp_path / "forged.tar.gz"

        manifest = {
            "format_version": 1,
            "created_at_utc": "2026-04-28T00:00:00+00:00",
            "embedder_stamp": {
                "provider": stamp.provider,
                "model": stamp.model,
                "dimension": stamp.dimension,
            },
            "schema_version": 1,
            "sqlcipher_encrypted": False,
        }
        with tarfile.open(forged, "w:gz") as tar:
            payload = json.dumps(manifest).encode("utf-8")
            info = tarfile.TarInfo("MANIFEST.json")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            # SQLite present, chroma/ absent.
            sqlite_payload = b"sqlite stub"
            info = tarfile.TarInfo("metadata.sqlite")
            info.size = len(sqlite_payload)
            tar.addfile(info, io.BytesIO(sqlite_payload))

        with pytest.raises(DatabaseError, match="missing 'chroma/'"):
            service.restore(forged, expected_stamp=stamp)


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


@pytest.mark.security
class TestSecurity:
    def test_output_file_is_owner_only(
        self,
        seeded: tuple[str, str, EmbedderStamp],
        service: BackupService,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "backup.tar.gz"
        service.backup(output)
        mode = output.stat().st_mode & 0o777
        assert mode == 0o600, (
            f"Backup file mode is {oct(mode)}; must be 0o600 to keep the artefact owner-only."
        )
        # Defence-in-depth: world / group bits are explicitly clear.
        assert (output.stat().st_mode & stat.S_IRWXG) == 0
        assert (output.stat().st_mode & stat.S_IRWXO) == 0

    def test_no_credential_shaped_attributes_on_service(
        self,
        chroma_path: str,
        metadata_db_path: str,
    ) -> None:
        """BackupService must not grow credential-bearing attributes.

        The encryption key is resolved on-demand from the environment
        rather than stored on the service so this scan stays clean —
        guarded against a future refactor that adds a long-lived
        secret-shaped attribute.
        """
        bare = BackupService(
            chroma_path=chroma_path,
            metadata_db_path=metadata_db_path,
        )
        forbidden = (
            "api_key",
            "apikey",
            "secret",
            "password",
            "bearer",
            "authorization",
            "credential",
            "encryption_key",
        )
        attrs = {a.lower() for a in vars(bare)}
        for hint in forbidden:
            assert hint not in attrs, f"BackupService grew a credential-shaped attribute: {hint}"
        assert "token" not in attrs
        assert "api_token" not in attrs

    def test_module_imports_no_pipeline_or_ui_dependencies(self) -> None:
        """``database/backup.py`` is surface-agnostic.

        No ``rich`` / ``typer`` / ``edgartools`` / pipeline imports —
        the CLI wrapper drives progress through an injected callback.
        """
        src = Path("src/sec_generative_search/database/backup.py").read_text(encoding="utf-8")
        for name in ("rich", "typer", "edgartools", "edgar"):
            assert f"import {name}" not in src, (
                f"backup.py must not import {name!r} — it is surface-agnostic"
            )
            assert f"from {name}" not in src, (
                f"backup.py must not import from {name!r} — it is surface-agnostic"
            )
        assert "sec_generative_search.pipeline" not in src, (
            "backup.py must not depend on the ingestion pipeline"
        )

    def test_refuses_symlink_in_lexical_parent_of_output(
        self,
        service: BackupService,
        seeded: tuple[str, str, EmbedderStamp],
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "real_dir"
        target.mkdir()
        link = tmp_path / "link_dir"
        link.symlink_to(target)
        with pytest.raises(DatabaseError, match="symlink"):
            service.backup(link / "backup.tar.gz")

    def test_refuses_symlink_in_lexical_parent_of_input(
        self,
        service: BackupService,
        stamp: EmbedderStamp,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "real_dir"
        target.mkdir()
        link = tmp_path / "link_dir"
        link.symlink_to(target)
        # The file does not need to exist — the symlink check fires first.
        with pytest.raises(DatabaseError, match="symlink"):
            service.restore(link / "backup.tar.gz", expected_stamp=stamp)

    def test_manifest_does_not_carry_encryption_key(
        self,
        chroma_path: str,
        metadata_db_path: str,
        stamp: EmbedderStamp,
        service: BackupService,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Even with an encryption key configured, the manifest is plain.

        The key value must never be serialised into MANIFEST.json —
        the manifest carries only the *fact* of encryption via
        ``sqlcipher_encrypted``.
        """
        # Key set *before* seeding: with pysqlcipher3 (the API image) the
        # store is genuinely encrypted; without it (CI) it falls back to
        # plain.  The manifest must be key-free either way.
        monkeypatch.setenv("DB_ENCRYPTION_KEY", "this-is-a-long-test-key-not-a-secret")
        _seed_storage(chroma_path, metadata_db_path, stamp)
        output = tmp_path / "backup.tar.gz"
        service.backup(output)

        with tarfile.open(output) as tar:
            fobj = tar.extractfile(tar.getmember("MANIFEST.json"))
            assert fobj is not None
            raw = fobj.read().decode("utf-8")

        forbidden_substrings = (
            "this-is-a-long-test-key-not-a-secret",
            "encryption_key",
            "api_key",
            "password",
        )
        for needle in forbidden_substrings:
            assert needle not in raw, f"MANIFEST.json leaked secret-shaped string {needle!r}"

    def test_refuses_path_traversal_archive(
        self,
        service: BackupService,
        stamp: EmbedderStamp,
        tmp_path: Path,
    ) -> None:
        """Tarball entries that try to escape the staging dir are refused.

        Python 3.12's ``data_filter`` rejects absolute and ``../`` paths;
        an older fallback path mirrors the rejection.  Either way the
        restore operation refuses without writing any of the live state.
        """
        bogus = tmp_path / "evil.tar.gz"
        manifest = {
            "format_version": 1,
            "created_at_utc": "2026-04-28T00:00:00+00:00",
            "embedder_stamp": {
                "provider": stamp.provider,
                "model": stamp.model,
                "dimension": stamp.dimension,
            },
            "schema_version": 1,
            "sqlcipher_encrypted": False,
        }
        with tarfile.open(bogus, "w:gz") as tar:
            payload = json.dumps(manifest).encode("utf-8")
            info = tarfile.TarInfo("MANIFEST.json")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            # An entry trying to escape via ``..`` — extract should refuse.
            evil_payload = b"escape"
            info = tarfile.TarInfo("../escape.txt")
            info.size = len(evil_payload)
            tar.addfile(info, io.BytesIO(evil_payload))

        with pytest.raises(DatabaseError):
            service.restore(bogus, expected_stamp=stamp)


# ---------------------------------------------------------------------------
# SQLCipher snapshot — pysqlcipher3 has no ``Connection.backup()``
# ---------------------------------------------------------------------------


class _RecordingSqlcipher:
    """Stand-in for the pysqlcipher3 module that records what the snapshot runs."""

    class Error(Exception):
        pass

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.connect_kwargs: dict = {}
        self.statements: list[tuple[str, tuple]] = []
        self.closed = False

    def connect(self, path: str, **kwargs: object) -> _RecordingSqlcipher:
        self.connect_kwargs = kwargs
        return self

    def execute(self, sql: str, params: tuple = ()) -> _RecordingSqlcipher:
        self.statements.append((sql, params))
        if self.fail_on is not None and self.fail_on in sql:
            raise self.Error("simulated sqlcipher failure")
        return self

    def fetchall(self) -> list:
        return []

    def close(self) -> None:
        self.closed = True


_SNAPSHOT_KEY = "unit-test-key-not-a-secret"


class TestSqlcipherSnapshotOrchestration:
    """Runs everywhere: the encrypted branch's statement contract."""

    def test_encrypted_snapshot_exports_inside_one_read_transaction(
        self,
        service: BackupService,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _RecordingSqlcipher()
        monkeypatch.setattr(
            "sec_generative_search.database.backup._get_sqlite_module", lambda _key: fake
        )
        dest = tmp_path / "snap.sqlite"

        service._snapshot_sqlite(dest, encryption_key=_SNAPSHOT_KEY, encrypted=True)

        key_literal = f"x'{_SNAPSHOT_KEY.encode().hex()}'"
        assert fake.statements == [
            (f'PRAGMA key = "{key_literal}"', ()),
            ("ATTACH DATABASE ? AS snapshot KEY ?", (str(dest), key_literal)),
            ("BEGIN", ()),
            ("SELECT sqlcipher_export('snapshot')", ()),
            ("COMMIT", ()),
        ]
        # No implicit driver transactions around ATTACH / the export.
        assert fake.connect_kwargs["isolation_level"] is None
        assert fake.closed is True

    @pytest.mark.security
    def test_export_failure_is_typed_closes_and_never_echoes_the_key(
        self,
        service: BackupService,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _RecordingSqlcipher(fail_on="sqlcipher_export")
        monkeypatch.setattr(
            "sec_generative_search.database.backup._get_sqlite_module", lambda _key: fake
        )

        with pytest.raises(DatabaseError, match="sqlcipher_export") as excinfo:
            service._snapshot_sqlite(
                tmp_path / "snap.sqlite", encryption_key=_SNAPSHOT_KEY, encrypted=True
            )

        assert fake.closed is True
        rendered = str(excinfo.value)
        assert _SNAPSHOT_KEY not in rendered
        assert _SNAPSHOT_KEY.encode().hex() not in rendered
        # The key is bound into ATTACH, never interpolated into its SQL.
        attach_sql = next(sql for sql, _ in fake.statements if sql.startswith("ATTACH"))
        assert _SNAPSHOT_KEY.encode().hex() not in attach_sql


def _sqlcipher_dump(path: Path, key: str) -> dict:
    from pysqlcipher3 import dbapi2 as sqlcipher

    conn = sqlcipher.connect(str(path))
    try:
        conn.execute(f"PRAGMA key = \"x'{key.encode().hex()}'\"")
        tables = [
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        ]
        return {
            "schema": sorted(
                tuple(row) for row in conn.execute("SELECT type, name, sql FROM sqlite_master")
            ),
            "rows": {t: sorted(conn.execute(f"SELECT * FROM {t}").fetchall()) for t in tables},  # noqa: S608
        }
    finally:
        conn.close()


@pytest.mark.skipif(
    importlib.util.find_spec("pysqlcipher3") is None,
    reason="needs pysqlcipher3 (the [encryption] extra; present in the API image, not in CI)",
)
class TestSqlcipherBackupRoundTrip:
    """The real driver: an encrypted deployment backs up and restores."""

    @pytest.mark.parametrize(
        "key",
        [
            # A passphrase (PBKDF2-derived), then 32 bytes (SQLCipher's raw-key form).
            "unit-test-key-not-a-secret",
            "0123456789abcdef0123456789abcdef",  # pragma: allowlist secret
        ],
    )
    def test_encrypted_round_trip(
        self,
        chroma_path: str,
        metadata_db_path: str,
        stamp: EmbedderStamp,
        service: BackupService,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        key: str,
    ) -> None:
        monkeypatch.setenv("DB_ENCRYPTION_KEY", key)
        _seed_storage(chroma_path, metadata_db_path, stamp)
        registry = MetadataRegistry(db_path=metadata_db_path)
        assert registry.encrypted is True
        registry.save_task_history(
            "task-1", status="completed", tickers=["AAPL"], form_types=["10-K"], results=[]
        )
        registry.close()
        source = _sqlcipher_dump(Path(metadata_db_path), key)

        output = tmp_path / "backup.tar.gz"
        assert service.backup(output).sqlcipher_encrypted is True

        extracted = tmp_path / "extracted"
        with tarfile.open(output) as tar:
            tar.extract("metadata.sqlite", extracted, filter="data")
        snapshot = extracted / "metadata.sqlite"
        with pytest.raises(sqlite3.DatabaseError):
            sqlite3.connect(snapshot).execute("SELECT * FROM sqlite_master").fetchall()
        assert _sqlcipher_dump(snapshot, key) == source

        shutil.rmtree(chroma_path)
        for suffix in ("", "-wal", "-shm"):
            Path(f"{metadata_db_path}{suffix}").unlink(missing_ok=True)
        assert service.restore(output, expected_stamp=stamp).sqlcipher_encrypted is True

        restored = MetadataRegistry(db_path=metadata_db_path)
        try:
            assert restored.encrypted is True
            assert restored.count() == 1
            assert restored.get_task_history("task-1") is not None
        finally:
            restored.close()
