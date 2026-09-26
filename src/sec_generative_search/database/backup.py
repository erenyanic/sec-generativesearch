"""Byte-faithful tarball backup and restore for the storage layer.

:class:`BackupService` produces a single ``.tar.gz`` containing:

- ``MANIFEST.json`` at the root: ``format_version``, ``created_at_utc``,
  ``embedder_stamp``, ``schema_version``, ``sqlcipher_encrypted``.
- ``metadata.sqlite``: live-consistent SQLite snapshot — the DB-API
  ``Connection.backup()`` for a plain database; SQLCipher's
  ``sqlcipher_export()`` under the same key for an encrypted one
  (pysqlcipher3 has no ``backup()``), so it is encrypted from the first
  page and plaintext never touches disk.
- ``chroma/``: recursive copy of the ChromaDB persistence directory.

The operator must quiesce writers before backup — Chroma exposes no
atomic-snapshot primitive, and a partially-written ChromaDB during
backup is the same operator-scope concern that :class:`ReindexService`
already documents.

Restore validates the manifest against the host's current state
*before any filesystem mutation* and refuses on:

- Stamp mismatch with the host's configured embedder
  (:class:`EmbeddingCollectionMismatchError`).
- ``schema_version > current`` — forward-only.
- Encryption-state mismatch (encrypted artefact onto a host without
  configured encryption, or vice versa).

A ``schema_version`` *lower* than the host's latest available is the
lossless-upgrade case: the restored SQLite file is honoured as-is, and
a subsequent :class:`MetadataRegistry` open will run the pending
migrations naturally.

Surface-agnostic and credential-free.  The encryption key is resolved
on-demand via :func:`_resolve_runtime_encryption_key` rather than held
on the service so that no long-lived attribute carries the secret —
mirrors the operator-scope contract from :class:`ReindexService`.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb

from sec_generative_search.config.constants import COLLECTION_NAME
from sec_generative_search.config.settings import get_settings
from sec_generative_search.core.exceptions import (
    DatabaseError,
    EmbeddingCollectionMismatchError,
)
from sec_generative_search.core.logging import get_logger
from sec_generative_search.core.types import (
    BackupReport,
    EmbedderStamp,
    RestoreReport,
)
from sec_generative_search.database.metadata import (
    _get_sqlite_module,
    _resolve_runtime_encryption_key,
)
from sec_generative_search.database.migrations import MIGRATIONS

if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["BackupService"]


logger = get_logger(__name__)


_FORMAT_VERSION = 1
_MANIFEST_NAME = "MANIFEST.json"
_CHROMA_SUBDIR = "chroma"
_SQLITE_NAME = "metadata.sqlite"


def _latest_available_schema_version() -> int:
    """Return the highest schema version this build can produce.

    v1 is implicit (the baseline shape created by
    :meth:`MetadataRegistry._initialise_schema`); v2+ live in
    :data:`MIGRATIONS`.  Used by restore to refuse forward-only — an
    artefact stamped at a higher version was produced by a newer build
    than the host can drive forward, and silent acceptance would land
    a future-shaped database the running registry cannot read.
    """
    if not MIGRATIONS:
        return 1
    return max(max(v for v, _ in MIGRATIONS), 1)


def _refuse_symlink_lexical_parents(path: Path, *, label: str) -> None:
    """Refuse if any existing lexical parent of *path* is a symlink.

    Mirrors :meth:`DatabaseSettings._validate_paths`.  Walks the
    *lexical* parent chain — calling :meth:`Path.resolve` first would
    have already followed every symlink, defeating the check, which
    is exactly the path-traversal class this guard exists to block.
    """
    check = path.absolute()
    while check != check.parent:
        if check.exists() and check.is_symlink():
            raise DatabaseError(
                f"{label} contains a symlink at '{check}'. "
                f"Symlinks are not permitted in backup paths for security.",
            )
        check = check.parent


class BackupService:
    """Tarball backup / restore over the dual-store on-disk layout.

    Constructed with paths only; opens nothing until ``backup()`` or
    ``restore()`` is called.  The encryption key is resolved on-demand
    from the environment so the service never holds a long-lived
    credential — keep it that way.

    Example:
        >>> service = BackupService()
        >>> report = service.backup("/backups/2026-04-28.tar.gz")
        >>> service.restore(
        ...     "/backups/2026-04-28.tar.gz",
        ...     expected_stamp=stamp,
        ... )
    """

    _FORMAT_VERSION = _FORMAT_VERSION
    _MANIFEST_NAME = _MANIFEST_NAME
    _CHROMA_SUBDIR = _CHROMA_SUBDIR
    _SQLITE_NAME = _SQLITE_NAME

    def __init__(
        self,
        *,
        chroma_path: str | None = None,
        metadata_db_path: str | None = None,
    ) -> None:
        """Construct a backup service over the configured storage paths.

        Args:
            chroma_path: ChromaDB persistence directory.  Falls through
                to ``settings.database.chroma_path`` when ``None``.
            metadata_db_path: SQLite file path.  Falls through to
                ``settings.database.metadata_db_path`` when ``None``.
        """
        settings = get_settings()
        self._chroma_path = Path(
            chroma_path if chroma_path is not None else settings.database.chroma_path
        )
        self._metadata_db_path = Path(
            metadata_db_path if metadata_db_path is not None else settings.database.metadata_db_path
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backup(
        self,
        output_path: str | Path,
        *,
        force: bool = False,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> BackupReport:
        """Write a backup tarball to *output_path*.

        Reads the live ChromaDB collection's stamp and the SQLite
        ``schema_version`` to populate the manifest, then writes a
        ``.tar.gz`` containing the manifest, a live SQLite snapshot,
        and a recursive copy of the ChromaDB directory.  The output
        file is ``chmod 0600`` after writing.

        Args:
            output_path: Destination tarball path.
            force: Overwrite an existing file at ``output_path``.
                Refuses by default — backups are append-only at the
                surface level so an operator never silently shadows a
                previous artefact with the same name.
            progress_callback: Optional ``(step, current, total)``
                callback.  Steps emitted: ``"backup-sqlite"``,
                ``"backup-chroma"``, ``"backup-archive"``.

        Returns:
            :class:`BackupReport` with the manifest values and timing.

        Raises:
            DatabaseError: Source stores missing or unreadable, output
                path refused (symlink-parent / exists-without-force /
                directory), or the SQLite snapshot failed.
        """
        started_at = time.monotonic()
        output_path = Path(output_path)

        _refuse_symlink_lexical_parents(output_path, label="Backup output path")

        if output_path.exists():
            if output_path.is_dir():
                raise DatabaseError(
                    f"Backup output path is a directory: '{output_path}'. "
                    "Choose a file path inside that directory instead.",
                )
            if not force:
                raise DatabaseError(
                    f"Backup output path already exists: '{output_path}'. "
                    "Pass force=True (CLI: --force) to overwrite.",
                )

        if not self._chroma_path.exists():
            raise DatabaseError(
                f"ChromaDB path not found at '{self._chroma_path}'. "
                "Cannot back up an empty deployment.",
            )
        if not self._metadata_db_path.exists():
            raise DatabaseError(
                f"Metadata SQLite file not found at '{self._metadata_db_path}'. "
                "Cannot back up an empty deployment.",
            )

        # Resolve the encryption key on-demand from the environment
        # rather than holding it on ``self`` — keeps the service
        # credential-free for the parametrised security test.
        encryption_key = _resolve_runtime_encryption_key(get_settings().database.encryption_key)

        embedder_stamp = self._read_embedder_stamp()
        schema_version = self._read_schema_version(encryption_key)
        encrypted = self._is_encrypted_runtime(encryption_key)

        manifest = {
            "format_version": self._FORMAT_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "embedder_stamp": {
                "provider": embedder_stamp.provider,
                "model": embedder_stamp.model,
                "dimension": embedder_stamp.dimension,
            },
            "schema_version": schema_version,
            "sqlcipher_encrypted": encrypted,
        }

        with tempfile.TemporaryDirectory() as staging_dir:
            staging = Path(staging_dir)

            # Manifest first (smallest, validates fastest on restore).
            manifest_path = staging / self._MANIFEST_NAME
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )

            # SQLite live snapshot (``backup()`` or ``sqlcipher_export()``).
            self._snapshot_sqlite(
                staging / self._SQLITE_NAME,
                encryption_key=encryption_key,
                encrypted=encrypted,
            )
            if progress_callback is not None:
                progress_callback("backup-sqlite", 1, 1)

            # ChromaDB recursive copy.  Operator has quiesced writers;
            # this is a plain filesystem walk.
            self._copy_chroma_tree(
                staging / self._CHROMA_SUBDIR,
                progress_callback=progress_callback,
            )

            # Pack into the destination tarball.
            self._write_tarball(
                staging=staging,
                output_path=output_path,
                progress_callback=progress_callback,
            )

        # Restrict file mode to owner-only.  Done after the tarfile
        # context exits so the file definitely exists and is closed.
        os.chmod(output_path, 0o600)

        duration = time.monotonic() - started_at
        size_bytes = output_path.stat().st_size

        logger.info(
            "Backup complete: %s (%d bytes) embedder=%s/%s schema=%d encrypted=%s in %.2fs",
            output_path,
            size_bytes,
            embedder_stamp.provider,
            embedder_stamp.model,
            schema_version,
            encrypted,
            duration,
        )

        return BackupReport(
            output_path=str(output_path),
            embedder_stamp=embedder_stamp,
            schema_version=schema_version,
            sqlcipher_encrypted=encrypted,
            size_bytes=size_bytes,
            duration_seconds=duration,
        )

    def restore(
        self,
        input_path: str | Path,
        *,
        expected_stamp: EmbedderStamp,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> RestoreReport:
        """Restore a backup tarball into the configured storage paths.

        Validates the manifest against the host's current configuration
        before any filesystem mutation.  Refuses with a typed exception
        on mismatch — the existing live state is never touched on the
        refusal path.

        Args:
            input_path: Source tarball path.
            expected_stamp: The host's currently-configured embedder
                stamp.  Must equal the artefact's stamp; otherwise
                refused with :class:`EmbeddingCollectionMismatchError`.
            progress_callback: Optional ``(step, current, total)``
                callback.  Steps emitted: ``"restore-extract"``,
                ``"restore-chroma"``, ``"restore-sqlite"``.

        Returns:
            :class:`RestoreReport` with manifest values and timing.

        Raises:
            DatabaseError: Input path refused, manifest malformed,
                forward-only schema, encryption-state mismatch, or
                inner-archive structure invalid.
            EmbeddingCollectionMismatchError: Stamp mismatch with the
                host's configured embedder.
        """
        started_at = time.monotonic()
        input_path = Path(input_path)

        _refuse_symlink_lexical_parents(input_path, label="Backup input path")

        if not input_path.is_file():
            raise DatabaseError(
                f"Backup file not found: '{input_path}'.",
            )

        # Read MANIFEST.json without unpacking the rest — a malformed
        # tarball never touches the live filesystem.
        manifest = self._read_manifest_from_tarball(input_path)
        self._validate_format_version(manifest)
        artefact_stamp = self._parse_manifest_stamp(manifest)
        artefact_schema = self._parse_manifest_schema(manifest)
        artefact_encrypted = bool(manifest.get("sqlcipher_encrypted", False))

        # Validation order: stamp → schema → encryption.
        # Stamp first because that is the most common operator error
        # (wrong artefact for the host's embedder).
        if artefact_stamp != expected_stamp:
            raise EmbeddingCollectionMismatchError(
                expected=expected_stamp,
                actual=artefact_stamp,
                details=(
                    "Backup artefact was produced under a different "
                    "embedder than this host is configured for."
                ),
            )

        latest_available = _latest_available_schema_version()
        if artefact_schema > latest_available:
            raise DatabaseError(
                f"Backup schema_version {artefact_schema} is newer than this "
                f"build supports (latest available: {latest_available}). "
                "Upgrade the application before restoring.",
            )

        host_encryption_key = _resolve_runtime_encryption_key(
            get_settings().database.encryption_key
        )
        host_encrypted = self._is_encrypted_runtime(host_encryption_key)
        if artefact_encrypted != host_encrypted:
            raise DatabaseError(
                f"Encryption mismatch: artefact sqlcipher_encrypted="
                f"{artefact_encrypted}, host has encryption "
                f"{'configured' if host_encrypted else 'unset'}. "
                "Set or unset DB_ENCRYPTION_KEY{,_FILE} to match the "
                "artefact, or restore on a host with the matching state.",
            )

        # All validations passed; extract and replace.
        with tempfile.TemporaryDirectory() as staging_dir:
            staging = Path(staging_dir)
            self._extract_tarball(input_path, staging)
            if progress_callback is not None:
                progress_callback("restore-extract", 1, 1)

            extracted_chroma = staging / self._CHROMA_SUBDIR
            extracted_sqlite = staging / self._SQLITE_NAME
            if not extracted_chroma.is_dir():
                raise DatabaseError(
                    f"Backup is missing '{self._CHROMA_SUBDIR}/' — refusing to restore.",
                )
            if not extracted_sqlite.is_file():
                raise DatabaseError(
                    f"Backup is missing '{self._SQLITE_NAME}' — refusing to restore.",
                )

            self._replace_target_paths(
                extracted_chroma=extracted_chroma,
                extracted_sqlite=extracted_sqlite,
                progress_callback=progress_callback,
            )

        duration = time.monotonic() - started_at

        logger.info(
            "Restore complete from %s: embedder=%s/%s schema=%d encrypted=%s in %.2fs",
            input_path,
            artefact_stamp.provider,
            artefact_stamp.model,
            artefact_schema,
            artefact_encrypted,
            duration,
        )

        return RestoreReport(
            input_path=str(input_path),
            embedder_stamp=artefact_stamp,
            schema_version=artefact_schema,
            sqlcipher_encrypted=artefact_encrypted,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Internal — source-side reads (backup phase)
    # ------------------------------------------------------------------

    def _read_embedder_stamp(self) -> EmbedderStamp:
        """Read the live ChromaDB collection's stamp via raw client.

        Bypasses :class:`ChromaDBClient` because ``ChromaDBClient``
        verifies the stamp against a *configured* stamp passed at
        construction; the backup service should report whatever
        actually sealed the collection.  The raw read is non-mutating.
        """
        try:
            client = chromadb.PersistentClient(path=str(self._chroma_path))
            collection = client.get_collection(name=COLLECTION_NAME)
        except Exception as exc:
            raise DatabaseError(
                f"Failed to open ChromaDB collection '{COLLECTION_NAME}' for backup",
                details=str(exc),
            ) from exc

        metadata = dict(collection.metadata or {})
        try:
            return EmbedderStamp.from_metadata(metadata)
        except ValueError as exc:
            raise DatabaseError(
                "ChromaDB collection has no valid embedder stamp; "
                "refusing to back up an unstamped collection.",
                details=str(exc),
            ) from exc

    def _read_schema_version(self, encryption_key: str | None) -> int:
        """Read ``MAX(version)`` from ``schema_version`` on a fresh connection.

        Does not go through :class:`MetadataRegistry` — that would
        re-fire bootstrap and migration apply, which is destructive
        from the backup-service perspective.  A small read-only
        connection is enough.
        """
        sqlite_module = _get_sqlite_module(encryption_key)
        try:
            conn = sqlite_module.connect(
                str(self._metadata_db_path),
                check_same_thread=False,
            )
        except sqlite_module.Error as exc:
            raise DatabaseError(
                "Failed to open SQLite for schema_version read",
                details=str(exc),
            ) from exc

        try:
            if encryption_key and sqlite_module is not sqlite3:
                hex_key = encryption_key.encode().hex()
                # PRAGMA key does not accept ``?`` parameter binding;
                # the value is hex-encoded into a blob literal — same
                # pattern used by :class:`MetadataRegistry`.
                conn.execute(f"PRAGMA key = \"x'{hex_key}'\"")
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        except sqlite_module.Error as exc:
            raise DatabaseError(
                "Failed to read schema_version from SQLite for backup",
                details=str(exc),
            ) from exc
        finally:
            conn.close()

        if row is None or row[0] is None:
            raise DatabaseError(
                "schema_version table is empty — database not bootstrapped. "
                "Open the application once to initialise the registry, then "
                "retry the backup.",
            )
        return int(row[0])

    @staticmethod
    def _is_encrypted_runtime(encryption_key: str | None) -> bool:
        """Return ``True`` iff the runtime selected the SQLCipher driver.

        Mirrors :class:`MetadataRegistry`'s contract: a configured key
        with no ``pysqlcipher3`` driver falls back to plain ``sqlite3``,
        and the manifest must reflect what was actually used at backup
        time, not the operator's intent.
        """
        if not encryption_key:
            return False
        sqlite_module = _get_sqlite_module(encryption_key)
        return sqlite_module is not sqlite3

    def _snapshot_sqlite(
        self,
        dest_path: Path,
        *,
        encryption_key: str | None,
        encrypted: bool,
    ) -> None:
        """Live-snapshot the metadata SQLite file into *dest_path*.

        A plain database uses the DB-API ``Connection.backup()``, which is
        consistent under WAL without quiescing writers.  An encrypted one
        goes through :meth:`_export_sqlcipher_snapshot` — pysqlcipher3's
        ``Connection`` has no ``backup()``.
        """
        if encrypted and encryption_key is not None:
            self._export_sqlcipher_snapshot(dest_path, encryption_key)
            return

        try:
            src = sqlite3.connect(str(self._metadata_db_path), check_same_thread=False)
        except sqlite3.Error as exc:
            raise DatabaseError(
                "Failed to open SQLite source for snapshot",
                details=str(exc),
            ) from exc

        try:
            try:
                dst = sqlite3.connect(str(dest_path), check_same_thread=False)
            except sqlite3.Error as exc:
                raise DatabaseError(
                    "Failed to open SQLite destination for snapshot",
                    details=str(exc),
                ) from exc

            try:
                src.backup(dst)
            except sqlite3.Error as exc:
                raise DatabaseError(
                    "SQLite snapshot via Connection.backup() failed",
                    details=str(exc),
                ) from exc
            finally:
                dst.close()
        finally:
            src.close()

    def _export_sqlcipher_snapshot(self, dest_path: Path, encryption_key: str) -> None:
        """Snapshot an encrypted database with SQLCipher's ``sqlcipher_export()``.

        The destination is ``ATTACH``ed under the same key literal
        :class:`MetadataRegistry` passes to ``PRAGMA key`` (bound, never
        interpolated), so it derives the identical key and is encrypted from
        its first page — restoring on a host with the matching key just
        works, and plaintext never touches disk.  The export runs inside one
        explicit read transaction, so every table comes from the same WAL
        snapshot even while the API keeps writing (verified in-image: a row
        committed mid-export is excluded).

        ``isolation_level=None`` stops the driver issuing its own
        ``BEGIN``/``COMMIT`` (and ``ATTACH`` is refused inside a
        transaction).  Never install a trace callback on this connection:
        pysqlcipher3 segfaults when one re-enters during the export.
        """
        sqlcipher = _get_sqlite_module(encryption_key)
        key_literal = f"x'{encryption_key.encode().hex()}'"
        try:
            src = sqlcipher.connect(
                str(self._metadata_db_path),
                check_same_thread=False,
                isolation_level=None,
            )
        except sqlcipher.Error as exc:
            raise DatabaseError(
                "Failed to open SQLite source for snapshot",
                details=str(exc),
            ) from exc

        try:
            # PRAGMA does not accept ``?`` binding; same literal as the registry.
            src.execute(f'PRAGMA key = "{key_literal}"')
            src.execute("ATTACH DATABASE ? AS snapshot KEY ?", (str(dest_path), key_literal))
            src.execute("BEGIN")
            src.execute("SELECT sqlcipher_export('snapshot')").fetchall()
            src.execute("COMMIT")
        except sqlcipher.Error as exc:
            raise DatabaseError(
                "SQLCipher snapshot via sqlcipher_export() failed",
                details=str(exc),
            ) from exc
        finally:
            # Closing rolls back an open transaction and detaches the snapshot.
            src.close()

    def _copy_chroma_tree(
        self,
        dest_root: Path,
        *,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> None:
        """Recursively copy chroma_path into *dest_root*.

        ``symlinks=False`` follows symlinks during copy — files at the
        destination are regular, so a malicious symlink inside the
        Chroma directory cannot point outside the staging area.  The
        symlink-parent check on the *output path* covers the inverse
        attack on the destination tarball location.
        """
        try:
            shutil.copytree(
                str(self._chroma_path),
                str(dest_root),
                symlinks=False,
                dirs_exist_ok=False,
            )
        except OSError as exc:
            raise DatabaseError(
                f"Failed to copy ChromaDB tree from '{self._chroma_path}'",
                details=str(exc),
            ) from exc
        if progress_callback is not None:
            progress_callback("backup-chroma", 1, 1)

    def _write_tarball(
        self,
        *,
        staging: Path,
        output_path: Path,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> None:
        """Pack the staging directory into a ``.tar.gz``.

        Manifest is added first so a streaming reader can validate it
        without unpacking the whole archive.
        """
        try:
            with tarfile.open(str(output_path), "w:gz") as tar:
                tar.add(
                    str(staging / self._MANIFEST_NAME),
                    arcname=self._MANIFEST_NAME,
                )
                tar.add(
                    str(staging / self._SQLITE_NAME),
                    arcname=self._SQLITE_NAME,
                )
                tar.add(
                    str(staging / self._CHROMA_SUBDIR),
                    arcname=self._CHROMA_SUBDIR,
                )
        except OSError as exc:
            raise DatabaseError(
                f"Failed to write backup tarball at '{output_path}'",
                details=str(exc),
            ) from exc
        if progress_callback is not None:
            progress_callback("backup-archive", 1, 1)

    # ------------------------------------------------------------------
    # Internal — manifest parsing (restore phase)
    # ------------------------------------------------------------------

    def _read_manifest_from_tarball(self, input_path: Path) -> dict[str, Any]:
        """Extract ``MANIFEST.json`` from the tarball without full unpack."""
        try:
            with tarfile.open(str(input_path), "r:*") as tar:
                try:
                    member = tar.getmember(self._MANIFEST_NAME)
                except KeyError as exc:
                    raise DatabaseError(
                        f"Backup archive is missing '{self._MANIFEST_NAME}' — "
                        "not a valid sec-rag backup.",
                    ) from exc
                fobj = tar.extractfile(member)
                if fobj is None:
                    raise DatabaseError(
                        f"'{self._MANIFEST_NAME}' is not a regular file in the archive.",
                    )
                payload = fobj.read()
        except (OSError, tarfile.TarError) as exc:
            raise DatabaseError(
                f"Failed to read backup archive '{input_path}'",
                details=str(exc),
            ) from exc

        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DatabaseError(
                f"'{self._MANIFEST_NAME}' is not valid JSON; archive is corrupt.",
                details=str(exc),
            ) from exc
        if not isinstance(decoded, dict):
            raise DatabaseError(
                f"'{self._MANIFEST_NAME}' must be a JSON object; archive is corrupt.",
            )
        return decoded

    def _validate_format_version(self, manifest: dict[str, Any]) -> None:
        """Refuse manifests with an unrecognised ``format_version``.

        Forward-only on the format axis as well — a future build may
        widen the manifest, and silently accepting an unknown version
        would mask the gap.
        """
        version = manifest.get("format_version")
        if version != self._FORMAT_VERSION:
            raise DatabaseError(
                f"Backup format_version is {version!r}; this build supports "
                f"{self._FORMAT_VERSION}. Upgrade or downgrade to match.",
            )

    def _parse_manifest_stamp(self, manifest: dict[str, Any]) -> EmbedderStamp:
        raw = manifest.get("embedder_stamp")
        if not isinstance(raw, dict):
            raise DatabaseError(
                "Backup MANIFEST.json is missing or has malformed "
                "embedder_stamp; archive is corrupt.",
            )
        try:
            return EmbedderStamp(
                provider=raw["provider"],
                model=raw["model"],
                dimension=int(raw["dimension"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DatabaseError(
                "Backup MANIFEST.json embedder_stamp is malformed.",
                details=str(exc),
            ) from exc

    def _parse_manifest_schema(self, manifest: dict[str, Any]) -> int:
        raw = manifest.get("schema_version")
        try:
            value = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise DatabaseError(
                "Backup MANIFEST.json has malformed schema_version.",
                details=str(exc),
            ) from exc
        if value <= 0:
            raise DatabaseError(
                f"Backup MANIFEST.json schema_version is non-positive ({value}).",
            )
        return value

    # ------------------------------------------------------------------
    # Internal — extraction and replacement (restore phase)
    # ------------------------------------------------------------------

    def _extract_tarball(self, input_path: Path, dest: Path) -> None:
        """Extract *input_path* into *dest* with path-traversal guards.

        Python 3.12's ``tarfile.data_filter`` enforces the safe-extract
        contract (no absolute paths, no ``..``, no symlinks pointing
        outside the destination).  Older runtimes fall through to a
        manual guard that resolves each member path against *dest*.
        """
        try:
            with tarfile.open(str(input_path), "r:*") as tar:
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(str(dest), filter="data")
                else:  # pragma: no cover — defensive on older runtimes
                    self._safe_extractall(tar, dest)
        except (OSError, tarfile.TarError) as exc:
            raise DatabaseError(
                f"Failed to extract backup archive '{input_path}'",
                details=str(exc),
            ) from exc

    @staticmethod
    def _safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
        """Manual path-traversal guard for runtimes without ``data_filter``.

        Resolves each member's destination path against *dest* and
        refuses anything that escapes — covers absolute paths,
        ``../`` traversal, and symlinked tar entries.
        """
        dest_resolved = dest.resolve()
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not target.is_relative_to(dest_resolved):
                raise DatabaseError(
                    f"Backup contains a path-traversal entry: '{member.name}'.",
                )
        # Each member's resolved destination has been validated above
        # against ``dest`` — anything pointing outside has already
        # raised.  Suppression is justified by that pre-check.
        tar.extractall(str(dest))  # noqa: S202

    def _replace_target_paths(
        self,
        *,
        extracted_chroma: Path,
        extracted_sqlite: Path,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> None:
        """Replace the live ChromaDB and SQLite paths with the extracted contents.

        ChromaDB is replaced first because it is the bigger of the two
        and the more likely to fail mid-write (filesystem fills up,
        permission error).  A crash between the two replacements
        leaves a fresh Chroma plus the previous SQLite — a subsequent
        :class:`MetadataRegistry` open would surface a stamp mismatch
        on the next ChromaDBClient open and the operator restores
        again.

        SQLite ``-wal`` and ``-shm`` sidecars are removed before the
        replacement so a stale write-ahead log from the old database
        does not corrupt the freshly-restored file.
        """
        self._chroma_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata_db_path.parent.mkdir(parents=True, exist_ok=True)

        if self._chroma_path.exists():
            shutil.rmtree(str(self._chroma_path))
        shutil.move(str(extracted_chroma), str(self._chroma_path))
        if progress_callback is not None:
            progress_callback("restore-chroma", 1, 1)

        for suffix in ("", "-wal", "-shm"):
            sidecar = Path(f"{self._metadata_db_path}{suffix}")
            if sidecar.exists():
                sidecar.unlink()
        shutil.move(str(extracted_sqlite), str(self._metadata_db_path))
        if progress_callback is not None:
            progress_callback("restore-sqlite", 1, 1)
