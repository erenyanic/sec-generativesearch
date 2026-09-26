"""Encrypted-at-rest provider-credential store.

The store layers a small CRUD surface over the
``provider_credentials`` table that migration v2 creates inside the
SQLCipher-encrypted metadata database.  It honours the
:class:`~sec_generative_search.core.credentials.CredentialStore`
protocol so the resolver chain treats it identically to the in-memory
session store.

Design notes:

- **No per-row crypto.**  SQLCipher's whole-database encryption is the
    load-bearing control.  Layering Fernet on top would introduce a
    second key-management story without a clear threat that justifies
    it; revisit only when a concrete threat model demands compartment
    isolation between the credentials table and the rest of the metadata
    tables.

- **Reuses the registry's connection.**  Opening a second connection
    to the same SQLCipher file would duplicate the ``_get_sqlite_module``
    selection, the ``PRAGMA key`` ordering, and the WAL pragma.
    Sharing the registry's connection — under the registry's lock —
    keeps the encryption boilerplate single-sourced.  The coupling on
    ``MetadataRegistry._conn`` / ``_lock`` / ``_db_error`` is
    intentional; the migration ensures the table exists before the store
    is constructed.

- **Refuses construction without persistence + SQLCipher.**  The
  constructor reads the live ``DB_PERSIST_PROVIDER_CREDENTIALS`` and
  the registry's ``encrypted`` property.  If persistence is off, the
  store raises :class:`ConfigurationError` rather than silently
  swallowing writes — a no-op store would be an exfiltration trap
  ("set worked, but get returns nothing — where did my key go?").
  The settings validator in
  :class:`~sec_generative_search.config.settings.DatabaseSettings`
  already rejects ``persist=true`` without SQLCipher at load, so the
  registry's ``encrypted=True`` invariant is doubly enforced.

- **No raw key in any read API except ``get``.**  ``list_providers``
  returns provider names; ``count`` returns an integer; ``__repr__``
  carries no value.  Audit-log entries on ``set`` / ``delete`` /
  ``clear`` mask the credential tail via :func:`mask_secret`.

- **No background pruning.**  Unlike the in-memory store, persistent
    credentials are intentionally retained until the user removes them.
    Operators that want time-based expiry can implement it elsewhere;
    the store does not impose policy here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sec_generative_search.config.settings import (
    DatabaseSettings,
    get_settings,
)
from sec_generative_search.core.exceptions import (
    ConfigurationError,
    DatabaseError,
)
from sec_generative_search.core.logging import audit_log, get_logger
from sec_generative_search.core.security import mask_secret
from sec_generative_search.database.metadata import MetadataRegistry

__all__ = [
    "CredentialRecord",
    "EncryptedCredentialStore",
]

logger = get_logger(__name__)


@dataclass(frozen=True)
class CredentialRecord:
    """One row from the ``provider_credentials`` table.

    Used only by introspection paths (``count`` / future admin
    listings).  The ``api_key`` is intentionally **not** carried on
    this dataclass — a record returned to a listing context must never
    expose the secret.  The :meth:`EncryptedCredentialStore.get` path
    is the single sanctioned read site for the raw value.
    """

    user_id: str
    provider: str
    created_at: str
    updated_at: str


class EncryptedCredentialStore:
    """Persistent provider-credential store backed by SQLCipher.

    Conforms to
    :class:`~sec_generative_search.core.credentials.CredentialStore`.
    See module docstring for design rationale and the SQLCipher
    coupling.

    Construction contract:

    - The encrypted-credential table (created by migration v2) MUST
      already exist.  The standard call site is
      ``MetadataRegistry().__init__`` → migrations apply v2 → caller
      hands the registry to ``EncryptedCredentialStore(registry)``.
    - ``DB_PERSIST_PROVIDER_CREDENTIALS`` MUST be ``True`` and the
      registry MUST report ``encrypted=True``.  Either condition false
      raises :class:`ConfigurationError` — silent no-op writes are an
      operator footgun and refused by design.
    """

    _TABLE = "provider_credentials"

    def __init__(
        self,
        registry: MetadataRegistry,
        *,
        settings: DatabaseSettings | None = None,
    ) -> None:
        db_settings = settings or get_settings().database

        if not db_settings.persist_provider_credentials:
            raise ConfigurationError(
                "EncryptedCredentialStore requires "
                "DB_PERSIST_PROVIDER_CREDENTIALS=true.  Either enable "
                "the toggle (and configure SQLCipher) or use the "
                "in-memory session store via "
                "`InMemorySessionCredentialStore` instead."
            )
        if not registry.encrypted:
            # Defensive: settings validation should have caught this,
            # but an operator who handcrafts a registry with
            # ``encryption_key=None`` could land here.  Refuse rather
            # than silently writing plaintext credentials to disk.
            raise ConfigurationError(
                "EncryptedCredentialStore refuses to operate without "
                "SQLCipher.  The registry was opened without an "
                "encryption key; configure DB_ENCRYPTION_KEY (or "
                "DB_ENCRYPTION_KEY_FILE) and re-open the registry."
            )

        # Intentional intra-package coupling — see module docstring.
        self._registry = registry
        self._conn = registry._conn
        self._lock = registry._lock
        self._db_error = registry._db_error

    # ------------------------------------------------------------------
    # CredentialStore protocol
    # ------------------------------------------------------------------

    def get(self, key_id: str, provider: str) -> str | None:
        """Return the stored credential, or ``None`` if absent.

        Does not emit an audit log per call — that would flood under
        request rate.  The resolver-level audit
        (:func:`~sec_generative_search.core.credentials.encrypted_user_resolver`)
        logs each successful resolution exactly once.
        """
        sql = (
            f"SELECT api_key FROM {self._TABLE} "  # noqa: S608 — table name is a constant
            "WHERE user_id = ? AND provider = ? LIMIT 1"
        )
        try:
            with self._lock:
                row = self._conn.execute(sql, (key_id, provider)).fetchone()
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to read provider credential",
                details=str(exc),
            ) from exc
        if row is None:
            return None
        return row["api_key"]

    def set(self, key_id: str, provider: str, api_key: str) -> None:
        """Upsert the credential for ``(key_id, provider)``.

        Updates ``updated_at`` on conflict; preserves the original
        ``created_at`` so rotation history stays observable.
        """
        if not api_key:
            # Same rationale as the in-memory store: an empty string
            # would silently shadow downstream resolvers.
            raise ValueError(
                "api_key must be a non-empty string. Use delete() to remove a stored credential."
            )

        now = datetime.now(UTC).isoformat()
        # UPDATE-then-INSERT, not ``INSERT … ON CONFLICT DO UPDATE``: the
        # UPSERT clause needs SQLite >= 3.24, and the image's SQLCipher
        # bundles 3.15.2.  The UPDATE opens the write transaction even when
        # it matches no row, so another process cannot insert the same key
        # before our INSERT commits.
        sql_update = (
            f"UPDATE {self._TABLE} SET api_key = ?, updated_at = ? "  # noqa: S608 — table name is a constant
            "WHERE user_id = ? AND provider = ?"
        )
        sql_insert = (
            f"INSERT INTO {self._TABLE} "  # noqa: S608 — table name is a constant
            "(user_id, provider, api_key, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(sql_update, (api_key, now, key_id, provider))
                if cursor.rowcount == 0:
                    self._conn.execute(sql_insert, (key_id, provider, api_key, now, now))
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to upsert provider credential",
                details=str(exc),
            ) from exc

        audit_log(
            "credential_set",
            detail=(
                f"store=encrypted_persistent "
                f"user_id_tail={mask_secret(key_id)} "
                f"provider={provider} "
                f"key_tail={mask_secret(api_key)}"
            ),
        )

    def delete(self, key_id: str, provider: str) -> bool:
        sql = (
            f"DELETE FROM {self._TABLE} "  # noqa: S608 — table name is a constant
            "WHERE user_id = ? AND provider = ?"
        )
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(sql, (key_id, provider))
                removed = cursor.rowcount > 0
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to delete provider credential",
                details=str(exc),
            ) from exc

        if removed:
            audit_log(
                "credential_delete",
                detail=(
                    f"store=encrypted_persistent "
                    f"user_id_tail={mask_secret(key_id)} "
                    f"provider={provider}"
                ),
            )
        return removed

    def list_providers(self, key_id: str) -> set[str]:
        sql = (
            f"SELECT provider FROM {self._TABLE} "  # noqa: S608 — table name is a constant
            "WHERE user_id = ?"
        )
        try:
            with self._lock:
                rows = self._conn.execute(sql, (key_id,)).fetchall()
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to list stored providers",
                details=str(exc),
            ) from exc
        return {row["provider"] for row in rows}

    def clear(self, key_id: str) -> int:
        sql = (
            f"DELETE FROM {self._TABLE} "  # noqa: S608 — table name is a constant
            "WHERE user_id = ?"
        )
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(sql, (key_id,))
                removed = cursor.rowcount
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to clear stored credentials",
                details=str(exc),
            ) from exc

        if removed:
            audit_log(
                "credential_clear",
                detail=(
                    f"store=encrypted_persistent "
                    f"user_id_tail={mask_secret(key_id)} "
                    f"removed={removed}"
                ),
            )
        return removed

    # ------------------------------------------------------------------
    # Introspection (admin / metrics)
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the total number of stored credential rows.

        Does not return the rows themselves and does not group by
        ``user_id`` — the value is for operational metrics only and
        carries no per-tenant detail that could leak via aggregation.
        """
        sql = f"SELECT COUNT(*) FROM {self._TABLE}"  # noqa: S608 — constant table
        try:
            with self._lock:
                row = self._conn.execute(sql).fetchone()
            return row[0]
        except self._db_error as exc:
            raise DatabaseError(
                "Failed to count provider credentials",
                details=str(exc),
            ) from exc

    def __repr__(self) -> str:
        # Never echo the connection or any user-id; just the type and
        # the encryption invariant.
        return f"EncryptedCredentialStore(encrypted={self._registry.encrypted!r})"
