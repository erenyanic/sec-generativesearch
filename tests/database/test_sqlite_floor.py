"""SQL executed by ``src/`` must run on the API image's SQLite 3.15.2.

The image's ``pysqlcipher3`` links SQLCipher 3.4.1, which bundles SQLite
3.15.2 (Debian bookworm ``libsqlcipher0``).  CI and the dev venv have no
``pysqlcipher3`` and run the stdlib driver on a modern SQLite, so SQL that
needs anything newer passes every other test and fails only in a deployed
B/C image — as the ``EncryptedCredentialStore.set`` UPSERT did until
2026-09-26.  This static scan is the CI-visible guard for that floor.

It scans every non-docstring string literal (f-string parts joined) in the
modules that execute SQL, and pins that SQL execution stays inside
``database/`` so the scan's scope stays complete.  Single-word keywords are
matched upper-case only (the codebase writes SQL upper-case; prose such as
``=true`` in error messages must not trip it).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src" / "sec_generative_search"
_DATABASE = _SRC / "database"

# Feature → pattern, each needing a SQLite newer than 3.15.2.
_NEWER_THAN_3_15: dict[str, re.Pattern[str]] = {
    "UPSERT (3.24)": re.compile(r"\bON\s+CONFLICT\b.*?\bDO\s+(UPDATE|NOTHING)\b", re.I | re.S),
    "RETURNING (3.35)": re.compile(r"\bRETURNING\b"),
    "window function (3.25)": re.compile(r"\bOVER\s*\(", re.I),
    "IIF (3.32)": re.compile(r"\bIIF\s*\(", re.I),
    "RENAME COLUMN (3.25)": re.compile(r"\bRENAME\s+COLUMN\b", re.I),
    "DROP COLUMN (3.35)": re.compile(r"\bDROP\s+COLUMN\b", re.I),
    "NULLS FIRST/LAST (3.30)": re.compile(r"\bNULLS\s+(FIRST|LAST)\b", re.I),
    "aggregate FILTER (3.30)": re.compile(r"\bFILTER\s*\(\s*WHERE\b", re.I),
    "generated column (3.31)": re.compile(r"\bGENERATED\s+ALWAYS\b", re.I),
    "STRICT table (3.37)": re.compile(r"\)\s*STRICT\b"),
    "JSON ->> operator (3.38)": re.compile(r"->>"),
    "TRUE/FALSE literal (3.23)": re.compile(r"\b(TRUE|FALSE)\b"),
}

_SQL_EXEC_METHODS = {"execute", "executemany", "executescript"}


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _string_literals(source: str) -> list[tuple[int, str]]:
    """Every non-docstring string literal, f-strings joined with ``{}``."""
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            text = "".join(
                part.value if isinstance(part, ast.Constant) else "{}" for part in node.values
            )
            out.append((node.lineno, text))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            out.append((node.lineno, node.value))
    return out


def _violations(source: str) -> list[str]:
    # A set: an f-string and its merged literal part report the same line.
    found = set()
    for lineno, text in _string_literals(source):
        for feature, pattern in _NEWER_THAN_3_15.items():
            if pattern.search(text):
                found.add(f"line {lineno}: {feature}")
    return sorted(found)


def test_sql_stays_within_sqlite_3_15() -> None:
    offenders = []
    for path in sorted(_DATABASE.glob("*.py")):
        for violation in _violations(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(_SRC)} {violation}")
    assert offenders == [], (
        "SQL needs a SQLite newer than 3.15.2, the version bundled with the API "
        "image's SQLCipher — it would pass CI and fail in a deployed B/C image:\n"
        + "\n".join(offenders)
    )


def _sql_exec_lines(source: str) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _SQL_EXEC_METHODS
    ]


def test_sql_execution_stays_inside_the_scanned_package() -> None:
    """A ``.execute(...)`` outside ``database/`` would escape the scan above."""
    outside = [
        f"{path.relative_to(_SRC)}:{lineno}"
        for path in sorted(_SRC.rglob("*.py"))
        if _DATABASE not in path.parents
        for lineno in _sql_exec_lines(path.read_text(encoding="utf-8"))
    ]
    assert outside == []


def test_exec_guard_detects_every_sql_entry_point() -> None:
    source = "conn.execute('SELECT 1')\nconn.executemany(q, rows)\nconn.executescript(s)\n"
    assert _sql_exec_lines(source) == [1, 2, 3]


@pytest.mark.parametrize(
    "snippet",
    [
        # The shape that shipped broken (implicitly concatenated f-string).
        'sql = (f"INSERT INTO {t} (a, b) VALUES (?, ?) "\n'
        '       "ON CONFLICT(a) DO UPDATE SET b = excluded.b")',
        # Spans a placeholder: only the joined f-string text matches.
        'sql = f"INSERT INTO t VALUES (?) ON CONFLICT({key}) DO NOTHING"',
        'sql = "DELETE FROM t WHERE a = ? RETURNING b"',
        'sql = "SELECT a, ROW_NUMBER() OVER (ORDER BY a) FROM t"',
        'sql = "SELECT a FROM t WHERE flag = TRUE"',
    ],
)
def test_scanner_flags_post_3_15_sql(snippet: str) -> None:
    assert _violations(snippet), snippet


def test_scanner_ignores_docstrings_and_prose() -> None:
    source = (
        "def f():\n"
        '    """Avoid ``INSERT … ON CONFLICT DO UPDATE``: RETURNING is 3.35."""\n'
        '    raise ValueError("set DB_PERSIST_PROVIDER_CREDENTIALS=true first")\n'
    )
    assert _violations(source) == []


def test_scan_is_not_vacuous() -> None:
    literals = [
        text
        for path in _DATABASE.glob("*.py")
        for _, text in _string_literals(path.read_text(encoding="utf-8"))
    ]
    sql = [t for t in literals if re.match(r"\s*(SELECT|INSERT|UPDATE|DELETE|CREATE)\b", t)]
    assert len(sql) >= 30
