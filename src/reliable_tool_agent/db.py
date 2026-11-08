"""Database access. Read paths go through a connection that cannot write.

Two independent locks (defence in depth):
1. The file is opened with the URI `mode=ro`: SQLite itself refuses writes to it, and
   refuses to open a missing file (a plain `sqlite3.connect` would silently create an
   empty database and the tools would answer "no such order" instead of failing).
2. `PRAGMA query_only = ON` makes the connection reject any change, including to a
   database attached later with ATTACH.

The only write path is issuing a refund. It uses `connect_for_refunds`, whose SQLite
authorizer allows reads and `INSERT INTO refunds`, and refuses everything else.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "shop.db"


def connect_readonly(path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open the shop database read-only. Raises sqlite3.OperationalError if missing."""
    uri = f"{Path(path).resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only = ON")
    conn.row_factory = sqlite3.Row
    return conn


# Actions the refund connection may perform. Anything not listed is denied by SQLite
# when the statement is prepared, before it runs.
_REFUND_ALLOWED = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,  # COUNT, SUM, COALESCE in the checks
    sqlite3.SQLITE_TRANSACTION,  # BEGIN / COMMIT / ROLLBACK
}


def _refund_authorizer(action: int, arg1: str | None, *_: object) -> int:
    if action in _REFUND_ALLOWED:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_INSERT and arg1 == "refunds":
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def connect_for_refunds(path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open the shop database to issue refunds: reads plus INSERT INTO refunds, nothing else.

    `mode=rw` (not `rwc`) still refuses to create a missing file. Transactions are left to
    the caller (`isolation_level=None`), so that the checks and the insert can share one
    `BEGIN IMMEDIATE` transaction.
    """
    uri = f"{Path(path).resolve().as_uri()}?mode=rw"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.set_authorizer(_refund_authorizer)
    return conn
