"""Database access. Read paths go through a connection that cannot write.

Two independent locks (defence in depth):
1. The file is opened with the URI `mode=ro`: SQLite itself refuses writes to it, and
   refuses to open a missing file (a plain `sqlite3.connect` would silently create an
   empty database and the tools would answer "no such order" instead of failing).
2. `PRAGMA query_only = ON` makes the connection reject any change, including to a
   database attached later with ATTACH.
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
