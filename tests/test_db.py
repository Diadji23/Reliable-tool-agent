"""The read-only connection must refuse every kind of write."""

from __future__ import annotations

import sqlite3

import pytest

from reliable_tool_agent.db import connect_readonly

WRITES = [
    "INSERT INTO refunds VALUES (999, 'O10001', 100, 'x', '2026-10-15')",
    "UPDATE orders SET status = 'cancelled'",
    "DELETE FROM customers",
    "DROP TABLE orders",
    "CREATE TABLE evil (x INTEGER)",
]


def test_reads_work(shop_db):
    with connect_readonly(shop_db) as conn:
        (n,) = conn.execute("SELECT COUNT(*) FROM customers").fetchone()
    assert n > 0


@pytest.mark.parametrize("sql", WRITES)
def test_writes_are_refused(shop_db, sql):
    conn = connect_readonly(shop_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(sql)
    finally:
        conn.close()


def test_attached_database_is_also_read_only(shop_db, tmp_path):
    """ATTACH would open a second, writable file; query_only must block that too."""
    conn = connect_readonly(shop_db)
    try:
        conn.execute("ATTACH DATABASE ? AS other", (str(tmp_path / "other.db"),))
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE other.evil (x INTEGER)")
    finally:
        conn.close()


def test_missing_database_fails_instead_of_creating_an_empty_one(tmp_path):
    missing = tmp_path / "nope.db"
    with pytest.raises(sqlite3.OperationalError):
        connect_readonly(missing)
    assert not missing.exists()
