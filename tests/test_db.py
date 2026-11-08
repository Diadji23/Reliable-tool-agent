"""The read-only connection refuses every write; the refund one allows only INSERT INTO refunds."""

from __future__ import annotations

import sqlite3

import pytest

from reliable_tool_agent.db import connect_for_refunds, connect_readonly

REFUND_ROW = "INSERT INTO refunds VALUES (999, 'O10001', 'P001', 1, 0, 100, 'x', '2026-10-15')"

WRITES = [
    REFUND_ROW,
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


def test_refund_connection_can_read_and_insert_a_refund(writable_db):
    conn = connect_for_refunds(writable_db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        (before,) = conn.execute("SELECT COUNT(*) FROM refunds").fetchone()
        conn.execute(REFUND_ROW)
        conn.execute("COMMIT")
        (after,) = conn.execute("SELECT COALESCE(SUM(1), 0) FROM refunds").fetchone()
    finally:
        conn.close()
    assert after == before + 1


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE refunds SET amount_cents = 1",  # even on its own table, no update
        "DELETE FROM refunds",
        "UPDATE orders SET status = 'cancelled'",
        "INSERT INTO order_items VALUES ('O10001', 'P002', 1, 100)",
        "DELETE FROM customers",
        "DROP TABLE orders",
        "CREATE TABLE evil (x INTEGER)",
        "PRAGMA query_only = OFF",
    ],
)
def test_refund_connection_refuses_everything_else(writable_db, sql):
    conn = connect_for_refunds(writable_db)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            conn.execute(sql)
    finally:
        conn.close()


def test_refund_connection_cannot_attach_another_database(writable_db, tmp_path):
    conn = connect_for_refunds(writable_db)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            conn.execute("ATTACH DATABASE ? AS other", (str(tmp_path / "other.db"),))
    finally:
        conn.close()


def test_refund_connection_does_not_create_a_missing_database(tmp_path):
    missing = tmp_path / "nope.db"
    with pytest.raises(sqlite3.OperationalError):
        connect_for_refunds(missing)
    assert not missing.exists()
