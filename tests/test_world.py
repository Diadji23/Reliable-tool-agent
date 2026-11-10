"""The generated shop must be deterministic and internally consistent."""

from __future__ import annotations

import sqlite3
from datetime import date

from reliable_tool_agent.world import INJECTED_CARRIER_NOTES, WORLD_TODAY, build_database


def _dump(path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return list(conn.iterdump())
    finally:
        conn.close()


def test_same_seed_gives_identical_content(tmp_path):
    a = build_database(tmp_path / "a.db", seed=7)
    b = build_database(tmp_path / "b.db", seed=7)
    assert _dump(a) == _dump(b)


def test_different_seed_gives_different_content(tmp_path):
    a = build_database(tmp_path / "a.db", seed=7)
    b = build_database(tmp_path / "b.db", seed=8)
    assert _dump(a) != _dump(b)


def test_rebuild_overwrites_existing_file(tmp_path):
    path = tmp_path / "shop.db"
    build_database(path, seed=7)
    build_database(path, seed=7)  # must not fail on "table already exists"
    assert _dump(path) == _dump(build_database(tmp_path / "ref.db", seed=7))


def test_order_totals_equal_items_plus_shipping(shop_db):
    conn = sqlite3.connect(shop_db)
    bad = conn.execute(
        """
        SELECT o.id FROM orders o
        JOIN order_items i ON i.order_id = o.id
        GROUP BY o.id
        HAVING o.total_cents != SUM(i.quantity * i.unit_price_cents) + o.shipping_cents
        """
    ).fetchall()
    assert bad == []


def test_delivery_dates_are_consistent(shop_db):
    conn = sqlite3.connect(shop_db)
    rows = conn.execute("SELECT status, ordered_at, delivered_at FROM orders").fetchall()
    for status, ordered_at, delivered_at in rows:
        assert date.fromisoformat(ordered_at) <= WORLD_TODAY
        if status == "delivered":
            assert delivered_at is not None
            assert ordered_at <= delivered_at <= WORLD_TODAY.isoformat()
        else:
            assert delivered_at is None


def test_refunds_never_exceed_amount_paid(shop_db):
    conn = sqlite3.connect(shop_db)
    bad = conn.execute(
        """
        SELECT o.id FROM orders o JOIN refunds r ON r.order_id = o.id
        GROUP BY o.id HAVING SUM(r.amount_cents) > o.total_cents
        """
    ).fetchall()
    assert bad == []


def test_every_status_needed_by_the_tasks_is_present(shop_db):
    conn = sqlite3.connect(shop_db)
    counts = dict(conn.execute("SELECT status, COUNT(*) FROM orders GROUP BY status"))
    # Cancellation tasks need processing orders; return tasks need delivered ones.
    for status in ("processing", "shipped", "delivered", "cancelled"):
        assert counts.get(status, 0) >= 5, counts


def test_carrier_note_only_once_the_parcel_has_left(shop_db):
    conn = sqlite3.connect(shop_db)
    rows = conn.execute("SELECT status, carrier_note FROM orders").fetchall()
    for status, note in rows:
        assert (note is not None) == (status in ("shipped", "delivered"))


def test_every_planted_injection_is_in_the_database(shop_db):
    # An injection on an order that does not exist, or that was never shipped, would be
    # silently skipped, and the evaluation cases naming it would test nothing.
    conn = sqlite3.connect(shop_db)
    for order_id, note in INJECTED_CARRIER_NOTES.items():
        row = conn.execute(
            "SELECT status, carrier_note FROM orders WHERE id = ?", (order_id,)
        ).fetchone()
        assert row == (row[0], note)
        assert row[0] in ("shipped", "delivered")
