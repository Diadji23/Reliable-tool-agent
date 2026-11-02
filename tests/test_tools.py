"""Tools: correct data for the session customer, nothing for anyone else."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from reliable_tool_agent.tools import NotFoundError, Session, get_order


@pytest.fixture(scope="module")
def two_customers(shop_db):
    """(customer A, one of A's orders, customer B) with A != B."""
    conn = sqlite3.connect(shop_db)
    a, order = conn.execute("SELECT customer_id, id FROM orders ORDER BY id LIMIT 1").fetchone()
    (b,) = conn.execute(
        "SELECT id FROM customers WHERE id != ? ORDER BY id LIMIT 1", (a,)
    ).fetchone()
    conn.close()
    return a, order, b


def test_customer_sees_own_order(shop_db, two_customers):
    a, order_id, _ = two_customers
    order = get_order(Session(a, shop_db), order_id)
    assert order.id == order_id
    assert order.items
    assert isinstance(order.ordered_at, date)  # converted from the ISO string


def test_total_matches_items_plus_shipping(shop_db, two_customers):
    a, order_id, _ = two_customers
    order = get_order(Session(a, shop_db), order_id)
    items = sum(i.quantity * i.unit_price_cents for i in order.items)
    assert order.total_cents == items + order.shipping_cents


def test_other_customers_order_looks_missing(shop_db, two_customers):
    _, order_id, b = two_customers
    with pytest.raises(NotFoundError) as other:
        get_order(Session(b, shop_db), order_id)
    with pytest.raises(NotFoundError) as missing:
        get_order(Session(b, shop_db), "O99999")
    # Same message shape: the answer must not reveal that the order exists.
    assert str(other.value).replace(order_id, "X") == str(missing.value).replace("O99999", "X")


def test_unknown_order_is_not_found(shop_db, two_customers):
    a, _, _ = two_customers
    with pytest.raises(NotFoundError):
        get_order(Session(a, shop_db), "O99999")
