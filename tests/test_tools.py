"""Tools: correct data for the session customer, nothing for anyone else."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from reliable_tool_agent.tools import (
    NotFoundError,
    Session,
    get_order,
    list_my_orders,
    search_policies,
)


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


# --- list_my_orders -------------------------------------------------------------------


def test_list_my_orders_returns_only_own_orders_newest_first(shop_db, two_customers):
    a, _, _ = two_customers
    orders = list_my_orders(Session(a, shop_db))
    conn = sqlite3.connect(shop_db)
    own = {r[0] for r in conn.execute("SELECT id FROM orders WHERE customer_id = ?", (a,))}
    assert {o.id for o in orders} == own
    dates = [o.ordered_at for o in orders]
    assert dates == sorted(dates, reverse=True)


def test_list_my_orders_status_filter(shop_db):
    conn = sqlite3.connect(shop_db)
    (cid,) = conn.execute(
        "SELECT customer_id FROM orders GROUP BY customer_id "
        "HAVING COUNT(DISTINCT status) >= 2 ORDER BY customer_id LIMIT 1"
    ).fetchone()
    session = Session(cid, shop_db)
    everything = list_my_orders(session)
    delivered = list_my_orders(session, "delivered")
    assert delivered and all(o.status == "delivered" for o in delivered)
    assert len(delivered) < len(everything)


# --- search_policies ------------------------------------------------------------------


def _top(query: str, shop_db) -> tuple[str, int]:
    rule = search_policies(Session("C0001", shop_db), query)[0]
    return rule.policy, rule.rule


def test_search_finds_the_exact_rule(shop_db):
    assert _top("Can I return a final sale item that arrived damaged?", shop_db) == ("returns", 3)


def test_search_matches_word_forms(shop_db):
    # "returning" must match "returned"; "refund" must match "refunds" / "refunded".
    assert _top("returning a lamp", shop_db)[0] == "returns"
    assert _top("How do I get a refund?", shop_db)[0] == "refunds"


def test_search_unrelated_query_returns_nothing(shop_db):
    assert search_policies(Session("C0001", shop_db), "what is the capital of France") == []


def test_search_known_limit_no_synonyms(shop_db):
    """Documented limit of keyword search: 'send it back' does not find the returns
    policy (no shared word). If this test starts failing, the search got smarter."""
    results = search_policies(Session("C0001", shop_db), "send it back")
    assert all(r.policy != "returns" for r in results)
