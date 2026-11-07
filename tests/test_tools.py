"""Tools: correct data for the session customer, nothing for anyone else."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from reliable_tool_agent.tools import (
    NotFoundError,
    Session,
    check_return_eligibility,
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


# --- check_return_eligibility -------------------------------------------------------
# Fixed data from the deterministic world: C0001's O10001 was delivered on 2026-09-18,
# O10002 contains a final-sale item, C0006's O10009 is shipped.


def eligibility(customer: str, order_id: str, shop_db, today: date = date(2026, 10, 15)):
    return check_return_eligibility(Session(customer, shop_db, today=today), order_id)


def test_ten_days_after_delivery_is_returnable(shop_db):
    """The case the model got wrong in the chat: delivered 5 Oct, today 15 Oct."""
    result = eligibility("C0001", "O10003", shop_db)
    assert result.days_since_delivery == 10
    assert result.return_window_ends == date(2026, 11, 4)
    assert all(item.returnable for item in result.items)


@pytest.mark.parametrize(
    ("today", "day", "returnable"),
    [
        (date(2026, 9, 18), 0, True),  # delivery day counts as day 0
        (date(2026, 10, 18), 30, True),  # day 30 is accepted
        (date(2026, 10, 19), 31, False),  # day 31 is refused
    ],
)
def test_return_window_boundaries(shop_db, today, day, returnable):
    result = eligibility("C0001", "O10001", shop_db, today)
    assert result.days_since_delivery == day
    assert result.return_window_ends == date(2026, 10, 18)
    assert {item.returnable for item in result.items} == {returnable}
    assert {item.returnable_if_defective_or_wrong for item in result.items} == {returnable}


def test_final_sale_item_only_if_defective_or_wrong(shop_db):
    items = {i.product_name: i for i in eligibility("C0001", "O10002", shop_db).items}
    lights = items["String lights 10 m"]
    assert lights.final_sale and not lights.returnable
    assert lights.returnable_if_defective_or_wrong
    assert "Final sale" in lights.reason
    assert items["Chef's knife 20 cm"].returnable  # the rest of the order is not affected


def test_final_sale_item_after_the_window_is_not_returnable_at_all(shop_db):
    late = eligibility("C0001", "O10002", shop_db, date(2026, 12, 1))
    assert not any(i.returnable or i.returnable_if_defective_or_wrong for i in late.items)


def test_order_not_delivered_cannot_be_returned(shop_db):
    result = eligibility("C0006", "O10009", shop_db)
    assert result.status == "shipped"
    assert result.days_since_delivery is None and result.return_window_ends is None
    assert not any(i.returnable or i.returnable_if_defective_or_wrong for i in result.items)
    assert "shipped" in result.items[0].reason


def test_eligibility_of_another_customers_order_is_not_found(shop_db):
    with pytest.raises(NotFoundError):
        eligibility("C0001", "O10009", shop_db)
