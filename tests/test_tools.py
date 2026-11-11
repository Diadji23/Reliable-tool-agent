"""Tools: correct data for the session customer, nothing for anyone else."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from reliable_tool_agent.tools import (
    NotFoundError,
    RefundRefusedError,
    Session,
    check_return_eligibility,
    get_order,
    list_my_orders,
    quote_refund,
    refund_order,
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


def test_carrier_note_never_leaves_the_tool(shop_db):
    # O10021 (C0013) holds a planted injection in its carrier note (experiment 03: the
    # model obeyed such notes). The order is returned without it.
    order = get_order(Session("C0013", shop_db), "O10021")
    assert "carrier" not in order.model_dump_json().lower()
    assert "SUPPORT ASSISTANT" not in order.model_dump_json()


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


# --- Refunds ---------------------------------------------------------------------------
# O10040 (C0021, delivered 13 Oct, shipping 590): P010 x1 at 4500, P019 x1 at 2600 final sale.
# O10032 (C0018, delivered 15 Oct): P014 x2 at 14500, one unit already refunded.

# For refunds refused by a policy rule: the refusal comes before the approved amount is
# compared, so its value does not matter.
ANY = 1


def refunds_in(db) -> list[tuple]:
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT * FROM refunds ORDER BY id").fetchall()
    conn.close()
    return rows


def test_changed_mind_refunds_the_item_price_only(writable_db):
    refund = refund_order(
        Session("C0021", writable_db),
        "O10040",
        "P010",
        1,
        "changed_mind",
        approved_amount_cents=4500,
    )
    assert (refund.item_cents, refund.shipping_cents, refund.amount_cents) == (4500, 0, 4500)
    last = refunds_in(writable_db)[-1]
    assert last[1:6] == ("O10040", "P010", 1, 0, 4500)
    assert last[6] == "changed_mind: 1 x P010"
    assert last[7] == "2026-10-15"


def test_defective_item_also_refunds_shipping_but_only_once(writable_db):
    session = Session("C0021", writable_db)
    first = refund_order(
        session, "O10040", "P010", 1, "defective_or_wrong", approved_amount_cents=5090
    )
    assert (first.shipping_cents, first.amount_cents) == (590, 5090)
    # The final-sale item can be refunded when defective, without a second shipping fee.
    second = refund_order(
        session, "O10040", "P019", 1, "defective_or_wrong", approved_amount_cents=2600
    )
    assert (second.shipping_cents, second.amount_cents) == (0, 2600)
    assert second.already_refunded_cents == 5090


def test_final_sale_item_is_refused_when_customer_changed_mind(writable_db):
    with pytest.raises(RefundRefusedError, match="Final sale"):
        refund_order(
            Session("C0021", writable_db),
            "O10040",
            "P019",
            1,
            "changed_mind",
            approved_amount_cents=ANY,
        )


def test_refund_after_the_return_window_is_refused(writable_db):
    late = Session("C0001", writable_db, today=date(2026, 10, 19))  # day 31 for O10001
    for condition in ("changed_mind", "defective_or_wrong"):
        with pytest.raises(RefundRefusedError, match="window ended"):
            refund_order(late, "O10001", "P003", 1, condition, approved_amount_cents=ANY)


def test_order_not_delivered_cannot_be_refunded(writable_db):
    with pytest.raises(RefundRefusedError, match="shipped"):
        refund_order(
            Session("C0006", writable_db),
            "O10009",
            "P015",
            1,
            "defective_or_wrong",
            approved_amount_cents=ANY,
        )


def test_already_refunded_units_are_deducted(writable_db):
    session = Session("C0018", writable_db)
    with pytest.raises(RefundRefusedError, match="1 already refunded, so at most 1"):
        refund_order(session, "O10032", "P014", 2, "changed_mind", approved_amount_cents=ANY)
    refund = refund_order(session, "O10032", "P014", 1, "changed_mind", approved_amount_cents=14500)
    assert refund.amount_cents == 14500


def test_the_same_units_cannot_be_refunded_twice(writable_db):
    session = Session("C0001", writable_db)
    refund_order(
        session,
        "O10003",
        "P012",
        2,
        "changed_mind",
        approved_amount_cents=quote_refund(
            session, "O10003", "P012", 2, "changed_mind"
        ).amount_cents,
    )
    with pytest.raises(RefundRefusedError, match="at most 0"):
        refund_order(session, "O10003", "P012", 1, "changed_mind", approved_amount_cents=ANY)


def test_total_refunded_never_exceeds_amount_paid(writable_db):
    """refunds.md rule 2 is checked on its own, even if per-item records were inconsistent."""
    conn = sqlite3.connect(writable_db)
    conn.execute(
        "INSERT INTO refunds (order_id, product_id, quantity, shipping_cents, amount_cents, "
        "reason, created_at) VALUES ('O10040', 'P019', 1, 0, 5000, 'legacy', '2026-10-14')"
    )
    conn.commit()
    conn.close()
    with pytest.raises(RefundRefusedError, match="cannot exceed the 7690 cents paid"):
        refund_order(
            Session("C0021", writable_db),
            "O10040",
            "P010",
            1,
            "changed_mind",
            approved_amount_cents=ANY,
        )


@pytest.mark.parametrize(
    ("customer", "order_id", "product_id"),
    [
        ("C0001", "O10040", "P010"),  # another customer's order looks missing
        ("C0001", "O10003", "P001"),  # product not in the order
    ],
)
def test_refund_of_something_not_owned_is_not_found(writable_db, customer, order_id, product_id):
    with pytest.raises(NotFoundError):
        refund_order(
            Session(customer, writable_db),
            order_id,
            product_id,
            1,
            "changed_mind",
            approved_amount_cents=ANY,
        )


def test_quantity_must_be_positive(writable_db):
    with pytest.raises(RefundRefusedError, match="at least 1"):
        refund_order(
            Session("C0021", writable_db),
            "O10040",
            "P010",
            0,
            "changed_mind",
            approved_amount_cents=ANY,
        )


def test_refused_refunds_leave_the_database_unchanged(writable_db):
    before = refunds_in(writable_db)
    session = Session("C0021", writable_db)
    for args in [("P019", 1, "changed_mind"), ("P010", 2, "changed_mind")]:
        with pytest.raises(RefundRefusedError):
            refund_order(session, "O10040", *args, approved_amount_cents=ANY)
    with pytest.raises(NotFoundError):
        refund_order(
            Session("C0001", writable_db),
            "O10040",
            "P010",
            1,
            "changed_mind",
            approved_amount_cents=ANY,
        )
    assert refunds_in(writable_db) == before


def test_quote_matches_the_refund_and_writes_nothing(writable_db):
    session = Session("C0021", writable_db)
    before = refunds_in(writable_db)
    quote = quote_refund(session, "O10040", "P010", 1, "defective_or_wrong")
    assert refunds_in(writable_db) == before
    refund = refund_order(
        session, "O10040", "P010", 1, "defective_or_wrong", approved_amount_cents=quote.amount_cents
    )
    assert refund.model_dump(exclude={"refund_id", "created_at"}) == quote.model_dump()


# --- The approved amount: what a human approved is what is issued, or nothing ---------


def test_refund_is_refused_if_the_amount_is_not_the_one_approved(writable_db):
    before = refunds_in(writable_db)
    with pytest.raises(RefundRefusedError, match="now 5090 cents, not the 4500 cents"):
        refund_order(
            Session("C0021", writable_db),
            "O10040",
            "P010",
            1,
            "defective_or_wrong",
            approved_amount_cents=4500,
        )
    assert refunds_in(writable_db) == before


def test_amount_changed_between_approval_and_refund(writable_db):
    """Two approvals given on quotes from the same moment: the shipping fee is refunded
    only once, so the second refund is now lower than approved, and is not issued."""
    session = Session("C0021", writable_db)
    pillow = quote_refund(session, "O10040", "P010", 1, "defective_or_wrong")
    candle = quote_refund(session, "O10040", "P019", 1, "defective_or_wrong")
    assert (pillow.amount_cents, candle.amount_cents) == (5090, 3190)  # both with shipping

    refund_order(
        session,
        "O10040",
        "P010",
        1,
        "defective_or_wrong",
        approved_amount_cents=pillow.amount_cents,
    )
    with pytest.raises(RefundRefusedError, match="now 2600 cents, not the 3190 cents"):
        refund_order(
            session,
            "O10040",
            "P019",
            1,
            "defective_or_wrong",
            approved_amount_cents=candle.amount_cents,
        )
    assert [r[2] for r in refunds_in(writable_db) if r[1] == "O10040"] == ["P010"]
