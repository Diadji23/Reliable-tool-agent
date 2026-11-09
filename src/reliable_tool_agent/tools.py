"""Tools the agent can call. Each one is narrow: it does one thing, with typed output.

Privilege separation: the customer identity comes from the `Session`, set by the
application after authentication, never from the model. No tool takes a `customer_id`
argument, so the model cannot ask for another customer's data, whatever a message says.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from reliable_tool_agent.db import DEFAULT_DB_PATH, connect_for_refunds, connect_readonly
from reliable_tool_agent.world import WORLD_TODAY

POLICIES_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"

OrderStatus = Literal["processing", "shipped", "delivered", "cancelled"]


@dataclass(frozen=True)
class Session:
    """Who the agent is serving. Built by the application, never by the model."""

    customer_id: str
    db_path: Path = DEFAULT_DB_PATH
    policies_dir: Path = POLICIES_DIR
    today: date = WORLD_TODAY  # the world's fixed date; tests move it to check boundaries


class NotFoundError(LookupError):
    """The requested object does not exist for this customer."""


class OrderItem(BaseModel):
    product_id: str
    product_name: str
    quantity: int
    unit_price_cents: int
    final_sale: bool


class Order(BaseModel):
    id: str
    status: OrderStatus
    ordered_at: date
    delivered_at: date | None
    shipping_cents: int
    total_cents: int
    items: list[OrderItem]


def get_order(session: Session, order_id: str) -> Order:
    """Return one of the session customer's orders.

    An order that belongs to someone else gets the same answer as a missing one:
    saying "not yours" would reveal that the order exists.
    """
    with connect_readonly(session.db_path) as conn:
        row = conn.execute(
            "SELECT * FROM orders WHERE id = ? AND customer_id = ?",
            (order_id, session.customer_id),
        ).fetchone()
        if row is None:
            raise NotFoundError(f"No order {order_id!r} found for this customer.")
        items = conn.execute(
            """
            SELECT i.product_id, p.name AS product_name, i.quantity,
                   i.unit_price_cents, p.final_sale
            FROM order_items i JOIN products p ON p.id = i.product_id
            WHERE i.order_id = ? ORDER BY i.product_id
            """,
            (order_id,),
        ).fetchall()

    # Pydantic validates and converts here: ISO strings -> date, 0/1 -> bool,
    # and an unexpected status would raise instead of passing silently.
    return Order(
        **{k: row[k] for k in Order.model_fields if k != "items"},
        items=[OrderItem(**dict(item)) for item in items],
    )


class OrderSummary(BaseModel):
    id: str
    status: OrderStatus
    ordered_at: date
    total_cents: int
    n_items: int


def list_my_orders(session: Session, status: OrderStatus | None = None) -> list[OrderSummary]:
    """The session customer's orders, newest first, optionally filtered by status."""
    with connect_readonly(session.db_path) as conn:
        rows = conn.execute(
            """
            SELECT o.id, o.status, o.ordered_at, o.total_cents, SUM(i.quantity) AS n_items
            FROM orders o JOIN order_items i ON i.order_id = o.id
            WHERE o.customer_id = ? AND (? IS NULL OR o.status = ?)
            GROUP BY o.id ORDER BY o.ordered_at DESC, o.id DESC
            """,
            (session.customer_id, status, status),
        ).fetchall()
    return [OrderSummary(**dict(r)) for r in rows]


class PolicyRule(BaseModel):
    policy: str  # file stem, e.g. "returns"
    rule: int  # rule number inside the policy (1, 2, ...); 0 = text before the first rule
    text: str


# Words too common to say anything about relevance.
_STOPWORDS = set(
    "a an and are be by can do does for from how i if in is it its my of on or the to "
    "what when where which who why with you your".split()
)


def _stem(word: str) -> str:
    """Crude stemming: returns/returned/returning -> return, cancelled -> cancel.

    Only needs to map the forms used in our short policies to the same key; a real
    stemmer (Porter, Snowball) would be overkill here.
    """
    for suffix in ("ing", "ed", "es", "s", "e"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2]:  # cancell -> cancel, shipp -> ship
        word = word[:-1]
    return word


def _words(text: str) -> set[str]:
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    return {_stem(w) for w in cleaned.split() if w not in _STOPWORDS}


def _split_rules(policy: str, markdown: str) -> list[PolicyRule]:
    """One PolicyRule per numbered item ("1. ..."), continuation lines included."""
    rules: list[PolicyRule] = []
    number, lines = 0, []
    for line in markdown.splitlines():
        if line.startswith("#"):
            continue
        head = line.split(".", 1)[0]
        if head.isdigit() and line[len(head) : len(head) + 2] == ". ":
            if lines:
                rules.append(PolicyRule(policy=policy, rule=number, text=" ".join(lines)))
            number, lines = int(head), [line[len(head) + 2 :].strip()]
        elif line.strip():
            lines.append(line.strip())
    if lines:
        rules.append(PolicyRule(policy=policy, rule=number, text=" ".join(lines)))
    return rules


def search_policies(session: Session, query: str, k: int = 3) -> list[PolicyRule]:
    """The k policy rules sharing the most words with `query` (keyword search).

    Deliberately simple: overlap of crudely stemmed words, no synonyms ("send back" does
    not match "return"). Rules with no word in common are never returned, so an unrelated query
    gives an empty list instead of irrelevant text.
    """
    query_words = _words(query)
    scored = []
    for path in sorted(session.policies_dir.glob("*.md")):
        for rule in _split_rules(path.stem, path.read_text(encoding="utf-8")):
            # The policy name counts as a word of the rule ("refund" matches refunds.md),
            # and breaks ties: on equal overlap, rules from the policy the query names
            # come first instead of whichever file sorts first alphabetically.
            title = query_words & _words(rule.policy)
            score = len(query_words & _words(rule.text) | title)
            if score:
                scored.append((-score, -len(title), rule.policy, rule.rule, rule))
    return [rule for *_, rule in sorted(scored)[:k]]


# --- Return eligibility: the returns policy, applied in code -----------------------------

RETURN_WINDOW_DAYS = 30  # returns.md rule 1: day 0 = delivery day, day 30 accepted


class ItemEligibility(BaseModel):
    product_id: str
    product_name: str
    quantity: int
    final_sale: bool
    returnable: bool  # can be returned for any reason today
    returnable_if_defective_or_wrong: bool  # can be returned if defective, damaged or wrong
    reason: str


class ReturnEligibility(BaseModel):
    order_id: str
    status: OrderStatus
    today: date
    delivered_at: date | None
    days_since_delivery: int | None
    return_window_ends: date | None  # last day a return is accepted
    items: list[ItemEligibility]


def check_return_eligibility(session: Session, order_id: str) -> ReturnEligibility:
    """Apply the returns policy (rules 1-4) to every item of an order.

    Date arithmetic and rule logic are done here, not by the model: a small model asked
    "delivered on 5 October, today is 15 October: within 30 days?" answered no, and when
    challenged, produced a confident but wrong day count. The model now only reads
    `returnable` and `reason`.
    """
    order = get_order(session, order_id)
    days = window_ends = None
    if order.status == "delivered" and order.delivered_at is not None:
        days = (session.today - order.delivered_at).days
        window_ends = order.delivered_at + timedelta(days=RETURN_WINDOW_DAYS)

    items = []
    for item in order.items:
        if order.status != "delivered":
            ok = ok_if_faulty = False
            reason = f"Only delivered orders can be returned; this order is {order.status}."
        elif days > RETURN_WINDOW_DAYS:
            ok = ok_if_faulty = False
            reason = f"The 30-day return window ended on {window_ends.isoformat()}."
        elif item.final_sale:
            ok, ok_if_faulty = False, True
            reason = (
                "Final sale: returnable only if defective, damaged or the wrong item, "
                f"until {window_ends.isoformat()}."
            )
        else:
            ok = ok_if_faulty = True
            reason = f"Returnable until {window_ends.isoformat()} (day {days} of 30)."
        items.append(
            ItemEligibility(
                product_id=item.product_id,
                product_name=item.product_name,
                quantity=item.quantity,
                final_sale=item.final_sale,
                returnable=ok,
                returnable_if_defective_or_wrong=ok_if_faulty,
                reason=reason,
            )
        )
    return ReturnEligibility(
        order_id=order.id,
        status=order.status,
        today=session.today,
        delivered_at=order.delivered_at,
        days_since_delivery=days,
        return_window_ends=window_ends,
        items=items,
    )


RefundCondition = Literal["changed_mind", "defective_or_wrong"]


class RefundRefusedError(ValueError):
    """The refund breaks a policy rule. The message says which one, for the model to relay."""


class RefundQuote(BaseModel):
    """What a refund would be: computed by code, shown to the human who approves it."""

    order_id: str
    product_id: str
    product_name: str
    quantity: int
    condition: RefundCondition
    item_cents: int  # unit price x quantity (returns.md rule 5)
    shipping_cents: int  # original shipping fee, only if defective or wrong, only once
    amount_cents: int  # item_cents + shipping_cents
    already_refunded_cents: int  # on this order, before this refund


class Refund(RefundQuote):
    """A refund that was issued."""

    refund_id: int
    created_at: date


def _quote(
    conn: sqlite3.Connection,
    session: Session,
    order_id: str,
    product_id: str,
    quantity: int,
    condition: RefundCondition,
) -> RefundQuote:
    """Apply the returns and refunds policies; refunds already issued are read from `conn`."""
    if quantity < 1:
        raise RefundRefusedError("The quantity to refund must be at least 1.")
    order = get_order(session, order_id)  # NotFoundError for someone else's order
    eligibility = check_return_eligibility(session, order_id)
    item = next((i for i in order.items if i.product_id == product_id), None)
    if item is None:
        raise NotFoundError(f"Order {order_id} has no product {product_id!r}.")
    status = next(i for i in eligibility.items if i.product_id == product_id)

    if condition == "changed_mind" and not status.returnable:
        raise RefundRefusedError(f"{item.product_name} cannot be refunded: {status.reason}")
    if condition == "defective_or_wrong" and not status.returnable_if_defective_or_wrong:
        raise RefundRefusedError(f"{item.product_name} cannot be refunded: {status.reason}")

    (item_refunded,) = conn.execute(
        "SELECT COALESCE(SUM(quantity), 0) FROM refunds WHERE order_id = ? AND product_id = ?",
        (order_id, product_id),
    ).fetchone()
    left = item.quantity - item_refunded
    if quantity > left:
        raise RefundRefusedError(
            f"{item.product_name}: {item.quantity} ordered, {item_refunded} already refunded, "
            f"so at most {left} can be refunded."
        )

    already, shipping_refunded = conn.execute(
        "SELECT COALESCE(SUM(amount_cents), 0), COALESCE(SUM(shipping_cents), 0) "
        "FROM refunds WHERE order_id = ?",
        (order_id,),
    ).fetchone()
    item_cents = item.unit_price_cents * quantity
    shipping = (
        order.shipping_cents if condition == "defective_or_wrong" and not shipping_refunded else 0
    )
    amount = item_cents + shipping
    if already + amount > order.total_cents:  # refunds.md rule 2, checked whatever the cause
        raise RefundRefusedError(
            f"Refunds on order {order_id} cannot exceed the {order.total_cents} cents paid "
            f"({already} cents already refunded)."
        )
    return RefundQuote(
        order_id=order_id,
        product_id=product_id,
        product_name=item.product_name,
        quantity=quantity,
        condition=condition,
        item_cents=item_cents,
        shipping_cents=shipping,
        amount_cents=amount,
        already_refunded_cents=already,
    )


def quote_refund(
    session: Session,
    order_id: str,
    product_id: str,
    quantity: int,
    condition: RefundCondition,
) -> RefundQuote:
    """Compute a refund without issuing it. Raises RefundRefusedError if a rule forbids it."""
    with connect_readonly(session.db_path) as conn:
        return _quote(conn, session, order_id, product_id, quantity, condition)


def refund_order(
    session: Session,
    order_id: str,
    product_id: str,
    quantity: int,
    condition: RefundCondition,
    *,
    approved_amount_cents: int,
) -> Refund:
    """Issue a refund for returned units of one product of an order.

    The model chooses what is returned and why; the code decides whether it is allowed and
    computes the amount. The checks and the insert share one `BEGIN IMMEDIATE` transaction,
    so two refunds issued at the same time cannot both pass the "already refunded" checks.
    The stored reason is built from the condition, never copied from model text.

    `approved_amount_cents` is the amount a human approved. Time passes between the quote
    they saw and this call (other refunds, the return window), so the amount is computed
    again here and the refund is issued only if it is still exactly that amount: the
    comparison and the insert are in the same transaction, with nothing in between.
    """
    conn = connect_for_refunds(session.db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            quote = _quote(conn, session, order_id, product_id, quantity, condition)
            if quote.amount_cents != approved_amount_cents:
                raise RefundRefusedError(
                    f"The refund is now {quote.amount_cents} cents, not the "
                    f"{approved_amount_cents} cents that were approved (refunds or dates "
                    "changed since). Nothing was issued. If the customer still wants it, "
                    "call refund_order again: the reviewer will see the new amount."
                )
            reason = f"{condition}: {quantity} x {product_id}"
            cursor = conn.execute(
                "INSERT INTO refunds "
                "(order_id, product_id, quantity, shipping_cents, amount_cents, reason, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    order_id,
                    product_id,
                    quantity,
                    quote.shipping_cents,
                    quote.amount_cents,
                    reason,
                    session.today.isoformat(),
                ),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    return Refund(**quote.model_dump(), refund_id=cursor.lastrowid, created_at=session.today)
