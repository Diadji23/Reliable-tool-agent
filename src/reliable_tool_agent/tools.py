"""Tools the agent can call. Each one is narrow: it does one thing, with typed output.

Privilege separation: the customer identity comes from the `Session`, set by the
application after authentication, never from the model. No tool takes a `customer_id`
argument, so the model cannot ask for another customer's data, whatever a message says.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from reliable_tool_agent.db import DEFAULT_DB_PATH, connect_readonly

POLICIES_DIR = Path(__file__).resolve().parents[2] / "data" / "policies"

OrderStatus = Literal["processing", "shipped", "delivered", "cancelled"]


@dataclass(frozen=True)
class Session:
    """Who the agent is serving. Built by the application, never by the model."""

    customer_id: str
    db_path: Path = DEFAULT_DB_PATH
    policies_dir: Path = POLICIES_DIR


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
