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


@dataclass(frozen=True)
class Session:
    """Who the agent is serving. Built by the application, never by the model."""

    customer_id: str
    db_path: Path = DEFAULT_DB_PATH


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
    status: Literal["processing", "shipped", "delivered", "cancelled"]
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
