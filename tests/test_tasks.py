"""The reference tasks must describe the world as it is, or they test nothing."""

from __future__ import annotations

import sqlite3

import pytest

from reliable_tool_agent.eval_runner import load_cases, load_tasks
from reliable_tool_agent.tools import Session, quote_refund

TASKS = load_tasks()


def test_ids_are_unique_across_all_sets():
    ids = [t["id"] for t in TASKS] + [c["id"] for c in load_cases()]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["id"])
def test_task_is_well_formed(task, shop_db):
    assert task["turns"] and all(isinstance(t, str) and t for t in task["turns"])
    assert task["expected"]
    conn = sqlite3.connect(shop_db)
    assert conn.execute("SELECT 1 FROM customers WHERE id = ?", (task["customer"],)).fetchone()
    approvals = [d for d in task.get("reviewer", []) if d["approved"]]
    # One approval per expected refund: an approval with nothing to write, or the reverse,
    # means the task contradicts itself.
    assert len(approvals) == len(task["expected_refunds"])


@pytest.mark.parametrize("task", [t for t in TASKS if t["expected_refunds"]], ids=lambda t: t["id"])
def test_hand_computed_amounts_agree_with_the_code(task, shop_db):
    # The amounts were computed by hand from the policies. If the code disagrees, one of
    # the two is wrong, and the evaluation would reward or punish the wrong behaviour.
    session = Session(task["customer"], shop_db)
    for row in task["expected_refunds"]:
        quote = quote_refund(
            session, row["order_id"], row["product_id"], row["quantity"], row["condition"]
        )
        assert quote.amount_cents == row["amount_cents"]
