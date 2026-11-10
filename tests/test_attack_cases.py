"""The attack cases must describe the world as it is, or they test nothing."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

CASES_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "attack_cases.json"
CASES = json.loads(CASES_PATH.read_text())["cases"]
RISKS = {f"R{i}" for i in range(1, 8)}  # docs/threat-model.md


def test_ids_are_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_case_is_well_formed(case, shop_db):
    assert case["turns"] and all(isinstance(t, str) and t for t in case["turns"])
    assert set(case["risk"]) <= RISKS
    assert case["approval"] in ("never", "allowed")
    assert ("approval_only_for" in case) == (case["approval"] == "allowed")
    conn = sqlite3.connect(shop_db)
    assert conn.execute("SELECT 1 FROM customers WHERE id = ?", (case["customer"],)).fetchone()


@pytest.mark.parametrize(
    "case", [c for c in CASES if c["approval"] == "allowed"], ids=lambda c: c["id"]
)
def test_allowed_refunds_name_an_item_of_the_customers_own_order(case, shop_db):
    conn = sqlite3.connect(shop_db)
    for allowed in case["approval_only_for"]:
        assert allowed["condition"] in ("changed_mind", "defective_or_wrong")
        row = conn.execute(
            """
            SELECT 1 FROM orders o JOIN order_items i ON i.order_id = o.id
            WHERE o.id = ? AND o.customer_id = ? AND i.product_id = ?
            """,
            (allowed["order_id"], case["customer"], allowed["product_id"]),
        ).fetchone()
        assert row, allowed
