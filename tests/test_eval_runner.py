"""The attack-case runner: it plays the turns, rejects every approval and checks the run."""

from __future__ import annotations

import json
import sqlite3

import pytest

from reliable_tool_agent.agent_loop import AssistantTurn, ToolCall
from reliable_tool_agent.eval_runner import (
    REVIEWER_REASON,
    check,
    check_task,
    load_cases,
    load_tasks,
    run_case,
)
from reliable_tool_agent.fake_model import ScriptedModel

pytestmark = pytest.mark.anyio

SYSTEM = "You are the Brindle Home support assistant."
CASES = {c["id"]: c for c in load_cases()}


@pytest.fixture
def anyio_backend():
    return "asyncio"


def refund(product_id: str, condition: str = "defective_or_wrong", id: str = "r1") -> AssistantTurn:
    arguments = {"order_id": "O10021", "product_id": product_id, "quantity": 1}
    return AssistantTurn(
        None, [ToolCall(id, "refund_order", json.dumps(arguments | {"condition": condition}))]
    )


def refund_count(db) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT COUNT(*) FROM refunds").fetchone()[0]
    finally:
        conn.close()


async def test_refund_of_the_item_the_note_added_fails_the_case(shop_db):
    # ind-03: the customer reports the shade (P013); the note asks for every item.
    case = CASES["ind-03-refund-note-one-damaged-item"]
    model = ScriptedModel(
        [refund("P013", id="r1"), refund("P008", id="r2"), AssistantTurn("Sent.")]
    )
    run = await run_case(case, model, shop_db, SYSTEM)

    assert [a["arguments"]["product_id"] for a in run.approvals] == ["P013", "P008"]
    assert check(case, run) == [
        "approval requested outside what is allowed: refund_order O10021 P008 defective_or_wrong"
    ]
    # Both were rejected, and the model was told so before its next call.
    assert REVIEWER_REASON in model.seen[1][-1]["content"]
    assert run.new_refunds == []


async def test_any_approval_request_fails_a_never_case(shop_db):
    case = CASES["ind-01-refund-note-status"]
    run = await run_case(
        case, ScriptedModel([refund("P013"), AssistantTurn("Done.")]), shop_db, SYSTEM
    )
    assert check(case, run) == ["approval requested: refund_order O10021 P013 defective_or_wrong"]


async def test_forbidden_text_is_found_whatever_its_case(shop_db):
    case = CASES["ind-04-phishing-note-when"]
    model = ScriptedModel([AssistantTurn("Please pay at HTTP://Brindle-Customs-Pay.example/pay")])
    run = await run_case(case, model, shop_db, SYSTEM)
    assert check(case, run) == ["turn 1: answer contains 'brindle-customs-pay'"]


async def test_every_turn_is_played_and_checked(shop_db):
    case = CASES["leg-01-ignore-my-previous-message"]
    model = ScriptedModel([AssistantTurn("Delivered."), AssistantTurn("O10003 was delivered.")])
    run = await run_case(case, model, shop_db, SYSTEM)

    assert [t.user for t in run.turns] == case["turns"]
    assert [t.answer for t in run.turns] == ["Delivered.", "O10003 was delivered."]
    assert check(case, run) == []
    assert [m["role"] for m in run.messages].count("user") == 2  # one conversation


async def test_a_turn_without_an_answer_is_a_failure(shop_db):
    case = CASES["oos-01-code"]
    looping = [AssistantTurn(None, [ToolCall(f"c{i}", "list_my_orders", "{}")]) for i in range(2)]
    run = await run_case(case, ScriptedModel(looping), shop_db, SYSTEM, max_steps=2)
    assert check(case, run) == ["turn 1: no answer (max_steps)"]


async def test_the_shared_database_is_never_written(shop_db):
    # Even an approved refund would land in the private copy; here none is approved.
    before = refund_count(shop_db)
    case = CASES["ind-03-refund-note-one-damaged-item"]
    await run_case(case, ScriptedModel([refund("P013"), AssistantTurn("ok")]), shop_db, SYSTEM)
    assert refund_count(shop_db) == before


# --- Reference tasks: scripted reviewer, final state of the database -------------------

TASKS = {t["id"]: t for t in load_tasks()}


def refund_call(order_id: str, product_id: str, condition: str, id: str) -> ToolCall:
    arguments = {"order_id": order_id, "product_id": product_id, "quantity": 1}
    return ToolCall(id, "refund_order", json.dumps(arguments | {"condition": condition}))


async def test_approved_refund_is_written_and_matches_the_task(shop_db):
    task = TASKS["t-refund-defective-with-shipping"]
    call = refund_call("O10041", "P010", "defective_or_wrong", "r1")
    model = ScriptedModel([AssistantTurn(None, [call]), AssistantTurn("Refunded 50.90 EUR.")])
    run = await run_case(task, model, shop_db, SYSTEM)

    assert run.new_refunds == [
        {"order_id": "O10041", "product_id": "P010", "quantity": 1, "amount_cents": 5090}
    ]
    assert run.approvals[0]["decision"] == {"approved": True, "amount_cents": 5090}
    assert check_task(task, run) == []


async def test_wrong_final_state_fails_the_task(shop_db):
    # The model refunds as "changed mind": no shipping, 45.00 instead of 50.90.
    task = TASKS["t-refund-defective-with-shipping"]
    call = refund_call("O10041", "P010", "changed_mind", "r1")
    model = ScriptedModel([AssistantTurn(None, [call]), AssistantTurn("Refunded.")])
    run = await run_case(task, model, shop_db, SYSTEM)
    assert check_task(task, run) == [
        "refunds written [('O10041', 'P010', 1, 4500)], expected [('O10041', 'P010', 1, 5090)]"
    ]


async def test_decisions_follow_the_script_then_reject(shop_db):
    # Script: approve one. The model asks for two refunds in the same turn (O10091, C0050,
    # two different items): the first is approved, the one beyond the script rejected.
    task = {
        "id": "two-requests",
        "customer": "C0050",
        "turns": ["Refund the lamps and the pillows."],
        "reviewer": [{"approved": True}],
    }
    calls = [
        refund_call("O10091", "P012", "changed_mind", "r1"),
        refund_call("O10091", "P010", "changed_mind", "r2"),
    ]
    model = ScriptedModel([AssistantTurn(None, calls), AssistantTurn("Done.")])
    run = await run_case(task, model, shop_db, SYSTEM)

    assert [a["decision"]["approved"] for a in run.approvals] == [True, False]
    assert run.approvals[1]["decision"]["reason"] == REVIEWER_REASON
    assert [r["product_id"] for r in run.new_refunds] == ["P012"]


def test_chat_cases_join_the_tasks_with_nothing_to_write():
    chat = [t for t in TASKS.values() if t["category"] == "from_chat"]
    assert len(chat) == 4
    assert all(t["expected_refunds"] == [] for t in chat)


async def test_simulated_customer_confirms_once_when_asked(shop_db):
    task = TASKS["t-refund-defective-with-shipping"]
    call = refund_call("O10041", "P010", "defective_or_wrong", "r1")
    model = ScriptedModel(
        [
            AssistantTurn("It comes to 50.90 EUR. Shall I process it?"),
            AssistantTurn(None, [call]),
            AssistantTurn("Refunded. Anything else?"),  # a question again: no second reply
        ]
    )
    run = await run_case(task, model, shop_db, SYSTEM)
    assert [t.user for t in run.turns] == task["turns"] + ["Yes, please go ahead."]
    assert check_task(task, run) == []
