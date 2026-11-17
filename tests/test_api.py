"""The HTTP API, with a scripted model and the real MCP server (no network, no model)."""

from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from reliable_tool_agent.agent_loop import AssistantTurn, ToolCall
from reliable_tool_agent.api import Caller, create_app, parse_tokens
from reliable_tool_agent.fake_model import ScriptedModel

TOKENS = {
    "tok-c21": Caller("customer", "C0021"),
    "tok-c09": Caller("customer", "C0009"),
    "tok-rev": Caller("reviewer"),
}
REFUND = {
    "order_id": "O10040",
    "product_id": "P010",
    "quantity": 1,
    "condition": "defective_or_wrong",
}


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def api(model_turns, db) -> TestClient:
    return TestClient(create_app(ScriptedModel(model_turns), TOKENS, db))


def refunds_in(db) -> list[tuple]:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT order_id, product_id, amount_cents FROM refunds").fetchall()


def start(client: TestClient, token: str = "tok-c21") -> str:
    response = client.post("/conversations", headers=auth(token))
    assert response.status_code == 201
    return response.json()["conversation_id"]


def paused_on_a_refund(client: TestClient) -> str:
    """A conversation of C0021 whose refund waits for the reviewer."""
    conversation = start(client)
    response = client.post(
        f"/conversations/{conversation}/messages",
        json={"text": "My pillow from O10040 arrived torn, please refund it."},
        headers=auth("tok-c21"),
    )
    assert response.json() == {
        "conversation_id": conversation,
        "status": "awaiting_approval",
        "answer": None,
    }
    return conversation


REFUND_TURNS = [
    AssistantTurn(None, [ToolCall("r1", "refund_order", json.dumps(REFUND))]),
    AssistantTurn("Your refund of 50.90 EUR has been issued."),
]


# --- who is calling --------------------------------------------------------------------


@pytest.mark.parametrize("headers", [{}, auth("not-a-token"), {"Authorization": "tok-c21"}])
def test_no_valid_token_is_401(writable_db, headers):
    response = api([], writable_db).post("/conversations", headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_each_role_is_refused_the_other_roles_routes(writable_db):
    client = api([], writable_db)
    assert client.post("/conversations", headers=auth("tok-rev")).status_code == 403
    assert client.get("/approvals", headers=auth("tok-c21")).status_code == 403


def test_tokens_are_read_from_one_line():
    assert parse_tokens("a=customer:C0021, b=reviewer") == {
        "a": Caller("customer", "C0021"),
        "b": Caller("reviewer"),
    }
    for bad in ["a=customer", "a=reviewer:C0021", "a=admin", "=reviewer"]:
        with pytest.raises(ValueError):
            parse_tokens(bad)


# --- customers -------------------------------------------------------------------------


def test_customer_gets_the_agents_answer(writable_db):
    client = api(
        [
            AssistantTurn(None, [ToolCall("g1", "get_order", '{"order_id": "O10040"}')]),
            AssistantTurn("Delivered on 2026-10-13."),
        ],
        writable_db,
    )
    conversation = start(client)
    response = client.post(
        f"/conversations/{conversation}/messages",
        json={"text": "Where is O10040?"},
        headers=auth("tok-c21"),
    )
    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": conversation,
        "status": "answered",
        "answer": "Delivered on 2026-10-13.",
    }


def test_the_customer_id_comes_from_the_token_not_from_the_body(writable_db):
    model_turns = [
        AssistantTurn(None, [ToolCall("g1", "get_order", '{"order_id": "O10040"}')]),
        AssistantTurn("..."),
    ]
    client = api(model_turns, writable_db)
    conversation = start(client, "tok-c09")  # C0009 does not own O10040
    client.post(
        f"/conversations/{conversation}/messages",
        json={"text": "Where is O10040?", "customer_id": "C0021"},  # an extra field: ignored
        headers=auth("tok-c09"),
    )
    tool_result = client.app.state.service.model.seen[1][-1]["content"]
    assert "No order 'O10040' found for this customer" in tool_result


def test_another_customers_conversation_looks_missing(writable_db):
    client = api([], writable_db)
    conversation = start(client, "tok-c21")
    for path in [f"/conversations/{conversation}/messages", "/conversations/c-unknown/messages"]:
        response = client.post(path, json={"text": "Hello"}, headers=auth("tok-c09"))
        assert response.status_code == 404


@pytest.mark.parametrize("body", [{}, {"text": ""}, {"text": "x" * 2001}, {"message": "hi"}])
def test_an_invalid_message_is_422_before_the_agent_runs(writable_db, body):
    client = api([], writable_db)  # no model turn: the agent must not be called
    conversation = start(client)
    response = client.post(
        f"/conversations/{conversation}/messages", json=body, headers=auth("tok-c21")
    )
    assert response.status_code == 422


def test_a_turn_without_answer_says_so(writable_db):
    calls = [AssistantTurn(None, [ToolCall(f"g{i}", "list_my_orders", "{}")]) for i in range(2)]
    client = TestClient(create_app(ScriptedModel(calls), TOKENS, writable_db, max_steps=2))
    conversation = start(client)
    response = client.post(
        f"/conversations/{conversation}/messages", json={"text": "Hi"}, headers=auth("tok-c21")
    )
    assert response.json()["status"] == "no_answer"


# --- a refund waiting for the reviewer -------------------------------------------------


def test_reviewer_sees_the_pending_refund_with_the_quote_and_the_customers_words(writable_db):
    client = api(REFUND_TURNS, writable_db)
    conversation = paused_on_a_refund(client)

    response = client.get("/approvals", headers=auth("tok-rev"))
    assert response.status_code == 200
    (pending,) = response.json()
    assert pending["conversation_id"] == conversation
    assert pending["customer_id"] == "C0021"
    (request,) = pending["requests"]
    assert request["tool"] == "refund_order" and request["arguments"] == REFUND
    assert request["preview"]["amount_cents"] == 5090  # computed by the code
    assert request["customer_said"] == ["My pillow from O10040 arrived torn, please refund it."]


def test_customer_cannot_write_while_a_refund_awaits_the_reviewer(writable_db):
    client = api(REFUND_TURNS, writable_db)
    conversation = paused_on_a_refund(client)
    before = refunds_in(writable_db)
    response = client.post(
        f"/conversations/{conversation}/messages", json={"text": "Hello?"}, headers=auth("tok-c21")
    )
    assert response.status_code == 409
    assert refunds_in(writable_db) == before
