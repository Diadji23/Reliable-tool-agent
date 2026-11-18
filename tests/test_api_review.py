"""The reviewer's decision: POST /approvals/{conversation_id}."""

from __future__ import annotations

from test_api import REFUND_TURNS, api, auth, paused_on_a_refund, refunds_in, start

from reliable_tool_agent.agent_loop import AssistantTurn


def test_approved_refund_is_written_and_the_customer_gets_the_answer(writable_db):
    client = api(REFUND_TURNS, writable_db)
    conversation = paused_on_a_refund(client)
    before = refunds_in(writable_db)

    response = client.post(
        f"/approvals/{conversation}",
        json={"decisions": {"r1": {"approved": True, "amount_cents": 5090}}},
        headers=auth("tok-rev"),
    )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": conversation,
        "status": "answered",
        "answer": "Your refund of 50.90 EUR has been issued.",
    }
    assert refunds_in(writable_db) == before + [("O10040", "P010", 5090)]
    assert client.get("/approvals", headers=auth("tok-rev")).json() == []  # nothing left


def test_rejected_refund_is_not_written(writable_db):
    client = api(REFUND_TURNS[:1] + [AssistantTurn("Sorry, it was declined.")], writable_db)
    conversation = paused_on_a_refund(client)
    before = refunds_in(writable_db)

    response = client.post(
        f"/approvals/{conversation}",
        json={"decisions": {"r1": {"approved": False, "reason": "need a photo"}}},
        headers=auth("tok-rev"),
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Sorry, it was declined."
    assert refunds_in(writable_db) == before


def test_the_customer_can_write_again_once_the_reviewer_has_decided(writable_db):
    client = api(REFUND_TURNS + [AssistantTurn("You're welcome.")], writable_db)
    conversation = paused_on_a_refund(client)
    client.post(
        f"/approvals/{conversation}",
        json={"decisions": {"r1": {"approved": True, "amount_cents": 5090}}},
        headers=auth("tok-rev"),
    )
    response = client.post(
        f"/conversations/{conversation}/messages", json={"text": "Thanks"}, headers=auth("tok-c21")
    )
    assert response.status_code == 200


def test_only_a_reviewer_may_decide(writable_db):
    client = api(REFUND_TURNS, writable_db)
    conversation = paused_on_a_refund(client)
    response = client.post(
        f"/approvals/{conversation}",
        json={"decisions": {"r1": {"approved": True, "amount_cents": 5090}}},
        headers=auth("tok-c21"),  # the customer approving their own refund
    )
    assert response.status_code == 403


def test_unknown_conversation_is_404_and_nothing_pending_is_409(writable_db):
    client = api([], writable_db)
    decisions = {"decisions": {"r1": {"approved": False}}}
    unknown = client.post("/approvals/c-unknown", json=decisions, headers=auth("tok-rev"))
    assert unknown.status_code == 404
    idle = client.post(f"/approvals/{start(client)}", json=decisions, headers=auth("tok-rev"))
    assert idle.status_code == 409


def test_an_approval_without_an_amount_is_422(writable_db):
    client = api(REFUND_TURNS, writable_db)
    conversation = paused_on_a_refund(client)
    response = client.post(
        f"/approvals/{conversation}",
        json={"decisions": {"r1": {"approved": True}}},
        headers=auth("tok-rev"),
    )
    assert response.status_code == 422
