"""Conversation memory of the LangGraph agent: threads kept by a checkpointer."""

from __future__ import annotations

import json
import sqlite3

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from mcp import Client

from reliable_tool_agent.agent_graph import GraphAgent
from reliable_tool_agent.agent_loop import AssistantTurn, McpToolbox, ToolCall
from reliable_tool_agent.fake_model import ScriptedModel
from reliable_tool_agent.mcp_server import build_server
from reliable_tool_agent.tools import Session

pytestmark = pytest.mark.anyio

SYSTEM = "You are the Brindle Home support assistant."


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def owner_and_order(shop_db):
    conn = sqlite3.connect(shop_db)
    row = conn.execute("SELECT customer_id, id FROM orders ORDER BY id LIMIT 1").fetchone()
    conn.close()
    return row


def get_order(order_id: str, id: str = "call_1") -> AssistantTurn:
    return AssistantTurn(None, [ToolCall(id, "get_order", json.dumps({"order_id": order_id}))])


def roles(messages) -> list[str]:
    return [m["role"] for m in messages]


async def test_second_message_sees_the_first_exchange(shop_db, owner_and_order):
    owner, order_id = owner_and_order
    model = ScriptedModel(
        [get_order(order_id), AssistantTurn("Delivered."), AssistantTurn("Yes, you can.")]
    )
    async with Client(build_server(Session(owner, shop_db))) as client:
        agent = GraphAgent(model, McpToolbox(client), SYSTEM, InMemorySaver())
        await agent.ask("t1", f"Where is {order_id}?")
        result = await agent.ask("t1", "Can I return it?")

    assert result.answer == "Yes, you can."
    # On the second turn the model saw the whole first exchange, tool result included,
    # with a single system prompt at the top.
    assert roles(model.seen[2]) == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert model.seen[2][-1]["content"] == "Can I return it?"
    assert roles(result.messages).count("system") == 1


async def test_threads_are_isolated(shop_db, owner_and_order):
    owner, order_id = owner_and_order
    model = ScriptedModel([get_order(order_id), AssistantTurn("Delivered."), AssistantTurn("Hi")])
    async with Client(build_server(Session(owner, shop_db))) as client:
        agent = GraphAgent(model, McpToolbox(client), SYSTEM, InMemorySaver())
        await agent.ask("thread-a", f"Where is {order_id}?")
        result = await agent.ask("thread-b", "Hello")

    assert roles(model.seen[2]) == ["system", "user"]  # nothing from thread-a
    assert order_id not in json.dumps(result.messages)


async def test_step_budget_applies_per_turn(shop_db, owner_and_order):
    """max_steps bounds one turn: a long conversation must not exhaust it."""
    owner, order_id = owner_and_order
    turn = [get_order(order_id), AssistantTurn("Delivered.")]  # 2 model calls per turn
    model = ScriptedModel(turn * 2)
    async with Client(build_server(Session(owner, shop_db))) as client:
        agent = GraphAgent(model, McpToolbox(client), SYSTEM, InMemorySaver(), max_steps=2)
        first = await agent.ask("t1", "Where is it?")
        second = await agent.ask("t1", "Where is it again?")

    assert (first.stop_reason, first.steps) == ("answer", 2)
    assert (second.stop_reason, second.steps) == ("answer", 2)


async def test_conversation_survives_a_new_agent_on_the_same_checkpointer(shop_db, owner_and_order):
    """What a web server does: a new request builds a new agent, the saver keeps the thread."""
    owner, _ = owner_and_order
    saver = InMemorySaver()
    async with Client(build_server(Session(owner, shop_db))) as client:
        await GraphAgent(
            ScriptedModel([AssistantTurn("Hello!")]), McpToolbox(client), SYSTEM, saver
        ).ask("t1", "Hi, I am Ada.")
        model = ScriptedModel([AssistantTurn("You are Ada.")])
        await GraphAgent(model, McpToolbox(client), SYSTEM, saver).ask("t1", "Who am I?")

    assert [m["content"] for m in model.seen[0]] == [
        SYSTEM,
        "Hi, I am Ada.",
        "Hello!",
        "Who am I?",
    ]


async def test_history_is_kept_whole_for_now(shop_db, owner_and_order):
    """Documents a known limit: nothing trims or summarises old messages yet.

    With an 8k-token context and ~1k-1.4k tokens per question, a local model holds about
    six turns. Context management comes in a later step and will change this test.
    """
    owner, _ = owner_and_order
    model = ScriptedModel([AssistantTurn(f"Answer {i}") for i in range(10)])
    async with Client(build_server(Session(owner, shop_db))) as client:
        agent = GraphAgent(model, McpToolbox(client), SYSTEM, InMemorySaver())
        for i in range(10):
            result = await agent.ask("t1", f"Question {i}")
    assert len(result.messages) == 1 + 2 * 10


async def test_history_returns_the_saved_conversation(shop_db, owner_and_order):
    owner, _ = owner_and_order
    async with Client(build_server(Session(owner, shop_db))) as client:
        agent = GraphAgent(
            ScriptedModel([AssistantTurn("Hello!")]), McpToolbox(client), SYSTEM, InMemorySaver()
        )
        assert await agent.history("t1") == []
        await agent.ask("t1", "Hi")
        assert roles(await agent.history("t1")) == ["system", "user", "assistant"]
        assert await agent.history("unknown") == []


# --- Human approval before a refund ----------------------------------------------------

REFUND = {
    "order_id": "O10040",
    "product_id": "P010",
    "quantity": 1,
    "condition": "defective_or_wrong",
}


def refund_call(id: str = "r1", **changes) -> ToolCall:
    return ToolCall(id, "refund_order", json.dumps(REFUND | changes))


def refunds_in(db) -> list[tuple]:
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT order_id, product_id, amount_cents FROM refunds").fetchall()
    conn.close()
    return rows


class CountingToolbox(McpToolbox):
    """The real toolbox, recording every tool it runs."""

    def __init__(self, client):
        super().__init__(client)
        self.ran: list[str] = []

    async def run(self, name, raw_arguments):
        self.ran.append(name)
        return await super().run(name, raw_arguments)


async def test_refund_waits_for_the_reviewer_then_runs_when_approved(writable_db):
    model = ScriptedModel([AssistantTurn(None, [refund_call()]), AssistantTurn("Refunded.")])
    before = refunds_in(writable_db)
    async with Client(build_server(Session("C0021", writable_db))) as client:
        agent = GraphAgent(model, McpToolbox(client), SYSTEM, InMemorySaver())
        paused = await agent.ask("t1", "My lamp arrived broken, refund it please.")

        assert paused.stop_reason == "awaiting_approval"
        (request,) = paused.pending
        assert request["tool"] == "refund_order" and request["arguments"] == REFUND
        assert request["preview"]["amount_cents"] == 5090  # computed by the code
        assert refunds_in(writable_db) == before  # nothing written while waiting
        assert len(model.seen) == 1
        with pytest.raises(RuntimeError, match="awaits a reviewer"):
            await agent.ask("t1", "Hello?")

        done = await agent.resume("t1", {"r1": {"approved": True}})

    assert done.stop_reason == "answer" and done.answer == "Refunded."
    assert refunds_in(writable_db) == before + [("O10040", "P010", 5090)]
    assert json.loads(model.seen[1][-1]["content"])["amount_cents"] == 5090


async def test_rejected_refund_is_not_run_and_the_model_gets_the_reason(writable_db):
    model = ScriptedModel([AssistantTurn(None, [refund_call()]), AssistantTurn("Sorry.")])
    before = refunds_in(writable_db)
    async with Client(build_server(Session("C0021", writable_db))) as client:
        agent = GraphAgent(model, McpToolbox(client), SYSTEM, InMemorySaver())
        await agent.ask("t1", "Refund my lamp.")
        done = await agent.resume("t1", {"r1": {"approved": False, "reason": "photo needed"}})

    assert done.answer == "Sorry."
    assert refunds_in(writable_db) == before
    tool = model.seen[1][-1]
    assert "rejected" in tool["content"] and "photo needed" in tool["content"]


async def test_a_missing_decision_counts_as_a_rejection(writable_db):
    model = ScriptedModel([AssistantTurn(None, [refund_call()]), AssistantTurn("Sorry.")])
    before = refunds_in(writable_db)
    async with Client(build_server(Session("C0021", writable_db))) as client:
        agent = GraphAgent(model, McpToolbox(client), SYSTEM, InMemorySaver())
        await agent.ask("t1", "Refund my lamp.")
        await agent.resume("t1", {"some-other-call": {"approved": True}})

    assert refunds_in(writable_db) == before
    assert "no decision was given" in model.seen[1][-1]["content"]


async def test_a_refund_the_policy_refuses_does_not_reach_the_reviewer(writable_db):
    """Final sale + changed mind: the quote already fails, so no human is asked."""
    model = ScriptedModel(
        [
            AssistantTurn(None, [refund_call(product_id="P019", condition="changed_mind")]),
            AssistantTurn("That item was final sale."),
        ]
    )
    before = refunds_in(writable_db)
    async with Client(build_server(Session("C0021", writable_db))) as client:
        toolbox = CountingToolbox(client)
        agent = GraphAgent(model, toolbox, SYSTEM, InMemorySaver())
        result = await agent.ask("t1", "Refund the final-sale vase, I changed my mind.")

    assert result.stop_reason == "answer"
    assert "Final sale" in model.seen[1][-1]["content"]
    assert toolbox.ran == ["quote_refund"]  # refund_order itself never ran
    assert refunds_in(writable_db) == before


async def test_other_tools_of_the_same_turn_run_once_despite_the_pause(writable_db):
    """On resume LangGraph re-runs the paused node: get_order must not be in it."""
    turn = AssistantTurn(
        None, [ToolCall("g1", "get_order", json.dumps({"order_id": "O10040"})), refund_call()]
    )
    model = ScriptedModel([turn, AssistantTurn("Done.")])
    async with Client(build_server(Session("C0021", writable_db))) as client:
        toolbox = CountingToolbox(client)
        agent = GraphAgent(model, toolbox, SYSTEM, InMemorySaver())
        await agent.ask("t1", "Check my order and refund the lamp.")
        await agent.resume("t1", {"r1": {"approved": True}})

    assert toolbox.ran.count("get_order") == 1
    assert toolbox.ran.count("refund_order") == 1
    # Results answer the calls in the order the model made them.
    assert [m["tool_call_id"] for m in model.seen[1][-2:]] == ["g1", "r1"]


async def test_the_reviewer_can_answer_from_a_new_agent(writable_db):
    """The pause is saved by the checkpointer: the decision can come after a restart."""
    saver = InMemorySaver()
    async with Client(build_server(Session("C0021", writable_db))) as client:
        first = GraphAgent(
            ScriptedModel([AssistantTurn(None, [refund_call()])]), McpToolbox(client), SYSTEM, saver
        )
        await first.ask("t1", "Refund my lamp.")
        later = GraphAgent(
            ScriptedModel([AssistantTurn("Refunded.")]), McpToolbox(client), SYSTEM, saver
        )
        done = await later.resume("t1", {"r1": {"approved": True}})
    assert done.answer == "Refunded."


async def test_resume_without_a_pending_request_is_an_error(shop_db, owner_and_order):
    owner, _ = owner_and_order
    async with Client(build_server(Session(owner, shop_db))) as client:
        agent = GraphAgent(
            ScriptedModel([AssistantTurn("Hi.")]), McpToolbox(client), SYSTEM, InMemorySaver()
        )
        await agent.ask("t1", "Hello")
        with pytest.raises(RuntimeError, match="not waiting"):
            await agent.resume("t1", {"r1": {"approved": True}})
