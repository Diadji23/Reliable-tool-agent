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
