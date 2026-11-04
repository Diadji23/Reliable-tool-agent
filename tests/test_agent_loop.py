"""The hand-written agent loop, driven by a scripted model and the real MCP server."""

from __future__ import annotations

import json
import sqlite3

import pytest
from mcp import Client

from reliable_tool_agent.agent_loop import AssistantTurn, McpToolbox, ToolCall, run_agent
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


def call(name: str, arguments: str, id: str = "call_1") -> AssistantTurn:
    return AssistantTurn(None, [ToolCall(id, name, arguments)])


async def run(shop_db, customer_id, turns, max_steps=8):
    model = ScriptedModel(turns)
    async with Client(build_server(Session(customer_id, shop_db))) as client:
        result = await run_agent(model, McpToolbox(client), "Help me", SYSTEM, max_steps)
    return result, model


def tool_messages(result):
    return [m for m in result.messages if m["role"] == "tool"]


async def test_direct_answer_takes_one_step(shop_db, owner_and_order):
    owner, _ = owner_and_order
    result, model = await run(shop_db, owner, [AssistantTurn("Hello! How can I help?")])
    assert result.stop_reason == "answer"
    assert result.steps == 1
    assert result.answer == "Hello! How can I help?"
    assert tool_messages(result) == []


async def test_tool_result_is_given_back_to_the_model(shop_db, owner_and_order):
    owner, order_id = owner_and_order
    result, model = await run(
        shop_db,
        owner,
        [call("get_order", json.dumps({"order_id": order_id})), AssistantTurn("Delivered.")],
    )
    assert result.steps == 2
    (tool,) = tool_messages(result)
    assert tool["tool_call_id"] == "call_1"  # the result answers that precise call
    assert json.loads(tool["content"])["id"] == order_id
    # On its second call, the model saw its own request followed by the tool result.
    assert [m["role"] for m in model.seen[1]] == ["system", "user", "assistant", "tool"]


async def test_tool_specs_are_translated_for_the_model(shop_db, owner_and_order):
    owner, _ = owner_and_order
    async with Client(build_server(Session(owner, shop_db))) as client:
        specs = await McpToolbox(client).specs()
    by_name = {s["function"]["name"]: s for s in specs}
    assert set(by_name) == {"get_order", "list_my_orders", "search_policies"}
    get_order = by_name["get_order"]
    assert get_order["type"] == "function"
    assert get_order["function"]["description"].startswith("Get one of the current")
    assert get_order["function"]["parameters"]["required"] == ["order_id"]


async def test_several_tool_calls_in_one_turn(shop_db, owner_and_order):
    owner, order_id = owner_and_order
    turn = AssistantTurn(
        None,
        [
            ToolCall("a", "get_order", json.dumps({"order_id": order_id})),
            ToolCall("b", "search_policies", '{"query": "return window"}'),
        ],
    )
    result, _ = await run(shop_db, owner, [turn, AssistantTurn("Done.")])
    assert [m["tool_call_id"] for m in tool_messages(result)] == ["a", "b"]


# --- The model makes mistakes: each one must come back as text, not crash the loop --


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("delete_everything", "{}", "unknown tool"),
        ("get_order", '{"order_id": "O10001"', "not valid JSON"),  # truncated JSON
        ("get_order", '["O10001"]', "must be a JSON object"),
        ("list_my_orders", '{"status": "lost"}', "validation error"),
        ("get_order", '{"order_id": "O99999"}', "No order 'O99999' found"),
    ],
)
async def test_model_mistakes_are_reported_back(
    shop_db, owner_and_order, name, arguments, expected
):
    owner, _ = owner_and_order
    result, model = await run(shop_db, owner, [call(name, arguments), AssistantTurn("Sorry.")])
    assert result.stop_reason == "answer"  # the loop survived and the model got a 2nd turn
    (tool,) = tool_messages(result)
    assert expected in tool["content"]
    assert model.seen[1][-1]["content"] == tool["content"]


async def test_loop_stops_after_max_steps(shop_db, owner_and_order):
    """A model that never stops calling tools must not run forever."""
    owner, order_id = owner_and_order
    looping = [call("get_order", json.dumps({"order_id": order_id})) for _ in range(3)]
    result, _ = await run(shop_db, owner, looping, max_steps=3)
    assert result.stop_reason == "max_steps"
    assert result.answer is None
    assert result.steps == 3
