"""The MCP server, seen from a real MCP client (no LLM involved)."""

from __future__ import annotations

import sqlite3
import sys

import pytest
from mcp import Client, StdioServerParameters

from reliable_tool_agent.mcp_server import build_server
from reliable_tool_agent.tools import Session

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def owner_and_order(shop_db):
    conn = sqlite3.connect(shop_db)
    owner, order = conn.execute("SELECT customer_id, id FROM orders ORDER BY id LIMIT 1").fetchone()
    (other,) = conn.execute(
        "SELECT id FROM customers WHERE id != ? ORDER BY id LIMIT 1", (owner,)
    ).fetchone()
    conn.close()
    return owner, order, other


async def test_schema_exposes_no_customer_id(shop_db, owner_and_order):
    owner, _, _ = owner_and_order
    async with Client(build_server(Session(owner, shop_db))) as client:
        listed = await client.list_tools()
    tools = {t.name: t for t in listed.tools}
    assert set(tools) == {
        "get_order",
        "list_my_orders",
        "search_policies",
        "check_return_eligibility",
    }
    schema = tools["get_order"].input_schema
    assert schema["required"] == ["order_id"]
    for tool in tools.values():
        assert "customer" not in str(tool.input_schema).lower()


async def test_invalid_argument_is_rejected_with_a_reason(shop_db, owner_and_order):
    """Pydantic checks the model's arguments before the tool runs."""
    owner, _, _ = owner_and_order
    async with Client(build_server(Session(owner, shop_db))) as client:
        result = await client.call_tool("list_my_orders", {"status": "lost"})
    assert result.is_error
    assert "delivered" in result.content[0].text  # the allowed values are listed


async def test_call_returns_structured_order(shop_db, owner_and_order):
    owner, order_id, _ = owner_and_order
    async with Client(build_server(Session(owner, shop_db))) as client:
        result = await client.call_tool("get_order", {"order_id": order_id})
    assert not result.is_error
    assert result.structured_content["id"] == order_id
    assert result.structured_content["items"]


async def test_other_customers_order_is_an_explained_error(shop_db, owner_and_order):
    _, order_id, other = owner_and_order
    async with Client(build_server(Session(other, shop_db))) as client:
        result = await client.call_tool("get_order", {"order_id": order_id})
    assert result.is_error
    # The model must get the reason (an expected failure), not a generic crash message.
    assert "No order" in result.content[0].text


async def test_stdio_transport_end_to_end(shop_db, owner_and_order):
    """The real deployment path: the server runs as a subprocess, JSON over stdin/stdout."""
    owner, order_id, _ = owner_and_order
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "reliable_tool_agent.mcp_server", "--customer-id", owner, "--db", str(shop_db)],
    )
    async with Client(params) as client:
        result = await client.call_tool("get_order", {"order_id": order_id})
    assert not result.is_error
    assert result.structured_content["id"] == order_id


async def test_return_eligibility_over_mcp(shop_db):
    async with Client(build_server(Session("C0001", shop_db))) as client:
        ok = await client.call_tool("check_return_eligibility", {"order_id": "O10003"})
        missing = await client.call_tool("check_return_eligibility", {"order_id": "O10009"})
    assert ok.structured_content["return_window_ends"] == "2026-11-04"
    assert ok.structured_content["items"][0]["returnable"] is True
    assert missing.is_error  # another customer's order: same answer as a missing one
    assert "No order 'O10009' found" in missing.content[0].text
