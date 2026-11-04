"""The llama-server adapter, against a fake HTTP server (no network, no model)."""

from __future__ import annotations

import json

import httpx
import pytest

from reliable_tool_agent.llama_server import LlamaServerModel

pytestmark = pytest.mark.anyio

TOOLS = [{"type": "function", "function": {"name": "get_order", "parameters": {}}}]
MESSAGES = [{"role": "user", "content": "Where is O10001?"}]


@pytest.fixture
def anyio_backend():
    return "asyncio"


def fake_server(message: dict, usage: dict | None = None):
    """A model whose server always answers `message`; records the requests it gets."""
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append({"url": str(request.url), "json": json.loads(request.content)})
        body = {"choices": [{"message": message}]}
        if usage:
            body["usage"] = usage
        return httpx.Response(200, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return LlamaServerModel(client=client), requests


async def test_request_carries_messages_tools_and_thinking_off():
    model, requests = fake_server({"role": "assistant", "content": "Hi"})
    await model.complete(MESSAGES, TOOLS)
    (sent,) = requests
    assert sent["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert sent["json"]["messages"] == MESSAGES
    assert sent["json"]["tools"] == TOOLS
    assert sent["json"]["temperature"] == 0.0
    assert sent["json"]["chat_template_kwargs"] == {"enable_thinking": False}


async def test_text_answer():
    model, _ = fake_server(
        {"role": "assistant", "content": "It was delivered."},
        usage={"prompt_tokens": 812, "completion_tokens": 9, "total_tokens": 821},
    )
    turn = await model.complete(MESSAGES, TOOLS)
    assert turn.content == "It was delivered."
    assert turn.tool_calls == []
    assert turn.usage == {"prompt_tokens": 812, "completion_tokens": 9}


async def test_tool_call_is_parsed():
    model, _ = fake_server(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "abc",
                    "type": "function",
                    "function": {"name": "get_order", "arguments": '{"order_id": "O10001"}'},
                }
            ],
        }
    )
    turn = await model.complete(MESSAGES, TOOLS)
    (call,) = turn.tool_calls
    assert (call.id, call.name, call.arguments) == ("abc", "get_order", '{"order_id": "O10001"}')


async def test_object_arguments_and_missing_id_are_normalised():
    model, _ = fake_server(
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "get_order", "arguments": {"order_id": "O1"}}}],
        }
    )
    (call,) = (await model.complete(MESSAGES, TOOLS)).tool_calls
    assert call.id == "call_0"
    assert json.loads(call.arguments) == {"order_id": "O1"}


async def test_server_error_raises():
    """A broken server is our problem, not the model's: it must not pass silently."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(httpx.HTTPStatusError):
        await LlamaServerModel(client=client).complete(MESSAGES, TOOLS)
