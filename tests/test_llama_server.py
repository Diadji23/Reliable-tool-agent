"""The llama-server adapter, against a fake HTTP server (no network, no model)."""

from __future__ import annotations

import json

import httpx
import pytest

from reliable_tool_agent.llama_server import LlamaServerModel, ModelUnavailableError

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
    with pytest.raises(ModelUnavailableError):
        await LlamaServerModel(client=client).complete(MESSAGES, TOOLS)


async def test_hosted_api_gets_key_and_options_but_no_template_switch():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    model = LlamaServerModel(
        base_url="https://api.example/v1",
        thinking=None,
        api_key="secret",
        extra={"response_format": {"type": "json_object"}},
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await model.complete(MESSAGES, [])
    (request,) = seen
    body = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer secret"
    assert body["response_format"] == {"type": "json_object"}
    assert "chat_template_kwargs" not in body
    assert "tools" not in body


async def test_rate_limited_or_failed_request_is_retried_after_the_wait_asked():
    answers = [
        httpx.Response(429, headers={"retry-after": "0"}),
        httpx.Response(500, headers={"retry-after": "0"}),
        httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
    ]
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: answers.pop(0)))
    turn = await LlamaServerModel(client=client, max_retries=3).complete(MESSAGES, [])
    assert turn.content == "ok"
    assert answers == []


async def test_rate_limit_that_never_lifts_ends_in_an_error():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(429, headers={"retry-after": "0"}))
    )
    with pytest.raises(ModelUnavailableError) as raised:
        await LlamaServerModel(client=client, max_retries=2).complete(MESSAGES, [])
    assert raised.value.status_code == 429
    assert raised.value.retry_after == 0


def tool_refused(message="parameters for tool quote_refund did not match schema"):
    error = {"message": message, "code": "tool_use_failed", "failed_generation": "..."}
    return httpx.Response(400, json={"error": error})


async def test_refused_tool_call_is_asked_again_with_the_servers_error():
    requests = []
    answers = [
        tool_refused(),
        httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
    ]

    def handler(request):
        requests.append(json.loads(request.content))
        return answers.pop(0)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    turn = await LlamaServerModel(client=client).complete(MESSAGES, TOOLS)
    assert turn.content == "ok"
    assert requests[0]["messages"] == MESSAGES
    retry = requests[1]["messages"]
    assert retry[:-1] == MESSAGES
    assert retry[-1]["role"] == "user"
    assert "not from the customer" in retry[-1]["content"]
    assert "did not match schema" in retry[-1]["content"]


async def test_tool_call_refused_every_time_ends_in_an_error():
    calls = []

    def handler(request):
        calls.append(1)
        return tool_refused()

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await LlamaServerModel(client=client).complete(MESSAGES, TOOLS)
    assert len(calls) == 3  # the first try and two corrections


async def test_other_bad_requests_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "bad", "code": "other"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await LlamaServerModel(client=client).complete(MESSAGES, TOOLS)
    assert len(calls) == 1


async def test_a_wait_longer_than_allowed_fails_at_once_instead_of_sleeping():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "3600"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = LlamaServerModel(client=client, max_retries=5, max_wait_s=60)
    with pytest.raises(ModelUnavailableError) as raised:
        await model.complete(MESSAGES, [])
    assert raised.value.retry_after == 3600
    assert len(calls) == 1


async def test_leaked_reasoning_is_cut_from_the_answer():
    leaked = "No, refunds go to the original payment method. </think> No, back to your card."
    model, _ = fake_server({"content": leaked})
    turn = await model.complete(MESSAGES, TOOLS)
    assert turn.content == "No, back to your card."


async def test_an_answer_without_reasoning_is_left_alone():
    model, _ = fake_server({"content": "Use <b>bold</b>, think about it."})
    assert (await model.complete(MESSAGES, TOOLS)).content == "Use <b>bold</b>, think about it."
