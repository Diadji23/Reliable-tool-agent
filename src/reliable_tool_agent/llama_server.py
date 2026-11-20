"""ChatModel backed by an OpenAI-compatible chat-completions server (llama.cpp's llama-server).

Start the server (tool calls need --jinja, which applies the model's own chat template):

    llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --jinja -c 8192 --port 8080

The same class works with any server speaking this API, local or hosted. For a hosted one
(the judge uses Groq): pass `api_key`, set `thinking=None` (the template switch is a
llama-server extension that other servers refuse), put provider options in `extra` and
allow a few `max_retries`.

Some hosted servers (Groq) check the generated tool arguments against the tool's schema
and answer "400 tool_use_failed" instead of returning the bad call. The agent loop would
have shown the model its mistake; here the mistake never reaches it. So the adapter does
the same thing itself: it asks again with the server's error message added, at most
`TOOL_ERROR_RETRIES` times, without keeping that message in the conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from reliable_tool_agent.agent_loop import AssistantTurn, Message, ToolCall

TOOL_ERROR_RETRIES = 2

log = logging.getLogger(__name__)


def without_reasoning(content: str | None) -> str | None:
    """The answer without the model's private reasoning, if some leaked into it.

    Qwen models reason between <think> tags; servers normally strip them, but the hosted
    Qwen once returned "...reasoning... </think> answer" (experiment 06): only what
    follows the last closing tag is meant for the customer."""
    if content is None or "</think>" not in content:
        return content
    return content.rsplit("</think>", 1)[1].strip() or None


class ModelUnavailableError(Exception):
    """The model server is out of quota or failing: not the request's fault, try later.

    `retry_after` is the wait the server asked for, in seconds (None if it gave none)."""

    def __init__(self, status_code: int, retry_after: float | None):
        super().__init__(f"model server answered {status_code}, retry after {retry_after} s")
        self.status_code = status_code
        self.retry_after = retry_after


def tool_use_error(response: httpx.Response) -> str | None:
    """The server's message if it refused the model's tool call, else None."""
    if response.status_code != 400:
        return None
    try:
        error = response.json().get("error", {})
    except ValueError:
        return None
    return error.get("message") if error.get("code") == "tool_use_failed" else None


def correction(error: str) -> Message:
    # A user turn because chat templates refuse a system message after the first one.
    # Written by this code, not by the customer, and never stored in the conversation.
    return {
        "role": "user",
        "content": f"[Automatic message, not from the customer] Your last tool call was "
        f"refused: {error}. Call the tool again with valid arguments.",
    }


class LlamaServerModel:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "local",
        temperature: float = 0.0,
        thinking: bool | None = False,
        timeout_s: float = 600.0,
        client: httpx.AsyncClient | None = None,
        api_key: str | None = None,
        extra: dict[str, Any] | None = None,
        max_retries: int = 0,
        max_wait_s: float | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        # Qwen3.5 "thinks aloud" by default: hundreds of tokens before each answer, i.e.
        # minutes per step at ~8 tok/s on this CPU. Off unless asked for; whether it
        # helps is a question for the evaluation.
        # None: do not send the switch at all (servers other than llama-server).
        self.thinking = thinking
        self.client = client or httpx.AsyncClient(timeout=timeout_s)
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.extra = extra or {}
        # 0 for the local server: an error there is a bug to see at once, not to wait out.
        self.max_retries = max_retries
        # A batch run can wait out a quota; a customer cannot. Beyond this wait, give up at
        # once (ModelUnavailableError) instead of sleeping. None: no limit.
        self.max_wait_s = max_wait_s

    async def complete(self, messages: list[Message], tools: list[dict]) -> AssistantTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            **self.extra,
        }
        if self.thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.thinking}
        if tools:
            payload["tools"] = tools
        response = await self._post(payload)
        for _ in range(TOOL_ERROR_RETRIES):
            error = tool_use_error(response)
            if error is None:
                break
            response = await self._post({**payload, "messages": [*messages, correction(error)]})
        response.raise_for_status()
        body = response.json()
        message = body["choices"][0]["message"]

        calls = []
        for i, raw in enumerate(message.get("tool_calls") or []):
            arguments = raw["function"].get("arguments", "")
            if not isinstance(arguments, str):  # some servers send an object, not text
                arguments = json.dumps(arguments)
            calls.append(ToolCall(raw.get("id") or f"call_{i}", raw["function"]["name"], arguments))

        usage = body.get("usage")
        return AssistantTurn(
            content=without_reasoning(message.get("content")),
            tool_calls=calls,
            usage={k: usage[k] for k in ("prompt_tokens", "completion_tokens") if k in usage}
            if usage
            else None,
        )

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        """POST, waiting and retrying on "too many requests" (429) and server errors (5xx).

        A free tier limits tokens per minute, and hosted APIs fail now and then on their
        side; in both cases the request itself was fine and waiting is enough. When the
        retries run out, or the wait asked is longer than `max_wait_s`, it raises
        ModelUnavailableError. Every retry is logged: a slow turn must say why.
        """
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            response = await self.client.post(url, json=payload, headers=self.headers)
            if response.status_code != 429 and response.status_code < 500:
                return response
            asked = response.headers.get("retry-after")
            wait = float(asked) if asked is not None else 10.0
            if attempt == self.max_retries or (
                self.max_wait_s is not None and wait > self.max_wait_s
            ):
                log.warning(
                    "model server %s, asked to wait %s s: giving up", response.status_code, asked
                )
                raise ModelUnavailableError(
                    response.status_code, float(asked) if asked is not None else None
                )
            log.warning(
                "model server %s, retry %d/%d in %.1f s",
                response.status_code,
                attempt + 1,
                self.max_retries,
                wait,
            )
            await asyncio.sleep(wait)
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self.client.aclose()
