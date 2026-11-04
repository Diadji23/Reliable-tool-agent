"""ChatModel backed by an OpenAI-compatible chat-completions server (llama.cpp's llama-server).

Start the server (tool calls need --jinja, which applies the model's own chat template):

    llama-server -m models/Qwen3.5-4B-Q4_K_M.gguf --jinja -c 8192 --port 8080

The same class works with any server speaking this API, local or hosted.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from reliable_tool_agent.agent_loop import AssistantTurn, Message, ToolCall


class LlamaServerModel:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "local",
        temperature: float = 0.0,
        thinking: bool = False,
        timeout_s: float = 600.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        # Qwen3.5 "thinks aloud" by default: hundreds of tokens before each answer, i.e.
        # minutes per step at ~8 tok/s on this CPU. Off unless asked for; whether it
        # helps is a question for the evaluation.
        self.thinking = thinking
        self.client = client or httpx.AsyncClient(timeout=timeout_s)

    async def complete(self, messages: list[Message], tools: list[dict]) -> AssistantTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if tools:
            payload["tools"] = tools
        response = await self.client.post(f"{self.base_url}/chat/completions", json=payload)
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
            content=message.get("content"),
            tool_calls=calls,
            usage={k: usage[k] for k in ("prompt_tokens", "completion_tokens") if k in usage}
            if usage
            else None,
        )

    async def aclose(self) -> None:
        await self.client.aclose()
