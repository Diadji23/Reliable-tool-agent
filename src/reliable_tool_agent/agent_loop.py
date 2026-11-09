"""A minimal agent loop, written by hand (no framework) to show the mechanism.

The whole idea of a tool-using agent fits in one loop:

    messages = [system prompt, user question]
    repeat:
        ask the model, giving it the messages and the tool descriptions
        if it answers with text only        -> that is the final answer, stop
        if it asks for tool calls           -> run each tool, append each result
                                               as a `tool` message, and loop
    stop anyway after `max_steps` model calls (a model can loop forever)

Messages use the OpenAI chat-completions format (roles system / user / assistant /
tool), spoken both by llama.cpp's `llama-server` and by most hosted APIs, so the same loop
works with a local or a remote model. The LangGraph version comes next; this file stays
as the reference for what the framework does for us.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from mcp import Client

Message = dict[str, Any]


# --- What the model returns ------------------------------------------------------------


@dataclass
class ToolCall:
    id: str  # chosen by the model; the tool result must quote it back
    name: str
    arguments: str  # raw JSON text, exactly as generated: it may be invalid


@dataclass
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Token counts reported by the server ({"prompt_tokens": ..., "completion_tokens": ...}),
    # kept for cost and latency accounting; None for the scripted model.
    usage: dict[str, int] | None = None


class ChatModel(Protocol):
    """Anything that can answer a conversation, possibly by asking for tool calls."""

    async def complete(self, messages: list[Message], tools: list[dict]) -> AssistantTurn: ...


# --- Tools, served by the MCP server ---------------------------------------------------


class McpToolbox:
    """Bridge between the loop and an MCP client: tool specs in, tool results out."""

    def __init__(self, client: Client):
        self.client = client
        self.names: set[str] = set()

    async def specs(self) -> list[dict]:
        """The MCP tools, translated into the format the model expects."""
        listed = (await self.client.list_tools()).tools
        self.names = {t.name for t in listed}
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": hide_approved_amount(t.input_schema),
                },
            }
            for t in listed
        ]

    async def call(self, name: str, raw_arguments: str) -> str:
        """Run one tool call and return the text the model will read."""
        return (await self.run(name, raw_arguments)).text

    async def run(self, name: str, raw_arguments: str) -> ToolOutput:
        """Run one tool call; say whether it succeeded, with the text and the data.

        Every failure the model can cause (unknown tool, invalid JSON, invalid
        arguments, expected tool error) comes back as text, so the model can correct
        itself. Only bugs on our side raise.
        """
        if name not in self.names:
            return ToolOutput(
                False, f"Error: unknown tool {name!r}. Available tools: {sorted(self.names)}."
            )
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            return ToolOutput(
                False, f"Error: the arguments are not valid JSON ({exc.msg}). Send a JSON object."
            )
        if not isinstance(arguments, dict):
            return ToolOutput(False, "Error: the arguments must be a JSON object.")

        result = await self.client.call_tool(name, arguments)
        if result.is_error or result.structured_content is None:
            text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
            return ToolOutput(False, text)
        data = result.structured_content
        return ToolOutput(True, json.dumps(data), data)


def hide_approved_amount(schema: dict) -> dict:
    """The tool's input schema without the argument only the approval may fill."""
    if APPROVED_AMOUNT not in schema.get("properties", {}):
        return schema
    shown = {**schema, "properties": dict(schema["properties"])}
    del shown["properties"][APPROVED_AMOUNT]
    if "required" in shown:
        shown["required"] = [r for r in shown["required"] if r != APPROVED_AMOUNT]
    return shown


@dataclass
class ToolOutput:
    ok: bool
    text: str  # what the model reads
    data: dict | None = None  # the structured result, when the call succeeded


# --- Sensitive tools -------------------------------------------------------------------

# Tools that change something a customer cannot undo, each with the read-only tool that
# shows a human what it would do. Decided here, in the agent: MCP annotations are hints
# written by the server, not a security boundary.
NEEDS_APPROVAL: dict[str, str] = {"refund_order": "quote_refund"}

# The argument a sensitive tool needs from the human approval. The agent fills it with the
# amount the reviewer approved; the model never sees it in the tool schema, and a value
# the model would send anyway is replaced.
APPROVED_AMOUNT = "approved_amount_cents"

NO_REVIEWER = (
    "Error: {name} needs a human reviewer's approval, and none is available here. "
    "It was not run. Tell the customer their request will be handled by the team."
)


# --- The loop --------------------------------------------------------------------------


@dataclass
class AgentResult:
    answer: str | None
    stop_reason: Literal["answer", "max_steps", "awaiting_approval"]
    steps: int  # number of model calls
    messages: list[Message]  # the full conversation: the trace of what happened
    # When stop_reason is "awaiting_approval": what the human reviewer must decide.
    pending: list[dict] = field(default_factory=list)


async def run_agent(
    model: ChatModel,
    toolbox: McpToolbox,
    user_message: str,
    system_prompt: str,
    max_steps: int = 8,
) -> AgentResult:
    tools = await toolbox.specs()
    messages: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    for step in range(1, max_steps + 1):
        turn = await model.complete(messages, tools)
        messages.append(assistant_message(turn))

        if not turn.tool_calls:  # text only: the model has answered
            return AgentResult(turn.content, "answer", step, messages)

        for call in turn.tool_calls:
            if call.name in NEEDS_APPROVAL:  # this loop has no human to ask: never run it
                output = NO_REVIEWER.format(name=call.name)
            else:
                output = await toolbox.call(call.name, call.arguments)
            # Tool output goes in a `tool` message, never in the system or user role:
            # it is data the model reads, not instructions it must follow.
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    return AgentResult(None, "max_steps", max_steps, messages)


def assistant_message(turn: AssistantTurn) -> Message:
    """The turn as an OpenAI-format `assistant` message, to append to the conversation."""
    message: Message = {"role": "assistant", "content": turn.content}
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in turn.tool_calls
        ]
    return message
