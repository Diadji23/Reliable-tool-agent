"""The same agent as `agent_loop.py`, orchestrated by a LangGraph state graph.

    START --> model --(no tool call)-----------------> END   stop_reason = "answer"
                |
          (tool calls)
                v
              tools --(steps >= max_steps)-------> END   stop_reason = "max_steps"
                |
                +--(otherwise)--> back to model

For a single question (`run_graph_agent`) behaviour is identical to the hand-written loop
(the same tests run on both). What the graph adds: a checkpointer that keeps a
conversation across turns (`GraphAgent`), and later `interrupt` to pause before a
sensitive action until a human approves it.

The model, the MCP toolbox and the OpenAI-format messages are the ones of the hand-written
loop: LangGraph only orchestrates.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from reliable_tool_agent.agent_loop import (
    AgentResult,
    ChatModel,
    McpToolbox,
    Message,
    assistant_message,
)


class AgentState(TypedDict):
    # The reducer `operator.add` appends what a node returns to the existing list, so a
    # node returns only its new messages.
    messages: Annotated[list[Message], operator.add]
    # Model calls in the current turn. No reducer: a value given replaces the old one, so
    # each new user message resets it and max_steps bounds one turn, not the conversation.
    steps: int


def build_graph(
    model: ChatModel,
    toolbox: McpToolbox,
    tools: list[dict],
    max_steps: int,
    checkpointer: BaseCheckpointSaver | None = None,
):
    async def call_model(state: AgentState) -> dict:
        turn = await model.complete(state["messages"], tools)
        return {"messages": [assistant_message(turn)], "steps": state["steps"] + 1}

    async def run_tools(state: AgentState) -> dict:
        results = []
        for call in state["messages"][-1]["tool_calls"]:
            output = await toolbox.call(call["function"]["name"], call["function"]["arguments"])
            # Tool output goes in a `tool` message: data the model reads, not instructions.
            results.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        return {"messages": results}

    def after_model(state: AgentState) -> str:
        return "tools" if state["messages"][-1].get("tool_calls") else END

    def after_tools(state: AgentState) -> str:
        return END if state["steps"] >= max_steps else "model"

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["model", END])
    return graph.compile(checkpointer=checkpointer)


async def run_graph_agent(
    model: ChatModel,
    toolbox: McpToolbox,
    user_message: str,
    system_prompt: str,
    max_steps: int = 8,
) -> AgentResult:
    graph = build_graph(model, toolbox, await toolbox.specs(), max_steps)
    initial: AgentState = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "steps": 0,
    }
    final = await graph.ainvoke(initial, {"recursion_limit": _recursion_limit(max_steps)})
    return _result(final)


class GraphAgent:
    """A conversation agent: each `thread_id` keeps its own history across `ask` calls.

    The checkpointer saves the graph state after every node, under the thread id. A new
    message on the same thread is appended (by the reducer) to the saved conversation.
    The thread id is chosen by the application, never by the model; serving a thread to
    the right customer only is the application's job (one MCP session per customer).
    """

    def __init__(
        self,
        model: ChatModel,
        toolbox: McpToolbox,
        system_prompt: str,
        checkpointer: BaseCheckpointSaver,
        max_steps: int = 8,
    ):
        self.model = model
        self.toolbox = toolbox
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer
        self.max_steps = max_steps
        self._graph = None

    async def ask(self, thread_id: str, user_message: str) -> AgentResult:
        if self._graph is None:  # tool specs need the MCP session: built on first use
            tools = await self.toolbox.specs()
            self._graph = build_graph(
                self.model, self.toolbox, tools, self.max_steps, self.checkpointer
            )
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": _recursion_limit(self.max_steps),
        }
        saved = await self._graph.aget_state(config)
        new: list[Message] = (
            [] if saved.values else [{"role": "system", "content": self.system_prompt}]
        )
        new.append({"role": "user", "content": user_message})
        final = await self._graph.ainvoke({"messages": new, "steps": 0}, config)
        return _result(final)


def _recursion_limit(max_steps: int) -> int:
    # Our own max_steps stops a turn cleanly; LangGraph's recursion limit (graph steps,
    # i.e. node runs per invocation) stays as a safety net and raises if our logic is wrong.
    return 2 * max_steps + 2


def _result(final: AgentState) -> AgentResult:
    last = final["messages"][-1]
    if last["role"] == "assistant" and not last.get("tool_calls"):
        return AgentResult(last["content"], "answer", final["steps"], final["messages"])
    return AgentResult(None, "max_steps", final["steps"], final["messages"])
