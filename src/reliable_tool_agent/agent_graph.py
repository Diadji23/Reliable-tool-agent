"""The same agent as `agent_loop.py`, orchestrated by a LangGraph state graph.

    START --> model --(no tool call)-----------------> END   stop_reason = "answer"
                |
          (tool calls)
                v
              tools --(steps >= max_steps)-------> END   stop_reason = "max_steps"
                |
                +--(otherwise)--> back to model

Behaviour is identical to the hand-written loop (the same tests run on both). What the
graph adds comes later: a checkpointer to keep a conversation across turns, and
`interrupt` to pause before a sensitive action until a human approves it.

The model, the MCP toolbox and the OpenAI-format messages are the ones of the hand-written
loop: LangGraph only orchestrates.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

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
    steps: int  # model calls so far; no reducer: a returned value replaces the old one


def build_graph(model: ChatModel, toolbox: McpToolbox, tools: list[dict], max_steps: int):
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
    return graph.compile()


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
    # Our own max_steps stops the run cleanly; LangGraph's recursion limit (graph steps,
    # i.e. node runs) stays as a safety net and would raise if our logic were wrong.
    final = await graph.ainvoke(initial, {"recursion_limit": 2 * max_steps + 2})
    last = final["messages"][-1]
    if last["role"] == "assistant" and not last.get("tool_calls"):
        return AgentResult(last["content"], "answer", final["steps"], final["messages"])
    return AgentResult(None, "max_steps", final["steps"], final["messages"])
