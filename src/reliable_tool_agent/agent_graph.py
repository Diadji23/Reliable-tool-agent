"""The same agent as `agent_loop.py`, orchestrated by a LangGraph state graph.

    START --> model --(no tool call)-----------------> END   stop_reason = "answer"
                |
                +--(a sensitive call, with a reviewer)--> approval --+
                |                                    (pauses here:  |
          (other tool calls)              stop_reason = "awaiting_approval")
                v                                                    |
              tools <------------------------------------------------+
                |
                +--(steps >= max_steps)-------> END   stop_reason = "max_steps"
                +--(otherwise)--> back to model

For a single question (`run_graph_agent`) behaviour is identical to the hand-written loop
(the same tests run on both), and sensitive tools are never run. What the graph adds: a
checkpointer that keeps a conversation across turns (`GraphAgent`), and `interrupt` to
pause before a sensitive action until a human reviewer decides.

The model, the MCP toolbox and the OpenAI-format messages are the ones of the hand-written
loop: LangGraph only orchestrates.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from reliable_tool_agent.agent_loop import (
    NEEDS_APPROVAL,
    NO_REVIEWER,
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
    # The outcome of each sensitive call of the last model turn, by tool call id: the text
    # the model reads if the call is not run, or None if the reviewer approved it.
    blocked: dict[str, str | None]


class Decision(BaseModel):
    """A human reviewer's answer to one approval request."""

    approved: bool
    reason: str = ""


REJECTED = (
    "Not done: a human reviewer rejected this {name} call ({reason}). It was not run. "
    "Tell the customer; do not try it again."
)
NO_DECISION = "no decision was given"


def build_graph(
    model: ChatModel,
    toolbox: McpToolbox,
    tools: list[dict],
    max_steps: int,
    checkpointer: BaseCheckpointSaver | None = None,
    human_review: bool = False,
):
    if human_review and checkpointer is None:
        raise ValueError("Pausing for a human reviewer needs a checkpointer.")

    async def call_model(state: AgentState) -> dict:
        turn = await model.complete(state["messages"], tools)
        return {"messages": [assistant_message(turn)], "steps": state["steps"] + 1}

    async def approval(state: AgentState) -> dict:
        # On resume, LangGraph runs this node again from the top: everything before
        # interrupt() is repeated. The previews are read-only, so that is harmless; this
        # is why the sensitive call itself waits in the `tools` node, after the decision.
        blocked: dict[str, str | None] = {}
        requests = []
        for call in sensitive_calls(state):
            name, arguments = call["function"]["name"], call["function"]["arguments"]
            preview = await toolbox.run(NEEDS_APPROVAL[name], arguments)
            if not preview.ok:
                # Refused by the policy or badly formed: no need to bother a human, and
                # the call is not run either (its refusal is the preview's).
                blocked[call["id"]] = preview.text
                continue
            requests.append(
                {
                    "tool_call_id": call["id"],
                    "tool": name,
                    "arguments": json.loads(arguments),
                    "preview": preview.data,
                }
            )
        if requests:
            answers = interrupt(requests)  # pauses here; returns the reviewer's decisions
            for request in requests:
                raw = answers.get(request["tool_call_id"]) if isinstance(answers, dict) else None
                decision = Decision.model_validate(raw) if raw is not None else None
                if decision is not None and decision.approved:
                    blocked[request["tool_call_id"]] = None
                else:  # rejected, or no decision for this call: never run by default
                    reason = decision.reason if decision and decision.reason else NO_DECISION
                    blocked[request["tool_call_id"]] = REJECTED.format(
                        name=request["tool"], reason=reason
                    )
        return {"blocked": blocked}

    async def run_tools(state: AgentState) -> dict:
        blocked = state.get("blocked", {}) if human_review else {}
        results = []
        for call in state["messages"][-1]["tool_calls"]:
            name = call["function"]["name"]
            if name not in NEEDS_APPROVAL:
                output = await toolbox.call(name, call["function"]["arguments"])
            elif not human_review:
                output = NO_REVIEWER.format(name=name)
            elif call["id"] in blocked and blocked[call["id"]] is None:  # approved
                output = await toolbox.call(name, call["function"]["arguments"])
            else:
                output = blocked.get(call["id"]) or REJECTED.format(name=name, reason=NO_DECISION)
            # Tool output goes in a `tool` message: data the model reads, not instructions.
            results.append({"role": "tool", "tool_call_id": call["id"], "content": output})
        return {"messages": results, "blocked": {}}

    def after_model(state: AgentState) -> str:
        if not state["messages"][-1].get("tool_calls"):
            return END
        return "approval" if human_review and sensitive_calls(state) else "tools"

    def after_tools(state: AgentState) -> str:
        return END if state["steps"] >= max_steps else "model"

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_node("approval", approval)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["approval", "tools", END])
    graph.add_edge("approval", "tools")
    graph.add_conditional_edges("tools", after_tools, ["model", END])
    return graph.compile(checkpointer=checkpointer)


def sensitive_calls(state: AgentState) -> list[dict]:
    calls = state["messages"][-1].get("tool_calls") or []
    return [c for c in calls if c["function"]["name"] in NEEDS_APPROVAL]


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

    Before a sensitive tool runs, the turn stops with stop_reason "awaiting_approval" and
    the requests in `pending`. The state is saved, so the reviewer can answer later, even
    from another process sharing the checkpointer: `resume` continues the same turn.
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
        graph, config = await self._ready(thread_id)
        saved = await graph.aget_state(config)
        if saved.next:
            raise RuntimeError(f"Thread {thread_id!r} awaits a reviewer's decision: resume it.")
        new: list[Message] = (
            [] if saved.values else [{"role": "system", "content": self.system_prompt}]
        )
        new.append({"role": "user", "content": user_message})
        return _result(await graph.ainvoke({"messages": new, "steps": 0}, config))

    async def resume(self, thread_id: str, decisions: dict[str, Decision | dict]) -> AgentResult:
        """Give the reviewer's decisions, by tool call id, and finish the paused turn.

        A pending call without a decision is treated as rejected.
        """
        graph, config = await self._ready(thread_id)
        if not (await graph.aget_state(config)).next:
            raise RuntimeError(f"Thread {thread_id!r} is not waiting for a decision.")
        answers = {
            call_id: Decision.model_validate(d).model_dump() for call_id, d in decisions.items()
        }
        return _result(await graph.ainvoke(Command(resume=answers), config))

    async def _ready(self, thread_id: str):
        if self._graph is None:  # tool specs need the MCP session: built on first use
            tools = await self.toolbox.specs()
            self._graph = build_graph(
                self.model,
                self.toolbox,
                tools,
                self.max_steps,
                self.checkpointer,
                human_review=True,
            )
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": _recursion_limit(self.max_steps),
        }
        return self._graph, config

    async def history(self, thread_id: str) -> list[Message]:
        """The saved conversation of a thread, as the model sees it ([] if none yet)."""
        if self._graph is None:
            return []
        saved = await self._graph.aget_state({"configurable": {"thread_id": thread_id}})
        return saved.values.get("messages", [])


def _recursion_limit(max_steps: int) -> int:
    # Our own max_steps stops a turn cleanly; LangGraph's recursion limit (graph steps,
    # i.e. node runs per invocation) stays as a safety net and raises if our logic is wrong.
    return 3 * max_steps + 2  # model, approval and tools in each step


def _result(final: dict) -> AgentResult:
    if final.get("__interrupt__"):  # paused in the approval node
        (paused,) = final["__interrupt__"]
        return AgentResult(
            None, "awaiting_approval", final["steps"], final["messages"], pending=paused.value
        )
    last = final["messages"][-1]
    if last["role"] == "assistant" and not last.get("tool_calls"):
        return AgentResult(last["content"], "answer", final["steps"], final["messages"])
    return AgentResult(None, "max_steps", final["steps"], final["messages"])
