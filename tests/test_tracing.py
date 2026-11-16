"""The traces of a real graph turn (scripted model, real MCP server), read back from MLflow."""

from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from mcp import Client

from reliable_tool_agent.agent_graph import GraphAgent
from reliable_tool_agent.agent_loop import AssistantTurn, McpToolbox, ToolCall
from reliable_tool_agent.fake_model import ScriptedModel
from reliable_tool_agent.mcp_server import build_server
from reliable_tool_agent.tools import Session

mlflow = pytest.importorskip("mlflow")  # local-only dependency group, not in CI

from reliable_tool_agent.tracing import (  # noqa: E402
    TracedAgent,
    TracedModel,
    TracedToolbox,
    enable_tracing,
    flush_traces,
)

pytestmark = pytest.mark.anyio

REFUND = {
    "order_id": "O10040",
    "product_id": "P010",
    "quantity": 1,
    "condition": "defective_or_wrong",
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


def turn(content, calls=(), prompt=1000, completion=20):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion}
    return AssistantTurn(content, list(calls), usage)


def spans(trace) -> list[tuple[str, str, str | None]]:
    """(span name, span type, parent span name), in start order."""
    by_id = {s.span_id: s for s in trace.data.spans}
    ordered = sorted(trace.data.spans, key=lambda s: s.start_time_ns)
    return [
        (s.name, s.span_type, by_id[s.parent_id].name if s.parent_id else None) for s in ordered
    ]


async def test_a_paused_refund_gives_one_trace_per_turn_with_model_and_tool_spans(
    writable_db, tmp_path
):
    enable_tracing("test-traces", f"sqlite:///{tmp_path / 'mlflow.db'}")
    model = ScriptedModel(
        [
            turn(None, [ToolCall("r1", "refund_order", json.dumps(REFUND))], 1200, 40),
            turn("Refunded.", prompt=1400, completion=8),
        ]
    )
    async with Client(build_server(Session("C0021", writable_db))) as client:
        toolbox = TracedToolbox(McpToolbox(client))
        graph = GraphAgent(TracedModel(model), toolbox, "System.", InMemorySaver())
        agent = TracedAgent(graph, user="C0021")
        paused = await agent.ask("t1", "My pillow arrived torn, refund it please.")
        assert paused.stop_reason == "awaiting_approval"
        await agent.resume("t1", {"r1": {"approved": True, "amount_cents": 5090}})

    flush_traces()
    asked, resumed = sorted(
        mlflow.search_traces(return_type="list"), key=lambda t: t.info.request_time
    )
    # Turn 1: the model asks for the refund; the code quotes it for the reviewer; pause.
    assert spans(asked) == [
        ("agent ask", "AGENT", None),
        ("model", "LLM", "agent ask"),
        ("quote_refund", "TOOL", "agent ask"),
    ]
    # Turn 2, after the decision. LangGraph re-runs the interrupted approval node from its
    # start, so the quote runs again (which is why the approved amount comes from the
    # decision, not from this new quote); then the refund runs and the model answers.
    assert spans(resumed) == [
        ("agent resume", "AGENT", None),
        ("quote_refund", "TOOL", "agent resume"),
        ("refund_order", "TOOL", "agent resume"),
        ("model", "LLM", "agent resume"),
    ]
    for trace in (asked, resumed):
        assert trace.info.request_metadata["mlflow.trace.session"] == "t1"
        assert trace.info.request_metadata["mlflow.trace.user"] == "C0021"

    llm = next(s for s in asked.data.spans if s.span_type == "LLM")
    assert llm.get_attribute("mlflow.chat.tokenUsage") == {
        "input_tokens": 1200,
        "output_tokens": 40,
        "total_tokens": 1240,
    }
    root = next(s for s in asked.data.spans if s.parent_id is None)
    assert root.outputs["stop_reason"] == "awaiting_approval"
    assert root.outputs["pending"] == ["refund_order"]
    tool = next(s for s in resumed.data.spans if s.name == "refund_order")
    assert tool.outputs["ok"] is True
    assert json.loads(tool.outputs["text"])["amount_cents"] == 5090
