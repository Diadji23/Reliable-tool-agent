"""Trace the agent in MLflow: one trace per customer turn, one span per model or tool call.

Optional (dependency group `tracking`), and kept outside the agent: the wrappers below
have the same interface as what they wrap, so the agent code neither imports MLflow nor
knows it is traced. A script turns tracing on with `enable_tracing()` and wraps its
model, toolbox and agent.

What one trace shows, for one `ask` (or `resume` after a reviewer's decision):

    agent turn (AGENT)          customer message -> answer, stop reason; session = thread
    ├── model (LLM)             messages in, text or tool calls out, tokens
    ├── get_order (TOOL)        arguments in, result text out, ok or not
    ├── model (LLM)
    └── ...

The trace's session is the conversation thread, so the turns of one conversation are
grouped; its user is the authenticated customer.
"""

from __future__ import annotations

import os
from typing import Any

from reliable_tool_agent.agent_loop import AgentResult, AssistantTurn, Message, ToolOutput

TRACES_EXPERIMENT = "agent-traces"


def enable_tracing(experiment: str = TRACES_EXPERIMENT, tracking_uri: str | None = None) -> None:
    """Send traces to the local MLflow database (or to MLFLOW_TRACKING_URI)."""
    import mlflow

    from reliable_tool_agent.tracking import DEFAULT_TRACKING_URI

    mlflow.set_tracking_uri(
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    )
    found = mlflow.set_experiment(experiment)
    # Opposite of the evaluation experiments: here the traces page is the one to open.
    mlflow.MlflowClient().set_experiment_tag(
        found.experiment_id, "mlflow.experimentKind", "genai_development"
    )


def flush_traces() -> None:
    """Wait until every finished trace is written (MLflow exports them in the background)."""
    import mlflow

    mlflow.flush_trace_async_logging()


class TracedModel:
    """A ChatModel whose every call is an LLM span with its token usage."""

    def __init__(self, inner, name: str = "model"):
        self.inner = inner
        self.name = name

    async def complete(self, messages: list[Message], tools: list[dict]) -> AssistantTurn:
        import mlflow
        from mlflow.entities import SpanType
        from mlflow.tracing.constant import SpanAttributeKey

        with mlflow.start_span(name=self.name, span_type=SpanType.LLM) as span:
            span.set_inputs({"messages": messages})
            span.set_attribute(SpanAttributeKey.CHAT_TOOLS, tools)
            turn = await self.inner.complete(messages, tools)
            span.set_outputs(
                {
                    "content": turn.content,
                    "tool_calls": [
                        {"name": c.name, "arguments": c.arguments} for c in turn.tool_calls
                    ],
                }
            )
            if turn.usage:
                inputs = turn.usage.get("prompt_tokens", 0)
                outputs = turn.usage.get("completion_tokens", 0)
                span.set_attribute(
                    SpanAttributeKey.CHAT_USAGE,
                    {
                        "input_tokens": inputs,
                        "output_tokens": outputs,
                        "total_tokens": inputs + outputs,
                    },
                )
            return turn


class TracedToolbox:
    """A toolbox whose every tool call is a TOOL span (the quote shown to a reviewer too)."""

    def __init__(self, inner):
        self.inner = inner

    @property
    def names(self) -> set[str]:
        return self.inner.names

    async def specs(self) -> list[dict]:
        return await self.inner.specs()

    async def call(self, name: str, raw_arguments: str) -> str:
        return (await self.run(name, raw_arguments)).text

    async def run(self, name: str, raw_arguments: str) -> ToolOutput:
        import mlflow
        from mlflow.entities import SpanType

        with mlflow.start_span(name=name, span_type=SpanType.TOOL) as span:
            span.set_inputs({"arguments": raw_arguments})
            output = await self.inner.run(name, raw_arguments)
            span.set_outputs({"ok": output.ok, "text": output.text})
            return output


class TracedAgent:
    """A GraphAgent whose every turn (ask, or resume after a decision) is one trace."""

    def __init__(self, inner, user: str | None = None, metadata: dict[str, str] | None = None):
        self.inner = inner
        self.user = user
        self.metadata = metadata or {}

    async def ask(self, thread_id: str, user_message: str) -> AgentResult:
        return await self._turn("ask", thread_id, {"message": user_message}, user_message)

    async def resume(self, thread_id: str, decisions: dict[str, Any]) -> AgentResult:
        shown = {k: v if isinstance(v, dict) else v.model_dump() for k, v in decisions.items()}
        return await self._turn("resume", thread_id, {"decisions": shown}, decisions)

    async def history(self, thread_id: str) -> list[Message]:
        return await self.inner.history(thread_id)

    async def _turn(self, kind: str, thread_id: str, inputs: dict, argument) -> AgentResult:
        import mlflow
        from mlflow.entities import SpanType

        with mlflow.start_span(name=f"agent {kind}", span_type=SpanType.AGENT) as span:
            span.set_inputs(inputs)
            mlflow.update_current_trace(
                session_id=thread_id, user=self.user, metadata=self.metadata
            )
            result = await getattr(self.inner, kind)(thread_id, argument)
            span.set_outputs(
                {
                    "answer": result.answer,
                    "stop_reason": result.stop_reason,
                    "pending": [p.get("tool") for p in result.pending or []],
                }
            )
            return result
