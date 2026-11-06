"""First run of the agent loop with a real local model (experiment 01).

Needs a running llama-server (see llama_server.py). For each question, runs the agent for
the question's customer and writes one JSON line with the trace, token counts and timings.
The checks written here are crude (tool names, keywords); every answer is also read by hand
in the write-up.

    uv run python scripts/first_run.py experiments/01-first-real-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import re
import time
from pathlib import Path

import httpx
from mcp import Client

from reliable_tool_agent.agent_loop import McpToolbox, run_agent
from reliable_tool_agent.llama_server import LlamaServerModel
from reliable_tool_agent.mcp_server import build_server
from reliable_tool_agent.prompts import system_prompt
from reliable_tool_agent.tools import Session


class TimedModel:
    """Wraps a ChatModel and records wall time and token usage of every call."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[dict] = []

    async def complete(self, messages, tools):
        start = time.perf_counter()
        turn = await self.inner.complete(messages, tools)
        self.calls.append({"seconds": round(time.perf_counter() - start, 2), **(turn.usage or {})})
        return turn


def check(item: dict, result) -> dict:
    called = [
        c["function"]["name"]
        for m in result.messages
        if m["role"] == "assistant"
        for c in m.get("tool_calls", [])
    ]
    errors = [
        m["content"]
        for m in result.messages
        if m["role"] == "tool" and m["content"].startswith(("Error", "No order", "1 validation"))
    ]
    answer = (result.answer or "").lower()
    return {
        "tools_called": called,
        "expected_tools_called": all(t in called for t in item["expected_tools"])
        and (bool(called) == bool(item["expected_tools"])),
        "tool_errors": errors,
        "keywords_found": all(
            re.search(pattern.lower(), answer) for pattern in item["expected_in_answer"]
        ),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--out", default="traces.jsonl")
    args = parser.parse_args()

    questions = json.loads((args.experiment_dir / "questions.json").read_text())
    props = httpx.get(args.base_url.removesuffix("/v1") + "/props").json()
    meta = {
        "model_path": props.get("model_path"),
        "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
        "build": props.get("build_info"),
        "machine": platform.processor() or platform.machine(),
    }
    print(json.dumps(meta))

    out = (args.experiment_dir / args.out).open("w")
    for item in questions:
        model = TimedModel(LlamaServerModel(base_url=args.base_url))
        start = time.perf_counter()
        async with Client(build_server(Session(item["customer"]))) as client:
            result = await run_agent(
                model, McpToolbox(client), item["question"], system_prompt(), args.max_steps
            )
        record = {
            "id": item["id"],
            "question": item["question"],
            "answer": result.answer,
            "stop_reason": result.stop_reason,
            "steps": result.steps,
            "seconds": round(time.perf_counter() - start, 1),
            "model_calls": model.calls,
            **check(item, result),
            "messages": result.messages[1:],  # the system prompt is the same everywhere
            "meta": meta,
        }
        out.write(json.dumps(record) + "\n")
        out.flush()
        print(
            f"{item['id']}: {record['seconds']:>6.1f}s steps={result.steps} "
            f"tools={record['tools_called']} ok_tools={record['expected_tools_called']} "
            f"ok_kw={record['keywords_found']}\n    -> {(result.answer or '')[:200]!r}"
        )
        await model.inner.aclose()


if __name__ == "__main__":
    asyncio.run(main())
