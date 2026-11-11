"""Run the attack cases with a real local model and save one JSON line per case.

Needs a running llama-server (see llama_server.py):

    uv run python scripts/attack_run.py experiments/03-guardrail-baseline

The deterministic checks are applied here; every answer is also read by hand in the
experiment's write-up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import platform
import time
from pathlib import Path

import httpx

from reliable_tool_agent.attack_eval import check, load_cases, run_case
from reliable_tool_agent.db import DEFAULT_DB_PATH
from reliable_tool_agent.llama_server import LlamaServerModel
from reliable_tool_agent.prompts import system_prompt


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


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--only", nargs="*", help="case ids to run (default: all)")
    parser.add_argument("--out", default="traces.jsonl")
    args = parser.parse_args()
    logging.getLogger("mcp").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    props = httpx.get(args.base_url.removesuffix("/v1") + "/props").json()
    meta = {
        "model_path": props.get("model_path"),
        "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
        "build": props.get("build_info"),
        "machine": platform.processor() or platform.machine(),
    }
    print(json.dumps(meta))

    cases = [c for c in load_cases() if not args.only or c["id"] in args.only]
    passed = 0
    with (args.experiment_dir / args.out).open("w") as out:
        for case in cases:
            model = TimedModel(LlamaServerModel(base_url=args.base_url))
            run = await run_case(case, model, DEFAULT_DB_PATH, system_prompt(), args.max_steps)
            failures = check(case, run)
            passed += not failures
            record = run.to_json()
            record["messages"] = record["messages"][1:]  # same system prompt everywhere
            out.write(
                json.dumps(
                    {**record, "failures": failures, "model_calls": model.calls, "meta": meta}
                )
                + "\n"
            )
            out.flush()
            seconds = sum(t.seconds for t in run.turns)
            print(f"{case['id']}: {seconds:6.1f}s {'PASS' if not failures else failures}")
            for t in run.turns:
                print(f"    -> {' '.join((t.answer or '').split())[:200]!r}")
            await model.inner.aclose()
    print(f"\n{passed}/{len(cases)} cases pass the deterministic checks")


if __name__ == "__main__":
    asyncio.run(main())
