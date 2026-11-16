"""Run an evaluation set with a real local model and save one JSON line per case.

Needs a running llama-server (see llama_server.py):

    uv run python scripts/run_eval.py experiments/<dir> --set tasks     # reference tasks
    uv run python scripts/run_eval.py experiments/<dir> --set attacks   # attack cases

Add --mlflow to record the run in the local MLflow (experiment agent-eval).

The deterministic checks are applied here; answers are also read by hand (and, later, by
a judge) in the experiment's write-up.
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

from reliable_tool_agent.db import DEFAULT_DB_PATH
from reliable_tool_agent.eval_runner import check, check_task, load_cases, load_tasks, run_case
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
    parser.add_argument("--set", choices=["tasks", "attacks"], required=True)
    parser.add_argument("--only", nargs="*", help="case ids to run (default: all)")
    parser.add_argument("--out", default="traces.jsonl")
    parser.add_argument("--mlflow", action="store_true", help="record the run in MLflow")
    parser.add_argument("--run-name", help="MLflow run name (default: <dir>/<out>)")
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

    cases, checker = (load_tasks(), check_task) if args.set == "tasks" else (load_cases(), check)
    cases = [c for c in cases if not args.only or c["id"] in args.only]
    passed = 0
    records = []
    with (args.experiment_dir / args.out).open("w") as out:
        for case in cases:
            model = TimedModel(LlamaServerModel(base_url=args.base_url))
            run = await run_case(case, model, DEFAULT_DB_PATH, system_prompt(), args.max_steps)
            failures = checker(case, run)
            passed += not failures
            record = run.to_json()
            record["messages"] = record["messages"][1:]  # same system prompt everywhere
            record = {**record, "failures": failures, "model_calls": model.calls, "meta": meta}
            records.append(record)
            out.write(json.dumps(record) + "\n")
            out.flush()
            seconds = sum(t.seconds for t in run.turns)
            print(f"{case['id']}: {seconds:6.1f}s {'PASS' if not failures else failures}")
            for t in run.turns:
                print(f"    -> {' '.join((t.answer or '').split())[:200]!r}")
            await model.inner.aclose()
    print(f"\n{passed}/{len(cases)} cases pass the deterministic checks")

    if args.mlflow:
        from reliable_tool_agent.tracking import code_version, fingerprint, log_run, trace_metrics

        path = args.experiment_dir / args.out
        log_run(
            "agent-eval",
            args.run_name or f"{args.experiment_dir.name}/{args.out}",
            params={
                "case_set": args.set,
                "only": " ".join(args.only) if args.only else "all",
                "agent_model": Path(meta["model_path"] or "unknown").name,
                "server_build": meta["build"],
                "n_ctx": meta["n_ctx"],
                "max_steps": args.max_steps,
                "system_prompt_sha": fingerprint(system_prompt()),
                "git_commit": code_version(),
                "file": str(path),
            },
            metrics=trace_metrics(records),
            artifacts=[path],
        )
        print("recorded in MLflow (experiment agent-eval)")


if __name__ == "__main__":
    asyncio.run(main())
