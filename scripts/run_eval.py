"""Run an evaluation set with a real local model and save one JSON line per case.

Needs a running llama-server (see llama_server.py):

    uv run python scripts/run_eval.py experiments/<dir> --set tasks     # reference tasks
    uv run python scripts/run_eval.py experiments/<dir> --set attacks   # attack cases

or a hosted OpenAI-compatible model (the key is read from GROQ_API_KEY, in the environment
or in .env):

    uv run python scripts/run_eval.py experiments/<dir> --set tasks \
        --base-url https://api.groq.com/openai/v1 --model qwen/qwen3.8-27b

Add --mlflow to record the run in the local MLflow (experiment agent-eval), and --trace to
trace every turn (experiment agent-traces, one session per case).

The deterministic checks are applied here; answers are also read by hand (and, later, by
a judge) in the experiment's write-up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import time
from pathlib import Path

import httpx

from reliable_tool_agent.db import DEFAULT_DB_PATH
from reliable_tool_agent.eval_runner import check, check_task, load_cases, load_tasks, run_case
from reliable_tool_agent.llama_server import LlamaServerModel
from reliable_tool_agent.prompts import system_prompt

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path = ROOT / ".env") -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip())


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


def make_model(base_url: str, name: str | None, api_key: str | None) -> LlamaServerModel:
    if name is None:
        return LlamaServerModel(base_url=base_url)
    # Same settings as the deployed API (api.app_from_env): retries for rate limits.
    return LlamaServerModel(
        base_url=base_url, model=name, thinking=None, api_key=api_key, max_retries=8
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument(
        "--model", help="hosted model name (key from GROQ_API_KEY); default: local llama-server"
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--set", choices=["tasks", "attacks"], required=True)
    parser.add_argument("--only", nargs="*", help="case ids to run (default: all)")
    parser.add_argument("--out", default="traces.jsonl")
    parser.add_argument("--mlflow", action="store_true", help="record the run in MLflow")
    parser.add_argument("--run-name", help="MLflow run name (default: <dir>/<out>)")
    parser.add_argument(
        "--trace", action="store_true", help="trace every turn in MLflow (agent-traces)"
    )
    args = parser.parse_args()
    logging.getLogger("mcp").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.model:  # hosted: no /props, the model name is all we know
        load_env()
        api_key = os.environ["GROQ_API_KEY"]
        meta = {"model_path": args.model, "n_ctx": None, "build": args.base_url, "machine": None}
    else:
        api_key = None
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
    if args.trace:
        from reliable_tool_agent.tracing import enable_tracing

        enable_tracing()
    batch = time.strftime("%Y%m%d-%H%M%S")  # names this run's sessions in MLflow
    with (args.experiment_dir / args.out).open("w") as out:
        for case in cases:
            model = TimedModel(make_model(args.base_url, args.model, api_key))
            session = f"eval-{batch}-{case['id']}" if args.trace else None
            run = await run_case(
                case, model, DEFAULT_DB_PATH, system_prompt(), args.max_steps, session
            )
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
    if args.trace:
        from reliable_tool_agent.tracing import flush_traces

        flush_traces()
        print(f"traced in MLflow: experiment agent-traces, sessions eval-{batch}-<case id>")

    if args.mlflow:
        from reliable_tool_agent.tracking import code_version, fingerprint, log_run, trace_metrics

        path = args.experiment_dir / args.out
        log_run(
            "agent-eval",
            args.run_name or f"{args.experiment_dir.name}/{args.out}",
            params={
                "case_set": args.set,
                "only": " ".join(args.only) if args.only else "all",
                "agent_model": args.model or Path(meta["model_path"] or "unknown").name,
                "server_build": meta["build"],
                "n_ctx": meta["n_ctx"],
                "max_steps": args.max_steps,
                "system_prompt_sha": fingerprint(system_prompt()),
                "git_commit": code_version(),
                "file": str(path),
                "trace_sessions": f"eval-{batch}-<case id>" if args.trace else "not traced",
            },
            metrics=trace_metrics(records),
            artifacts=[path],
        )
        print("recorded in MLflow (experiment agent-eval)")


if __name__ == "__main__":
    asyncio.run(main())
