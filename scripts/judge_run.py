"""Judge recorded conversations with a hosted model and compare with the hand labels.

    uv run python scripts/judge_run.py experiments/05-judge-calibration \\
        --trace experiments/03-guardrail-baseline/traces.jsonl \\
        --trace experiments/03-guardrail-baseline/traces_after.jsonl

Needs GROQ_API_KEY in the environment (or in .env). Writes one JSON line per conversation
(label, verdict, critique, tokens) to <experiment_dir>/<out>, then prints how many failures
the judge caught and how many passes it accepted, with every disagreement. Conversations
without a hand label are judged too (for a new run of the agent) and only counted.
Add --mlflow to record the run: experiment judge-calibration if some conversations are
labelled, judge otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from reliable_tool_agent.eval_runner import load_cases, load_tasks
from reliable_tool_agent.judge import (
    JUDGE_SYSTEM,
    JUDGE_USER,
    JudgeOutputError,
    judge,
    load_policies,
)
from reliable_tool_agent.llama_server import LlamaServerModel
from reliable_tool_agent.tracking import judgment_metrics

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "evaluation" / "judge_labels.json"


class UsageModel:
    """Wraps a ChatModel and adds up the tokens of its calls."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[dict] = []

    async def complete(self, messages, tools):
        turn = await self.inner.complete(messages, tools)
        self.calls.append(turn.usage or {})
        return turn


def load_env(path: Path = ROOT / ".env") -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip())


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--trace", action="append", required=True, help="traces file to judge")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    parser.add_argument("--out", default="judgments.jsonl")
    parser.add_argument("--only", nargs="*", help="case ids to judge (default: all)")
    parser.add_argument("--mlflow", action="store_true", help="record the run in MLflow")
    parser.add_argument("--run-name", help="MLflow run name (default: <dir>/<out>)")
    args = parser.parse_args()
    load_env()

    expected = {c["id"]: c["expected"] for c in load_cases() + load_tasks()}
    labels = {
        (label["trace"], label["case_id"]): label
        for label in json.loads(LABELS.read_text())["labels"]
    }
    policies = load_policies()
    model = LlamaServerModel(
        base_url=args.base_url,
        model=args.model,
        thinking=None,
        api_key=os.environ["GROQ_API_KEY"],
        extra={"response_format": {"type": "json_object"}, "seed": 0},
        max_retries=8,
    )

    rows = []
    with (args.experiment_dir / args.out).open("w") as out:
        for trace in args.trace:
            for line in (ROOT / trace).read_text().splitlines():
                run = json.loads(line)
                if args.only and run["case_id"] not in args.only:
                    continue
                label = labels.get((trace, run["case_id"]), {}).get("verdict")  # None: unlabelled
                counted = UsageModel(model)
                start = time.perf_counter()
                try:
                    verdict = await judge(
                        counted,
                        expected[run["case_id"]],
                        run["messages"],
                        policies,
                        run["approvals"],
                    )
                    judged, critique = verdict.verdict, verdict.critique
                except JudgeOutputError as error:
                    judged, critique = "error", str(error)
                row = {
                    "trace": trace,
                    "case_id": run["case_id"],
                    "label": label,
                    "judge": judged,
                    "critique": critique,
                    "calls": counted.calls,
                    "seconds": round(time.perf_counter() - start, 2),
                    "model": args.model,
                }
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                mark = "XX" if label and judged != label else "  "
                print(f"{mark} {run['case_id']}: label {label}, judge {judged}")
    await model.aclose()

    m = judgment_metrics(rows)
    print(f"\njudge passes {m['judge_pass']}/{len(rows)}   errors: {m['judge_errors']}", end="")
    print(f"   tokens: {m['judge_tokens']}")
    if "agreement" in m:
        print(
            f"failures caught: {m['failures_caught']}/{m['failures_labelled']}   "
            f"passes accepted: {m['passes_accepted']}/{m['passes_labelled']}"
        )
    for r in rows:
        if r["label"] and r["judge"] != r["label"]:
            print(f"\n[{r['case_id']}] label {r['label']}, judge {r['judge']}\n  {r['critique']}")

    if args.mlflow:
        from reliable_tool_agent.tracking import code_version, fingerprint, log_run

        path = args.experiment_dir / args.out
        log_run(
            "judge-calibration" if "agreement" in m else "judge",
            args.run_name or f"{args.experiment_dir.name}/{args.out}",
            params={
                "judge_model": args.model,
                "judge_prompt_sha": fingerprint(JUDGE_SYSTEM + JUDGE_USER),
                "traces_judged": ", ".join(args.trace),
                "only": " ".join(args.only) if args.only else "all",
                "git_commit": code_version(),
                "file": str(path),
            },
            metrics=m,
            artifacts=[path],
        )
        print("recorded in MLflow")


if __name__ == "__main__":
    asyncio.run(main())
