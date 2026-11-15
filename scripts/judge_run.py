"""Judge recorded conversations with a hosted model and compare with the hand labels.

    uv run python scripts/judge_run.py experiments/05-judge-calibration \\
        --trace experiments/03-guardrail-baseline/traces.jsonl \\
        --trace experiments/03-guardrail-baseline/traces_after.jsonl

Needs GROQ_API_KEY in the environment (or in .env). Writes one JSON line per conversation
(label, verdict, critique, tokens) to <experiment_dir>/<out>, then prints how many failures
the judge caught and how many passes it accepted, with every disagreement.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from reliable_tool_agent.eval_runner import load_cases, load_tasks
from reliable_tool_agent.judge import JudgeOutputError, judge, load_policies
from reliable_tool_agent.llama_server import LlamaServerModel

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
                label = labels[(trace, run["case_id"])]
                counted = UsageModel(model)
                start = time.perf_counter()
                try:
                    verdict = await judge(
                        counted, expected[run["case_id"]], run["messages"], policies
                    )
                    judged, critique = verdict.verdict, verdict.critique
                except JudgeOutputError as error:
                    judged, critique = "error", str(error)
                row = {
                    "trace": trace,
                    "case_id": run["case_id"],
                    "label": label["verdict"],
                    "judge": judged,
                    "critique": critique,
                    "calls": counted.calls,
                    "seconds": round(time.perf_counter() - start, 2),
                    "model": args.model,
                }
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                mark = "  " if judged == label["verdict"] else "XX"
                print(f"{mark} {run['case_id']}: label {label['verdict']}, judge {judged}")
    await model.aclose()

    fails = [r for r in rows if r["label"] == "fail"]
    passes = [r for r in rows if r["label"] == "pass"]
    caught = sum(r["judge"] == "fail" for r in fails)
    accepted = sum(r["judge"] == "pass" for r in passes)
    errors = sum(r["judge"] == "error" for r in rows)
    tokens = sum(
        c.get("prompt_tokens", 0) + c.get("completion_tokens", 0) for r in rows for c in r["calls"]
    )
    print(f"\nfailures caught: {caught}/{len(fails)}   passes accepted: {accepted}/{len(passes)}")
    print(f"agreement: {caught + accepted}/{len(rows)}   judge errors: {errors}   tokens: {tokens}")
    for r in rows:
        if r["judge"] != r["label"]:
            print(f"\n[{r['case_id']}] label {r['label']}, judge {r['judge']}\n  {r['critique']}")


if __name__ == "__main__":
    asyncio.run(main())
