"""Record the runs of experiments 03-05 in MLflow, from their committed files.

    uv run python scripts/backfill_mlflow.py
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # then open http://127.0.0.1:5000

No model is called: the numbers come from the traces and judgments already in the repo.
A file already recorded (same run name in the same experiment) is skipped, so the script
can be run again after adding an experiment.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

from reliable_tool_agent.tracking import (
    DEFAULT_TRACKING_URI,
    judgment_metrics,
    log_run,
    trace_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = json.loads((ROOT / "evaluation" / "judge_labels.json").read_text())["labels"]

# (file, run name, case set)
TRACES = [
    ("experiments/03-guardrail-baseline/traces.jsonl", "03 attacks, before the fix", "attacks"),
    (
        "experiments/03-guardrail-baseline/traces_after.jsonl",
        "03 attacks, after the fix",
        "attacks",
    ),
    ("experiments/04-reference-tasks/traces.jsonl", "04 reference tasks", "tasks"),
    (
        "experiments/04-reference-tasks/traces_after.jsonl",
        "04 refund tasks, simulated customer",
        "tasks",
    ),
]
# (file, run name, split, judge prompt version)
JUDGMENTS = [
    ("judgments_tuning_v1.jsonl", "05 tuning set, prompt v1", "tuning", "v1"),
    ("judgments_tuning_v2.jsonl", "05 tuning set, prompt v2", "tuning", "v2"),
    ("judgments_test.jsonl", "05 test set, frozen prompt v3", "test", "v3"),
    (
        "judgments_test_after_fix.jsonl",
        "05 test set, reviewer decisions shown (not held out)",
        "test",
        "v3 + reviewer decisions",
    ),
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def added_in(path: str) -> tuple[str, int]:
    """The commit that added the file, and its time in ms: when the run was made."""
    out = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%h %cI", "--", path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return out[-2], int(datetime.fromisoformat(out[-1]).timestamp() * 1000)


def already_recorded(experiment: str, run_name: str) -> bool:
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=DEFAULT_TRACKING_URI)
    found = client.get_experiment_by_name(experiment)
    if not found:
        return False
    runs = client.search_runs(
        [found.experiment_id], filter_string=f"attributes.run_name = '{run_name}'"
    )
    return bool(runs)


def main() -> None:
    for path, name, case_set in TRACES:
        if already_recorded("agent-eval", name):
            print(f"skip   {name}")
            continue
        runs = read_jsonl(ROOT / path)
        labels = {lab["case_id"]: lab["verdict"] for lab in LABELS if lab["trace"] == path}
        meta = runs[0].get("meta", {})
        commit, when = added_in(path)
        log_run(
            "agent-eval",
            name,
            params={
                "case_set": case_set,
                "agent_model": Path(meta.get("model_path", "unknown")).name,
                "server_build": meta.get("build", "unknown"),
                "n_ctx": meta.get("n_ctx", "unknown"),
                "git_commit": commit,
                "file": path,
            },
            metrics=trace_metrics(runs, labels or None),
            artifacts=[ROOT / path],
            start_time_ms=when,
        )
        print(f"logged {name}")

    for file, name, split, version in JUDGMENTS:
        if already_recorded("judge-calibration", name):
            print(f"skip   {name}")
            continue
        path = f"experiments/05-judge-calibration/{file}"
        rows = read_jsonl(ROOT / path)
        commit, when = added_in(path)
        log_run(
            "judge-calibration",
            name,
            params={
                "judge_model": rows[0]["model"],
                "judge_prompt": version,
                "split": split,
                "traces_judged": ", ".join(sorted({r["trace"] for r in rows})),
                "git_commit": commit,
                "file": path,
            },
            metrics=judgment_metrics(rows),
            artifacts=[ROOT / path],
            start_time_ms=when,
        )
        print(f"logged {name}")


if __name__ == "__main__":
    main()
