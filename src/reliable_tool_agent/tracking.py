"""Record evaluation runs in MLflow (local tracking, optional dependency group `tracking`).

The experiment write-ups stay the reference: they say what was expected and what it means.
MLflow keeps the numbers of every run side by side, with the files that produced them,
so that two runs can be compared without re-reading two READMEs.

Two MLflow experiments:
- `agent-eval`: one run per traces file (the agent on a case set): state check, words read
  by hand when labels exist, tokens, latency;
- `judge-calibration`: one run per judgments file: the judge against the hand labels.

The metrics are computed by plain functions (`trace_metrics`, `judgment_metrics`), tested
without MLflow; only `log_run` touches it.
"""

from __future__ import annotations

import hashlib
import os
import statistics
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# MLFLOW_TRACKING_URI, if set, sends runs elsewhere (a throw-away database for a trial run).
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
# Files attached to runs. Both git-ignored; MLFLOW_ARTIFACTS moves them with a trial database.
DEFAULT_ARTIFACTS = Path(os.environ.get("MLFLOW_ARTIFACTS", REPO_ROOT / "mlruns"))


def trace_metrics(runs: list[dict], labels: dict[str, str] | None = None) -> dict[str, float]:
    """Metrics of one traces file. `labels`: case id -> hand verdict on the words, if any."""
    seconds = [sum(t["seconds"] for t in r["turns"]) for r in runs]
    tokens = [
        sum(c.get("prompt_tokens", 0) + c.get("completion_tokens", 0) for c in r["model_calls"])
        for r in runs
    ]
    metrics = {
        "cases": len(runs),
        "state_check_pass": sum(not r["failures"] for r in runs),
        "state_check_rate": sum(not r["failures"] for r in runs) / len(runs),
        "model_calls_per_case": statistics.mean(len(r["model_calls"]) for r in runs),
        "tokens_per_case": statistics.mean(tokens),
        "seconds_per_case_median": statistics.median(seconds),
    }
    if labels:
        words = [labels[r["case_id"]] == "pass" for r in runs]
        metrics["words_pass_by_hand"] = sum(words)
        metrics["words_rate_by_hand"] = sum(words) / len(runs)
    return metrics


def judgment_metrics(rows: list[dict]) -> dict[str, float]:
    """The judge's verdicts (rows written by scripts/judge_run.py).

    Always: how many conversations the judge passes. When rows carry a hand label
    (`label` not None), also the judge against the labels, on the labelled rows only.
    """
    metrics: dict[str, float] = {
        "conversations": len(rows),
        "judge_pass": sum(r["judge"] == "pass" for r in rows),
        "judge_pass_rate": sum(r["judge"] == "pass" for r in rows) / len(rows),
        "judge_errors": sum(r["judge"] == "error" for r in rows),
        "judge_tokens": sum(
            c.get("prompt_tokens", 0) + c.get("completion_tokens", 0)
            for r in rows
            for c in r["calls"]
        ),
    }
    fails = [r for r in rows if r.get("label") == "fail"]
    passes = [r for r in rows if r.get("label") == "pass"]
    if not fails and not passes:
        return metrics
    caught = sum(r["judge"] == "fail" for r in fails)
    accepted = sum(r["judge"] == "pass" for r in passes)
    metrics |= {
        "agreement": (caught + accepted) / (len(fails) + len(passes)),
        "failures_caught": caught,
        "failures_labelled": len(fails),
        "passes_accepted": accepted,
        "passes_labelled": len(passes),
    }
    if fails:
        metrics["failures_caught_rate"] = caught / len(fails)
    if passes:
        metrics["passes_accepted_rate"] = accepted / len(passes)
    return metrics


def code_version() -> str:
    """The commit the code ran from, marked "-dirty" if tracked files had local changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    commit = git("rev-parse", "--short", "HEAD")
    return commit + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def fingerprint(text: str) -> str:
    """A short hash of a prompt: two runs with the same value used the same prompt."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def log_run(
    experiment: str,
    run_name: str,
    params: dict[str, object],
    metrics: dict[str, float],
    artifacts: list[Path],
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: Path = DEFAULT_ARTIFACTS,
    start_time_ms: int | None = None,
) -> str:
    """Create one finished MLflow run and return its id.

    `start_time_ms` dates the run when it is recorded after the fact (backfill): the
    commit time of the traces, not the time of the import.
    """
    from mlflow import MlflowClient  # optional dependency: imported only when used
    from mlflow.entities import Metric, Param

    client = MlflowClient(tracking_uri=tracking_uri)
    found = client.get_experiment_by_name(experiment)
    if found:
        experiment_id = found.experiment_id
    else:
        location = (artifact_root / experiment).as_uri()
        experiment_id = client.create_experiment(experiment, artifact_location=location)
    # MLflow 3 opens an experiment on its GenAI tracing page, empty here: these are
    # evaluation runs (params, metrics, files), so ask for the classic runs table.
    client.set_experiment_tag(experiment_id, "mlflow.experimentKind", "custom_model_development")
    run = client.create_run(experiment_id, run_name=run_name, start_time=start_time_ms)
    run_id = run.info.run_id
    timestamp = start_time_ms or run.info.start_time
    client.log_batch(
        run_id,
        metrics=[Metric(k, float(v), timestamp, 0) for k, v in metrics.items()],
        params=[Param(k, str(v)) for k, v in params.items()],
    )
    for path in artifacts:
        client.log_artifact(run_id, str(path))
    client.set_terminated(run_id, end_time=timestamp)
    return run_id
