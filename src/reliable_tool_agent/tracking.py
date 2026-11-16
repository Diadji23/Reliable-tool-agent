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

import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
DEFAULT_ARTIFACTS = REPO_ROOT / "mlruns"  # files attached to runs (both git-ignored)


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
    """The judge against the hand labels (rows written by scripts/judge_run.py)."""
    fails = [r for r in rows if r["label"] == "fail"]
    passes = [r for r in rows if r["label"] == "pass"]
    caught = sum(r["judge"] == "fail" for r in fails)
    accepted = sum(r["judge"] == "pass" for r in passes)
    metrics = {
        "conversations": len(rows),
        "agreement": (caught + accepted) / len(rows),
        "failures_caught": caught,
        "failures_labelled": len(fails),
        "passes_accepted": accepted,
        "passes_labelled": len(passes),
        "judge_errors": sum(r["judge"] == "error" for r in rows),
        "judge_tokens": sum(
            c.get("prompt_tokens", 0) + c.get("completion_tokens", 0)
            for r in rows
            for c in r["calls"]
        ),
    }
    if fails:
        metrics["failures_caught_rate"] = caught / len(fails)
    if passes:
        metrics["passes_accepted_rate"] = accepted / len(passes)
    return metrics


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
