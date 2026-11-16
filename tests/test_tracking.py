"""Evaluation metrics for MLflow, and one run written to a throw-away tracking database."""

import pytest

from reliable_tool_agent.tracking import judgment_metrics, log_run, trace_metrics


def trace(case_id, failures, seconds, tokens):
    return {
        "case_id": case_id,
        "failures": failures,
        "turns": [{"seconds": s} for s in seconds],
        "model_calls": [{"prompt_tokens": t, "completion_tokens": 10} for t in tokens],
    }


RUNS = [
    trace("a", [], [10.0, 5.0], [100, 200]),
    trace("b", ["no refund"], [30.0], [300]),
    trace("c", [], [2.0], [50]),
]


def test_trace_metrics_count_the_state_check_and_the_cost():
    m = trace_metrics(RUNS)
    assert m["state_check_pass"] == 2
    assert m["state_check_rate"] == pytest.approx(2 / 3)
    assert m["seconds_per_case_median"] == 15.0  # per case, all turns added
    assert m["tokens_per_case"] == pytest.approx(((110 + 210) + 310 + 60) / 3)
    assert "words_pass_by_hand" not in m  # no labels, no hand score


def test_hand_labels_give_a_words_score_apart_from_the_state_check():
    m = trace_metrics(RUNS, {"a": "pass", "b": "pass", "c": "fail"})
    assert m["words_pass_by_hand"] == 2
    assert m["state_check_pass"] == 2  # b fails the state, c the words: different cases


def test_judgment_metrics_report_failures_caught_and_passes_accepted_apart():
    rows = [
        {"label": "fail", "judge": "fail", "calls": [{"prompt_tokens": 5, "completion_tokens": 1}]},
        {"label": "fail", "judge": "pass", "calls": []},
        {"label": "pass", "judge": "pass", "calls": []},
        {"label": "pass", "judge": "error", "calls": []},
    ]
    m = judgment_metrics(rows)
    assert (m["failures_caught"], m["failures_labelled"]) == (1, 2)
    assert (m["passes_accepted"], m["passes_labelled"]) == (1, 2)
    assert m["agreement"] == 0.5
    assert m["judge_errors"] == 1
    assert m["judge_tokens"] == 6


def test_a_run_is_recorded_with_its_params_metrics_file_and_date(tmp_path):
    mlflow = pytest.importorskip("mlflow")  # local-only dependency group, not in CI
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    traces = tmp_path / "traces.jsonl"
    traces.write_text("{}\n")
    run_id = log_run(
        "agent-eval",
        "04 reference tasks",
        params={"git_commit": "6897fd2"},
        metrics={"state_check_pass": 26},
        artifacts=[traces],
        tracking_uri=uri,
        artifact_root=tmp_path / "artifacts",
        start_time_ms=1_794_000_000_000,
    )
    client = mlflow.MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)
    assert run.info.run_name == "04 reference tasks"
    assert run.info.start_time == 1_794_000_000_000
    assert run.info.status == "FINISHED"
    assert run.data.params == {"git_commit": "6897fd2"}
    assert run.data.metrics == {"state_check_pass": 26.0}
    assert [f.path for f in client.list_artifacts(run_id)] == ["traces.jsonl"]
