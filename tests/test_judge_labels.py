"""The hand labels the judge is calibrated against point at real conversations."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = json.loads((ROOT / "evaluation/judge_labels.json").read_text())["labels"]


def test_every_label_points_at_a_conversation_of_its_trace_file():
    for trace in {label["trace"] for label in LABELS}:
        lines = (ROOT / trace).read_text().splitlines()
        case_ids = [json.loads(line)["case_id"] for line in lines]
        labelled = [label["case_id"] for label in LABELS if label["trace"] == trace]
        assert labelled == case_ids, trace  # each conversation once, in file order


def test_verdicts_are_binary_and_every_failure_says_why():
    assert {label["verdict"] for label in LABELS} == {"pass", "fail"}
    assert all(label.get("reason") for label in LABELS if label["verdict"] == "fail")
