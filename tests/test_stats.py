"""Uncertainty of evaluation results.

Reference bounds were computed independently, as the roots of (p_hat - p)^2 = z^2 p(1-p)/n
found numerically with scipy, not with the closed form used by the module.
"""

from __future__ import annotations

import pytest

from reliable_tool_agent.stats import (
    PairedTable,
    mcnemar_exact,
    paired_table,
    wilson_interval,
)


@pytest.mark.parametrize(
    ("successes", "n", "rate", "low", "high"),
    [
        (26, 28, 0.929, 0.7735, 0.9802),  # experiment 04, final database state
        (24, 28, 0.857, 0.6851, 0.9430),  # experiment 06, by hand
        (2, 7, 0.286, 0.0822, 0.6411),  # experiment 03, indirect injections before
    ],
)
def test_wilson_interval_matches_reference_values(successes, n, rate, low, high):
    result = wilson_interval(successes, n)
    assert result.rate == pytest.approx(rate, abs=5e-4)
    assert result.low == pytest.approx(low, abs=1e-4)
    assert result.high == pytest.approx(high, abs=1e-4)


def test_a_perfect_score_still_has_an_interval():
    """28/28 is not 'always right': the lower bound is n / (n + z^2)."""
    result = wilson_interval(28, 28)
    assert result.high == 1.0
    assert result.low == pytest.approx(28 / (28 + 1.959963984540054**2))
    assert result.low < 0.88


def test_wilson_interval_rejects_impossible_counts():
    for successes, n in [(-1, 5), (6, 5), (0, 0)]:
        with pytest.raises(ValueError):
            wilson_interval(successes, n)


def test_paired_table_counts_each_kind_of_pair():
    before = {"t1": True, "t2": True, "t3": False, "t4": False, "t5": False}
    after = {"t1": True, "t2": False, "t3": True, "t4": True, "t5": False}
    assert paired_table(before, after) == PairedTable(both_pass=1, only_a=1, only_b=2, both_fail=1)


def test_paired_table_matches_tasks_by_id_not_by_order():
    before = {"t1": True, "t2": False}
    after = {"t2": True, "t1": True}  # same tasks, another order
    assert paired_table(before, after) == PairedTable(1, 0, 1, 0)


def test_paired_table_refuses_runs_on_different_tasks():
    with pytest.raises(ValueError, match="t3"):
        paired_table({"t1": True, "t2": True}, {"t1": True, "t3": True})


def test_indirect_injections_before_and_after():
    """Experiment 03: 2/7 then 7/7 on the same cases; the five that changed all improved."""
    before = {f"ind-0{i}": i in (3, 7) for i in range(1, 8)}
    after = {case: True for case in before}
    table = paired_table(before, after)
    assert table == PairedTable(both_pass=2, only_a=0, only_b=5, both_fail=0)
    assert mcnemar_exact(table) == pytest.approx(0.0625)  # 2 x (1/2)^5


def test_mcnemar_is_symmetric_and_neutral_without_discordant_pairs():
    assert mcnemar_exact(PairedTable(3, 4, 1, 2)) == mcnemar_exact(PairedTable(3, 1, 4, 2))
    assert mcnemar_exact(PairedTable(10, 0, 0, 3)) == 1.0
    assert mcnemar_exact(PairedTable(0, 3, 3, 0)) == 1.0  # capped, not 2 x 0.66
