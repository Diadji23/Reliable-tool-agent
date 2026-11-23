"""Uncertainty of evaluation results: how much a score on a few dozen tasks can be trusted.

Following "Adding Error Bars to Evals" (Miller, Anthropic, 2024): the tasks are a sample
of the questions the agent could get, so a score comes with an interval, and two versions
run on the same tasks are compared task by task (paired), not by their two scores.

Standard library only: scipy is not a dependency of the package (CI does not install it).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

Z_95 = 1.959963984540054  # two-sided 95 % quantile of the standard normal


@dataclass(frozen=True)
class Proportion:
    successes: int
    n: int
    low: float
    high: float

    @property
    def rate(self) -> float:
        return self.successes / self.n


def wilson_interval(successes: int, n: int, z: float = Z_95) -> Proportion:
    """Wilson score interval for `successes` out of `n` (95 % by default).

    Preferred to the normal "p ± z·SE" interval for small n and rates near 0 or 1: it
    never leaves [0, 1] and does not shrink to a point at 28/28.
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n and n > 0, got {successes}/{n}")
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    # At 0/n and n/n one bound is exactly 0 or 1; floating point would give 0.9999999999999998.
    low = 0.0 if successes == 0 else max(0.0, centre - half_width)
    high = 1.0 if successes == n else min(1.0, centre + half_width)
    return Proportion(successes, n, low, high)


@dataclass(frozen=True)
class PairedTable:
    """Two versions run on the same tasks, counted pair by pair.

    `both_pass` and `both_fail` say nothing about the difference; only the discordant
    pairs do: `only_a` (a passes, b fails) and `only_b` (b passes, a fails).
    """

    both_pass: int
    only_a: int
    only_b: int
    both_fail: int

    @property
    def discordant(self) -> int:
        return self.only_a + self.only_b


def paired_table(a: Mapping[str, bool], b: Mapping[str, bool]) -> PairedTable:
    """Cross-tabulate per-task results of two versions, matched by task id.

    Keyed by task id rather than by position, so a reordered or incomplete run cannot be
    paired with the wrong task silently: both runs must cover exactly the same tasks.
    """
    if a.keys() != b.keys():
        missing = sorted(a.keys() ^ b.keys())
        raise ValueError(f"the two runs do not cover the same tasks: {missing}")
    counts = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for task in a:
        counts[(bool(a[task]), bool(b[task]))] += 1
    return PairedTable(
        both_pass=counts[(True, True)],
        only_a=counts[(True, False)],
        only_b=counts[(False, True)],
        both_fail=counts[(False, False)],
    )


def mcnemar_exact(table: PairedTable) -> float:
    """Two-sided exact McNemar p-value: is one version better on the same tasks?

    Under "no difference", each discordant pair is equally likely to favour a or b, so
    the smaller count follows Binomial(discordant, 1/2). With no discordant pair there is
    no evidence either way: p = 1.
    """
    n = table.discordant
    if n == 0:
        return 1.0
    k = min(table.only_a, table.only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
