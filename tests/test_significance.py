"""Paired significance tests (eval/significance.py, D92)."""
import itertools

import pytest

from eval import significance as S


def brute_force_p(diffs):
    """Two-sided sign-flip p by explicit enumeration, as an independent reference."""
    n = len(diffs)
    obs = abs(sum(diffs) / n)
    ge = 0
    for signs in itertools.product((1, -1), repeat=n):
        s = sum(sg * d for sg, d in zip(signs, diffs))
        if abs(s) / n >= obs - 1e-12:
            ge += 1
    return ge / (2 ** n)


def test_paired_differences_pairs_and_drops_none():
    a = {"q1": 0.5, "q2": None, "q3": 0.2, "q4": 0.9}
    b = {"q1": 0.7, "q2": 0.4, "q3": None, "q4": 0.9}
    assert S.paired_differences(a, b) == [pytest.approx(0.2), pytest.approx(0.0)]  # q1, q4


def test_exact_matches_independent_enumeration():
    diffs = [0.3, -0.1, 0.2, 0.05, -0.02, 0.4]
    out = S.randomization_test(diffs)
    assert out["method"] == "exact"
    assert out["p"] == pytest.approx(brute_force_p(diffs))


def test_all_positive_differences_give_the_smallest_possible_p():
    diffs = [0.1, 0.2, 0.3]
    out = S.randomization_test(diffs)
    # only the all-plus pattern reaches the observed |mean|; two-sided also counts all-minus
    assert out["p"] == pytest.approx(2 / 2 ** 3)


def test_zero_mean_difference_is_not_significant():
    out = S.randomization_test([0.2, -0.2, 0.1, -0.1])
    assert out["p"] == pytest.approx(1.0)


def test_monte_carlo_for_large_n_is_close_to_exact():
    diffs = [0.1, -0.05, 0.2, 0.0, 0.3, -0.1, 0.15, 0.05]
    exact = S.randomization_test(diffs)["p"]
    mc = S.randomization_test(diffs, exact_max_n=4, n_mc=20000, seed=1)
    assert mc["method"] == "monte_carlo"
    assert mc["p"] == pytest.approx(exact, abs=0.03)


def test_monte_carlo_is_seeded_and_reproducible():
    diffs = [0.1, -0.05, 0.2, 0.0, 0.3, -0.1]
    a = S.randomization_test(diffs, exact_max_n=1, n_mc=5000, seed=7)
    b = S.randomization_test(diffs, exact_max_n=1, n_mc=5000, seed=7)
    assert a["p"] == b["p"]


def test_bootstrap_ci_brackets_the_mean_and_is_ordered():
    diffs = [0.1, 0.2, 0.15, 0.05, 0.25]
    lo, hi = S.bootstrap_ci(diffs, n=2000, seed=3)
    assert lo <= sum(diffs) / len(diffs) <= hi


def test_bootstrap_ci_empty():
    assert S.bootstrap_ci([]) == (None, None)


def test_holm_step_down_is_monotone_and_capped():
    adj = S.holm([0.01, 0.04, 0.03])
    # sorted 0.01,0.03,0.04 -> 0.03, 0.06, 0.04 -> monotone 0.03, 0.06, 0.06
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)  # 0.04 was raised to the running max 0.06
    assert adj[1] == pytest.approx(0.06)
    assert all(0.0 <= p <= 1.0 for p in adj)


def test_holm_single_pvalue_unchanged():
    assert S.holm([0.2]) == [pytest.approx(0.2)]


def test_ttest_crosscheck_runs_or_returns_none():
    p = S.ttest_rel_p([0.1, 0.2, 0.15, 0.05, 0.25])
    assert p is None or 0.0 <= p <= 1.0
    assert S.ttest_rel_p([0.1]) is None  # n < 2


def test_compare_and_holm_over():
    a = {"q1": 0.5, "q2": 0.5, "q3": 0.5}
    b = {"q1": 0.7, "q2": 0.6, "q3": 0.8}
    comp = S.compare("a->b", a, b)  # diffs 0.2, 0.1, 0.3 -> mean 0.2
    assert comp.n == 3 and comp.mean_diff == pytest.approx(0.2)
    paired = S.holm_over([comp])
    assert paired[0][1] == pytest.approx(comp.p)
