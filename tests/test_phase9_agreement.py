"""Agreement between two scorers (eval/phase9/agreement.py, D124), against hand-computed values."""
import pytest

from eval.phase9 import agreement as A


def test_exact_and_within_one():
    pairs = [(1, 1), (2, 3), (5, 3), (4, 4)]
    assert A.exact(pairs) == 0.5
    assert A.within_one(pairs) == 0.75
    assert A.exact([]) is None and A.within_one([]) is None


def test_binary_kappa_hand_computed():
    # 3 both-yes, 3 both-no, 1 and 1 disagreements: po = 6/8, pe = 0.5, kappa = 0.5
    a = [1, 1, 0, 0, 1, 0, 1, 0]
    b = [1, 0, 0, 0, 1, 1, 1, 0]
    assert A.binary_kappa(list(zip(a, b, strict=True))) == pytest.approx(0.5)


def test_quadratic_weighted_kappa_hand_computed():
    assert A.weighted_kappa([(3, 3), (1, 1), (5, 5)]) == pytest.approx(1.0)
    # extremes swapped half the time: po = pe = 0.5 -> kappa 0
    assert A.weighted_kappa([(1, 1), (5, 5), (1, 5), (5, 1)]) == pytest.approx(0.0)
    # adjacent swap: w(d=1) = 1 - (1/4)^2 = 0.9375, po = 0.9375, pe = 0.96875 -> kappa = -1
    assert A.weighted_kappa([(1, 2), (2, 1)]) == pytest.approx(-1.0)


def test_kappa_is_undefined_when_both_scorers_use_one_score():
    assert A.weighted_kappa([(4, 4), (4, 4), (4, 4)]) is None
    assert A.weighted_kappa([]) is None
    assert A.binary_kappa([(1, 1), (1, 1)]) is None


def test_linear_weights_differ_from_quadratic_for_a_two_step_gap():
    pairs = [(1, 3), (3, 1), (2, 2), (4, 4), (5, 5)]
    quad = A.weighted_kappa(pairs, weights="quadratic")
    lin = A.weighted_kappa(pairs, weights="linear")
    assert quad is not None and lin is not None and quad > lin


def test_agreement_uses_only_the_keys_both_scored():
    a = {"r1": 3, "r2": 4, "r3": 5}
    b = {"r2": 4, "r3": 4, "r4": 1}
    rep = A.agreement(a, b)
    assert rep["n"] == 2 and rep["exact"] == 0.5 and rep["within_one"] == 1.0


def test_agreement_by_dimension_pools_every_dimension():
    a = {"x": {"feasibility": 5, "groundedness": 3}, "y": {"feasibility": 4, "groundedness": 2}}
    b = {"x": {"feasibility": 5, "groundedness": 4}, "y": {"feasibility": 4, "groundedness": 2}}
    rep = A.agreement_by_dimension(a, b, ["feasibility", "groundedness"])
    assert rep["feasibility"]["exact"] == 1.0 and rep["groundedness"]["exact"] == 0.5
    assert rep["pooled"]["n"] == 4 and rep["pooled"]["exact"] == 0.75
