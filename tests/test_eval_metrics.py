"""Metric correctness against values worked out by hand (D73)."""
import math
from datetime import date

import pytest

from eval import metrics as m

# Q: labels a=2, b=1, c=0 (judged, not relevant), d=2; e is unjudged.
QREL = {"a": 2, "b": 1, "c": 0, "d": 2}


def test_dcg_hand_value():
    # ranking [b, a, c]: gains 1, 2, 0 -> 1/log2(2) + 2/log2(3) + 0
    got = m.ndcg_at_k(["b", "a", "c"], QREL, k=3)
    dcg = 1.0 + 2.0 / math.log2(3)
    ideal = 2.0 + 2.0 / math.log2(3) + 1.0 / math.log2(4)  # ideal gains 2, 2, 1
    assert got == pytest.approx(dcg / ideal)
    assert got == pytest.approx(2.2619 / 3.7619, abs=1e-3)  # = 0.6013


def test_ndcg_perfect_and_cut():
    assert m.ndcg_at_k(["a", "d", "b", "c"], QREL, k=10) == pytest.approx(1.0)
    # k=1: only a (gain 2); ideal at 1 is 2 -> 1.0
    assert m.ndcg_at_k(["a", "x"], QREL, k=1) == pytest.approx(1.0)
    assert m.ndcg_at_k(["b", "a"], QREL, k=1) == pytest.approx(0.5)


def test_ndcg_ideal_uses_all_judged_not_only_retrieved():
    # only b retrieved; ideal still contains a and d (both 2): 1 / (2 + 2/log2(3) + 1/log2(4))
    ideal = 2.0 + 2.0 / math.log2(3) + 1.0 / 2.0
    assert m.ndcg_at_k(["b"], QREL, k=10) == pytest.approx(1.0 / ideal)


def test_unjudged_counts_as_zero():
    assert m.ndcg_at_k(["e", "f"], QREL, k=10) == 0.0
    assert m.precision_at_k(["e", "f", "a"], QREL, k=5) == pytest.approx(1 / 5)


def test_no_relevant_query_returns_none():
    zero = {"a": 0, "b": 0}
    assert m.ndcg_at_k(["a"], zero) is None
    assert m.reciprocal_rank(["a"], zero) is None
    assert m.recall_at_k(["a"], zero, 10) is None
    assert m.precision_at_k(["a"], zero, 5) == 0.0
    assert m.ndcg_at_k(["a"], {}) is None


def test_mrr_first_relevant_rank():
    assert m.reciprocal_rank(["c", "e", "b"], QREL) == pytest.approx(1 / 3)
    assert m.reciprocal_rank(["a"], QREL) == 1.0
    assert m.reciprocal_rank(["c", "e"], QREL) == 0.0
    # threshold 2: b (grade 1) no longer counts, d at rank 2
    assert m.reciprocal_rank(["b", "d"], QREL, threshold=2) == pytest.approx(0.5)


def test_precision_at_5_denominator_is_5():
    assert m.precision_at_k(["a", "b"], QREL, k=5) == pytest.approx(2 / 5)
    assert m.precision_at_k(["a", "b", "c", "d", "e", "x"], QREL, k=5) == pytest.approx(3 / 5)


def test_recall_at_k_uses_all_relevant():
    # relevant (>=1): a, b, d -> 3
    assert m.recall_at_k(["a", "c", "x"], QREL, 10) == pytest.approx(1 / 3)
    assert m.recall_at_k(["a", "b", "d"], QREL, 10) == pytest.approx(1.0)
    assert m.recall_at_k(["c", "x", "d"], QREL, 2) == 0.0
    assert m.recall_at_k(["c", "x", "d"], QREL, 3) == pytest.approx(1 / 3)


def test_unjudged_share():
    assert m.unjudged_at_k(["a", "e", "f", "c"], QREL, 10) == pytest.approx(0.5)
    assert m.unjudged_at_k(["a", "e", "f", "c"], QREL, 2) == pytest.approx(0.5)
    assert m.unjudged_at_k([], QREL, 10) == 0.0


def test_duplicate_chunk_counts_once():
    assert m.dedupe(["a", "a", "b", "a"]) == ["a", "b"]
    assert m.precision_at_k(["a", "a", "a", "a", "a"], QREL, 5) == pytest.approx(1 / 5)


def test_months_before_and_window():
    assert m.months_before(date(2026, 10, 18), 18) == date(2025, 4, 18)
    assert m.months_before(date(2026, 3, 31), 1) == date(2026, 2, 28)
    assert m.months_before(date(2024, 3, 31), 1) == date(2024, 2, 29)


def test_freshness_counts_published_dates_only():
    ref = date(2026, 10, 18)
    rows = [
        {"date_kind": "published", "date": "2025-06-01"},   # in window
        {"date_kind": "published", "date": "2025-04-17"},   # one day before the window
        {"date_kind": "published", "date": "2025-04-18"},   # first day of the window
        {"date_kind": "revised", "date": None},             # revised: never counts
        {"date_kind": "sitemap_lastmod", "date": "2026-01-01"},
    ]
    assert m.freshness_share(rows, ref) == pytest.approx(2 / 5)
    assert m.freshness_share([], ref) == 0.0
    # a revised chunk that carries a recent date still does not count
    assert m.freshness_share([{"date_kind": "revised", "date": "2026-01-01"}], ref) == 0.0


def test_diversity_and_excluded_share():
    rows = [{"source_type": "blog", "place": "Hampi"}, {"source_type": "blog", "place": "Hampi"},
            {"source_type": "reference", "place": "Munnar"}]
    assert m.diversity(rows) == {"source_types": 2, "places": 2}
    assert m.excluded_share(rows, {"Munnar"}) == pytest.approx(1 / 3)
    assert m.excluded_share([], {"Munnar"}) == 0.0


def test_provenance_counts():
    rows = [{"chunk_id": "a", "provenance": "llm"}, {"chunk_id": "c", "provenance": "llm"},
            {"chunk_id": "e", "provenance": "rule_only"}]
    got = m.provenance_counts(rows, QREL)
    assert got["llm"] == {"n": 2, "judged": 2, "grade_sum": 2, "relevant": 1}
    assert got["rule_only"] == {"n": 1}


def test_mean_skips_none_and_counts_them():
    assert m.mean([1.0, None, 0.0, None]) == (0.5, 2)
    assert m.mean([None]) == (None, 1)
    assert m.mean(x for x in [1.0, 3.0]) == (2.0, 0)
