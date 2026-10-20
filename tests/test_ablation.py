"""Ablation table and consecutive-row significance wiring (eval/ablation.py, D84/D92).

The pure helpers are tested with fabricated evaluate()-shaped results so no index is needed.
"""
import pytest

from eval import ablation


def make_result(ndcg_overall, per_query, **overall):
    g = {"n_queries": len(per_query), "n_zero_relevant": 0, "ndcg@10": ndcg_overall,
         "mrr": 0.9, "p@5": 0.6, "recall@10": 0.5, "unjudged@10": 0.0, "excluded@10": 0.1,
         "src_types@10": 2.0, "places@10": 4.0}
    g.update(overall)
    return {"groups": {"overall": g}, "per_query": per_query}


ROW_A = make_result(0.60, [{"id": "q1", "ndcg@10": 0.6}, {"id": "q2", "ndcg@10": 0.5},
                           {"id": "q3", "ndcg@10": 0.7}])
ROW_B = make_result(0.72, [{"id": "q1", "ndcg@10": 0.8}, {"id": "q2", "ndcg@10": 0.7},
                           {"id": "q3", "ndcg@10": 0.9}])


def test_headline_row_pulls_the_overall_group():
    row = ablation.headline_row("row2", ROW_B)
    assert row["row"] == "row2" and row["n"] == 3
    assert row["ndcg@10"] == 0.72 and row["unjudged@10"] == 0.0


def test_ndcg_by_query_keeps_none():
    res = make_result(0.5, [{"id": "q1", "ndcg@10": 0.5}, {"id": "q2", "ndcg@10": None}])
    assert ablation.ndcg_by_query(res) == {"q1": 0.5, "q2": None}


def test_significance_rows_compares_consecutive_pairs():
    sig = ablation.significance_rows([("A", ROW_A), ("B", ROW_B)])
    assert len(sig) == 1
    assert sig[0]["comparison"] == "A -> B"
    assert sig[0]["delta_ndcg@10"] == pytest.approx(0.2)  # each query improved by 0.2
    assert 0.0 <= sig[0]["p"] <= 1.0 and 0.0 <= sig[0]["p_holm"] <= 1.0


def test_significance_rows_empty_for_a_single_row():
    assert ablation.significance_rows([("A", ROW_A)]) == []


def test_markdown_has_rows_and_the_caveat():
    table = [ablation.headline_row("A", ROW_A), ablation.headline_row("B", ROW_B)]
    sig = ablation.significance_rows([("A", ROW_A), ("B", ROW_B)])
    md = ablation.to_markdown(table, sig, "dev")
    assert "A -> B" in md and "Holm" in md
    assert "round 2" in md and "descriptive" in md


def test_csv_header_and_blank_for_none():
    res = make_result(None, [{"id": "q1", "ndcg@10": None}], ndcg="x")
    csv = ablation.to_csv([ablation.headline_row("A", res)])
    header, row = csv.strip().split("\n")
    assert header.startswith("row,n,0rel,ndcg@10")
    assert ",," in row  # the None nDCG is an empty field
