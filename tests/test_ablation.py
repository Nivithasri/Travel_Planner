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


def test_condensed_ndcg_ignores_unjudged_chunks():
    # ranking: u1 (unjudged), a (grade 2), u2 (unjudged), b (grade 1)
    res = {"groups": {"overall": {}}, "per_query": [{"id": "q", "ranked": ["u1", "a", "u2", "b"]}]}
    qrels = {"q": {"a": 2, "b": 1, "c": 2}}
    got = ablation.condensed_ndcg(res, qrels)
    # judged-only list [a, b]: dcg = 2 + 1/log2(3); ideal over all judged = [2, 2, 1]
    import math
    want = (2 + 1 / math.log2(3)) / (2 + 2 / math.log2(3) + 1 / 2)
    assert got == pytest.approx(want)


def test_condensed_ndcg_skips_queries_without_relevant_chunks():
    res = {"groups": {"overall": {}}, "per_query": [{"id": "q", "ranked": ["a"]}]}
    assert ablation.condensed_ndcg(res, {"q": {"a": 0}}) is None


def test_headline_row_carries_the_condensed_column():
    res = make_result(0.5, [{"id": "q1", "ndcg@10": 0.5, "ranked": ["a"]}])
    assert ablation.headline_row("A", res)["cond_ndcg@10"] is None  # no qrels given
    row = ablation.headline_row("A", res, {"q1": {"a": 2}})
    assert row["cond_ndcg@10"] == pytest.approx(1.0)


def test_run_names_the_table_files_after_the_stem_not_the_last_row(tmp_path, monkeypatch):
    # regression: the row loop variable once shadowed the file-name parameter
    from datetime import date

    from eval import run_eval

    def with_ranked(res):
        return {**res, "per_query": [{**r, "ranked": []} for r in res["per_query"]]}

    results = iter([with_ranked(ROW_A), with_ranked(ROW_B)])
    monkeypatch.setattr(ablation, "load_reference_date", lambda: date(2026, 10, 18))
    monkeypatch.setattr(ablation, "load_queries", lambda split: [])
    monkeypatch.setattr(ablation, "load_qrels", lambda split, allow_test=False: {})
    monkeypatch.setattr(ablation, "load_chunk_info", lambda *a: {})
    monkeypatch.setattr(ablation, "load_excluded", lambda: set())
    monkeypatch.setattr(ablation, "load_retriever", lambda spec: None)
    monkeypatch.setattr(run_eval, "evaluate", lambda *a, **k: next(results))
    rc = ablation.run([["rowA", "x:y"], ["rowB", "x:y"]], "dev", tmp_path, "mystem")
    assert rc == 0
    assert (tmp_path / "mystem_dev.md").exists() and (tmp_path / "mystem_dev.csv").exists()
    assert not (tmp_path / "rowB_dev.md").exists()
    assert (tmp_path / "dev_rowA.json").exists() and (tmp_path / "dev_rowB.json").exists()
